import data_parse
import trainer
import model
from datasets import load_dataset
import numpy as np
import threading
import time
import sys
import os
import gc
from tensorflow import keras

training_active = False
stop_event = threading.Event()
current_accuracy = 0.0
worker_thread = None
train_iterator = None
test_iterator = None
pair_buffer = []
test_pair_buffer = []
model_lock = threading.Lock()

model_path = "transformer_model_final.keras"
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    try:
        m = keras.models.load_model(model_path, custom_objects={
            "TokenAndPositionEmbedding": model.TokenAndPositionEmbedding
        })
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Building new model...")
        m = model.build_transformer_model()
else:
    print("No saved model found. Building new model...")
    m = model.build_transformer_model()

def generate_response(input_text):
    data_parse.load_vocab()
    
    start_token_id = data_parse.get_special_token_id("<TALK_START>")
    end_token_id = data_parse.get_special_token_id("<TALK_END>")
    pad_token_id = data_parse.get_special_token_id("<PAD>")

    tokenized_seq = data_parse.tonkenizer([input_text])[0]
    
    enc_pad_len = model.MAX_LENGTH - len(tokenized_seq)
    
    if enc_pad_len < 0:
        enc_in = tokenized_seq[-model.MAX_LENGTH:]
    else:
        enc_in = tokenized_seq + [pad_token_id] * enc_pad_len
        
    encoder_input = np.array([enc_in])
    
    output_seq = [start_token_id]
    
    for _ in range(model.MAX_LENGTH):
        dec_pad_len = model.MAX_LENGTH - len(output_seq)
        if dec_pad_len < 0:
            break
            
        decoder_input = output_seq + [pad_token_id] * dec_pad_len
        dec_in_array = np.array([decoder_input])
        
        with model_lock:
            preds = m([encoder_input, dec_in_array], training=False)
            preds = preds.numpy()
        
        current_step_idx = len(output_seq) - 1
        next_token_probs = preds[0, current_step_idx, :]
        next_token = np.argmax(next_token_probs)
        
        if next_token == end_token_id:
            break
        
        output_seq.append(next_token)

    return data_parse.decode(output_seq[1:])

import re

def parse_chatml(text):
    parts = text.split("<|im_start|>")
    user_msg = ""
    assistant_msg = ""
    
    for part in parts:
        if part.startswith("user"):
            user_msg = part.replace("user\n", "").replace("<|im_end|>\n", "").replace("<|im_end|>", "").strip()
        elif part.startswith("assistant"):
            raw = part.replace("assistant\n", "").replace("<|im_end|>\n", "").replace("<|im_end|>", "").strip()
            assistant_msg = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            
    if user_msg and assistant_msg:
        return [(user_msg, assistant_msg)]
    return []


import random

def get_train_batch(limit=100):
    global train_iterator, pair_buffer
    
    # Initialize / Load data if empty
    if not pair_buffer and train_iterator is None:
        convs = data_parse.load_all_conversations("data")
        all_pairs = data_parse.generate_pairs(convs)
        random.shuffle(all_pairs)
        pair_buffer = all_pairs
    
    batch = []
    
    global all_training_pairs
    if 'all_training_pairs' not in globals() or not all_training_pairs:
        convs = data_parse.load_all_conversations("data")
        all_training_pairs = data_parse.generate_pairs(convs)
        random.shuffle(all_training_pairs)
        pair_buffer = list(all_training_pairs)
    
    if not pair_buffer:
         random.shuffle(all_training_pairs)
         pair_buffer = list(all_training_pairs)
         
    take = min(len(pair_buffer), limit)
    batch = pair_buffer[:take]
    pair_buffer = pair_buffer[take:]
        
    return batch

def training_cycle():
    global current_accuracy, stop_event, training_active, m
    cycle_count = 0
    
    while not stop_event.is_set():
        cycle_count += 1
        pairs = get_train_batch(limit=100) # Returns (input, target) list
        if not pairs:
            continue
            
        inputs = [p[0] for p in pairs]
             
        targets = [p[1] for p in pairs]

        tokenized_inputs = data_parse.tonkenizer(inputs)
        tokenized_targets = data_parse.tonkenizer(targets)
        
        tokenized_pairs = list(zip(tokenized_inputs, tokenized_targets))
        
        with model_lock:
            t_start = data_parse.get_special_token_id("<TALK_START>")
            t_end = data_parse.get_special_token_id("<TALK_END>")
            t_pad = data_parse.get_special_token_id("<PAD>")
            
            trainer.train_pairs(
                tokenized_pairs, 
                model=m,
                start_token_id=t_start,
                end_token_id=t_end,
                pad_token_id=t_pad
            )
        
        acc = test(samples=50)
        current_accuracy = acc
        
        gc.collect()

        sleep_time = 60 if cycle_count % 20 == 0 else 10
        if cycle_count % 20 == 0:
            data_parse.clear_memory()
            
        if stop_event.wait(timeout=sleep_time):
            break
    
    with model_lock:
        m.save("transformer_model_final.keras")
    training_active = False

def start_training():
    global training_active, stop_event, worker_thread
    if training_active:
        return

    stop_event.clear()
    training_active = True
    worker_thread = threading.Thread(target=training_cycle)
    worker_thread.daemon = True
    worker_thread.start()

def stop_training():
    global stop_event
    if not training_active:
        return
    stop_event.set()

def get_current_accuracy():
    return current_accuracy

def test(samples=50):
    global test_iterator, test_pair_buffer
    if test_iterator is None:
        dataset = load_dataset("Bossologist/reddit-conversations-processed", split='train', streaming=True)
        test_iterator = iter(dataset)
    
    pairs = []
    if test_pair_buffer:
        take = min(len(test_pair_buffer), samples)
        pairs.extend(test_pair_buffer[:take])
        test_pair_buffer = test_pair_buffer[take:]
    
    try:
        while len(pairs) < samples:
            example = next(test_iterator)
            text = example['text']
            new_pairs = parse_chatml(text)

            needed = samples - len(pairs)
            if len(new_pairs) > needed:
                pairs.extend(new_pairs[:needed])
                test_pair_buffer.extend(new_pairs[needed:])
                break
            else:
                pairs.extend(new_pairs)

    except StopIteration:
         dataset = load_dataset("Bossologist/reddit-conversations-processed", split='train', streaming=True)
         test_iterator = iter(dataset)
    
    if not pairs:
        return 0.0

    inputs = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]

    tokenized_inputs = [] 
    for inp in inputs:
        tokenized_inputs.append(data_parse.create_context(inp))
        
    tokenized_targets = data_parse.tonkenizer(targets)
    
    pad_token_id = data_parse.get_special_token_id("<PAD>")
    start_token_id = data_parse.get_special_token_id("<TALK_START>")
    end_token_id = data_parse.get_special_token_id("<TALK_END>")
    
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    
    for input_seq, target_seq in zip(tokenized_inputs, tokenized_targets):
        
        enc_pad_len = model.MAX_LENGTH - len(input_seq)
        if enc_pad_len < 0:
            input_seq = input_seq[-model.MAX_LENGTH:]
            enc_pad_len = 0
        enc_in = input_seq + [pad_token_id] * enc_pad_len
        
        target_seq_chopped = target_seq[:model.MAX_LENGTH-1] # reserve 1 for START/END
        
        dec_pad_len = model.MAX_LENGTH - (len(target_seq_chopped) + 1)
        if dec_pad_len < 0: dec_pad_len = 0 
        
        dec_in = [start_token_id] + target_seq_chopped + [pad_token_id] * dec_pad_len
        
        target_seq_for_loss = target_seq_chopped + [end_token_id]
        target = target_seq_for_loss + [pad_token_id] * dec_pad_len
        
        encoder_inputs.append(enc_in)
        decoder_inputs.append(dec_in)
        decoder_targets.append(target)
        
    encoder_inputs = np.array(encoder_inputs)
    decoder_inputs = np.array(decoder_inputs)
    decoder_targets = np.array(decoder_targets)
    
    with model_lock:
        results = m.evaluate([encoder_inputs, decoder_inputs], decoder_targets, verbose=0)
    
    accuracy = results[1]
    return accuracy

if __name__ == "__main__":
    try:
        data_parse.load_vocab()
        optimizer = keras.optimizers.Adam(learning_rate=trainer.LEARNING_RATE)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        m.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
        training_cycle()
    finally:
        test_iterator = None
        train_iterator = None
        gc.collect()