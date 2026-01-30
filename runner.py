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

def get_train_batch(limit=100):
    global train_iterator
    if train_iterator is None:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split='train', streaming=True)
        train_iterator = iter(dataset)
    
    texts = []
    try:
        for _ in range(limit):
            example = next(train_iterator)
            text = example['text']
            if text.strip():
                texts.append(text)
    except StopIteration:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split='train', streaming=True)
        train_iterator = iter(dataset)
        
    return texts

def training_cycle():
    global current_accuracy, stop_event, training_active, m
    cycle_count = 0
    
    while not stop_event.is_set():
        cycle_count += 1
        texts = get_train_batch(limit=100)
        if not texts:
            continue
            
        tokenized_texts = data_parse.tonkenizer(texts)
        
        with model_lock:
            t_start = data_parse.get_special_token_id("<TALK_START>")
            t_end = data_parse.get_special_token_id("<TALK_END>")
            t_pad = data_parse.get_special_token_id("<PAD>")
            
            trainer.pretrain_autoencoder(
                tokenized_texts, 
                model=m,
                start_token_id=t_start,
                end_token_id=t_end,
                pad_token_id=t_pad
            )
        
        acc = test(samples=50)
        current_accuracy = acc
        
        gc.collect()

        sleep_time = 300 if cycle_count % 10 == 0 else 60
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
    global test_iterator
    if test_iterator is None:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split='validation', streaming=True)
        test_iterator = iter(dataset)
    
    texts = []
    count = 0
    try:
        for _ in range(samples * 2):
            example = next(test_iterator)
            text = example['text']
            if text.strip():
                texts.append(text)
                count += 1
            if count >= samples:
                break
    except StopIteration:
         dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split='validation', streaming=True)
         test_iterator = iter(dataset)
    
    tokenized_texts = data_parse.tonkenizer(texts)
    
    pad_token_id = data_parse.get_special_token_id("<PAD>")
    start_token_id = data_parse.get_special_token_id("<TALK_START>")
    end_token_id = data_parse.get_special_token_id("<TALK_END>")
    
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    
    for seq in tokenized_texts:
        seq = list(seq[-(model.MAX_LENGTH-2):])
        enc_pad_len = model.MAX_LENGTH - len(seq)
        enc_in = seq + [pad_token_id] * enc_pad_len
        
        dec_in_pad = enc_pad_len - 1 if enc_pad_len > 0 else 0
        dec_in = [start_token_id] + seq + [pad_token_id] * dec_in_pad
        dec_in = dec_in[:model.MAX_LENGTH]
        
        target = seq + [end_token_id] + [pad_token_id] * dec_in_pad
        target = target[:model.MAX_LENGTH]
        
        encoder_inputs.append(enc_in)
        decoder_inputs.append(dec_in)
        decoder_targets.append(target)
        print(dec_in)
        
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
        print(test(samples = 50))
    finally:
        test_iterator = None
        train_iterator = None
        gc.collect()