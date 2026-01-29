import data_parse
import trainer
import model
from datasets import load_dataset
import numpy as np
import threading
import time
import sys
import gc

# Global state
training_active = False
stop_event = threading.Event()
current_accuracy = 0.0
worker_thread = None
train_iterator = None
test_iterator = None
model_lock = threading.Lock()

m = model.build_transformer_model()

def generate_response(input_text):
    """
    Generates a response from the model based on the input text.
    Uses a Greedy Decode strategy.
    """
    # Ensure tokenizer is loaded
    data_parse.load_vocab()
    
    # Get special token IDs dynamically
    start_token_id = data_parse.get_special_token_id("<TALK_START>")
    end_token_id = data_parse.get_special_token_id("<TALK_END>")
    pad_token_id = data_parse.get_special_token_id("<PAD>")

    # Tokenize input
    tokenized_seq = data_parse.tonkenizer([input_text])[0]
    
    # Prepare Encoder Input (Pad to MAX_LENGTH)
    enc_pad_len = model.MAX_LENGTH - len(tokenized_seq)
    if enc_pad_len < 0:
        enc_in = tokenized_seq[:model.MAX_LENGTH]
    else:
        enc_in = tokenized_seq + [pad_token_id] * enc_pad_len
        
    encoder_input = np.array([enc_in])
    
    # Decoder Input Initialization
    output_seq = [start_token_id]
    
    # Greedy Generation Loop
    for _ in range(model.MAX_LENGTH):
        dec_pad_len = model.MAX_LENGTH - len(output_seq)
        if dec_pad_len < 0:
            break
            
        decoder_input = output_seq + [pad_token_id] * dec_pad_len
        dec_in_array = np.array([decoder_input])
        
        # Predict (Thread-safe)
        with model_lock:
            # preds = m.predict([encoder_input, dec_in_array], verbose=0)
            preds = m([encoder_input, dec_in_array], training=False)
            preds = preds.numpy()
        
        # Get prediction for the last valid token
        current_step_idx = len(output_seq) - 1
        next_token_probs = preds[0, current_step_idx, :]
        next_token = np.argmax(next_token_probs)
        
        if next_token == end_token_id:
            break
        
        output_seq.append(next_token)

    # Decode the generated sequence (excluding start token)
    return data_parse.decode(output_seq[1:])

def get_train_batch(limit=100):
    global train_iterator
    if train_iterator is None:
        dataset = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
        train_iterator = iter(dataset)
    
    texts = []
    try:
        for _ in range(limit):
            example = next(train_iterator)
            text = example['text']
            if text.strip(): # Skip empty lines
                texts.append(text)
    except StopIteration:
        # If dataset ends, reload
        dataset = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
        train_iterator = iter(dataset)
        
    return texts

def training_cycle():
    global current_accuracy, stop_event, training_active, m
    cycle_count = 0
    
    while not stop_event.is_set():
        cycle_count += 1
        # 1. Train Step
        texts = get_train_batch(limit=100)
        if not texts:
            continue
            
        tokenized_texts = data_parse.tonkenizer(texts)
        
        with model_lock:
            # Get dynamic token IDs ensure we match the tokenizer
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
        
        # 2. Test Step
        acc = test(samples=50)
        current_accuracy = acc
        
        # Explicit Garbage Collection to prevent memory creep
        gc.collect()

        # Pause logic: 5 minute break every 5 cycles, otherwise 20s
        if cycle_count % 5 == 0:
            time.sleep(300)
        else:
            time.sleep(45)
    
    # Final cleanup
    with model_lock:
        m.save("transformer_model_final.keras")
    training_active = False
    # Worker thread ends here naturally

def start_training():
    global training_active, stop_event, worker_thread
    if training_active:
        return

    stop_event.clear()
    training_active = True
    worker_thread = threading.Thread(target=training_cycle)
    worker_thread.daemon = True # Close if main program closes violently
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
        dataset = load_dataset('roneneldan/TinyStories', split='validation', streaming=True)
        test_iterator = iter(dataset)
    
    # print(f"Collecting {samples} test samples...")
    texts = []
    count = 0
    try:
        for _ in range(samples * 2): # Look at up to 2x samples to find non-empty ones
            example = next(test_iterator)
            text = example['text']
            if text.strip(): # Skip empty lines
                texts.append(text)
                count += 1
            if count >= samples:
                break
    except StopIteration:
         # Reload if iterator runs out
         dataset = load_dataset('roneneldan/TinyStories', split='validation', streaming=True)
         test_iterator = iter(dataset)
    
    # Tokenize
    tokenized_texts = data_parse.tonkenizer(texts)
    
    # Prepare Data Arrays (Same logic as trainer.pretrain_autoencoder)
    # Get dynamic token IDs
    pad_token_id = data_parse.get_special_token_id("<PAD>")
    start_token_id = data_parse.get_special_token_id("<TALK_START>")
    end_token_id = data_parse.get_special_token_id("<TALK_END>")
    
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    
    for seq in tokenized_texts:
        seq = list(seq[:model.MAX_LENGTH-2])
        enc_pad_len = model.MAX_LENGTH - len(seq)
        enc_in = seq + [pad_token_id] * enc_pad_len
        
        # Decoder Input: <START> + seq
        dec_in_pad = enc_pad_len - 1 if enc_pad_len > 0 else 0
        dec_in = [start_token_id] + seq + [pad_token_id] * dec_in_pad
        dec_in = dec_in[:model.MAX_LENGTH]
        
        # Target: seq + <END>
        target = seq + [end_token_id] + [pad_token_id] * dec_in_pad
        target = target[:model.MAX_LENGTH]
        
        encoder_inputs.append(enc_in)
        decoder_inputs.append(dec_in)
        decoder_targets.append(target)
        
    encoder_inputs = np.array(encoder_inputs)
    decoder_inputs = np.array(decoder_inputs)
    decoder_targets = np.array(decoder_targets)
    
    # Evaluate returns [loss, accuracy]
    with model_lock:
        results = m.evaluate([encoder_inputs, decoder_inputs], decoder_targets, verbose=0)
    
    accuracy = results[1]
    return accuracy

