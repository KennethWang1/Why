import numpy as np
from tensorflow import keras
from model import build_transformer_model, MAX_LENGTH

LEARNING_RATE = 0.001
BATCH_SIZE = 12
EPOCHS = 4

def train_transformer(encoder_input_data, decoder_input_data, decoder_target_data, model=None, save_path="transformer_model.keras", sample_weight=None):
    if model is None:
        model = build_transformer_model()
    
    if not hasattr(model, "optimizer") or model.optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    
    history = model.fit(
        x=[encoder_input_data, decoder_input_data],
        y=decoder_target_data,
        sample_weight=sample_weight,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0
    )
    
    model.save(save_path)
    
    return model, history

def pretrain_autoencoder(tokenized_texts, model=None, start_token_id=0, end_token_id=1, pad_token_id=4):
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    
    for seq in tokenized_texts:
        if len(seq) > MAX_LENGTH:
             seq = seq[:MAX_LENGTH]
             
        enc_pad_len = MAX_LENGTH - len(seq)
        if enc_pad_len < 0: enc_pad_len = 0
        enc_in = seq + [pad_token_id] * enc_pad_len
        
        # seq is [START, content, END]
        # decoder inputs: [START, content] (remove last END)
        # decoder targets: [content, END] (remove first START)
        
        dec_in_seq = seq[:-1]
        dec_target_seq = seq[1:]
        
        dec_pad_len = MAX_LENGTH - len(dec_in_seq)
        if dec_pad_len < 0:
             dec_in_seq = dec_in_seq[:MAX_LENGTH]
             dec_target_seq = dec_target_seq[:MAX_LENGTH]
             dec_pad_len = 0
             
        dec_in = dec_in_seq + [pad_token_id] * dec_pad_len
        target = dec_target_seq + [pad_token_id] * dec_pad_len
        
        encoder_inputs.append(enc_in)
        decoder_inputs.append(dec_in)
        decoder_targets.append(target)
        
    encoder_inputs = np.array(encoder_inputs, dtype=np.int32)
    decoder_inputs = np.array(decoder_inputs, dtype=np.int32)
    decoder_targets = np.array(decoder_targets, dtype=np.int32)
    
    # Create sample weights: 0 for PAD tokens in target, 1 otherwise
    weights = np.where(decoder_targets == pad_token_id, 0.0, 1.0)
    
    return train_transformer(encoder_inputs, decoder_inputs, decoder_targets, save_path="pretrained_model.keras", model=model, sample_weight=weights)

def train_pairs(tokenized_pairs, model=None, start_token_id=0, end_token_id=1, pad_token_id=4):
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    
    for input_seq, target_seq in tokenized_pairs:
        # Encoder Input: just the input sequence
        if len(input_seq) > MAX_LENGTH:
            input_seq = input_seq[-MAX_LENGTH:] # Keep end of context
            
        enc_pad_len = MAX_LENGTH - len(input_seq)
        if enc_pad_len < 0: enc_pad_len = 0
            
        enc_in = input_seq + [pad_token_id] * enc_pad_len
        
        # Target Seq is [START, ..., END]
        dec_in_seq = target_seq[:-1]
        dec_target_seq = target_seq[1:]
        
        if len(dec_in_seq) > MAX_LENGTH:
             dec_in_seq = dec_in_seq[:MAX_LENGTH]
             dec_target_seq = dec_target_seq[:MAX_LENGTH]
             
        dec_pad_len = MAX_LENGTH - len(dec_in_seq)
        if dec_pad_len < 0: dec_pad_len = 0
        
        dec_in = dec_in_seq + [pad_token_id] * dec_pad_len
        target = dec_target_seq + [pad_token_id] * dec_pad_len
        
        encoder_inputs.append(enc_in)
        decoder_inputs.append(dec_in)
        decoder_targets.append(target)
        
    encoder_inputs = np.array(encoder_inputs, dtype=np.int32)
    decoder_inputs = np.array(decoder_inputs, dtype=np.int32)
    decoder_targets = np.array(decoder_targets, dtype=np.int32)
    
    # Create sample weights: 0 for PAD tokens in target, 1 otherwise
    weights = np.where(decoder_targets == pad_token_id, 0.0, 1.0)

    return train_transformer(encoder_inputs, decoder_inputs, decoder_targets, save_path="transformer_model_final.keras", model=model, sample_weight=weights)

