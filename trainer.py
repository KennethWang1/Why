import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import build_transformer_model, MAX_LENGTH

LEARNING_RATE = 0.001
BATCH_SIZE = 12
EPOCHS = 2

def train_transformer(encoder_input_data, decoder_input_data, decoder_target_data, model=None, save_path="transformer_model.keras"):
    if model is None:
        model = build_transformer_model()
    
    if not hasattr(model, "optimizer") or model.optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    
    history = model.fit(
        x=[encoder_input_data, decoder_input_data],
        y=decoder_target_data,
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
        seq = list(seq[-(MAX_LENGTH-2):])
        
        enc_pad_len = MAX_LENGTH - len(seq)
        enc_in = seq + [pad_token_id] * enc_pad_len
        
        dec_in = [start_token_id] + seq + [pad_token_id] * (enc_pad_len - 1 if enc_pad_len > 0 else 0)
        dec_in = dec_in[:MAX_LENGTH]
        
        target = seq + [end_token_id] + [pad_token_id] * (enc_pad_len - 1 if enc_pad_len > 0 else 0)
        target = target[:MAX_LENGTH]
        
        encoder_inputs.append(enc_in)
        decoder_inputs.append(dec_in)
        decoder_targets.append(target)
        
    encoder_inputs = np.array(encoder_inputs, dtype=np.int32)
    decoder_inputs = np.array(decoder_inputs, dtype=np.int32)
    decoder_targets = np.array(decoder_targets, dtype=np.int32)
    
    return train_transformer(encoder_inputs, decoder_inputs, decoder_targets, save_path="pretrained_model.keras", model=model)

