import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import build_transformer_model, MAX_LENGTH

# --- Training Parameters ---
LEARNING_RATE = 0.001    # Learning rate for optimizer
BATCH_SIZE = 64          # Batch size for training
EPOCHS = 10              # Number of training epochs

def train_transformer(encoder_input_data, decoder_input_data, decoder_target_data, model=None, save_path="transformer_model.h5"):
    """
    Training algorithm for the transformer.
    
    Args:
        encoder_input_data: Numpy array (samples, seq_len)
        decoder_input_data: Numpy array (samples, seq_len)
        decoder_target_data: Numpy array (samples, seq_len)
        model: Optional pre-existing keras model. If None, a new one is built.
        save_path: Path to save the trained model.
    """
    if model is None:
        model = build_transformer_model()
    
    # Compile (or re-compile) the model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    
    history = model.fit(
        x=[encoder_input_data, decoder_input_data],
        y=decoder_target_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1
    )
    
    model.save(save_path)
    
    return model, history

def pretrain_autoencoder(tokenized_texts, start_token_id=0, end_token_id=1, pad_token_id=4):
    """
    Pretrains the model to output readable text by learning to reconstruct input text.
    (Auto-Encoder Strategy)
    
    Args:
        tokenized_texts: List of lists/arrays containing token IDs for sentences.
        start_token_id: ID used for <START> (0)
        end_token_id: ID used for <END> (1)
        pad_token_id: ID for padding (4)
    """
    
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    
    for seq in tokenized_texts:
        # Truncate if necessary (keeping space for start/end tokens)
        seq = list(seq[:MAX_LENGTH-2])
        
        # Encoder Input: The raw sequence (padded)
        enc_pad_len = MAX_LENGTH - len(seq)
        enc_in = seq + [pad_token_id] * enc_pad_len
        
        # Decoder Input: <START> + sequence (padded)
        dec_in = [start_token_id] + seq + [pad_token_id] * (enc_pad_len - 1 if enc_pad_len > 0 else 0)
        # Ensure length is exactly MAX_LENGTH
        dec_in = dec_in[:MAX_LENGTH]
        
        # Decoder Target: sequence + <END> (padded)
        target = seq + [end_token_id] + [pad_token_id] * (enc_pad_len - 1 if enc_pad_len > 0 else 0)
        # Ensure length is exactly MAX_LENGTH
        target = target[:MAX_LENGTH]
        
        encoder_inputs.append(enc_in)
        decoder_inputs.append(dec_in)
        decoder_targets.append(target)
        
    encoder_inputs = np.array(encoder_inputs)
    decoder_inputs = np.array(decoder_inputs)
    decoder_targets = np.array(decoder_targets)
    
    # Train and save to a specific pretrain file
    return train_transformer(encoder_inputs, decoder_inputs, decoder_targets, save_path="pretrained_model.h5", model=model)

