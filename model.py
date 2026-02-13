import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
except:
    pass

VOCAB_SIZE = 10000
MAX_LENGTH = 128
EMBED_DIM = 256    
NUM_HEADS = 4
FF_DIM = 1024     
NUM_ENCODER_LAYERS = 6 
NUM_DECODER_LAYERS = 2 
DROPOUT_RATE = 0.1

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

def transformer_encoder_layer(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    x_ff = layers.Dense(ff_dim, activation="relu")(x)
    x_ff = layers.Dense(inputs.shape[-1])(x_ff)
    x_ff = layers.Dropout(dropout)(x_ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + x_ff)

def transformer_decoder_layer(inputs, enc_outputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs, use_causal_mask=True)
    x = layers.Dropout(dropout)(x)
    query = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(query, enc_outputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + query)

    x_ff = layers.Dense(ff_dim, activation="relu")(x)
    x_ff = layers.Dense(inputs.shape[-1])(x_ff)
    x_ff = layers.Dropout(dropout)(x_ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + x_ff)

def build_transformer_model():
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = TokenAndPositionEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(encoder_inputs)
    
    for _ in range(NUM_ENCODER_LAYERS):
        x = transformer_encoder_layer(x, EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT_RATE)
    
    encoder_outputs = x

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    x_dec = TokenAndPositionEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(decoder_inputs)
    
    for _ in range(NUM_DECODER_LAYERS):
        x_dec = transformer_decoder_layer(x_dec, encoder_outputs, EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT_RATE)

    decoder_outputs = layers.Dense(VOCAB_SIZE, activation="softmax", dtype="float32")(x_dec)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="transformer")
    return model

if __name__ == "__main__":
    model = build_transformer_model()
    model.summary()
