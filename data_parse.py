import os
import threading
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from datasets import load_dataset
import model

# Global tokenizer instance
tokenizer = None
tokenizer_lock = threading.Lock()
VOCAB_FILE = "tokenizer.json"

# Special Tokens - Matching the user's previous convention where possible
# <TALK_START>=0, <TALK_END>=1, <THINK_START>=2, <THINK_END>=3, <PAD>=4
SPECIAL_TOKENS = ["<TALK_START>", "<TALK_END>", "<THINK_START>", "<THINK_END>", "<PAD>"]

def get_training_corpus():
    # Stream a small portion of the dataset for training the tokenizer
    try:
        raw_dataset = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
        for i, item in enumerate(raw_dataset):
            if i >= 2000: # Train on 2000 samples
                break
            yield item['text']
    except Exception as e:
        print(f"Error loading dataset for tokenizer training: {e}")
        yield "This is a fallback sentence to ensure tokenizer has something to train on."

def load_or_train_tokenizer():
    global tokenizer
    with tokenizer_lock:
        if tokenizer is not None:
            return tokenizer

        if os.path.exists(VOCAB_FILE):
             try:
                 tokenizer = Tokenizer.from_file(VOCAB_FILE)
             except Exception:
                 print("Failed to load tokenizer, retraining...")
                 tokenizer = None
        
        if tokenizer is None:
            print("Training BPE Tokenizer...")
            # Initialize BPE
            tokenizer = Tokenizer(models.BPE(unk_token="<PAD>")) 
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            tokenizer.decoder = decoders.ByteLevel()
            
            trainer = trainers.BpeTrainer(
                vocab_size=model.VOCAB_SIZE, 
                special_tokens=SPECIAL_TOKENS,
                show_progress=True
            )
            
            tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
            
            # Post-processing to automatically add Start/End tokens
            tokenizer.post_processor = processors.TemplateProcessing(
                single="<TALK_START> $A <TALK_END>",
                special_tokens=[
                    ("<TALK_START>", 0),
                    ("<TALK_END>", 1),
                ],
            )
            
            tokenizer.save(VOCAB_FILE)
            print("Tokenizer trained and saved.")
            
    return tokenizer

# Compatibility APIs
def load_vocab():
    load_or_train_tokenizer()

def save_vocab():
    pass

vocab_lock = tokenizer_lock 
vocab = {} 


def tonkenizer(texts):
    """
    Tokenizes a list of texts into list of lists of IDs.
    """
    global tokenizer
    if tokenizer is None:
        load_or_train_tokenizer()
        
    encoded_batch = tokenizer.encode_batch(texts)
    return [enc.ids for enc in encoded_batch]

def decode(ids):
    """
    Decodes a list of IDs back to string.
    """
    global tokenizer
    if tokenizer is None:
        load_or_train_tokenizer()
    return tokenizer.decode(ids, skip_special_tokens=True)

def get_special_token_id(token):
    global tokenizer
    if tokenizer is None:
        load_or_train_tokenizer()
    # If token exists, return id, else return padding id (4)
    id = tokenizer.token_to_id(token)
    return id if id is not None else 4


async def getMessage(message):

    await message.channel.send(f'Received your message: {message.content}')
    return True