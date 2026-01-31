import os
import threading
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from datasets import load_dataset
import model
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

tokenizer = None
tokenizer_lock = threading.Lock()
VOCAB_FILE = "tokenizer.json"

index = faiss.IndexFlatL2(384)
stored_tokens_map = {}
entry_count = 0

SPECIAL_TOKENS = ["<TALK_START>", "<TALK_END>", "<THINK_START>", "<THINK_END>", "<PAD>"]

def get_training_corpus():
    try:
        raw_dataset = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
        for i, item in enumerate(raw_dataset):
            if i >= 2000:
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
            tokenizer = Tokenizer(models.BPE(unk_token="<PAD>")) 
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            tokenizer.decoder = decoders.ByteLevel()
            
            trainer = trainers.BpeTrainer(
                vocab_size=model.VOCAB_SIZE, 
                special_tokens=SPECIAL_TOKENS,
                show_progress=True
            )
            
            tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
            
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

def load_vocab():
    load_or_train_tokenizer()

vocab_lock = tokenizer_lock 
vocab = {} 


def tonkenizer(texts):
    global tokenizer
    if tokenizer is None:
        load_or_train_tokenizer()
        
    encoded_batch = tokenizer.encode_batch(texts)
    return [enc.ids for enc in encoded_batch]

def decode(ids):
    global tokenizer
    if tokenizer is None:
        load_or_train_tokenizer()
    return tokenizer.decode(ids, skip_special_tokens=True)

def get_special_token_id(token):
    global tokenizer
    if tokenizer is None:
        load_or_train_tokenizer()
    id = tokenizer.token_to_id(token)
    return id if id is not None else 4

#legacy
async def getMessage(message):
    #await message.channel.send(f'Received your message: {message.content}')
    return True

def add_embeddings(text):
    global m, index, stored_tokens_map, entry_count
    
    # Encode and normalize
    embedding = m.encode(text)
    faiss.normalize_L2(np.array([embedding])) # Verify length is not 1 by normalizing to 1
    
    # Add to FAISS
    index.add(np.array([embedding]))
    
    # Store text correspondence
    stored_tokens_map[entry_count] = tonkenizer([text])[0]
    entry_count += 1
    return True

def rag(message):
    global m
    message_embeddings = m.encode(message)
    return message_embeddings

def create_context(message):
    global m, index, stored_tokens_map
    
    query_vec = m.encode([message])
    faiss.normalize_L2(query_vec)
    
    if index.ntotal > 0:
        D, I = index.search(query_vec, k=min(5, index.ntotal))
        indices = I[0]
        distances = D[0]
    else:
        indices = []
        distances = []
    
    tokenized_query = tonkenizer([message])[0]
    start_token_id = get_special_token_id("<TALK_START>")
    
    available_tokens = 126 - len(tokenized_query)
    context_tokens = []

    for i, idx in enumerate(indices):
        if idx == -1: continue
        
        if distances[i] > 1.2: 
            continue
            
        retrieved_seq = stored_tokens_map[idx]
        
        if available_tokens - len(retrieved_seq) < 0:
            continue

        context_tokens.extend(retrieved_seq)
        available_tokens -= len(retrieved_seq)
    
    final_input = context_tokens + tokenized_query
    
    if len(final_input) > 128:
       final_input = final_input[-128:]
       
    return final_input