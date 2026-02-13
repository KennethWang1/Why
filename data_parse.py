import os
import threading
import json
import glob
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

SPECIAL_TOKENS = ["<TALK_START>", "<TALK_END>", "<THINK_START>", "<THINK_END>", "<PAD>"]

class RAGStore:
    def __init__(self, max_entries=500):
        self.index = faiss.IndexFlatL2(384)
        self.stored_tokens = []
        self.max_entries = max_entries

    def add(self, text, encoder):
        embedding = encoder.encode(text)
        embedding = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(embedding)

        if self.index.ntotal > 0:
            D, _ = self.index.search(embedding, k=1)
            if D[0][0] < 0.05:
                return False

        if self.index.ntotal >= self.max_entries:
            self._evict_oldest()

        self.index.add(embedding)
        self.stored_tokens.append(tonkenizer([text])[0])
        return True

    def create_context(self, message, encoder, max_context_tokens=126):
        query_vec = encoder.encode([message])
        faiss.normalize_L2(query_vec)

        tokenized_query = tonkenizer([message])[0]
        available = max_context_tokens - len(tokenized_query)
        context_tokens = []

        if self.index.ntotal > 0:
            D, I = self.index.search(query_vec, k=min(5, self.index.ntotal))
            for i, idx in enumerate(I[0]):
                if idx == -1 or D[0][i] > 1.2:
                    continue
                retrieved = self.stored_tokens[idx]
                if available - len(retrieved) < 0:
                    continue
                context_tokens.extend(retrieved)
                available -= len(retrieved)

        final_input = context_tokens + tokenized_query
        if len(final_input) > 256:
            final_input = final_input[-256:]
        return final_input

    def clear(self):
        self.index = faiss.IndexFlatL2(384)
        self.stored_tokens.clear()

    def _evict_oldest(self):
        keep_count = self.max_entries // 2
        entries_to_keep = self.stored_tokens[-keep_count:]

        new_index = faiss.IndexFlatL2(384)
        for i in range(self.index.ntotal - keep_count, self.index.ntotal):
            vec = self.index.reconstruct(i)
            new_index.add(np.array([vec], dtype=np.float32))

        self.index = new_index
        self.stored_tokens = entries_to_keep


chat_rag = RAGStore(max_entries=500)
train_rag = RAGStore(max_entries=500)

def get_training_corpus():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    files = glob.glob(os.path.join(data_dir, "*.json"))

    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for msg in data.get('messages', []):
                content = msg.get('content', '').strip()
                if content and not msg.get('author', {}).get('isBot', False):
                    yield content
        except Exception as e:
            print(f"Error reading {filepath} for tokenizer training: {e}")

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

def add_embeddings(text):
    return chat_rag.add(text, m)

def create_context(message):
    return chat_rag.create_context(message, m)

def clear_chat_memory():
    chat_rag.clear()
    print("Chat RAG memory cleared.")

def add_train_embedding(text):
    return train_rag.add(text, m)

def create_train_context(message):
    return train_rag.create_context(message, m)

def clear_memory():
    train_rag.clear()
    print("Train RAG memory cleared.")

def load_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = data.get('messages', [])
    
    conversation = []
    
    for msg in messages:
        if not msg.get('content'):
            continue
            
        author = msg.get('author', {}).get('name', 'Unknown')
        content = msg.get('content', '').strip()
        
        if content:
            conversation.append(f"{author}: {content}")
            
    return conversation

def load_all_conversations(data_dir):
    all_conversations = []
    files = glob.glob(os.path.join(data_dir, "*.json"))
    
    for f in files:
        try:
            conv = load_from_json(f)
            if conv:
                all_conversations.append(conv)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return all_conversations

def generate_pairs(conversations, window_size=1):
    pairs = []
    for conv in conversations:
        
        merged_conv = []
        if not conv: continue
        
        current_msg = conv[0]
        current_author = current_msg.split(':')[0]
        
        for i in range(1, len(conv)):
            msg = conv[i]
            author = msg.split(':')[0]
            content = ":".join(msg.split(':')[1:]).strip()
            
            if author == current_author:
                current_msg += " \n " + content
            else:
                merged_conv.append(current_msg)
                current_msg = msg
                current_author = author
        merged_conv.append(current_msg)
        
        for i in range(len(merged_conv) - 1):
            target_msg = merged_conv[i+1]
            if ':' in target_msg:
                target_content = target_msg.split(':', 1)[1].strip()
            else:
                target_content = target_msg
                
            pairs.append((merged_conv[i], target_content))
            
    return pairs
