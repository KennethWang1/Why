import re
import json
import os

vocab = {"<TALK_START>": 0, "<TALK_END>": 1, "<THINK_START>": 2, "<THINK_END>": 3, "<PAD>": 4}
TOKEN_PATTERN = re.compile(r'\w+|[^\w\s]')

def save_vocab(path='vocab.json'):
    global vocab
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=4)

def load_vocab(path='vocab.json'):
    global vocab
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
        except json.JSONDecodeError:
            vocab = {"<TALK_START>": 0, "<TALK_END>": 1, "<THINK_START>": 2, "<THINK_END>": 3, "<PAD>": 4}
    return True

def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())

def tonkenizer(texts):
    global vocab
    
    local_vocab = vocab
    start_token = local_vocab.get("<TALK_START>", 0)
    end_token = local_vocab.get("<TALK_END>", 1)
    next_id = len(local_vocab)
    
    last_tokens = []
    
    for i, text in enumerate(texts):
        matches = TOKEN_PATTERN.findall(text.lower())
        
        ids = [start_token]
        
        for token in matches:
            if token in local_vocab:
                ids.append(local_vocab[token])
            else:
                local_vocab[token] = next_id
                ids.append(next_id)
                next_id += 1
                
        ids.append(end_token)
        
        texts[i] = ids
                
    return texts

def encode(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

async def getMessage(message):

    await message.channel.send(f'Received your message: {message.content}')
    return True