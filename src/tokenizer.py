from collections import Counter
import re
from typing import List
import json
import os

class FastTokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
    def preprocess(self, text: str) -> str:
        # Basic cleaning
        text = text.lower().strip()
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text
        
    def build_vocab(self, texts: List[str]):
        word_freqs = Counter()
        for text in texts:
            text = self.preprocess(text)
            words = text.split()
            word_freqs.update(words)
            
        # Add most common words to vocabulary
        for word, _ in word_freqs.most_common(self.vocab_size - 4):  # -4 for special tokens
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        self.vocab_size = len(self.word2idx)
        
    def save_vocab(self, path: str):
        vocab_data = {
            'vocab_size': self.vocab_size,
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()}  # Convert int keys to str for JSON
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            
    def load_vocab(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            
        self.vocab_size = vocab_data['vocab_size']
        self.word2idx = vocab_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}  # Convert str keys back to int
        
    def encode(self, text: str, max_length: int = None, truncation: bool = True) -> List[int]:
        text = self.preprocess(text)
        tokens = text.split()
        
        if max_length and truncation and len(tokens) > max_length - 2:  # -2 for sos/eos
            tokens = tokens[:max_length-2]
            
        ids = [self.sos_token_id]
        ids.extend([self.word2idx.get(token, self.unk_token_id) for token in tokens])
        ids.append(self.eos_token_id)
        
        return ids
        
    def decode(self, ids: List[int]) -> str:
        tokens = [self.idx2word[idx] for idx in ids]
        # Remove special tokens
        tokens = [t for t in tokens if t not in ['<pad>', '<sos>', '<eos>']]
        return ' '.join(tokens) 