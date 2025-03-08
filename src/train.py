import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, RMSprop
import numpy as np
from tqdm import tqdm
from tokenizer import FastTokenizer
from summarizer import AbstractiveSummarizer
from torch.cuda.amp import autocast, GradScaler


EPOCHS = 5
BATCH_SIZE = 64
MODEL_NAME = 'second_model'
TRAIN_SIZE = 0.010

train_files = {'src': './data/train.txt.src', 'tgt': './data/train.txt.tgt'}
val_files = {'src': './data/val.txt.src', 'tgt': './data/val.txt.tgt'}
test_files = {'src': './data/test.txt.src'}

class SummarizationDataset(Dataset):
    def __init__(self, articles_path, summaries_path, tokenizer):
        self.articles = open(articles_path).readlines()
        self.articles = self.articles[:int(len(self.articles) * TRAIN_SIZE)]
        self.summaries = open(summaries_path).readlines()
        self.summaries = self.summaries[:int(len(self.summaries) * TRAIN_SIZE)]
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.articles)
        
    def __getitem__(self, idx):
        article = self.tokenizer.encode(self.articles[idx], max_length=512, truncation=True)
        summary = self.tokenizer.encode(self.summaries[idx], max_length=128, truncation=True)
        return {'article': torch.tensor(article), 'summary': torch.tensor(summary)}

def pad_sequence(sequences, padding_value=0):
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        padded = torch.full((max_len,), padding_value, dtype=torch.long)
        padded[:len(seq)] = torch.tensor(seq)
        padded_seqs.append(padded)
    return torch.stack(padded_seqs)

def collate_fn(batch):
    articles = [item['article'] for item in batch]
    summaries = [item['summary'] for item in batch]
    
    # Pad sequences
    articles_padded = pad_sequence(articles)
    summaries_padded = pad_sequence(summaries)
    
    return {
        'article': articles_padded,
        'summary': summaries_padded
    }

def train_epoch(model, dataloader, optimizer, criterion, device, gradient_accumulation_steps=4):
    model.train()
    total_loss = 0
    scaler = GradScaler()
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader)):
        src = batch['article'].to(device)
        tgt = batch['summary'].to(device)
        
        with autocast():
            output = model(src, tgt[:, :-1])
            output = output.contiguous().view(-1, output.size(-1))
            target = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, target) / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['article'].to(device)
            tgt = batch['summary'].to(device)
            
            output = model(src, tgt[:, :-1], teacher_forcing_ratio=0.0)
            output = output.contiguous().view(-1, output.size(-1))
            target = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def inference(model, article, tokenizer, device, max_length=128):
    model.eval()
    with torch.no_grad():
        src = tokenizer.encode(article, max_length=512, truncation=True)
        src = torch.tensor(src).unsqueeze(0).to(device)
        
        # Initialize with SOS token
        output_sequence = [tokenizer.bos_token_id]
        
        for _ in range(max_length):
            tgt = torch.tensor([output_sequence]).to(device)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            next_token = output[0, -1].argmax().item()
            
            if next_token == tokenizer.eos_token_id:
                break
                
            output_sequence.append(next_token)
            
    return tokenizer.decode(output_sequence)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab_path = './data/vocab.json'
    tokenizer = FastTokenizer()
    
    if os.path.exists(vocab_path):
        print("Loading existing vocabulary...")
        tokenizer.load_vocab(vocab_path)
    else:
        print("Building vocabulary...")
        train_articles = open(train_files['src']).readlines()
        train_summaries = open(train_files['tgt']).readlines()
        tokenizer.build_vocab(train_articles + train_summaries)
        tokenizer.save_vocab(vocab_path)
    
    # Rest of the code remains same, just replace BertTokenizer with SpacyTokenizer
    model = AbstractiveSummarizer(
        vocab_size=tokenizer.vocab_size,
        embed_size=256,
        hidden_size=512
    ).to(device)
    
    # Data loading
    train_dataset = SummarizationDataset(
        train_files['src'],
        train_files['tgt'],
        tokenizer
    )
    val_dataset = SummarizationDataset(
        val_files['src'],
        val_files['tgt'],
        tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn
    )
    
    # Training setup
    optimizer = RMSprop(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in tqdm(range(EPOCHS), desc='Training'):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{MODEL_NAME}.pt')
            
   

if __name__ == '__main__':
    main() 