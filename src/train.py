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


import argparse

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (Default: 5)')
parser.add_argument('--batch_size', type=int, default=64, help='Training batch size (Default: 64)')
parser.add_argument('--model_name', type=str, default='model_weights', help='Name of the model (Default: model_weights)')
parser.add_argument('--train_size', type=float, default=0.010, help='Fraction of training data to use (Default: 0.010)')
parser.add_argument('--test_file', type=str, default='./data/test.txt.src', help='Path to the test file (Default: ./data/test.txt.src)')
parser.add_argument('--output_file', type=str, default='predictions.txt', help='Path to the output file (Default: predictions.txt)')
parser.add_argument('--test_size', type=float, default=1.0, help='Fraction of test data to use (Default: 1.0)')

args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
MODEL_NAME = args.model_name
TRAIN_SIZE = args.train_size
TEST_SIZE = args.test_size
TEST_FILE = args.test_file
OUTPUT_FILE = args.output_file

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


def batch_inference(model, articles, tokenizer, device, batch_size=16, max_length=128):
    model.eval()
    all_summaries = []
    
    # Process in batches
    for i in tqdm(range(0, len(articles), batch_size), desc="Generating summaries"):
        batch_articles = articles[i:i+batch_size]
        
        # Encode and pad all articles in the batch
        encoded_articles = [tokenizer.encode(article, max_length=512, truncation=True) for article in batch_articles]
        src = pad_sequence(encoded_articles).to(device)
        
        batch_summaries = []
        with torch.no_grad():
            # Initialize with start tokens for each article in batch
            batch_output_sequences = [[0] for _ in range(len(batch_articles))]
            
            # Generate tokens step by step
            for _ in range(max_length):
                # Prepare current output sequences
                tgt_list = [torch.tensor(seq) for seq in batch_output_sequences]
                tgt = pad_sequence(tgt_list).to(device)
                
                # Get model predictions
                output = model(src, tgt, teacher_forcing_ratio=0.0)
                next_tokens = output[:, -1].argmax(dim=1).cpu().tolist()
                
                # Check for end tokens and append next tokens
                finished = [False] * len(batch_articles)
                for j, token in enumerate(next_tokens):
                    if hasattr(tokenizer, 'eos_token_id') and token == tokenizer.eos_token_id:
                        finished[j] = True
                    else:
                        batch_output_sequences[j].append(token)
                
                # Stop if all sequences have reached EOS
                if all(finished):
                    break
        
        # Decode all sequences in the batch
        for seq in batch_output_sequences:
            all_summaries.append(tokenizer.decode(seq))
    
    return all_summaries


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
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


    
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test file not found at {TEST_FILE}")
    
    test_articles = open(TEST_FILE).readlines()
    test_articles = test_articles[:int(len(test_articles) * TEST_SIZE)] #sub setting because gpu resources are contested like Afghanistan
    batch_size = 64 #inference batch size
    print(f"Generating summaries in batches of {batch_size}...")
    
    predictions = batch_inference(model, test_articles, tokenizer, device, batch_size=batch_size)
    
    print(f"Saving predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(predictions))
    
    print(f"Successfully generated {len(predictions)} summaries!")
            
   

if __name__ == '__main__':
    main() 