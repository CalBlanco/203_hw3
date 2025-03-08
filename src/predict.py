import os
import torch
from tqdm import tqdm
from tokenizer import FastTokenizer
from summarizer import AbstractiveSummarizer

def pad_sequence(sequences, padding_value=0):
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        padded = torch.full((max_len,), padding_value, dtype=torch.long)
        padded[:len(seq)] = torch.tensor(seq)
        padded_seqs.append(padded)
    return torch.stack(padded_seqs)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    vocab_path = './data/vocab.json'
    tokenizer = FastTokenizer()
    
    if os.path.exists(vocab_path):
        print("Loading vocabulary...")
        tokenizer.load_vocab(vocab_path)
    else:
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    
    # Load model
    model = AbstractiveSummarizer(
        vocab_size=tokenizer.vocab_size,
        embed_size=256,
        hidden_size=512
    ).to(device)
    
    # Load saved weights
    model_path = 'best_model.pt'
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    # Load test data
    test_file = './data/test.txt.src'
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found at {test_file}")
    
    test_articles = open(test_file).readlines()
    
    # Batch size for predictions
    batch_size = 16
    print(f"Generating summaries in batches of {batch_size}...")
    
    predictions = batch_inference(model, test_articles, tokenizer, device, batch_size=batch_size)
    
    # Save predictions
    output_file = 'predictions.txt'
    print(f"Saving predictions to {output_file}...")
    with open(output_file, 'w') as f:
        f.write('\n'.join(predictions))
    
    print(f"Successfully generated {len(predictions)} summaries!")

if __name__ == '__main__':
    main() 