import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, src):
        embedded = self.embedding(src)  # [batch, seq_len, embed_size]
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Combine bidirectional states [batch, seq_len, hidden_size*2]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = torch.tanh(self.fc(hidden))  # [batch, hidden_size]
        
        # Convert outputs to correct shape [batch, seq_len, hidden_size]
        outputs = self.fc(outputs)
        
        return outputs, hidden 