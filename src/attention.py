import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        
    def forward(self, query, encoder_outputs):
        # query: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size]
        
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Calculate attention scores
        query_hidden = self.W1(query)  # [batch_size, 1, hidden_size]
        encoder_hidden = self.W2(encoder_outputs)  # [batch_size, src_len, hidden_size]
        
        # Broadcast query to match encoder outputs
        energy = torch.tanh(query_hidden + encoder_hidden)  # [batch_size, src_len, hidden_size]
        attention = self.V(energy).squeeze(-1)  # [batch_size, src_len]
        
        return F.softmax(attention, dim=1).unsqueeze(1)  # [batch_size, 1, src_len] 