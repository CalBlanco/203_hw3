import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, input, hidden, encoder_outputs, attn_weights):
        # input: [batch_size]
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size]
        # attn_weights: [batch_size, 1, src_len]
        
        # Reshape hidden for LSTM which expects [num_layers, batch, hidden_size]
        hidden_reshaped = (
            hidden.unsqueeze(0),  # [1, batch_size, hidden_size]
            torch.zeros_like(hidden).unsqueeze(0)  # Initial cell state
        )
        
        embedded = self.embedding(input.unsqueeze(1))  # [batch_size, 1, embed_size]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_size]
        
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden_n, cell_n) = self.lstm(lstm_input, hidden_reshaped)
        
        # Reshape hidden back to [batch_size, hidden_size]
        hidden_out = hidden_n.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        return prediction, hidden_out 