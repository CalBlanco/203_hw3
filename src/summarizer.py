import torch
import torch.nn as nn
from encoder import Encoder
from attention import BahdanauAttention
from decoder import Decoder

class AbstractiveSummarizer(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()  # Important: call parent constructor
        self.encoder = Encoder(vocab_size, embed_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size)
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # Encoder
        encoder_outputs, hidden = self.encoder(src)
        
        # Decoder with attention
        outputs = []
        for t in range(tgt.size(1)):
            attn_weights = self.attention(hidden, encoder_outputs)
            output, hidden = self.decoder(tgt[:, t], hidden, encoder_outputs, attn_weights)
            outputs.append(output)
            
        return torch.stack(outputs, dim=1) 