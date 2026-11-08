import torch
import torch.nn as nn
from cs336_basics.multiheadattention import MultiHeadAttention
from cs336_basics.feedforward import FeedForward
from cs336_basics.rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, rope)
        self.feedforward = FeedForward(d_model, d_ff)
        self.rms_norm_1 = RMSNorm(d_model)
        self.rms_norm_2 = RMSNorm(d_model)

    def forward(self, x, token_positions=None):
        f1 = x + self.multi_head_attention(self.rms_norm_1(x), token_positions)
        return f1 + self.feedforward(self.rms_norm_2(f1))
