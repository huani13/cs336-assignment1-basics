import torch
import torch.nn as nn
from cs336_basics.multiheadattention import MultiHeadAttention
from cs336_basics.feedforward import FeedForward
from cs336_basics.rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.feedforward = FeedForward(d_model, d_ff, device=device, dtype=dtype)
        self.rms_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.multi_head_attention(self.rms_norm_1(x), token_positions)
        return x + self.feedforward(self.rms_norm_2(x))
