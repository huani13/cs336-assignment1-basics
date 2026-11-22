from einops import rearrange
import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.attention import attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        self.qkv = Linear(d_model, 3*d_model, device=device, dtype=dtype)
        self.proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope

    def forward(self, x, token_positions=None):

        seq = x.shape[-2]
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, "... seq (three h d_h) -> three ... h seq d_h",
            three = 3, h=self.num_heads
        ).unbind(0)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device)
            elif token_positions.ndim > 1:
                token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.ones(seq, seq, dtype=torch.bool, device=x.device).tril()
        out = attention(q, k, v, mask)
        out = rearrange(out, "... h seq d_h -> ... seq (h d_h)")
        return self.proj(out)
