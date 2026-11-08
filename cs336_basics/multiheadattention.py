from einops import rearrange
import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.attention import attention
from cs336_basics.rope import Rope

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None):
        super().__init__()
        self.queries = Linear(d_model, d_model)
        self.keys = Linear(d_model, d_model)
        self.values = Linear(d_model, d_model)
        self.proj = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_model = d_model
        self.rope = rope

    def forward(self, x, token_positions=None):
        Q = self.queries(x)
        K = self.keys(x)
        V = self.values(x)

        multihead_Q = rearrange(Q, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)
        multihead_K = rearrange(K, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)
        multihead_V = rearrange(V, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[1])
            multihead_Q = self.rope(multihead_Q, token_positions)
            multihead_K = self.rope(multihead_K, token_positions)

        mask = torch.triu(torch.ones(multihead_Q.shape[-2], multihead_Q.shape[-2])).transpose(0, 1).bool()
        multihead_att = attention(multihead_Q, multihead_K, multihead_V, mask)
        att = rearrange(multihead_att, "... h seq d_h -> ... seq (h d_h)")
        return self.proj(att)
