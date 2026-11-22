import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.silu import silu
from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w13_weight = Linear(d_model, 2*d_ff, device, dtype)
        self.w2_weight = Linear(d_ff, d_model, device, dtype)

    def forward(self, x):
        gate, up = rearrange(self.w13_weight(x), "... (two d_ff) -> two ... d_ff", two=2).unbind(0)
        return self.w2_weight( silu(gate) * up)
