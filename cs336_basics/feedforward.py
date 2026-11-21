import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.silu import silu

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1_weight = Linear(d_model, d_ff, device, dtype)
        self.w2_weight = Linear(d_ff, d_model, device, dtype)
        self.w3_weight = Linear(d_model, d_ff, device, dtype)

    def forward(self, x):
        return self.w2_weight( silu(self.w1_weight(x)) * self.w3_weight(x))
