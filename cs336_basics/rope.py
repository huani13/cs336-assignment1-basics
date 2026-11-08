import torch
import torch.nn as nn

class Rope(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        exponents = torch.arange(0, d_k, 2, device=device).float() / d_k
        inv_freq = 1.0 / (theta ** exponents)

        positions = torch.arange(max_seq_len, device=device).float()
        angles = positions[:,None] * inv_freq[None, :]

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos

        return torch.stack((y_even, y_odd), dim=-1).flatten(-2)
