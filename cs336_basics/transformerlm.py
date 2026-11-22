import torch
import torch.nn as nn

from cs336_basics.embedding import Embedding
from cs336_basics.rope import Rope
from cs336_basics.transformerblock import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                    device=None, dtype=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        rope = Rope(rope_theta, d_model // num_heads, context_length, device=device)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        )
        self.final_layer_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        positions = torch.arange(token_ids.shape[-1], device=token_ids.device)
        for block in self.transformer_blocks:
            x = block(x, positions)
        return self.output(self.final_layer_norm(x))
