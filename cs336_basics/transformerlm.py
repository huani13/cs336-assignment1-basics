import torch
import torch.nn as nn

from cs336_basics.embedding import Embedding
from cs336_basics.rope import Rope
from cs336_basics.transformerblock import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        rope = Rope(rope_theta, d_model // num_heads, context_length)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, rope) for i in range(num_layers)]
        )
        self.final_layer_norm = RMSNorm(d_model)
        self.output = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        return self.output(x)
