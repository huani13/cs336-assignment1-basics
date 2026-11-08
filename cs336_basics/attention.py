import torch
import torch.nn as nn
from cs336_basics.softmax import softmax
from einops import einsum

def attention(Q, K, V, mask):
    dk = K.size()[-1]
    QKT = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scaled_QKT = QKT / dk**0.5
    scaled_mask_QKT = scaled_QKT + torch.where(mask, 0.0, float("-inf"))
    return einsum(softmax(scaled_mask_QKT, -1), V, "... queries keys, ... keys d_v -> ... queries d_v")
