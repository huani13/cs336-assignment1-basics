import torch

def softmax(x, dim):
    shifted = x-torch.max(x, dim=dim, keepdim=True).values
    exponents = torch.exp(shifted)
    return exponents / exponents.sum(dim=dim, keepdim=True)
