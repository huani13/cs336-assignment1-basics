import torch

def cross_entropy(logits, targets):
    m = logits.amax(dim=-1, keepdim=True)
    lse = (logits - m).exp().sum(dim=-1).log() + m.squeeze(-1)
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (lse - target_logits).mean()
