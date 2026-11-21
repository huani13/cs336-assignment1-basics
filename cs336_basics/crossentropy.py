import torch

def cross_entropy(logits, targets):
    shifted = logits - torch.max(logits, dim=-1, keepdim=True).values
    denom = torch.exp(shifted).sum(dim=-1, keepdim=True)
    neg_log_prob = -(shifted - denom.log())
    return neg_log_prob.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).mean()
