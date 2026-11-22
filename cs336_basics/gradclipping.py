import torch

@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    norms = torch._foreach_norm(grads)
    total_norm = torch.linalg.vector_norm(torch.stack([n.float() for n in norms]))
    clip_coef = (max_l2_norm / (total_norm + eps)).clamp(max=1.0)
    torch._foreach_mul_(grads, clip_coef)
