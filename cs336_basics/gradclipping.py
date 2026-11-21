def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    param_local = []
    for param in parameters:
        param_local.append(param)
    l2_norm = 0
    for param in param_local:
        if not param.requires_grad:
            continue
        grad = param.grad
        l2_norm += (grad**2).sum()
    l2_norm = l2_norm**0.5
    if l2_norm <= max_l2_norm:
        return
    multiplier = max_l2_norm / (l2_norm + eps)
    for param in param_local:
        if not param.requires_grad:
            continue
        param.grad *= multiplier
