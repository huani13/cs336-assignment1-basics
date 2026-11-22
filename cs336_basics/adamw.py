from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8):
        defaults = {"lr": lr, "betas":betas, "weight_decay": weight_decay, "eps":eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]
                m.lerp_(grad, 1-b1)
                v.mul_(b2).addcmul_(grad, grad, value=1-b2)
                step_size = lr * math.sqrt(1-b2**t) / (1-b1**t)
                p.mul_(1-lr*weight_decay)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
        return loss
