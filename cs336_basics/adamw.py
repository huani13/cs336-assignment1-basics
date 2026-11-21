from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8):
        defaults = {"lr": lr, "betas":betas, "weight_decay": weight_decay, "eps":eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                at = lr * (1-b2**t)**0.5 / (1-b1**t)
                p.data *= (1-lr*weight_decay)
                m = state.get("m", 0)
                m = b1*m + (1-b1)*grad
                v = state.get("v", 0)
                v = b2*v + (1-b2)*grad**2
                p.data -= at*m / (v**0.5 + eps)
                state["m"] = m
                state["v"] = v
                state["t"] = t+1

        return loss
