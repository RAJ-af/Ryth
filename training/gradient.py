"""Gradient utilities — accumulation, clipping, and NaN/Inf detection.

Trainer inhe use karta hai taaki gradient handling ek jagah, testable, aur safe
rahe. NaN/Inf detection important hai — mixed precision me training kabhi-kabhi
diverge karti hai, aur usse jaldi pakadna chahiye (skip ya stop).
"""

from __future__ import annotations

import torch


def is_finite_loss(loss: torch.Tensor) -> bool:
    return bool(torch.isfinite(loss).all())


def grads_are_finite(model) -> bool:
    """True agar saare gradients finite hain (koi NaN/Inf nahi)."""
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def clip_grad_norm(model, max_norm: float) -> float:
    """Gradients ko clip karo aur (clipping se pehle ka) total norm return karo."""
    if max_norm <= 0:
        # sirf norm compute karo (clip nahi)
        total = torch.norm(torch.stack([
            p.grad.detach().norm(2) for p in model.parameters()
            if p.grad is not None]), 2) if any(
                p.grad is not None for p in model.parameters()) else torch.tensor(0.0)
        return float(total)
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))


class GradientManager:
    """Accumulation + clipping + NaN detection ko encapsulate karta hai.

    Usage per optimizer-step:
        gm.zero_grad(optimizer)
        for _ in range(accum):
            loss = compute_loss() / accum
            gm.backward(loss, scaler)          # returns False agar loss NaN
        ok = gm.step(model, optimizer, scaler) # clip + step; False agar grads NaN
    """

    def __init__(self, grad_clip: float = 1.0):
        self.grad_clip = grad_clip
        self.last_grad_norm = 0.0
        self.nan_skips = 0

    def zero_grad(self, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss, scaler) -> bool:
        if not is_finite_loss(loss):
            self.nan_skips += 1
            return False
        scaler.scale(loss).backward()
        return True

    def step(self, model, optimizer, scaler) -> bool:
        # fp16 scaler ke saath clip se pehle unscale zaroori
        scaler.unscale_(optimizer)
        if not grads_are_finite(model):
            self.nan_skips += 1
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return False
        self.last_grad_norm = clip_grad_norm(model, self.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        return True
