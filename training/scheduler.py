"""Learning-rate scheduler factory — linear warmup + cosine decay.

Steps 0..warmup   : lr 0 -> peak (linear)
Steps warmup..max : peak -> peak*min_lr_ratio (cosine)
Steps > max       : floor (constant)

Scheduler ek plain object hai jisme `lr_at(step)` hai — state save/resume trivial
(sirf step number chahiye). Trainer optimizer ka lr manually set karta hai.
"""

from __future__ import annotations

import math


class WarmupCosine:
    def __init__(self, lr: float, warmup_steps: int, max_steps: int,
                 min_lr_ratio: float = 0.1):
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.floor = lr * min_lr_ratio

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step + 1) / max(1, self.warmup_steps)
        if step >= self.max_steps:
            return self.floor
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))    # 1 -> 0
        return self.floor + coeff * (self.lr - self.floor)


def build_warmup_cosine(config):
    return WarmupCosine(config.lr, config.warmup_steps, config.max_steps,
                        config.min_lr_ratio)


SCHEDULERS = {"cosine": build_warmup_cosine}


def build_scheduler(config):
    if config.scheduler not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {config.scheduler!r} "
                         f"(known: {sorted(SCHEDULERS)})")
    return SCHEDULERS[config.scheduler](config)


def set_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr
