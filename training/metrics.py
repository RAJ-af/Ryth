"""Training metrics tracker — throughput + running averages.

Loss, tokens/sec, aur running average track karta hai ek logging window ke liye.
Trainer log karne se pehle isse read karta hai, phir reset.
"""

from __future__ import annotations

import time

from .loss import perplexity


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._t0 = time.perf_counter()
        self.tokens = 0
        self.loss_sum = 0.0
        self.n = 0

    def update(self, loss_value: float, n_tokens: int):
        self.loss_sum += loss_value
        self.tokens += n_tokens
        self.n += 1

    def avg_loss(self) -> float:
        return self.loss_sum / max(1, self.n)

    def tokens_per_sec(self) -> float:
        elapsed = time.perf_counter() - self._t0
        return self.tokens / max(1e-9, elapsed)

    def snapshot(self, lr: float, grad_norm: float) -> dict:
        loss = self.avg_loss()
        return {
            "loss": loss,
            "ppl": perplexity(loss),
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": round(self.tokens_per_sec(), 1),
        }
