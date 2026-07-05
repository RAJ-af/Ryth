"""Loss — causal language-modeling cross entropy (+ perplexity helper).

logits: [B, T, vocab]  targets: [B, T]  -> flatten -> cross_entropy.
`ignore_index` se pad tokens ko loss se hata sakte ho.
"""

from __future__ import annotations

import math

import torch.nn.functional as F


def lm_cross_entropy(logits, targets, ignore_index: int = -100):
    b, t, v = logits.shape
    return F.cross_entropy(
        logits.reshape(b * t, v).float(),
        targets.reshape(b * t),
        ignore_index=ignore_index,
    )


def perplexity(loss_value: float) -> float:
    """exp(loss), overflow ke against clamped."""
    return math.exp(min(20.0, loss_value))
