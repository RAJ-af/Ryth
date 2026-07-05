"""Evaluator — validation loss + perplexity.

Model ko eval mode me daal ke kuch val batches par average cross-entropy nikalta
hai (no grad). Perplexity = exp(loss). Trainer isse early-stopping + best-model
selection ke liye use karta hai.
"""

from __future__ import annotations

import torch

from .loss import lm_cross_entropy, perplexity


@torch.no_grad()
def evaluate(model, val_loader, device, autocast, max_batches: int) -> dict:
    was_training = model.training
    model.eval()
    total_loss, n = 0.0, 0
    for i, (inp, tgt) in enumerate(val_loader):
        if i >= max_batches:
            break
        inp, tgt = inp.to(device), tgt.to(device)
        with autocast():
            logits, _ = model(inp)
            loss = lm_cross_entropy(logits, tgt)
        total_loss += loss.item()
        n += 1
    if was_training:
        model.train()
    avg = total_loss / max(1, n)
    return {"val_loss": avg, "val_ppl": perplexity(avg)}
