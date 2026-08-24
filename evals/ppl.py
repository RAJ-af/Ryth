"""Held-out perplexity — C vs Python files alag-alag report karne ke liye."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(model, tok, text: str, *, seq_len: int = 512,
               device: str = "cpu") -> float:
    """Non-overlapping windows ka mean-NLL -> exp. <2 tokens ho toh inf."""
    ids = tok.encode(text)
    if len(ids) < 2:
        return float("inf")
    dev = next(model.parameters()).device
    total_nll, n_pred = 0.0, 0
    for start in range(0, len(ids) - 1, seq_len):
        window = ids[start:start + seq_len + 1]
        if len(window) < 2:
            break
        x = torch.tensor([window[:-1]], dtype=torch.long, device=dev)
        y = torch.tensor([window[1:]], dtype=torch.long, device=dev)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                               reduction="sum")
        total_nll += loss.item()
        n_pred += y.numel()
    if n_pred == 0:
        return float("inf")
    return math.exp(total_nll / n_pred)


@torch.no_grad()
def evaluate_files(model, tok, files: dict[str, str], **kw) -> dict[str, float]:
    """label -> filepath dict; label -> ppl dict wapas (per-language reporting)."""
    out = {}
    for label, path in files.items():
        with open(path, encoding="utf-8", errors="replace") as f:
            out[label] = perplexity(model, tok, f.read(), **kw)
    return out
