"""Base attention interface + shared helpers.

Har attention backend (GQA, MLA-future, ...) `BaseAttention` extend karta hai aur
ek hi forward signature deta hai, taaki `TransformerBlock` ko pata na chale kaunsa
backend chal raha hai. Isse future me naye attention types plug karna aasaan.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, n_kv, T, hd] -> [B, n_kv * n_rep, T, hd] (KV heads ko query heads tak repeat)."""
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return (x[:, :, None, :, :]
            .expand(b, n_kv, n_rep, t, hd)
            .reshape(b, n_kv * n_rep, t, hd))


def causal_mask(q_len: int, k_len: int, device) -> torch.Tensor:
    """Bool mask [q_len, k_len], True = attend allowed.

    Query row i ki absolute position = (k_len - q_len) + i (KV-cache offset handle).
    Us position tak ke saare keys allowed hain.
    """
    i = torch.arange(q_len, device=device).unsqueeze(1)
    j = torch.arange(k_len, device=device).unsqueeze(0)
    return j <= (k_len - q_len) + i


class BaseAttention(nn.Module):
    """Attention backend interface.

    Contract:
        forward(x, cos, sin, past_kv=None, use_cache=False)
            x        : [B, T, d_model]
            cos, sin : [T, head_dim]  (positions [offset, offset+T))
            past_kv  : optional (k, v) cache
            returns  : (out [B, T, d_model], present_kv or None)
    """

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        raise NotImplementedError("attention backend must implement forward()")
