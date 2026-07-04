"""Grouped-Query Attention (GQA) — the default attention backend.

Features:
  * GQA: query heads zyada, key/value heads kam -> KV-cache memory kam (fast infer).
  * RoPE: q, k par rotary embeddings.
  * KV-cache: decoding me purane k/v dobara compute nahi, cache me jodte hain.
  * FlashAttention: `F.scaled_dot_product_attention` (SDPA) use hota hai agar
    config.use_flash_attention True ho; warna manual fallback.
  * QK-norm (optional): q aur k ko RMSNorm karke stability badhaate hain
    (config.use_qk_norm). Bade models me training stable rehti hai.
  * Causal masking: har token sirf apne + pichhle tokens dekh sakta hai.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RythConfig
from ..rmsnorm import RMSNorm
from ..rope import apply_rotary
from .base import BaseAttention, causal_mask, repeat_kv


class GQAttention(BaseAttention):
    def __init__(self, config: RythConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_rep
        self.head_dim = config.head_dim
        self.use_flash = config.use_flash_attention
        self.attn_dropout = config.attn_dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        d_model = config.d_model
        self.q_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        # optional QK normalization (feature flag)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.norm_eps)

    # ------------------------------------------------------------------ #
    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE (naye tokens ki positions par)
        q, k = apply_rotary(q, k, cos, sin)

        # KV-cache: purane k/v ke saath naye ko jodo
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present_kv = (k, v) if use_cache else None

        # GQA: kv heads ko query heads tak repeat karo
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        out = self._attend(q, k, v, q.size(2), k.size(2))
        out = out.transpose(1, 2).reshape(b, t, -1)      # heads wapas merge
        return self.resid_dropout(self.o_proj(out)), present_kv

    # ------------------------------------------------------------------ #
    def _attend(self, q, k, v, q_len, k_len):
        dropout_p = self.attn_dropout if self.training else 0.0

        # ---- FlashAttention / SDPA path ----
        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            if q_len == k_len:
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=dropout_p)
            mask = causal_mask(q_len, k_len, q.device)      # bool, True=keep
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p)

        # ---- manual fallback ----
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = causal_mask(q_len, k_len, q.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, v)
