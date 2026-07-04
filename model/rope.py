"""Rotary Positional Embeddings (RoPE).

Position ki jaankari model me daalne ka tarika: har (query, key) vector ko uski
position ke hisaab se ek angle se "rotate" kar dete hain. Isse attention ko do
tokens ke beech ki *relative* doori pata chal jaati hai.

`base theta` configurable hai — bada theta => lambe context ko better handle karta hai.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Vector ke doosre half ko negate karke aage laata hai: [a, b] -> [-b, a]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor):
    """RoPE ko q aur k par apply karo.

    q, k : [batch, n_heads, seq_len, head_dim]
    cos, sin : [seq_len, head_dim]
    """
    cos = cos[None, None, :, :]   # broadcast over batch & heads
    sin = sin[None, None, :, :]
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim even hona chahiye"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # inv_freq: har dimension-pair ke liye ek frequency
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, offset: int = 0):
        """Positions [offset, offset+seq_len) ke liye cos/sin tables banao.

        `offset` KV-cache decoding ke liye hai (jab hum ek-ek naya token dete hain).
        Returns cos, sin : [seq_len, head_dim]
        """
        t = torch.arange(offset, offset + seq_len, device=device,
                         dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # [seq_len, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)        # [seq_len, head_dim]
        return emb.cos(), emb.sin()
