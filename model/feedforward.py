"""SwiGLU feed-forward network.

Standard transformer FFN ki jagah SwiGLU: ek "gate" branch (SiLU se guzarti hai)
aur ek "up" branch, dono ko multiply karke "down" projection. Ye modern coding
models (Llama, StarCoder2) me better perform karta hai.

    FFN(x) = W_down( SiLU(W_gate x) * W_up x )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_up = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))     # activation branch
        up = self.w_up(x)                 # linear branch
        return self.dropout(self.w_down(gate * up))
