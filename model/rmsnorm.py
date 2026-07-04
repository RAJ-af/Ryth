"""RMSNorm — Root Mean Square Layer Normalization.

LayerNorm se simple aur fast: sirf vector ki RMS (magnitude) se normalize karta
hai, mean subtract nahi karta. Llama/StarCoder isi ka use karte hain.
Computation float32 me hota hai (stability ke liye), phi input dtype me wapas.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learnable per-dim scale (gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)
