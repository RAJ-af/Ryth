"""Language-model head: hidden states -> vocabulary logits.

Weight tying support: agar embedding ki weight di jaaye to LM head usi ko share
karta hai (naya parameter nahi banata). Isse params kam hote hain aur quality
aksar behtar hoti hai — ye standard practice hai (GPT-2, Llama).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int,
                 tied_weight: nn.Parameter | None = None, init_std: float = 0.02):
        super().__init__()
        self.tied = tied_weight is not None
        if tied_weight is not None:
            # Same Parameter object share -> gradients bhi tie ho jaate hain
            self.weight = tied_weight
        else:
            self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
            nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model] -> logits: [B, T, vocab_size]
        return F.linear(x, self.weight)
