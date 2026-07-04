"""Token embedding layer.

Simple learned lookup table: har token id ke liye ek d_model-dim vector. Weight
tying ke liye `.weight` property expose ki hai taaki LM head isse share kar sake.
Initialization central `model.init.apply_init` karta hai (single source of truth).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    @property
    def weight(self) -> nn.Parameter:
        """Underlying [vocab_size, d_model] weight — LM head weight tying ke liye."""
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, seq_len] (long) -> [batch, seq_len, d_model]
        return self.embedding(input_ids)
