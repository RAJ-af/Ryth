"""Multi-head Latent Attention (MLA) — PLACEHOLDER (future).

MLA (DeepSeek-V2 style) KV-cache ko latent space me compress karta hai — bahut
lambe context ke liye memory-efficient. Abhi implement NAHI kiya; ye stub isliye
hai taaki factory/config structure future me change na karna pade.

Backend chuno: `RythConfig(attention_backend="mla")` — abhi NotImplementedError dega.
"""

from __future__ import annotations

from ..config import RythConfig
from .base import BaseAttention


class MLAttention(BaseAttention):
    def __init__(self, config: RythConfig):
        super().__init__()
        raise NotImplementedError(
            "MLA (Multi-head Latent Attention) is planned for a future version. "
            "Use attention_backend='gqa' for now.")
