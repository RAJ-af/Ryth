"""Attention factory — config.attention_backend ke hisaab se backend chunta hai.

Isse `TransformerBlock` attention type se decoupled rehta hai. Naya backend add
karo -> yahan register karo -> config me naam do. Bas.
"""

from __future__ import annotations

from ..config import RythConfig
from .base import BaseAttention
from .gqa import GQAttention
from .mla import MLAttention

# Registry: backend name -> class
_BACKENDS = {
    "gqa": GQAttention,
    "mla": MLAttention,      # future (instantiation par NotImplementedError)
}


def build_attention(config: RythConfig) -> BaseAttention:
    backend = config.attention_backend
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown attention_backend {backend!r} (known: {sorted(_BACKENDS)})")
    return _BACKENDS[backend](config)
