"""Attention subpackage — pluggable attention backends via a factory.

    base.py     BaseAttention interface + shared helpers (repeat_kv, causal_mask)
    gqa.py      GQAttention  (default)
    mla.py      MLAttention  (future stub)
    factory.py  build_attention(config)
"""

from .base import BaseAttention, causal_mask, repeat_kv
from .factory import build_attention
from .gqa import GQAttention
from .mla import MLAttention

__all__ = [
    "BaseAttention", "build_attention", "GQAttention", "MLAttention",
    "repeat_kv", "causal_mask",
]
