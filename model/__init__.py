"""Ryth model core — a coding-first LLM built from scratch in pure PyTorch.

Public API:
    RythConfig, RythForCausalLM, RythDecoder
plus the individual building blocks (for unit testing / research).
"""

from .attention import (BaseAttention, GQAttention, MLAttention,
                        build_attention, causal_mask, repeat_kv)
from .checkpoint import (build_checkpoint_metadata, load_checkpoint,
                         save_checkpoint)
from .config import RythConfig
from .decoder import RythDecoder, RythForCausalLM
from .embedding import TokenEmbedding
from .feedforward import SwiGLU
from .generate import generate
from .init import apply_init
from .lm_head import LMHead
from .metrics import format_metrics, model_metrics
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary, rotate_half
from .transformer_block import TransformerBlock

__all__ = [
    "RythConfig", "RythForCausalLM", "RythDecoder",
    "TokenEmbedding", "RotaryEmbedding", "apply_rotary", "rotate_half",
    "RMSNorm", "SwiGLU", "TransformerBlock", "LMHead", "apply_init",
    "BaseAttention", "GQAttention", "MLAttention", "build_attention",
    "repeat_kv", "causal_mask",
    "generate", "model_metrics", "format_metrics",
    "build_checkpoint_metadata", "save_checkpoint", "load_checkpoint",
]
