"""A single Transformer block (Pre-Norm architecture).

Pre-Norm matlab: normalization sub-layer ke *pehle* lagti hai, aur residual
(skip) connection normalize kiye bina add hoti hai. Ye deep models ko stable
tarike se train karne deta hai.

    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))

Attention backend factory se aata hai (GQA default). Har block me 4 hook points
(before/after attention/ffn) hote hain, aur optional gradient checkpointing.
"""

from __future__ import annotations

import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .attention import build_attention
from .config import RythConfig
from .feedforward import SwiGLU
from .hooks import BlockHooks
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: RythConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = build_attention(config)              # ① factory
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.ffn_dim, config.resid_dropout)
        self.hooks = BlockHooks()                        # ⑧ debug/research hooks
        self.use_grad_ckpt = config.use_gradient_checkpointing

    # ------------------------------------------------------------------ #
    def _attn_ffn(self, x, cos, sin, past_kv, use_cache):
        ctx = {"layer": self.layer_idx}
        # attention sub-layer (pre-norm + residual)
        h = self.hooks.before_attention(self.attn_norm(x), **ctx)
        attn_out, present_kv = self.attn(h, cos, sin, past_kv=past_kv,
                                         use_cache=use_cache)
        attn_out = self.hooks.after_attention(attn_out, **ctx)
        x = x + attn_out
        # feed-forward sub-layer (pre-norm + residual)
        h2 = self.hooks.before_ffn(self.ffn_norm(x), **ctx)
        ffn_out = self.hooks.after_ffn(self.ffn(h2), **ctx)
        x = x + ffn_out
        return x, present_kv

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        # Gradient checkpointing: sirf training + no-cache par (activation memory bachao)
        if (self.use_grad_ckpt and self.training and not use_cache
                and past_kv is None):
            def run(x_):
                return self._attn_ffn(x_, cos, sin, None, False)[0]
            return checkpoint.checkpoint(run, x, use_reentrant=False), None
        return self._attn_ffn(x, cos, sin, past_kv, use_cache)
