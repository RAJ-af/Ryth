"""The Ryth decoder stack + the top-level causal-LM assembly.

`RythDecoder`      : embedding -> N transformer blocks -> final RMSNorm  (hidden states)
`RythForCausalLM`  : RythDecoder + LMHead, with optional weight tying. Ye poora model
                     hai jise aap train / infer karoge.

Ye v0.2.0 model core hai — architecture + forward pass + KV-cache. Training loop
alag Training Engine me aayega (ROADMAP Phase 3).
"""

from __future__ import annotations

import torch.nn as nn

from .config import RythConfig
from .embedding import TokenEmbedding
from .hooks import HOOK_POINTS
from .init import apply_init, init_output_weight
from .lm_head import LMHead
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding
from .transformer_block import TransformerBlock


class RythDecoder(nn.Module):
    def __init__(self, config: RythConfig):
        super().__init__()
        self.config = config
        self.embed = TokenEmbedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, config.norm_eps)

    def forward(self, input_ids, past_kvs=None, use_cache=False):
        """
        input_ids : [B, T] long
        past_kvs  : optional list (len n_layers) of (k, v) caches, ya None
        returns   : (hidden_states [B, T, d_model], new_caches or None)
        """
        # KV-cache offset: kitne tokens pehle se cache me hain
        offset = 0
        if past_kvs is not None and past_kvs[0] is not None:
            offset = past_kvs[0][0].size(2)

        x = self.embed(input_ids)
        cos, sin = self.rope(input_ids.size(1), input_ids.device, offset=offset)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)

        new_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past = past_kvs[i] if past_kvs is not None else None
            x, present = layer(x, cos, sin, past_kv=past, use_cache=use_cache)
            if use_cache:
                new_caches.append(present)

        return self.norm(x), new_caches


class RythForCausalLM(nn.Module):
    """Top-level model: decoder + LM head (+ weight tying)."""

    def __init__(self, config: RythConfig):
        super().__init__()
        self.config = config
        self.decoder = RythDecoder(config)
        tied = self.decoder.embed.weight if config.tie_embeddings else None
        self.lm_head = LMHead(config.d_model, config.vocab_size,
                              tied_weight=tied, init_std=config.init_std)
        apply_init(self, config)               # ② central weight init (scheme from config)
        # Untied lm_head weight is a bare Parameter (not a Linear/Embedding module),
        # so apply_init's module-walk skips it — init it with the scheme explicitly.
        if not config.tie_embeddings:
            init_output_weight(self.lm_head.weight, config)

    def forward(self, input_ids, past_kvs=None, use_cache=False):
        """returns : (logits [B, T, vocab_size], new_caches or None)"""
        hidden, new_caches = self.decoder(input_ids, past_kvs=past_kvs,
                                          use_cache=use_cache)
        return self.lm_head(hidden), new_caches

    # ---- research/debug hooks (⑧) ----
    def register_hook(self, point: str, fn):
        """Register `fn` at `point` on EVERY block. Returns fn (decorator-friendly)."""
        if point not in HOOK_POINTS:
            raise ValueError(f"hook point must be one of {HOOK_POINTS}")
        for block in self.decoder.layers:
            getattr(block.hooks, point).register(fn)
        return fn

    def clear_hooks(self):
        for block in self.decoder.layers:
            block.hooks.clear()

    # ---- param count ----
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.decoder.embed.weight.numel()
        return n
