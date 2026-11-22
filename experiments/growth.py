"""Opt-in progressive depth growth for an actual Ryth causal LM.

Example (run from the repository root)::

    from model import RythConfig, RythForCausalLM
    from experiments.growth import grow_depth
    model = RythForCausalLM(RythConfig(vocab_size=64, d_model=32,
                                       n_layers=2, n_heads=4, n_kv_heads=2))
    grown = grow_depth(model, 4)
    # Construct a NEW optimizer and scheduler for grown.parameters().

Growth returns an independent deep copy; original layers remain first, and new
layers are independent copies of original layers in round-robin order. It is
not function-preserving: adding residual blocks changes logits. Existing KV
caches must be discarded. Optimizers, schedulers, distributed wrappers and any
parameter references belong to the old model and must be rebuilt; optimizer
state is not migrated. Save the returned model with *its* updated config.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from model.decoder import RythForCausalLM


def grow_depth(model: RythForCausalLM, n_layers: int) -> RythForCausalLM:
    """Return a separate decoder expanded to ``n_layers`` by copying blocks.

    Only increasing a positive, consistent existing depth is supported. Model
    weights, dtype, device, training mode, and tied/untied embeddings are
    preserved. New layers inherit source weights, including any module hooks.
    """
    if not isinstance(model, RythForCausalLM):
        raise TypeError("model must be a RythForCausalLM")
    old_depth = len(model.decoder.layers)
    if (type(model.config.n_layers) is not int or old_depth < 1
            or model.config.n_layers != old_depth
            or model.decoder.config != model.config):
        raise ValueError("model config must match its non-empty decoder depth")
    if type(n_layers) is not int or n_layers <= old_depth:
        raise ValueError("n_layers must be an integer greater than the current depth")
    # replace invokes RythConfig's own validation and avoids mutating the source.
    config = replace(model.config, n_layers=n_layers)
    grown = copy.deepcopy(model)
    grown.config = config
    grown.decoder.config = config
    for index in range(old_depth, n_layers):
        block = copy.deepcopy(grown.decoder.layers[(index - old_depth) % old_depth])
        block.layer_idx = index
        grown.decoder.layers.append(block)
    return grown
