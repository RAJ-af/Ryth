"""Optimizer factory — AdamW with parameter grouping.

Weight decay sirf 2D+ weights (matmuls, embeddings) par; biases aur norm gains par
nahi (standard practice — GPT/Llama). Fused AdamW use hota hai agar CUDA par mile.
Factory pattern taaki future me naye optimizers add karna aasaan ho.
"""

from __future__ import annotations

import inspect

import torch


def _param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_adamw(model, config):
    groups = _param_groups(model, config.weight_decay)
    fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_ok and torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=config.lr,
                             betas=(config.beta1, config.beta2), eps=config.eps,
                             fused=use_fused)


OPTIMIZERS = {"adamw": build_adamw}


def build_optimizer(model, config):
    if config.optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {config.optimizer!r} "
                         f"(known: {sorted(OPTIMIZERS)})")
    return OPTIMIZERS[config.optimizer](model, config)
