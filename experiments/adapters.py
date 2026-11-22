"""LoRA for Ryth's PyTorch linear projections (not quantized/QLoRA).

Example, from the repository root::

    from experiments.adapters import inject_lora, save_adapter, merge_lora
    from model import RythConfig, RythForCausalLM
    model = RythForCausalLM(RythConfig.ryth_30m())
    inject_lora(model, rank=8, alpha=16)
    # Train with an optimizer over p for p in model.parameters() if p.requires_grad.
    save_adapter(model, "adapter.pt")
    model.eval(); merge_lora(model)  # inference; permanently folds into base

Load an adapter with ``load_adapter(base_model, path)`` using the *same base
checkpoint* and architecture used to train it. Adapter files contain no base
weights or tokenizer; shape checks cannot identify a different base checkpoint.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen linear plus alpha/rank scaled low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("alpha must be positive and dropout must be in [0, 1)")
        self.base = base
        self.rank, self.alpha = rank, float(alpha)
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * (self.alpha / self.rank)


def _replace(model, name, module):
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, leaf, module)


def _validate_targets(model, names):
    """Reject shared modules or weights: merging those could alter another layer."""
    modules = dict(model.named_modules(remove_duplicate=False))
    params = list(model.named_parameters(remove_duplicate=False))
    for name in names:
        layer = modules.get(name)
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"{name} is not an unwrapped Linear")
        if sum(m is layer for m in modules.values()) != 1:
            raise ValueError(f"{name} is a shared module")
        if sum(p is layer.weight for _, p in params) != 1:
            raise ValueError(f"{name} has a shared weight and cannot be safely merged")
        if not layer.weight.is_floating_point():
            raise ValueError(f"{name} must have floating-point weights (LoRA is not QLoRA)")


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                dropout: float = 0.0, targets=("q_proj", "v_proj")):
    """Freeze the whole base and wrap selected linears in-place; return model.

    Targets are exact full module paths or leaf names. The default selects GQA
    query and value projections; pass other projection names for other backends.
    Inject before creating the optimizer. Shared weights/modules are rejected.
    """
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("model already has LoRA adapters")
    if not targets or any(not isinstance(t, str) or not t for t in targets):
        raise ValueError("targets must contain module names")
    modules = dict(model.named_modules(remove_duplicate=False))
    names = [n for n, m in modules.items() if n and isinstance(m, nn.Linear)
             and any(n == t or n.rsplit(".", 1)[-1] == t for t in targets)]
    if any(not any(n == t or n.rsplit(".", 1)[-1] == t for n in names) for t in targets):
        raise ValueError("one or more targets did not match a Linear")
    _validate_targets(model, names)
    # Construct before changing model or trainability, including hyperparameter validation.
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("invalid LoRA rank, alpha or dropout")
    model.requires_grad_(False)
    for name in names:
        layer = LoRALinear(modules[name], rank, alpha, dropout)
        layer.train(modules[name].training)
        _replace(model, name, layer)
    return model


def save_adapter(model: nn.Module, path):
    """Save only adapter weights and configuration, not the base model."""
    layers = {}
    for name, layer in model.named_modules():
        if isinstance(layer, LoRALinear):
            layers[name] = {"rank": layer.rank, "alpha": layer.alpha,
                            "dropout": layer.dropout.p,
                            "A": layer.lora_A.detach().cpu().clone(),
                            "B": layer.lora_B.detach().cpu().clone()}
    if not layers:
        raise ValueError("model has no LoRA adapters")
    torch.save({"format": "ryth-lora-v1", "layers": layers}, path)


def load_adapter(model: nn.Module, path, map_location="cpu"):
    """Load onto a fresh base or matching injected model, validating all layers.

    The caller must supply the original base checkpoint and tokenizer. Loading
    on a fresh model freezes it and injects the saved adapters in-place.
    """
    data = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(data, dict) or data.get("format") != "ryth-lora-v1" or not isinstance(data.get("layers"), dict) or not data["layers"]:
        raise ValueError("invalid LoRA adapter file")
    layers = data["layers"]
    existing = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    if existing and set(existing) != set(layers):
        raise ValueError("adapter layer names do not match model")
    if not existing:
        _validate_targets(model, layers)
    for name, record in layers.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError("invalid adapter layer")
        layer = existing.get(name) if existing else model.get_submodule(name)
        base = layer.base if existing else layer
        rank, alpha, dropout = record.get("rank"), record.get("alpha"), record.get("dropout")
        if (not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0
                or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0
                or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1):
            raise ValueError(f"invalid adapter configuration for {name}")
        if existing and (layer.rank != rank or layer.alpha != alpha or layer.dropout.p != dropout):
            raise ValueError(f"adapter configuration mismatch for {name}")
        a, b = record.get("A"), record.get("B")
        if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                or a.shape != (rank, base.in_features) or b.shape != (base.out_features, rank)
                or not a.is_floating_point() or not b.is_floating_point()
                or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise ValueError(f"invalid adapter tensors for {name}")
    if not existing:
        model.requires_grad_(False)
        for name, record in layers.items():
            base = model.get_submodule(name)
            layer = LoRALinear(base, record["rank"], record["alpha"], record["dropout"])
            layer.train(base.training)
            _replace(model, name, layer)
    with torch.no_grad():
        for name, record in layers.items():
            layer = model.get_submodule(name)
            layer.lora_A.copy_(record["A"])
            layer.lora_B.copy_(record["B"])
    return model


def merge_lora(model: nn.Module):
    """Permanently merge and remove all adapters; use eval mode if dropout > 0."""
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    if not layers:
        raise ValueError("model has no LoRA adapters")
    if any(m.training and m.dropout.p for _, m in layers):
        raise ValueError("call eval() before merging adapters with dropout")
    with torch.no_grad():
        for name, layer in layers:
            layer.base.weight.add_((layer.lora_B @ layer.lora_A) * (layer.alpha / layer.rank))
            _replace(model, name, layer.base)
    return model
