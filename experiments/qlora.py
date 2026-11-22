"""Optional CUDA NF4 QLoRA fine-tuning of an existing floating-point Ryth model.

Install a CUDA-compatible PyTorch and ``pip install bitsandbytes``. Example::

    import torch
    from model import RythForCausalLM
    from experiments.qlora import prepare_qlora, save_qlora_adapter, load_qlora_adapter
    model = RythForCausalLM(config)  # config from the original checkpoint
    # For an FSDP weights-only export; a raw state_dict needs no ['model'].
    checkpoint = torch.load('base.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.cuda()                      # move BEFORE preparation
    prepare_qlora(model, rank=8, alpha=16)
    optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    # fine-tune, then:
    save_qlora_adapter(model, 'adapter.pt')
    # To reload: construct and load the SAME float base, move to CUDA, prepare
    # with the same options, then load_qlora_adapter(model, 'adapter.pt').

All ordinary Linear projections are converted from their existing float weights
using bitsandbytes Linear4bit NF4 with nested statistics. Only selected layers
(default GQA q_proj and v_proj) get trainable adapters; embeddings, LM head,
normalizations and biases remain floating point and frozen. Tied embedding/head
weights are preserved. This is fine-tuning, not quantized scratch training.
Conversion stages all replacements while retaining the float model, so peak GPU
memory includes both and may exceed the initial model footprint. Adapter files
do not identify or contain the base or tokenizer; retain the original checkpoint.
Do not move/cast the prepared model or merge adapters;
recreate it on the desired CUDA device instead. This module does not use the
float LoRA save/load/merge APIs (their weight allocation and merge assume float).
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.adapters import LoRALinear, _replace


class QLoRALinear(nn.Module):
    """A quantized frozen base and independent float32 LoRA parameters."""

    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, float(alpha)
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        residual = F.linear(F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return self.base(x) + residual.to(x.dtype) * (self.alpha / self.rank)


def _bitsandbytes():
    try:
        import bitsandbytes as bnb
    except (ImportError, OSError, RuntimeError) as exc:
        raise ImportError("QLoRA requires CUDA-compatible bitsandbytes: pip install bitsandbytes") from exc
    if not hasattr(bnb, "nn") or not hasattr(bnb.nn, "Linear4bit"):
        raise ImportError("QLoRA requires bitsandbytes with nn.Linear4bit; install a compatible version")
    return bnb


def _make_4bit(bnb, source):
    # Follow the Linear4bit documented lifecycle: load floating weights on CPU,
    # then move to CUDA exactly once to initialize quant_state and packed data.
    layer = bnb.nn.Linear4bit(source.in_features, source.out_features,
                              bias=source.bias is not None, quant_type="nf4",
                              compress_statistics=True, compute_dtype=torch.float32)
    layer.load_state_dict({k: v.detach().cpu() for k, v in source.state_dict().items()})
    layer = layer.to(source.weight.device)
    if (getattr(layer.weight, "quant_state", None) is None
            or layer.weight.dtype != torch.uint8
            or layer.weight.quant_state.quant_type != "nf4"):
        raise RuntimeError("bitsandbytes failed to initialize packed NF4 weights")
    layer.requires_grad_(False)
    return layer


def _require_cuda(model, params):
    devices = {p.device for _, p in params} | {b.device for b in model.buffers()}
    if len(devices) != 1 or next(iter(devices)).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("model parameters and buffers must be on one available CUDA device")


def prepare_qlora(model: nn.Module, rank=8, alpha=16.0, dropout=0.0,
                  targets=("q_proj", "v_proj")):
    """Convert Linear layers and attach LoRA in-place, returning model.

    The complete model must already be on one CUDA device. Targets are exact
    paths or leaf names. Build the optimizer after preparation. Conversion is
    staged before mutating the model; unsupported/shared projections fail.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if (not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0
            or isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or alpha <= 0
            or isinstance(dropout, bool) or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout) or not 0 <= dropout < 1):
        raise ValueError("invalid LoRA rank, alpha or dropout")
    if (not isinstance(targets, (tuple, list)) or not targets
            or any(not isinstance(t, str) or not t for t in targets)):
        raise ValueError("targets must be a nonempty list or tuple of module names")
    modules = dict(model.named_modules(remove_duplicate=False))
    if any(isinstance(m, (LoRALinear, QLoRALinear)) for m in modules.values()):
        raise ValueError("model already has adapters")
    linears = {n: m for n, m in modules.items() if n and isinstance(m, nn.Linear)}
    if not linears:
        raise ValueError("model has no Linear projections")
    selected = {n for n in linears if any(n == t or n.rsplit('.', 1)[-1] == t for t in targets)}
    if any(not any(n == t or n.rsplit('.', 1)[-1] == t for n in selected) for t in targets):
        raise ValueError("one or more targets did not match a Linear")
    params = list(model.named_parameters(remove_duplicate=False))
    for name, layer in linears.items():
        if type(layer) is not nn.Linear:
            raise ValueError(f"{name} is not an ordinary unquantized Linear")
        if sum(m is layer for m in modules.values()) != 1:
            raise ValueError(f"{name} is a shared module")
        if layer.weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"{name} must have float32, float16 or bfloat16 weights")
        # Reject identical Parameters and separate Parameter views sharing storage.
        for label, param in (("weight", layer.weight), ("bias", layer.bias)):
            if param is None:
                continue
            storage = param.untyped_storage().data_ptr()
            if sum(p.untyped_storage().data_ptr() == storage and p.device == param.device
                   for _, p in params) != 1:
                raise ValueError(f"{name} has shared {label} storage")
    _require_cuda(model, params)
    bnb = _bitsandbytes()
    replacements = {}
    for name, source in linears.items():
        layer = _make_4bit(bnb, source)
        if name in selected:
            layer = QLoRALinear(layer, rank, alpha, dropout)
        layer.train(source.training)
        replacements[name] = layer
    model.requires_grad_(False)
    for name, layer in replacements.items():
        _replace(model, name, layer)
    return model


def save_qlora_adapter(model: nn.Module, path):
    """Save only float adapters; the identical float base and preparation are required."""
    layers = {name: {"rank": m.rank, "alpha": m.alpha, "dropout": m.dropout.p,
                     "A": m.lora_A.detach().cpu().clone(), "B": m.lora_B.detach().cpu().clone()}
              for name, m in model.named_modules() if isinstance(m, QLoRALinear)}
    if not layers:
        raise ValueError("model has no QLoRA adapters")
    torch.save({"format": "ryth-qlora-nf4-v1", "layers": layers}, path)


def load_qlora_adapter(model: nn.Module, path, map_location="cpu"):
    """Load into an already prepared model after validating every layer."""
    data = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(data, dict) or data.get("format") != "ryth-qlora-nf4-v1" or not isinstance(data.get("layers"), dict):
        raise ValueError("invalid QLoRA adapter file")
    layers = {name: m for name, m in model.named_modules() if isinstance(m, QLoRALinear)}
    records = data["layers"]
    if not layers or set(records) != set(layers):
        raise ValueError("adapter layer names do not match prepared model")
    converted = {}
    for name, layer in layers.items():
        record = records[name]
        if not isinstance(record, dict) or (record.get("rank"), record.get("alpha"), record.get("dropout")) != (layer.rank, layer.alpha, layer.dropout.p):
            raise ValueError(f"adapter configuration mismatch for {name}")
        converted[name] = {}
        for key, param in (("A", layer.lora_A), ("B", layer.lora_B)):
            value = record.get(key)
            if (not isinstance(value, torch.Tensor) or value.shape != param.shape
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(f"invalid adapter tensor {key} for {name}")
            # Finite float64 values may overflow the adapter's float32 dtype.
            value = value.to(device=param.device, dtype=param.dtype)
            if not torch.isfinite(value).all():
                raise ValueError(f"invalid adapter tensor {key} for {name}")
            converted[name][key] = value
    with torch.no_grad():
        for name, layer in layers.items():
            layer.lora_A.copy_(converted[name]["A"])
            layer.lora_B.copy_(converted[name]["B"])
    return model
