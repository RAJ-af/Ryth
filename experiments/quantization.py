"""Inference-only per-output-channel symmetric INT8 weight storage for Ryth."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import RythConfig
from model.decoder import RythForCausalLM
from experiments.retrieval import load_model, _tokens, _windows


class Int8Linear(nn.Module):
    """Real int8 storage, per-row FP32 scale; dequantizes before floating matmul."""
    def __init__(self, weight, bias=None):
        super().__init__()
        if weight.ndim != 2 or not weight.is_floating_point() or not torch.isfinite(weight).all():
            raise ValueError("weight must be a finite floating point matrix")
        w = weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True) / 127
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        self.register_buffer("qweight", (w / scale).round().clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    def forward(self, x):
        weight = (self.qweight.to(x.dtype) * self.scale.to(x.dtype))
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, weight, bias)


def quantize_model(model):
    """Copy and convert linear layers and untied LM head; embeddings remain float.

    Reject all shared parameters to avoid silently breaking weight tying.
    """
    seen = set()
    for _, param in model.named_parameters(remove_duplicate=False):
        storage = (str(param.device), param.untyped_storage().data_ptr())
        if storage in seen:
            raise ValueError("shared/tied parameters are not supported; use an untied model")
        seen.add(storage)
    converted = copy.deepcopy(model).eval()
    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, Int8Linear(child.weight, child.bias))
            else:
                replace(child)
    replace(converted)
    converted.lm_head = Int8Linear(converted.lm_head.weight)
    return converted


def tensor_bytes(model):
    """Unique parameter and buffer storage bytes, excluding transient activations."""
    storages = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


@torch.no_grad()
def compare(model, quantized, documents, *, seq_len=None):
    length = model.config.max_seq_len if seq_len is None else seq_len
    if not 1 <= length <= model.config.max_seq_len:
        raise ValueError("seq_len must be within model context")
    total_abs = max_abs = 0.0
    count = 0
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for doc in documents:
            t = _tokens(doc, model.config.vocab_size)
            for x, _ in _windows(t, length):
                x = x.unsqueeze(0).to(device)
                a, _ = model(x)
                b, _ = quantized(x)
                diff = (a.float() - b.float()).abs()
                total_abs += diff.sum().item()
                max_abs = max(max_abs, diff.max().item())
                count += diff.numel()
    finally:
        model.train(was_training)
    if not count:
        raise ValueError("documents need at least one next-token pair")
    return {"mean_abs_logit_drift": total_abs / count, "max_abs_logit_drift": max_abs,
            "float_tensor_bytes": tensor_bytes(model), "int8_tensor_bytes": tensor_bytes(quantized)}


def save_quantized(path, model):
    torch.save({"config": asdict(model.config), "model": model.state_dict()}, path)


def load_quantized(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = quantize_model(RythForCausalLM(RythConfig(**state["config"])))
    model.load_state_dict(state["model"])
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", help="model config JSON if checkpoint lacks model config")
    p.add_argument("--eval", required=True, help="held-out JSON list of token-ID lists")
    p.add_argument("--output", required=True, help="quantized checkpoint output")
    p.add_argument("--seq-len", type=int)
    args = p.parse_args()
    model = load_model(args.checkpoint, args.config)
    quantized = quantize_model(model)
    docs = json.loads(Path(args.eval).read_text())
    print(json.dumps(compare(model, quantized, docs, seq_len=args.seq_len), indent=2))
    save_quantized(args.output, quantized)


if __name__ == "__main__":
    main()
