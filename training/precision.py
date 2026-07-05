"""Mixed-precision helpers — bf16 / fp16 / fp32.

autocast context aur GradScaler ko ek jagah handle karta hai taaki trainer clean
rahe. fp16 me GradScaler chahiye (underflow se bachne ke liye); bf16 me nahi.
"""

from __future__ import annotations

import contextlib

import torch

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"dtype must be one of {list(_DTYPES)}, got {name!r}")
    return _DTYPES[name]


def autocast_context(device: str, dtype_name: str):
    """Ek callable jo autocast context deta hai (mixed precision me), warna no-op."""
    if dtype_name == "fp32":
        return contextlib.nullcontext
    td = resolve_dtype(dtype_name)
    dev = "cuda" if device == "cuda" else "cpu"
    return lambda: torch.autocast(device_type=dev, dtype=td)


def make_grad_scaler(device: str, dtype_name: str):
    """fp16-on-CUDA ke liye enabled GradScaler; warna disabled (no-op) scaler."""
    enabled = (device == "cuda" and dtype_name == "fp16")
    try:                                    # newer API: torch.amp.GradScaler
        return torch.amp.GradScaler("cuda" if device == "cuda" else "cpu",
                                    enabled=enabled)
    except (AttributeError, TypeError):     # fallback older API
        return torch.cuda.amp.GradScaler(enabled=enabled)
