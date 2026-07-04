"""Benchmark helpers for the Ryth model core.

    forward.py  — forward pass + metrics (CPU / GPU)
    memory.py   — measured memory footprint
    speed.py    — prefill + decode throughput

Common builder/argparser lives here so each script stays small.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import RythConfig, RythForCausalLM

PRESETS = ["ryth_30m", "ryth_125m", "ryth_350m", "ryth_1b"]


def pick_device(want: str) -> str:
    if want != "auto":
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(preset: str, device: str, vocab_size: int = 32000, **overrides):
    cfg = getattr(RythConfig, preset)(vocab_size=vocab_size, **overrides)
    model = RythForCausalLM(cfg).to(device).eval()
    return model, cfg


def common_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--preset", default="ryth_30m", choices=PRESETS)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    return ap
