"""Training throughput benchmark — synthetic data, real train steps.

    python -m training.benchmark --preset ryth_30m --seq_len 256 --device cpu

Builds a model + random token batches and times full training steps
(forward + backward + optimizer step), reporting steps/sec and tokens/sec.
No dataset needed — measures the compute path only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import RythConfig, RythForCausalLM

from .loss import lm_cross_entropy
from .optimizer import build_optimizer
from .precision import autocast_context, make_grad_scaler
from .config import TrainConfig


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description="Ryth training throughput benchmark")
    ap.add_argument("--preset", default="ryth_30m")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")

    cfg = getattr(RythConfig, args.preset)(vocab_size=args.vocab)
    if args.seq_len > cfg.max_seq_len:
        cfg.max_seq_len = args.seq_len
    model = RythForCausalLM(cfg).to(device).train()
    opt = build_optimizer(model, TrainConfig(lr=3e-4))
    autocast = autocast_context(device, args.dtype)
    scaler = make_grad_scaler(device, args.dtype)

    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len + 1), device=device)
    inp, tgt = ids[:, :-1], ids[:, 1:]

    def train_step():
        opt.zero_grad(set_to_none=True)
        with autocast():
            logits, _ = model(inp)
            loss = lm_cross_entropy(logits, tgt)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(2):                       # warmup
        train_step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        train_step()
    _sync(device)
    elapsed = time.perf_counter() - t0

    steps_s = args.steps / elapsed
    tokens_s = steps_s * args.batch * args.seq_len
    print("=" * 56)
    print(f"Ryth training benchmark — {args.preset} (device={device}, {args.dtype})")
    print("=" * 56)
    print(f"  params      : {model.num_params()/1e6:.1f}M")
    print(f"  batch/seq   : {args.batch} x {args.seq_len}")
    print(f"  step time   : {elapsed/args.steps*1000:.1f} ms")
    print(f"  throughput  : {steps_s:.2f} steps/sec | {tokens_s:.0f} tokens/sec")


if __name__ == "__main__":
    main()
