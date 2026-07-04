"""Forward-pass driver + metrics report (CPU or GPU).

    python -m benchmarks.forward --preset ryth_30m --seq_len 256 --device cpu
    python -m benchmarks.forward --preset ryth_125m --device cuda

Prints: Parameters, Trainable Params, Estimated FLOPs, KV Cache Size, Peak
Activation Memory, Context Length — plus the real output shape.
"""

from __future__ import annotations

import torch

from model import format_metrics, model_metrics
from . import build_model, common_parser, pick_device


def main():
    args = common_parser("Ryth forward pass + metrics").parse_args()
    device = pick_device(args.device)

    model, cfg = build_model(args.preset, device, vocab_size=args.vocab)
    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len), device=device)

    with torch.no_grad():
        logits, _ = model(ids)

    print("=" * 56)
    print(f"Ryth forward pass — {args.preset}  (device={device})")
    print("=" * 56)
    print(f"  input  : {tuple(ids.shape)}")
    print(f"  logits : {tuple(logits.shape)}")
    dtype_bytes = 2 if device == "cuda" else 4
    m = model_metrics(model, cfg, seq_len=args.seq_len, batch=args.batch,
                      dtype_bytes=dtype_bytes)
    print(format_metrics(m))


if __name__ == "__main__":
    main()
