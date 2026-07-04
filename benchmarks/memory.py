"""Measured memory footprint of a forward pass (CPU or GPU).

    python -m benchmarks.memory --preset ryth_30m --seq_len 256 --device cpu

On CUDA: reports torch peak allocated. On CPU: reports process RSS delta
(approximate). Also prints the analytic KV-cache / activation estimates.
"""

from __future__ import annotations

import resource

import torch

from model import model_metrics
from . import build_model, common_parser, pick_device


def _rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS.
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    return (kb / 1024) if sys.platform != "darwin" else (kb / 1024 / 1024)


def main():
    args = common_parser("Ryth memory benchmark").parse_args()
    device = pick_device(args.device)
    model, cfg = build_model(args.preset, device, vocab_size=args.vocab)
    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len), device=device)

    print("=" * 56)
    print(f"Ryth memory — {args.preset}  (device={device})")
    print("=" * 56)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with torch.no_grad():
            model(ids)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"  CUDA peak allocated : {peak:.1f} MB")
    else:
        before = _rss_mb()
        with torch.no_grad():
            model(ids)
        print(f"  process RSS         : {_rss_mb():.1f} MB "
              f"(+{_rss_mb() - before:.1f} MB during forward, approx)")

    dtype_bytes = 2 if device == "cuda" else 4
    m = model_metrics(model, cfg, seq_len=args.seq_len, batch=args.batch,
                      dtype_bytes=dtype_bytes)
    print(f"  params              : {m['parameters']/1e6:.2f}M")
    print(f"  KV cache (analytic) : {m['kv_cache_bytes']/1024**2:.1f} MB")
    print(f"  activations (rough) : {m['activation_bytes']/1024**2:.1f} MB")


if __name__ == "__main__":
    main()
