"""Throughput benchmark: prefill + incremental decode (CPU or GPU).

    python -m benchmarks.speed --preset ryth_30m --seq_len 256 --device cpu

Reports:
    prefill tokens/sec  — full forward over `seq_len`
    decode  tokens/sec  — one-token-at-a-time with KV cache
"""

from __future__ import annotations

import time

import torch

from . import build_model, common_parser, pick_device


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    args = common_parser("Ryth speed benchmark").parse_args()
    device = pick_device(args.device)
    model, cfg = build_model(args.preset, device, vocab_size=args.vocab)

    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len), device=device)
    warmup, iters = 2, 5

    # ---- prefill ----
    with torch.no_grad():
        for _ in range(warmup):
            model(ids)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(ids)
        _sync(device)
        prefill_s = (time.perf_counter() - t0) / iters
    prefill_tps = args.batch * args.seq_len / prefill_s

    # ---- decode (KV cache, 32 new tokens) ----
    new_tokens = 32
    with torch.no_grad():
        logits, past = model(ids, use_cache=True)
        cur = logits[:, -1:].argmax(-1)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(new_tokens):
            logits, past = model(cur, past_kvs=past, use_cache=True)
            cur = logits[:, -1:].argmax(-1)
        _sync(device)
        decode_s = (time.perf_counter() - t0) / new_tokens
    decode_tps = args.batch / decode_s

    print("=" * 56)
    print(f"Ryth speed — {args.preset}  (device={device})")
    print("=" * 56)
    print(f"  prefill : {prefill_tps:>10.0f} tokens/sec  "
          f"(seq={args.seq_len}, batch={args.batch})")
    print(f"  decode  : {decode_tps:>10.0f} tokens/sec  (KV cache, 1 token/step)")


if __name__ == "__main__":
    main()
