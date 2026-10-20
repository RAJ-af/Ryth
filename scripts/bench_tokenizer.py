"""Benchmark script for BPE tokenizer training on 2 MB and 10 MB code corpora.

Usage:
    python scripts/bench_tokenizer.py [--vocab 1024] [--workers N]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tokenizer.bpe import BPETokenizer


def get_code_corpus(target_bytes: int) -> list[str]:
    """Gather Python code files up to target_bytes from repo and stdlib."""
    files = sorted(glob.glob(os.path.join(_REPO, "**", "*.py"), recursive=True))
    lib_dir = os.path.join(sys.prefix, "lib", "python" + sys.version[:4])
    if os.path.exists(lib_dir):
        files.extend(sorted(glob.glob(os.path.join(lib_dir, "**", "*.py"), recursive=True)))

    texts: list[str] = []
    total = 0
    for f in files:
        if not os.path.isfile(f):
            continue
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                t = fp.read()
        except OSError:
            continue
        b = t.encode("utf-8")
        if not b:
            continue
        if total + len(b) > target_bytes:
            rem = target_bytes - total
            if rem > 0:
                sliced = b[:rem].decode("utf-8", errors="ignore")
                if sliced:
                    texts.append(sliced)
                    total += len(sliced.encode("utf-8"))
            break
        texts.append(t)
        total += len(b)

    # If corpus fell short, duplicate existing texts to hit target_bytes
    if total < target_bytes and texts:
        orig = list(texts)
        i = 0
        while total < target_bytes:
            t = orig[i % len(orig)]
            b = t.encode("utf-8")
            if total + len(b) > target_bytes:
                rem = target_bytes - total
                sliced = b[:rem].decode("utf-8", errors="ignore")
                if sliced:
                    texts.append(sliced)
                    total += len(sliced.encode("utf-8"))
                break
            texts.append(t)
            total += len(b)
            i += 1

    return texts


def bench(size_bytes: int, label: str, vocab_size: int, workers: int | None = None) -> float:
    texts = get_code_corpus(size_bytes)
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    print(f"\n--- Benchmarking {label} ({total_bytes:,} bytes, {len(texts)} chunks/files) ---")
    t0 = time.time()
    tok = BPETokenizer()
    tok.train(texts, vocab_size=vocab_size, verbose=False, workers=workers)
    elapsed = time.time() - t0
    rate_mb = (total_bytes / (1024 * 1024)) / max(elapsed, 1e-6)
    print(f"[{label}] Vocab: {vocab_size} | Merges: {len(tok.merges)} | "
          f"Time: {elapsed:.2f} s | Speed: {rate_mb:.2f} MB/s")
    return elapsed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=1024, help="target vocab size (default: 1024)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="number of parallel workers (default: cpu_count)")
    args = ap.parse_args(argv)

    print(f"BPE Tokenizer Benchmark (Workers: {args.workers}, Target Vocab: {args.vocab})")

    t_2mb = bench(2_000_000, "2 MB Corpus", vocab_size=args.vocab, workers=args.workers)
    t_10mb = bench(10_000_000, "10 MB Corpus", vocab_size=args.vocab, workers=args.workers)

    print("\n================ BENCHMARK SUMMARY ================")
    print(f"  2 MB Corpus Training Time : {t_2mb:.2f} s")
    print(f" 10 MB Corpus Training Time : {t_10mb:.2f} s")
    print("===================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
