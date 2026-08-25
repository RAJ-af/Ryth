"""W1 tokenizer training — scratch BPE @24k on a STRATIFIED C+Python sample.

Spec risk-table ke mutabiq poore corpus pe train karna CPU pe bahut dheema hai;
isliye stratified sample (default 60MB) + ~1MB slice ka time-probe jo ETA
batata hai. `--probe-only` se sirf estimate, training nahi.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time


_CODE_EXTS = (".py", ".c")                   # sirf code files bucket hoti hain


def _bucket_of(path: str) -> str | None:
    name = os.path.basename(path)
    if name.startswith("_"):                 # _DONE jaise markers skip
        return None
    for e in _CODE_EXTS:
        if name.endswith(e):
            return e
    return None                              # .json/logs waghera training me nahi


def stratified_sample(root: str, target_chars: int,
                      seed: int = 1234) -> list[str]:
    """Extension-buckets (.py vs .c) me round-robin — dono bhashaein barabar."""
    buckets: dict[str, list[str]] = {".py": [], ".c": []}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            b = _bucket_of(fn)
            if b is not None:
                buckets[b].append(os.path.join(dp, fn))
    rng = random.Random(seed)
    for files in buckets.values():
        rng.shuffle(files)
    texts: list[str] = []
    got = 0
    idx = 0
    while got < target_chars:
        progressed = False
        for files in buckets.values():
            if idx < len(files):
                progressed = True
                with open(files[idx], encoding="utf-8", errors="replace") as f:
                    t = f.read()
                texts.append(t)
                got += len(t)
        if not progressed:
            break                                   # buckets khatam ho gaye
        idx += 1
    return texts


def time_probe(texts: list[str], tok=None) -> float:
    """~1MB slice pe train karke chars/sec estimate wapas lao."""
    from tokenizer.bpe import BPETokenizer

    tok = tok or BPETokenizer()
    probe: list[str] = []
    size = 0
    for t in texts:
        probe.append(t[: 1_000_000 - size])
        size += len(probe[-1])
        if size >= 1_000_000:
            break
    t0 = time.time()
    tok.train(probe, vocab_size=2048, verbose=False)
    return size / max(1e-9, time.time() - t0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="corpus_out/stage")
    p.add_argument("--vocab", type=int, default=24576)
    p.add_argument("--sample-mb", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="tok/tokenizer.json")
    p.add_argument("--probe-only", action="store_true",
                   help="sirf chars/sec + ETA print karo, train mat karo")
    a = p.parse_args(argv)

    target = int(a.sample_mb * 1_000_000)
    texts = stratified_sample(a.raw, target_chars=target, seed=a.seed)
    total = sum(len(t) for t in texts)
    print(f"[sample] files={len(texts)} chars={total:,}")
    rate = time_probe(texts)
    eta_min = total / max(rate, 1.0) / 60.0
    print(f"[probe] ~{rate:,.0f} chars/sec -> ETA ~{eta_min:.0f} min")
    if a.probe_only:
        return 0

    from tokenizer.bpe import BPETokenizer

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(texts, vocab_size=a.vocab, verbose=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok.save(a.out)
    meta = {"vocab_size": tok.vocab_size, "sample_chars": total,
            "train_seconds": round(time.time() - t0, 1),
            "seed": a.seed, "sources_root": a.raw}
    with open(a.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
