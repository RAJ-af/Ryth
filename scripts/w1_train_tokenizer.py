"""W1 tokenizer training — multilingual+code BPE on a SOURCE-balanced sample.

Har source-dir (stage/<source>/...) apna char-quota bharta hai — 12 Indic
bhashaen + English + 14 code sources balanced representation paate hain
(W1-revision; pehle sirf .py/.c ext-buckets the). ~1MB slice ka time-probe
ETA deta hai. `--probe-only` se sirf estimate, training nahi.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

# --- make the repo importable when run from a clone (scripts/..) ------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# pipes block-buffer hote hain — Kaggle cell me progress live dikhe
if hasattr(sys.stdout, "reconfigure"):                 # pragma: no cover
    sys.stdout.reconfigure(line_buffering=True)


def stratified_sample(root: str, target_chars: int,
                      seed: int = 1234) -> list[str]:
    """SOURCE-level buckets: stage/<source>/ ke files round-robin, har source
    apna char-quota (target // n_sources) bharta hai. Chhoti source apni
    files khatam hone par ruk jati hai (balance best-effort per-source).
    Deterministic: sorted source order + seed-shuffled files."""
    sources: dict[str, list[str]] = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.startswith("_"):            # _DONE/_SUMMARY markers skip
                continue
            rel = os.path.relpath(dp, root)
            top = rel.split(os.sep)[0] if rel != "." else "_flat"
            sources.setdefault(top, []).append(os.path.join(dp, fn))
    names = sorted(sources)
    if not names:
        return []
    quota = max(1, target_chars // len(names))
    rng = random.Random(seed)
    texts: list[str] = []
    for name in names:
        files = sorted(sources[name])
        rng.shuffle(files)
        got = 0
        i = 0
        while got < quota and i < len(files):
            with open(files[i], encoding="utf-8", errors="replace") as f:
                t = f.read()
            texts.append(t)
            got += len(t)
            i += 1
    return texts


def time_probe(texts: list[str], impl: str = "fast") -> float:
    """~1MB slice pe train karke chars/sec estimate wapas lao."""
    probe: list[str] = []
    size = 0
    for t in texts:
        probe.append(t[: 1_000_000 - size])
        size += len(probe[-1])
        if size >= 1_000_000:
            break
    t0 = time.time()
    if impl == "fast":
        from tokenizer.fast_bpe import train_fast

        train_fast(probe, vocab_size=2048, verbose=False)
    else:
        from tokenizer.bpe import BPETokenizer

        BPETokenizer().train(probe, vocab_size=2048, verbose=False)
    return size / max(1e-9, time.time() - t0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="corpus_out/stage")
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--sample-mb", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="tok/tokenizer.json")
    p.add_argument("--impl", choices=("fast", "naive"), default="fast",
                   help="fast = incremental heap (bit-identical proven, "
                        "tests/test_fast_bpe.py); naive = O(merges x corpus)")
    p.add_argument("--checkpoint-every", type=int, default=2000,
                   help="fast impl: itne merges par partial save (0 = off)")
    p.add_argument("--probe-only", action="store_true",
                   help="sirf chars/sec + ETA print karo, train mat karo")
    a = p.parse_args(argv)

    if not os.path.isdir(a.raw):
        raise SystemExit(f"[tok] raw dir nahi mila: {a.raw!r} — corpus stage "
                         "pehle complete karo (w1_build_corpus.py)")
    target = int(a.sample_mb * 1_000_000)
    texts = stratified_sample(a.raw, target_chars=target, seed=a.seed)
    total = sum(len(t) for t in texts)
    if not texts:
        raise SystemExit(f"[tok] {a.raw!r} me koi text file nahi mili")
    print(f"[sample] sources se files={len(texts)} chars={total:,}")
    rate = time_probe(texts, impl=a.impl)
    eta_min = total / max(rate, 1.0) / 60.0
    print(f"[probe:{a.impl}] ~{rate:,.0f} chars/sec -> ETA >= {eta_min:.0f} min "
          f"(lower bound — bada vocab merges ko mehnat zyada maangta hai)")
    if a.probe_only:
        return 0

    from tokenizer import DEFAULT_SPECIAL_TOKENS
    from tokenizer.bpe import BPETokenizer

    tok = BPETokenizer()
    t0 = time.time()
    if a.impl == "fast":
        from tokenizer.fast_bpe import train_fast

        ckpt = (a.out + ".partial") if a.checkpoint_every > 0 else None
        tok = train_fast(texts, vocab_size=a.vocab, verbose=True,
                         checkpoint_path=ckpt,
                         checkpoint_every=a.checkpoint_every)
    else:
        tok.train(texts, vocab_size=a.vocab, verbose=True)
    # specials MERGES ke BAAD append hote hain — merge-ids untouched rehte
    # hain; W3 SFT/inference isi saved file se exactly yehi IDs dekhega
    # (W1-revision fix: pehle register ho hi nahi rahe the)
    tok.add_special_tokens(DEFAULT_SPECIAL_TOKENS)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok.save(a.out)
    meta = {"vocab_size": tok.vocab_size, "sample_chars": total,
            "train_seconds": round(time.time() - t0, 1),
            "impl": a.impl,
            "seed": a.seed, "sources_root": a.raw,
            "special_tokens": dict(tok.special_tokens)}
    with open(a.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # resume marker — pack/notebook isse 'tokenizer complete' verify karte hain
    with open(os.path.join(os.path.dirname(a.out) or ".", "_DONE"), "w",
              encoding="utf-8") as f:
        json.dump({"vocab_size": tok.vocab_size,
                   "specials": len(tok.special_tokens)}, f)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
