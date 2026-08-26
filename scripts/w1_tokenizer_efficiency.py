"""W1 tokenizer efficiency — held-out per-source samples par compression.

tokens/char, bytes/token, chars/token per source + aggregate — Indic vs
English vs code ka honest hisaab (W1-revision §2). Offline tool: sirf
trained tokenizer.json + val_src tree (per-source subdirs) chahiye.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# --- make the repo importable when run from a clone (scripts/..) ------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

if hasattr(sys.stdout, "reconfigure"):                 # pragma: no cover
    sys.stdout.reconfigure(line_buffering=True)


def load_texts(val_root: str, max_chars_per_source: int = 200_000
               ) -> dict[str, list[str]]:
    """val_src/<source>/... se per-source texts (deterministic sorted walk)."""
    if not os.path.isdir(val_root):
        raise SystemExit(f"[eff] val dir nahi mila: {val_root!r}")
    per_source: dict[str, list[str]] = {}
    for name in sorted(os.listdir(val_root)):
        d = os.path.join(val_root, name)
        if not os.path.isdir(d) or name.startswith("_"):
            continue
        texts: list[str] = []
        got = 0
        for dp, _, fns in _walk_sorted(d):
            for fn in fns:
                if fn.startswith("_"):
                    continue
                with open(os.path.join(dp, fn), encoding="utf-8",
                          errors="replace") as f:
                    t = f.read()
                texts.append(t)
                got += len(t)
                if got >= max_chars_per_source:
                    break
            if got >= max_chars_per_source:
                break
        if texts:
            per_source[name] = texts
    return per_source


def _walk_sorted(root: str):
    for dp, dns, fns in os.walk(root):
        yield dp, sorted(dns), sorted(fns)


def efficiency_report(tok, per_source: dict[str, list[str]]) -> dict:
    rows: dict = {}
    for name, texts in sorted(per_source.items()):
        chars = sum(len(t) for t in texts)
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        ntok = sum(len(tok.encode(t)) for t in texts)
        rows[name] = {
            "files": len(texts), "chars": chars, "bytes": nbytes,
            "tokens": ntok,
            "tokens_per_char": round(ntok / max(chars, 1), 4),
            "chars_per_token": round(chars / max(ntok, 1), 2),
            "bytes_per_token": round(nbytes / max(ntok, 1), 2),
        }
    chars = sum(r["chars"] for r in rows.values())
    nbytes = sum(r["bytes"] for r in rows.values())
    ntok = sum(r["tokens"] for r in rows.values())
    files = sum(r["files"] for r in rows.values())
    rows["_aggregate"] = {
        "sources": len(per_source), "files": files, "chars": chars,
        "bytes": nbytes, "tokens": ntok,
        "tokens_per_char": round(ntok / max(chars, 1), 4),
        "chars_per_token": round(chars / max(ntok, 1), 2),
        "bytes_per_token": round(nbytes / max(ntok, 1), 2),
    }
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tok", required=True, help="tokenizer.json")
    p.add_argument("--val", required=True, help="val_src tree (per-source dirs)")
    p.add_argument("--out", default=None, help="report.json likho (optional)")
    p.add_argument("--max-chars", type=int, default=200_000,
                   help="per-source sample cap")
    a = p.parse_args(argv)

    from tokenizer.bpe import BPETokenizer

    tok = BPETokenizer.load(a.tok)
    report = efficiency_report(tok, load_texts(a.val, a.max_chars))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        print(f"[eff] report -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
