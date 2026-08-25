"""ryth-sft CLI — generate subcommand (spec §6).

DRY-RUN (offline): FakeTeacher se poora pipeline — key/corpus ke bina bhi
wiring proof hoti hai, dataset sample banta hai.

REAL generation (gated on owner key):
    export RYTH_TEACHER_API_KEY=sk-...
    export RYTH_TEACHER_MODEL=<nemotron-class-backend-name>
    ryth-sft generate --src corpus_out --target 5000 \
        --out data/sft_v1.jsonl --tokenizer tok/tokenizer.json

⚠ SECURITY: test_gen ke asserts LOCAL subprocess me run hote hain
(evals.execution.run_program) — trusted machine par hi chalao.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from types import SimpleNamespace


def _record(content: str, language: str, path: str, **extra) -> SimpleNamespace:
    """Full record duck-type (corpus builders repository/split/license mangte)."""
    d = dict(content=content, language=language, path=path, hash="",
             repository="", split="train", license="")
    d.update(extra)
    return SimpleNamespace(**d)


def load_records(src: str) -> list:
    """--src: corpus dir (recursive .py/.c/.h) YA jsonl rows."""
    records = []
    if src.endswith((".jsonl", ".jsonl.gz")):
        from sft.schema import read_jsonl

        for d in read_jsonl(src):
            records.append(_record(d.get("content", ""),
                                   d.get("language", ""), d.get("path", ""),
                                   hash=d.get("hash", ""),
                                   repository=d.get("repository", ""),
                                   split=d.get("split", "train"),
                                   license=d.get("license", "")))
        return records
    for dp, _dns, fns in os.walk(src):
        for fn in sorted(fns):
            if not fn.endswith((".py", ".c", ".h")):
                continue
            p = os.path.join(dp, fn)
            with open(p, encoding="utf-8", errors="replace") as f:
                content = f.read()
            records.append(_record(
                content,
                "python" if fn.endswith(".py") else "c",
                os.path.relpath(p, src),
                hash=hashlib.sha256(content.encode()).hexdigest()[:16]))
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ryth-sft", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="seeds x teacher -> SFT JSONL")
    g.add_argument("--src", required=True,
                   help="corpus dir ya .jsonl rows (content/language/path)")
    g.add_argument("--out", default="data/sft_v1.jsonl")
    g.add_argument("--target", type=int, default=5000,
                   help="final examples ki max count (spec: ~5-10k)")
    g.add_argument("--tasks", default="",
                   help="comma-list subset; default sab 5")
    g.add_argument("--model", default=None)
    g.add_argument("--base-url", dest="base_url", default=None)
    g.add_argument("--tokenizer", default=None,
                   help="diya to token_ids bhi pack honge (real 24k tok)")
    g.add_argument("--limit-files", type=int, default=None)
    g.add_argument("--dry-run", action="store_true",
                   help="FakeTeacher — offline wiring proof, no network")
    args = ap.parse_args(argv)

    from sft.generate import build_seeds, generate, package
    from sft.schema import validate_example

    records = load_records(args.src)
    if args.limit_files:
        records = records[:args.limit_files]
    seeds = build_seeds(records,
                        tasks=(args.tasks.split(",") if args.tasks else None))
    print(f"[seeds] {len(seeds)} seeds from {len(records)} records")

    if args.dry_run:
        from sft.teacher import FakeTeacher
        teacher = FakeTeacher()
    else:
        from sft.teacher import OpenAICompatTeacher
        kw = {"base_url": args.base_url} if args.base_url else {}
        teacher = OpenAICompatTeacher(model=args.model, **kw)

    examples, stats = generate(seeds, teacher, progress=lambda *a, **k: None)
    if args.target and len(examples) > args.target:
        examples = examples[:args.target]

    tok = None
    if args.tokenizer:
        from dataset import load_bpe_tokenizer
        tok = load_bpe_tokenizer(args.tokenizer)
    rows = package(examples, tok)
    bad = [r for r in rows if validate_example(r)]
    if bad:
        raise SystemExit(f"internal error: {len(bad)} rows failed validation")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    from sft.schema import write_jsonl
    write_jsonl(rows, args.out)
    stats_path = args.out + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("[sft]", json.dumps({k: stats[k] for k in
                               ("n_generated", "n_passed", "pass_rate")}))
    print(f"[out] {args.out} ({len(rows)} rows) | stats: {stats_path}")
    if not args.dry_run and stats["pass_rate"] < 0.9:
        print("[warn] pass_rate < 0.9 — spec §6 acceptance ke neeche; "
              "filter_reasons dekho")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
