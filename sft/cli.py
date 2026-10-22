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

from sft.teacher import TeacherConfigError, TeacherError


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

    # --- build subcommand ---
    b = sub.add_parser("build", help="chat JSONL -> RDS shards with loss mask")
    b.add_argument("data_dir", help="Directory or file containing .jsonl data")
    b.add_argument("out_dir", help="Output directory for RDS shards")
    b.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON")
    b.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    b.add_argument("--val_frac", type=float, default=0.05, help="Validation fraction")

    # --- train subcommand ---
    t = sub.add_parser("train", help="SFT training on base model weights")
    t.add_argument("--data_dir", required=True, help="SFT RDS dataset folder")
    t.add_argument("--weights", required=True, help="Pretrained model weights file")
    t.add_argument("--out_dir", default="runs/sft", help="Output directory")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--micro_batch_size", type=int, default=4)
    t.add_argument("--grad_accum_steps", type=int, default=2)
    t.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    t.add_argument("--device", default="auto")
    t.add_argument("--seed", type=int, default=1337)

    # --- generate subcommand ---
    g = sub.add_parser("generate", help="seeds x teacher -> SFT JSONL")
    g.add_argument("--src", required=True,
                   help="corpus dir ya .jsonl rows (content/language/path)")
    g.add_argument("--out", default="data/sft_v1.jsonl")
    g.add_argument("--target", type=int, default=5000,
                   help="itne examples PASS hote hi generation rok do "
                        "(spec: ~5-10k) — API spend cap")
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

    if args.cmd == "build":
        from sft.build import build_sft_dataset

        meta = build_sft_dataset(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            tokenizer_path=args.tokenizer,
            seq_len=args.seq_len,
            val_frac=args.val_frac,
        )
        for fname, r in meta["file_reports"].items():
            print(f"[build] {fname}: read={r['read']} valid={r['valid']} "
                  f"malformed={r['malformed']} tokens={r['tokens']}")
        print(f"[build] Complete: train_chunks={meta['train_chunks']} "
              f"val_chunks={meta['val_chunks']} -> {args.out_dir}")
        return 0

    if args.cmd == "train":
        from sft.train import train_sft

        train_sft(
            data_dir=args.data_dir,
            weights_path=args.weights,
            out_dir=args.out_dir,
            epochs=args.epochs,
            lr=args.lr,
            micro_batch_size=args.micro_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            dtype=args.dtype,
            device=args.device,
            seed=args.seed,
        )
        return 0

    from sft.generate import build_seeds, generate
    from sft.schema import validate_example

    records = load_records(args.src)
    if args.limit_files:
        records = records[:args.limit_files]
    seeds = build_seeds(records,
                        tasks=(args.tasks.split(",") if args.tasks else None))
    print(f"[seeds] {len(seeds)} seeds from {len(records)} records")

    try:
        if args.dry_run:
            from sft.teacher import FakeTeacher
            teacher = FakeTeacher()
        else:
            from sft.teacher import OpenAICompatTeacher
            kw = {"base_url": args.base_url} if args.base_url else {}
            teacher = OpenAICompatTeacher(model=args.model, **kw)
    except TeacherError as e:                          # key/model gating
        raise SystemExit(f"[sft] {e}") from None

    tok = None
    if args.tokenizer:
        from dataset import load_bpe_tokenizer
        tok = load_bpe_tokenizer(args.tokenizer)

    # incremental write: har PASS turant flush — hard crash par bhi jo mila
    # wo + stats bach jaate hain (review fix; pehle sab end me tha)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_written = 0
    fobj = open(args.out, "w", encoding="utf-8")
    try:
        def _sink(ex):
            nonlocal n_written
            row = ex.to_row(tok)
            problems = validate_example(row)
            if problems:                               # bug-guard, data nahi
                raise SystemExit(f"internal error: invalid row {problems}")
            fobj.write(json.dumps(row, ensure_ascii=False) + "\n")
            fobj.flush()
            n_written += 1

        target = args.target if (args.target and args.target > 0) else None
        _, stats = generate(seeds, teacher, target_passed=target,
                            on_example=_sink,
                            progress=print if args.dry_run
                            else (lambda *a, **k: None))
    except TeacherConfigError as e:                    # key/model gating — loud
        fobj.close()
        raise SystemExit(f"[sft] {e}") from None
    except Exception as e:
        fobj.close()
        crash_stats = {"crashed": True, "error": f"{type(e).__name__}: {e}",
                       "n_written": n_written}
        with open(args.out + ".stats.json", "w", encoding="utf-8") as f:
            json.dump(crash_stats, f, indent=2)
        print(f"[sft] CRASH after {n_written} rows — partial output + "
              f"stats saved ({args.out}.stats.json)")
        raise
    fobj.close()

    stats["n_written"] = n_written
    stats["crashed"] = False
    with open(args.out + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("[sft]", json.dumps({k: stats[k] for k in
                               ("n_generated", "n_passed", "pass_rate")}))
    print(f"[out] {args.out} ({n_written} rows) | stats: {args.out}"
          f".stats.json")
    if not args.dry_run and stats["pass_rate"] < 0.9:
        print("[warn] pass_rate < 0.9 — spec §6 acceptance ke neeche; "
              "filter_reasons dekho")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
