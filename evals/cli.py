"""ryth-eval CLI — checkpoint ki quality napo, JSON results ke saath.

Subcommands: humaneval | mbpp | ppl. Results har run pe JSON file me jaate
hain (spec §5 — scores runs ke beech track hone chahiye).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from .datasets import load_problems
from .generation import load_model
from .humaneval import save_results


def apply_limit(problems: list, limit: int | None) -> list:
    """Baseline sweeps ke liye pehli N problems; None/<=0 => poori list."""
    if not limit or limit <= 0:
        return problems
    return problems[:limit]


def _auto_out(task: str, ckpt: str) -> str:
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    os.makedirs("results", exist_ok=True)
    return os.path.join("results", f"{task}_{stem}.json")


def key_metrics(data: dict) -> dict:
    """Result JSON ko flat {metric: value} me todta hai (report table ke liye).

    pass_at_k/perplexity jaise dicts flatten hote hain; top-level numeric
    fields (n_problems) aise hi copy; meta skip.
    """
    flat: dict = {}
    for k, v in data.items():
        if k == "meta":
            continue
        if isinstance(v, dict):
            for fk, fv in v.items():                 # flattened pehli value jeete —
                flat.setdefault(fk, fv)              # top-level overwrite na kare
        elif isinstance(v, (int, float)):
            flat.setdefault(k, v)                    # pehli value jeet-ti hai
    return flat


def cmd_report(results_dir: str, out_path: str | None) -> int:
    """results/*.json ka markdown comparison table — print ya --out file."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                metrics = key_metrics(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue                                 # corrupt/partial result skip
        cells = ", ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in sorted(metrics.items()))
        rows.append((os.path.basename(path), cells or "-"))
    lines = ["| file | metrics |", "|---|---|"]
    lines += [f"| {name} | {cells} |" for name, cells in rows]
    table = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table)
    else:
        print(table, end="")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ryth-eval",
                                 description="Ryth evaluation harness")
    sub = ap.add_subparsers(dest="task", required=True)

    def common(sp):
        sp.add_argument("--ckpt", required=True)
        sp.add_argument("--tokenizer", required=True)
        sp.add_argument("--out", default=None)
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--preset", default=None,
                        help="default: checkpoint khud batata hai "
                             "(config ya metadata.model_preset)")
        sp.add_argument("--seq_len", type=int, default=1024)

    for name in ("humaneval", "mbpp"):
        sp = sub.add_parser(name)
        common(sp)
        sp.add_argument("--problems_file", required=True)
        sp.add_argument("--limit", type=int, default=None,
                        help="pehli N problems (baseline/CPU sweeps)")
        sp.add_argument("--n_samples", type=int, default=20)
        sp.add_argument("--temperature", type=float, default=0.8)
        sp.add_argument("--top_k", type=int, default=40)
        sp.add_argument("--max_new_tokens", type=int, default=256)
        sp.add_argument("--timeout", type=float, default=10.0)
        sp.add_argument("--mode", choices=("base", "chat"), default="base")
        sp.add_argument("--ks", default="1,5,10")

    sp = sub.add_parser("ppl")
    common(sp)
    sp.add_argument("--files", action="append", required=True,
                    help="LABEL=PATH (repeatable)")

    sp = sub.add_parser("report",
                        help="results dir ke JSONs ka markdown table")
    sp.add_argument("results_dir")
    sp.add_argument("--out", default=None)

    args = ap.parse_args(argv)

    if args.task == "report":                        # ckpt/tokenizer ki zaroorat nahi
        return cmd_report(args.results_dir, args.out)

    from dataset import load_bpe_tokenizer
    tok = load_bpe_tokenizer(args.tokenizer)
    model = load_model(args.ckpt, tok.vocab_size, preset=args.preset,
                       seq_len=args.seq_len, device=args.device)
    ks = tuple(int(k) for k in args.ks.split(",")) if hasattr(args, "ks") else ()
    out = args.out or _auto_out(args.task, args.ckpt)

    if args.task == "ppl":
        from .ppl import evaluate_files
        files = {}
        for spec in args.files:
            label, _, path = spec.partition("=")
            files[label] = path
        scores = evaluate_files(model, tok, files, seq_len=args.seq_len,
                                device=args.device)
        result = {"meta": {"task": "ppl"}, "perplexity": scores}
    else:
        from .humaneval import evaluate as he_eval
        from . import mbpp as M
        problems = (M.load_mbpp(args.problems_file) if args.task == "mbpp"
                    else load_problems(args.problems_file))
        problems = apply_limit(problems, getattr(args, "limit", None))
        fn = M.evaluate if args.task == "mbpp" else he_eval
        result = fn(problems, model=model, tok=tok, n_samples=args.n_samples,
                    mode=args.mode, temperature=args.temperature,
                    top_k=args.top_k, max_new_tokens=args.max_new_tokens,
                    timeout_s=args.timeout, ks=ks)

    # provenance meta — spec §5: results runs ke beech compare hone chahiye
    result.setdefault("meta", {}).update({
        "ckpt": os.path.basename(args.ckpt),
        "tokenizer": os.path.basename(args.tokenizer),
        "device": args.device, "seq_len": args.seq_len})
    if args.preset:
        result["meta"]["preset"] = args.preset
    if hasattr(args, "limit"):
        result["meta"]["limit"] = getattr(args, "limit")

    save_results(result, out)                      # ppl/humaneval/mbpp sab yahi
    print(f"\n== RESULTS ({args.task}) ==")
    print(f"  written: {out}")
    if "pass_at_k" in result:
        for k, v in result["pass_at_k"].items():
            print(f"  {k}: {v:.4f}")
    else:
        for label, v in result["perplexity"].items():
            print(f"  ppl[{label}]: {v:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
