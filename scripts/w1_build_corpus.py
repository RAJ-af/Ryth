"""W1 corpus build — sources -> staged raw tree, idempotent stages.

Stages (har ek apne `_DONE` marker se skip hota hai — Kaggle session marne par
dobara wahi se shuru):
  download : har source ko uske stage dir me materialize karo (byte-budget ke saath)

Network HF sources ke liye ye script KAGGLE/local-with-`datasets` pe chalta hai;
tests `--config none --input DIR` (pure local sources) se offline validate hote hain.
"""

from __future__ import annotations

import argparse
import json
import os


DEFAULT_TOTAL_BYTES = 2_400_000_000          # ~2.4 GB text ≈ 600M+ code tokens


def plan_budget(entries: list[dict], total_bytes: int) -> dict[str, int]:
    """Total byte budget ko sources me `weight` ke proportional baant do."""
    weights = [max(1, int(e.get("weight", 1))) for e in entries]
    tot = sum(weights)
    return {e["id"]: int(total_bytes * w / tot) for e, w in zip(entries, weights)}


def _marker(stage_dir: str) -> str:
    return os.path.join(stage_dir, "_DONE")


def _stage_download(entry: dict, budget: int, stage_root: str,
                    local_input: str | None) -> dict:
    from corpus.sources.registry import Source

    src = Source(**entry)
    stage_dir = os.path.join(stage_root,
                             src.id.replace(":", "_").replace("/", "_"))
    os.makedirs(stage_dir, exist_ok=True)
    done = _marker(stage_dir)
    if os.path.exists(done):                       # idempotent: pehle se hua?
        info = json.load(open(done, encoding="utf-8"))
        return {"files": info["files"], "bytes": info["bytes"]}
    if src.kind == "huggingface":
        from corpus.download.huggingface import HuggingFaceDownloader

        dl = HuggingFaceDownloader(max_bytes=budget)
        staged = dl.fetch(src, stage_root)
        root = staged.root
    elif src.kind == "local":
        if not local_input:
            raise SystemExit(f"{src.id}: local source needs --input DIR")
        root = local_input                          # as-is tree
    else:
        raise SystemExit(f"{src.id}: kind {src.kind} not wired for W1 yet")
    files, total = 0, 0
    for dp, _, fn in os.walk(root):
        for f in fn:
            files += 1
            total += os.path.getsize(os.path.join(dp, f))
    with open(done, "w", encoding="utf-8") as f:
        json.dump({"files": files, "bytes": total}, f)
    return {"files": files, "bytes": total}


def build(args) -> dict:
    entries = (json.load(open(args.config, encoding="utf-8"))
               if getattr(args, "config", None) else
               [{"id": "local:w1", "kind": "local", "location": args.input,
                 "languages": ["python", "c"], "category": "code"}])
    total = int(getattr(args, "total_gb", 2.4) * 1_000_000_000)
    budgets = plan_budget(entries, total)
    per_source_cap = int(getattr(args, "per_source_bytes", 0) or 0)
    stage_root = os.path.join(args.out, "stage")
    os.makedirs(stage_root, exist_ok=True)
    summary = {"sources": {}}
    for e in entries:
        cap = (min(budgets[e["id"]], per_source_cap)
               if per_source_cap else budgets[e["id"]])
        summary["sources"][e["id"]] = _stage_download(
            e, cap, stage_root, getattr(args, "input", None))
    summary["total_bytes"] = sum(v["bytes"] for v in summary["sources"].values())
    summary_path = os.path.join(args.out, "_SUMMARY.json")
    if not os.path.exists(summary_path):           # idempotent: ek hi baar likho
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/w1_sources.json",
                   help="JSON Source list; pass 'none' for a single local dir")
    p.add_argument("--input", default=None, help="local dir (config 'none' mode)")
    p.add_argument("--out", default="corpus_out")
    p.add_argument("--total-gb", type=float, default=2.4)
    p.add_argument("--per-source-bytes", type=int, default=0,
                   help="hard cap per source (testing)")
    a = p.parse_args(argv)
    if a.config and a.config.lower() == "none":
        a.config = None
    s = build(a)
    print(json.dumps(s, indent=2))
    target = int(a.total_gb * 1e9)
    print(f"[w1] target {target} bytes; got {s['total_bytes']} "
          f"({'OK' if s['total_bytes'] >= int(target * 0.95) else 'SHORT'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
