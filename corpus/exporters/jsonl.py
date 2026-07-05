"""Export to JSONL — records (metadata + optional content) and task examples.

Splitwise export bhi support karta hai (train/validation/test alag files).
"""

from __future__ import annotations

import json
import os


def export_records_jsonl(records: list, path: str, include_content: bool = True) -> int:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            if r.drop_reason:
                continue
            f.write(json.dumps(r.to_dict(include_content=include_content),
                               ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def export_records_by_split(records: list, out_dir: str,
                            include_content: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    buckets: dict = {}
    for r in records:
        if not r.drop_reason:
            buckets.setdefault(r.split, []).append(r)
    counts = {}
    for split, recs in buckets.items():
        counts[split] = export_records_jsonl(
            recs, os.path.join(out_dir, f"{split}.jsonl"), include_content)
    return counts


def export_tasks_jsonl(examples: list, path: str) -> int:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    return len(examples)
