"""Problem-file loading (offline-first) + opt-in downloaders.

Real benchmarks (HumanEval MIT / MBPP CC-BY) manually ya download_* se lao;
tests kabhi network nahi maangte — fixtures humare khud ke toy problems hain.
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass

_HUMANEVAL_URL = ("https://raw.githubusercontent.com/openai/human-eval/master/"
                  "data/HumanEval.jsonl.gz")
# MBPP ka GitHub jsonl 404 ho chuka — HF datasets-server rows API hi source.
_MBPP_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_MBPP_DATASET = "google-research-datasets/mbpp"
_MBPP_SPLITS = ("train", "validation", "test")


@dataclass
class Problem:
    """Ek coding problem: prompt (completion-shuruat), test, entry point."""

    task_id: str
    prompt: str
    test: str
    entry_point: str = ""
    canonical_solution: str = ""


def _to_problem(d: dict) -> Problem:
    for key in ("task_id", "prompt", "test"):
        if key not in d:
            raise ValueError(f"problem record missing required key {key!r}: "
                             f"got keys {sorted(d)}")
    return Problem(task_id=str(d["task_id"]), prompt=d["prompt"], test=d["test"],
                   entry_point=d.get("entry_point", ""),
                   canonical_solution=d.get("canonical_solution", ""))


def load_problems(path: str) -> list[Problem]:
    """`.jsonl` ya `.jsonl.gz` -> list[Problem]. Canonical schema strict hai."""
    opener = gzip.open if path.endswith(".gz") else open
    problems: list[Problem] = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(_to_problem(json.loads(line)))
    return problems


def download_humaneval(dest_dir: str) -> str:
    """Opt-in downloader — tests isko KABHI nahi chalate."""
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, "humaneval.jsonl.gz")
    urllib.request.urlretrieve(_HUMANEVAL_URL, out)
    return out


def _fetch_mbpp_rows() -> list[dict]:
    """HF datasets-server se MBPP 'full' config ke sab rows (original schema).

    num_rows_total missing ho toh bhi poora split aata hai (empty page par
    break) — silent truncation nahi. Pathological server loop se bachne ko
    per-split page-cap bhi hai.
    """
    rows: list[dict] = []
    for split in _MBPP_SPLITS:
        offset = 0
        pages = 0
        while True:
            pages += 1
            if pages > 500:                          # 964 rows / 100 = 10 pages
                raise RuntimeError(                  # real cap se bahut door
                    f"mbpp fetch: split {split!r} me 500+ pages — API "
                    "paginated khatam nahi ho raha, abort")
            q = urllib.parse.urlencode({"dataset": _MBPP_DATASET,
                                        "config": "full", "split": split,
                                        "offset": offset, "length": 100})
            req = urllib.request.Request(f"{_MBPP_ROWS_URL}?{q}",
                                         headers={"User-Agent": "ryth-eval"})
            with urllib.request.urlopen(req, timeout=60) as r:
                page = json.load(r)
            got = page.get("rows", [])
            rows.extend(got)
            offset += len(got)
            if not got:
                break                                # empty page = split khatam
            total = page.get("num_rows_total")
            if isinstance(total, int) and total > 0 and offset >= total:
                break                                # normal end
    return [r["row"] for r in rows]


def download_mbpp(dest_dir: str) -> str:
    """Opt-in downloader — tests isko KABHI nahi chalate (fake urlopen use karte)."""
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, "mbpp.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for row in _fetch_mbpp_rows():
            f.write(json.dumps(row) + "\n")
    return out
