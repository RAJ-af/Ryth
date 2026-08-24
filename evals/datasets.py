"""Problem-file loading (offline-first) + opt-in downloaders.

Real benchmarks (HumanEval MIT / MBPP CC-BY) manually ya download_* se lao;
tests kabhi network nahi maangte — fixtures humare khud ke toy problems hain.
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
from dataclasses import dataclass

_HUMANEVAL_URL = ("https://raw.githubusercontent.com/openai/human-eval/master/"
                  "data/HumanEval.jsonl.gz")
_MBPP_URL = ("https://raw.githubusercontent.com/google-research/google-research-datasets/"
             "master/mbpp/mbpp.jsonl")


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


def download_mbpp(dest_dir: str) -> str:
    """Opt-in downloader — tests isko KABHI nahi chalate."""
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, "mbpp.jsonl")
    urllib.request.urlretrieve(_MBPP_URL, out)
    return out
