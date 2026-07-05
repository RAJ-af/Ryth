"""Task mixer — assemble a task dataset honoring configurable task ratios.

File-level + repo-level builders se candidates banata hai, phir ratios ke hisaab se
deterministically downsample karke final mix deta hai. Language balancer jaisa hi
approach (max feasible total N).
"""

from __future__ import annotations

import hashlib
import json

from . import builders as B
from ..config import CorpusConfig

_FILE_BUILDERS = {
    "next_token": lambda rec, cfg: B.next_token_examples(rec),
    "fim": lambda rec, cfg: B.fim_examples(rec, cfg.fim_rate),
    "completion": lambda rec, cfg: B.completion_examples(rec),
    "editing": lambda rec, cfg: B.editing_examples(rec),
    "docstring_to_code": lambda rec, cfg: B.docstring_to_code_examples(rec),
    "code_to_explanation": lambda rec, cfg: B.code_to_explanation_examples(rec),
    "bug_fixing": lambda rec, cfg: B.bug_fixing_examples(rec),
    "refactoring": lambda rec, cfg: B.refactoring_examples(rec),
}
_REPO_BUILDERS = {
    "readme_to_code": B.readme_to_code_examples,
    "unit_test_generation": B.unit_test_generation_examples,
}


def _key(example: dict) -> str:
    return hashlib.sha256(json.dumps(example, sort_keys=True).encode()).hexdigest()


def _budget_ok(example: dict, budget: int) -> bool:
    size = len(example.get("text", "")) + len(example.get("input", "")) \
        + len(example.get("target", ""))
    return size <= budget


def collect_candidates(records: list, config: CorpusConfig) -> dict:
    """Return {task: [examples]} for every task with ratio > 0."""
    by_repo: dict = {}
    for r in records:
        by_repo.setdefault(r.repository, []).append(r)

    cand: dict = {t: [] for t, v in config.task_ratios.items() if v > 0}
    for task in list(cand):
        if task in _FILE_BUILDERS:
            fn = _FILE_BUILDERS[task]
            for r in records:
                if r.content:
                    cand[task].extend(fn(r, config))
        elif task in _REPO_BUILDERS:
            fn = _REPO_BUILDERS[task]
            for repo_recs in by_repo.values():
                cand[task].extend(fn(repo_recs))
    # budget filter + deterministic ordering
    for task, items in cand.items():
        kept = [e for e in items if _budget_ok(e, config.seq_char_budget)]
        cand[task] = sorted(kept, key=_key)
    return cand


def build_task_dataset(records: list, config: CorpusConfig | None = None,
                       *, enforce_ratios: bool = True, max_examples: int | None = None) -> list:
    """Build the mixed task dataset. Deterministic order (by task then key)."""
    config = config or CorpusConfig()
    cand = collect_candidates(records, config)
    ratios = {t: config.task_ratios[t] for t in cand if config.task_ratios.get(t, 0) > 0}

    if not enforce_ratios:
        out = [e for t in sorted(cand) for e in cand[t]]
        return out[:max_examples] if max_examples else out

    avail = {t: len(cand[t]) for t in ratios}
    if not any(avail.values()):
        return []

    def feasible(N):
        return all(int(ratios[t] * N) <= avail[t] for t in ratios)

    lo, hi = 0, (sum(avail.values()) + 1) * 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid - 1
    N = lo
    if max_examples:
        N = min(N, max_examples)

    out = []
    for t in sorted(ratios):
        take = int(ratios[t] * N)
        out.extend(cand[t][:take])
    return out


def task_distribution(examples: list) -> dict:
    dist: dict = {}
    for e in examples:
        dist[e["task"]] = dist.get(e["task"], 0) + 1
    return dist
