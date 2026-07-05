"""Deterministic train / validation / test splitting at the *repository* level.

Repository leakage rokne ke liye: ek repo ki saari files ek hi split me jaati hain.
Assignment repo-name + seed ke hash se hota hai (deterministic, reproducible; koi
RNG / wall-clock nahi). Ratios config se aate hain.
"""

from __future__ import annotations

import hashlib

SPLITS = ("train", "validation", "test")


def _fraction(repo: str, seed: int) -> float:
    """Stable value in [0, 1) for a repo+seed."""
    h = hashlib.sha256(f"{seed}:{repo}".encode()).digest()
    # first 8 bytes -> big int -> [0,1)
    v = int.from_bytes(h[:8], "big")
    return v / float(1 << 64)


def assign_split(repo: str, ratios: dict, seed: int = 1) -> str:
    """Assign a repo to a split by cumulative ratio buckets."""
    order = [s for s in SPLITS if s in ratios] or list(ratios)
    total = sum(ratios[s] for s in order)
    x = _fraction(repo, seed) * total
    acc = 0.0
    for s in order:
        acc += ratios[s]
        if x < acc:
            return s
    return order[-1]


def split_records(records: list, ratios: dict, seed: int = 1) -> dict:
    """Set `record.split` for every record (repo-consistent). Returns per-split
    counts. All files of a repo land in the same split."""
    cache: dict = {}
    counts = {s: 0 for s in SPLITS}
    for r in records:
        if r.repository not in cache:
            cache[r.repository] = assign_split(r.repository, ratios, seed)
        r.split = cache[r.repository]
        counts[r.split] = counts.get(r.split, 0) + 1
    return counts


def verify_no_leakage(records: list) -> bool:
    """True if no repository appears in more than one split."""
    seen: dict = {}
    for r in records:
        if r.repository in seen and seen[r.repository] != r.split:
            return False
        seen[r.repository] = r.split
    return True
