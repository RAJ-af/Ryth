"""Exact deduplication — file-level (sha256) and repository-level (signature).

File dedup: same content hash => keep the first (deterministic order), drop rest.
Repo dedup: repo signature = hash of its sorted file-hashes; identical signatures
=> duplicate repositories (keep one). Near-identical repos near.py handle karta.
"""

from __future__ import annotations

import hashlib


def dedupe_files(records: list) -> tuple:
    """Return (kept, dropped). Dropped get drop_reason='duplicate_file'.

    First occurrence (by input order) wins; deterministic for a fixed input order.
    """
    seen = set()
    kept, dropped = [], []
    for r in records:
        h = r.hash
        if h and h in seen:
            r.drop_reason = "duplicate_file"
            dropped.append(r)
        else:
            seen.add(h)
            kept.append(r)
    return kept, dropped


def repo_signature(file_hashes) -> str:
    """Stable signature for a repo from its set of file content hashes."""
    joined = "\n".join(sorted(set(file_hashes)))
    return hashlib.sha256(joined.encode()).hexdigest()


def dedupe_repos(repo_hashes: dict) -> dict:
    """`repo_hashes`: {repo: iterable_of_file_hashes}. Returns {repo: keep_bool}.

    Exact-duplicate repos (same signature) collapse to the first repo (sorted for
    determinism)."""
    sig_to_repo: dict = {}
    keep: dict = {}
    for repo in sorted(repo_hashes):
        sig = repo_signature(repo_hashes[repo])
        if sig in sig_to_repo:
            keep[repo] = False
        else:
            sig_to_repo[sig] = repo
            keep[repo] = True
    return keep
