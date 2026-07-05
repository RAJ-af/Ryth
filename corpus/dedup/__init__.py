"""Deduplication — exact (sha256) + near-duplicate (MinHash/LSH)."""

from .exact import dedupe_files, dedupe_repos, repo_signature
from .near import (NearDeduper, dedupe_near, jaccard_estimate,
                   minhash_signature)

__all__ = [
    "dedupe_files", "dedupe_repos", "repo_signature",
    "NearDeduper", "dedupe_near", "minhash_signature", "jaccard_estimate",
]
