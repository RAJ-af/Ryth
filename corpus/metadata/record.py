"""Per-file and per-repository metadata records.

`FileRecord` corpus ka atomic unit hai. Har file ka poora metadata isme hota hai
(spec: repository, license, language, path, hash, size, quality score, source,
split, timestamp) plus transient `content` (serialize nahi hota by default).

`RepoRecord` ek repository ka roll-up hai (license, quality, file count).

Timestamps caller se aate hain (`created_at`) — is package me kabhi wall-clock
read nahi hota, taaki builds deterministic/reproducible rahein.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict


def content_hash(data: bytes) -> str:
    """sha256 hex digest — file identity + exact dedup ke liye."""
    return hashlib.sha256(data).hexdigest()


@dataclass
class FileRecord:
    # identity / provenance
    repository: str                       # "owner/name" ya local repo id
    path: str                             # repo-relative path
    source: str = "local"                 # local | github | huggingface | http
    license: str = "UNKNOWN"              # SPDX id
    language: str = "unknown"

    # content stats
    hash: str = ""                        # sha256 of raw bytes
    size: int = 0                         # bytes

    # scoring / routing
    quality_score: float = 0.0            # 0..100 (repo-level, copied per file)
    file_quality: float = 0.0             # 0..100 (this file)
    split: str = "train"                  # train | validation | test

    # bookkeeping
    created_at: str = ""                  # ISO ts, passed in by caller
    drop_reason: str = ""                 # non-empty => filtered out

    # transient (not serialized)
    content: str | None = field(default=None, repr=False, compare=False)

    @property
    def kept(self) -> bool:
        return not self.drop_reason

    def to_dict(self, include_content: bool = False) -> dict:
        d = asdict(self)
        if not include_content:
            d.pop("content", None)
        return d

    @classmethod
    def from_bytes(cls, repository: str, path: str, data: bytes, **kw) -> "FileRecord":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        return cls(repository=repository, path=path, hash=content_hash(data),
                   size=len(data), content=text, **kw)


@dataclass
class RepoRecord:
    repository: str
    source: str = "local"
    license: str = "UNKNOWN"
    quality_score: float = 0.0
    n_files: int = 0
    n_bytes: int = 0
    languages: dict = field(default_factory=dict)   # language -> file count
    split: str = "train"
    signals: dict = field(default_factory=dict)     # quality signal breakdown
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
