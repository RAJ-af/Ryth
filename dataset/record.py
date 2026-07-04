"""Core data structures jo pipeline ke stages ke beech flow karte hain.

FileRecord  — ek source file (clean/validate/quality/encode stages isko enrich karte hain)
Chunk       — seq_len token ka ek training example + uska metadata (Module 10)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileRecord:
    path: str                          # relative path (repo ke andar)
    repo: str                          # repository name
    text: str = ""                     # file content (UTF-8)
    size_bytes: int = 0
    language: str = "unknown"          # Module 5
    quality: int = 0                   # Module 3 score 0-100
    difficulty: str = "medium"         # Module 9: easy | medium | hard
    token_ids: list[int] = field(default_factory=list)   # Module 7
    meta: dict[str, Any] = field(default_factory=dict)    # extra (imports/functions/…)
    dropped: bool = False
    drop_reason: str = ""

    def drop(self, reason: str) -> "FileRecord":
        self.dropped = True
        self.drop_reason = reason
        return self


@dataclass
class Chunk:
    token_ids: list[int]
    meta: dict[str, Any] = field(default_factory=dict)   # Module 10 (per-chunk)

    @property
    def length(self) -> int:
        return len(self.token_ids)
