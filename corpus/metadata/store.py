"""Record store — append / read metadata as JSONL (streaming, mmap-free).

Metadata ko JSONL me rakhte hain taaki laakhon records bina RAM bhare stream ho
sakein. `RecordStore` file+repo records dono ke liye simple, deterministic I/O
deta hai.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

from .record import FileRecord, RepoRecord


class RecordStore:
    """JSONL-backed store for FileRecords (and optionally RepoRecords)."""

    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    # ---- write ----
    def write(self, records: Iterable[FileRecord], include_content: bool = False) -> int:
        n = 0
        with open(self.path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(include_content=include_content),
                                   ensure_ascii=False, sort_keys=True) + "\n")
                n += 1
        return n

    def append(self, record: FileRecord, include_content: bool = False) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(include_content=include_content),
                               ensure_ascii=False, sort_keys=True) + "\n")

    # ---- read ----
    def read(self) -> Iterator[FileRecord]:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield FileRecord(**json.loads(line))

    def count(self) -> int:
        return sum(1 for _ in self.read())


def write_repo_records(path: str, repos: Iterable[RepoRecord]) -> int:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in repos:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_repo_records(path: str) -> Iterator[RepoRecord]:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield RepoRecord(**json.loads(line))
