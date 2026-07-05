"""Downloader interface + a uniform staging model.

Har downloader source ko ek local *staging directory* me materialize karta hai
aur `StagedRepo` return karta hai. Uske baad pipeline sab sources ko ek hi tarah
padhta hai (`iter_files`), chahe woh GitHub se aaye ya local disk se. Isse cleaning
/ filtering / dedup sab downloader-agnostic ho jaata hai.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator


@dataclass
class StagedRepo:
    """A repository materialized on local disk, ready to be ingested."""
    repo: str            # repo id, e.g. "pallets/click"
    source: str          # source kind: local | github | huggingface | http
    root: str            # directory containing the files
    license_hint: str = "UNKNOWN"

    def iter_files(self) -> Iterator[tuple[str, bytes]]:
        """Yield (repo-relative path, raw bytes) for every regular file."""
        base = os.path.abspath(self.root)
        for dirpath, _dirs, names in os.walk(base):
            for name in sorted(names):
                full = os.path.join(dirpath, name)
                if not os.path.isfile(full) or os.path.islink(full):
                    continue
                rel = os.path.relpath(full, base).replace(os.sep, "/")
                try:
                    with open(full, "rb") as f:
                        yield rel, f.read()
                except OSError:
                    continue

    def read_text_files(self, limit: int = 50) -> dict:
        """Small helper for license detection: {path: utf-8 text} (best effort)."""
        out = {}
        for rel, data in self.iter_files():
            try:
                out[rel] = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if len(out) >= limit:
                break
        return out


class Downloader:
    """Base downloader. Subclasses implement `available` + `fetch`."""
    kind = "base"

    def available(self) -> bool:
        """True if this downloader can run in the current environment."""
        return True

    def fetch(self, source, stage_dir: str) -> StagedRepo:  # pragma: no cover
        raise NotImplementedError


class DownloadError(RuntimeError):
    """Raised when a source cannot be fetched (network / missing dependency)."""
