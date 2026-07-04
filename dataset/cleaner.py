"""Module 1 — Dataset Cleaner.

Raw files me se junk hatata hai:
  ✅ vendor / build folders   ✅ binary files       ✅ empty files
  ✅ generated code           ✅ non-UTF-8          ✅ duplicate files (content hash)
"""

from __future__ import annotations

import hashlib

from .config import RDEConfig
from .record import FileRecord


def is_vendor_path(path: str, vendor_dirs: set) -> bool:
    parts = path.replace("\\", "/").split("/")
    return any(p in vendor_dirs for p in parts)


def looks_binary(raw: bytes) -> bool:
    if b"\x00" in raw:                      # null byte => binary
        return True
    if not raw:
        return False
    # bahut zyada non-text bytes => binary
    text_chars = bytes(range(32, 127)) + b"\n\r\t\f\b"
    nontext = sum(1 for b in raw[:4096] if b not in text_chars)
    return nontext / min(len(raw), 4096) > 0.30


def is_generated(text: str, markers) -> bool:
    head = text[:2000].lower()
    return any(m in head for m in markers)


class Cleaner:
    def __init__(self, config: RDEConfig):
        self.cfg = config
        self._seen_hashes: set[str] = set()   # content-hash dedup (Module 1)
        self.stats = {"vendor": 0, "binary": 0, "empty": 0, "non_utf8": 0,
                      "generated": 0, "duplicate": 0, "kept": 0}

    def clean_bytes(self, path: str, repo: str, raw: bytes) -> FileRecord | None:
        """Ek raw file ko check karo. Keep -> FileRecord, drop -> None."""
        rec = FileRecord(path=path, repo=repo, size_bytes=len(raw))

        if is_vendor_path(path, self.cfg.vendor_dirs):
            self.stats["vendor"] += 1; return None
        if looks_binary(raw):
            self.stats["binary"] += 1; return None

        # UTF-8 validation
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            self.stats["non_utf8"] += 1; return None

        if not text.strip():
            self.stats["empty"] += 1; return None
        if is_generated(text, self.cfg.generated_markers):
            self.stats["generated"] += 1; return None

        # duplicate file (exact content)
        h = hashlib.sha256(raw).hexdigest()
        if h in self._seen_hashes:
            self.stats["duplicate"] += 1; return None
        self._seen_hashes.add(h)

        rec.text = text
        rec.meta["content_sha256"] = h
        self.stats["kept"] += 1
        return rec
