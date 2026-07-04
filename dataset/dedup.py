"""Module 11 — Deduplication (chunk-level).

Do jagah dedup hota hai: Cleaner me file-level (content hash), aur yahan
chunk-level. Same token-sequence wale chunks ek hi baar rakhte hain.

Default SHA-256 (stdlib). Agar `xxhash` installed ho to wo use hota hai (tez,
bade datasets ke liye behtar) — bina dependency toote.
"""

from __future__ import annotations

import hashlib
import struct

try:
    import xxhash                      # optional, fast
    _HAS_XX = True
except Exception:
    _HAS_XX = False


def chunk_hash(token_ids) -> str:
    raw = struct.pack(f"<{len(token_ids)}I", *token_ids)
    if _HAS_XX:
        return xxhash.xxh64(raw).hexdigest()
    return hashlib.sha256(raw).hexdigest()


class ChunkDeduper:
    def __init__(self):
        self._seen: set[str] = set()
        self.n_seen = 0
        self.n_dropped = 0

    def is_duplicate(self, token_ids) -> bool:
        self.n_seen += 1
        h = chunk_hash(token_ids)
        if h in self._seen:
            self.n_dropped += 1
            return True
        self._seen.add(h)
        return False

    @property
    def backend(self) -> str:
        return "xxhash" if _HAS_XX else "sha256"
