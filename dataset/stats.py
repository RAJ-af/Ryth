"""Module 19 — Statistics (+ Packing Statistics).

Pipeline ke dauran sab kuch count karta hai aur ek clean summary deta hai jo
manifest + validation report me jaata hai.

Packing statistics (Module 7 wali) khaas important hai: agar 4096-context me sirf
1800 tokens use ho rahe hain, to yahan turant dikh jaayega (efficiency waste).
"""

from __future__ import annotations

from collections import Counter


class StatsCollector:
    def __init__(self, seq_len: int, itemsize: int):
        self.seq_len = seq_len
        self.itemsize = itemsize
        # file-level
        self.total_files_seen = 0
        self.files_kept = 0
        self.fim_docs = 0
        self.languages = Counter()
        self.difficulties = Counter()
        self.file_drops = Counter()
        # aggregates for report
        self.quality_sum = 0
        self.functions_sum = 0
        self.imports_sum = 0
        self.comments_sum = 0
        self.raw_bytes = 0                 # kept files ke original UTF-8 bytes
        self.longest_seq = 0               # sabse lamba single-doc token length
        # token / chunk
        self.total_tokens = 0
        self.total_chunks = 0
        self.real_tokens = 0               # non-pad tokens in shards
        self.pad_tokens = 0
        self.chunk_dupe_seen = 0
        self.chunk_dupe_dropped = 0

    # ---- file stage ----
    def add_file(self, rec):
        self.files_kept += 1
        self.languages[rec.language] += 1
        self.difficulties[rec.difficulty] += 1
        self.quality_sum += rec.quality
        self.functions_sum += rec.meta.get("functions", 0)
        self.imports_sum += len(rec.meta.get("imports", []))
        self.comments_sum += rec.meta.get("comment_lines", 0)
        self.raw_bytes += len(rec.text.encode("utf-8"))
        self.total_tokens += len(rec.token_ids)
        self.longest_seq = max(self.longest_seq, len(rec.token_ids))
        if rec.meta.get("kind") == "fim":
            self.fim_docs += 1

    # ---- chunk stage (packing) ----
    def add_chunk(self, chunk):
        self.total_chunks += 1
        pad = chunk.meta.get("padded", 0)
        self.pad_tokens += pad
        self.real_tokens += self.seq_len - pad

    # ---- summaries ----
    def _avg(self, total):
        return round(total / self.files_kept, 2) if self.files_kept else 0

    def packing(self) -> dict:
        capacity = self.total_chunks * self.seq_len
        eff = (100 * self.real_tokens / capacity) if capacity else 0.0
        return {
            "seq_len": self.seq_len,
            "capacity_tokens": capacity,
            "real_tokens": self.real_tokens,
            "pad_tokens": self.pad_tokens,
            "padding_pct": round(100 * self.pad_tokens / capacity, 3) if capacity else 0,
            "packing_efficiency_pct": round(eff, 2),
            "avg_context_usage": round(self.real_tokens / self.total_chunks, 1)
            if self.total_chunks else 0,
        }

    def compression_ratio(self) -> float:
        stored = self.real_tokens * self.itemsize
        return round(self.raw_bytes / stored, 3) if stored else 0.0

    def summary(self) -> dict:
        dupe_pct = (100 * self.chunk_dupe_dropped / self.chunk_dupe_seen
                    if self.chunk_dupe_seen else 0.0)
        return {
            "total_files_seen": self.total_files_seen,
            "files_kept": self.files_kept,
            "fim_docs": self.fim_docs,
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "longest_sequence": self.longest_seq,
            "avg_tokens_per_file": self._avg(self.total_tokens),
            "avg_quality": self._avg(self.quality_sum),
            "avg_functions": self._avg(self.functions_sum),
            "avg_imports": self._avg(self.imports_sum),
            "avg_comments": self._avg(self.comments_sum),
            "languages": dict(self.languages),
            "difficulty_split": dict(self.difficulties),
            "chunk_duplicate_pct": round(dupe_pct, 2),
            "file_drop_reasons": dict(self.file_drops),
            "packing": self.packing(),
            "compression_ratio": self.compression_ratio(),
        }
