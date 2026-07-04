"""Module 7 — Encoder.

Raw text -> token IDs (tokenizer se) -> optional BOS/EOS wrap. Binary packing
baad me RDS writer karta hai (uint16). Yahan sirf ids bante hain.
"""

from __future__ import annotations

from .config import RDEConfig
from .record import FileRecord


class Encoder:
    def __init__(self, tokenizer, config: RDEConfig):
        self.tok = tokenizer
        self.cfg = config

    def encode_record(self, rec: FileRecord) -> FileRecord:
        ids = self.tok.encode(rec.text)
        if self.cfg.add_bos and hasattr(self.tok, "bos_id"):
            ids = [self.tok.bos_id] + ids
        if self.cfg.add_eos and hasattr(self.tok, "eos_id"):
            ids = ids + [self.tok.eos_id]
        rec.token_ids = ids
        rec.meta["n_tokens"] = len(ids)
        return rec
