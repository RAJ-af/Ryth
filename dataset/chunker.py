"""Module 8 — Chunk Builder.

Encoded token streams ko fixed `seq_len` chunks me todta hai. Do modes:

  * pack (default): documents ko back-to-back jodo (EOS separator ke saath) aur
    exactly seq_len ke chunks kaato. Ye LM pre-training ka standard hai — koi
    padding waste nahi. Har chunk ka metadata us document ka hota hai jahan chunk
    shuru hua.
  * overlap > 0: sliding window (context continuity ke liye).

Har Chunk ke saath Module-10 metadata (language/repo/quality/difficulty/path/…)
attach hota hai.
"""

from __future__ import annotations

from .config import RDEConfig
from .record import Chunk, FileRecord


class ChunkBuilder:
    def __init__(self, config: RDEConfig):
        self.cfg = config

    def build(self, records: list[FileRecord]):
        """FileRecords (already curriculum-ordered) -> yield Chunk objects."""
        seq = self.cfg.seq_len
        step = seq - self.cfg.overlap if self.cfg.overlap else seq

        buf: list[int] = []
        buf_meta: list[dict] = []          # har token ke source-file ki meta

        def make_meta(start_idx: int) -> dict:
            m = buf_meta[start_idx] if start_idx < len(buf_meta) else {}
            return dict(m)

        for rec in records:
            m = self._record_meta(rec)
            buf.extend(rec.token_ids)
            buf_meta.extend([m] * len(rec.token_ids))

            # jitne complete chunks ban sakte hain, nikaal do
            while len(buf) >= seq:
                chunk_ids = buf[:seq]
                meta = make_meta(0)
                yield Chunk(token_ids=chunk_ids, meta=meta)
                buf = buf[step:]
                buf_meta = buf_meta[step:]

        # bacha hua remainder (seq se chhota) — pad karke bhej do taaki data na khoye
        if buf:
            pad_id = getattr(self, "_pad_id", 0)
            meta = make_meta(0)
            meta["padded"] = seq - len(buf)
            chunk_ids = buf + [pad_id] * (seq - len(buf))
            yield Chunk(token_ids=chunk_ids, meta=meta)

    def set_pad_id(self, pad_id: int):
        self._pad_id = pad_id

    @staticmethod
    def _record_meta(rec: FileRecord) -> dict:
        return {
            "repo": rec.repo,
            "path": rec.path,
            "language": rec.language,
            "quality": rec.quality,
            "difficulty": rec.difficulty,
            "functions": rec.meta.get("functions", 0),
            "classes": rec.meta.get("classes", 0),
            "imports": rec.meta.get("imports", []),
        }
