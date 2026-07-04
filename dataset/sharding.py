"""Module 15 — Sharding (+ Module 13 streaming write, Module 17 versioning).

Chunks ko multiple shard files me likhta hai. Jab ek shard `shard_max_bytes` tak
pahunch jaata hai, naya shard shuru ho jaata hai — isse chhoti machines par bhi
data manage hota hai aur training multiple workers me baant sakta hai.

Ek `manifest.json` bhi likhta hai: RDS version, tokenizer version, shard list,
per-shard chunk counts + checksums, aur dataset statistics.
"""

from __future__ import annotations

import json
import os

from .config import RDEConfig
from .rds import ShardWriter


class ShardManager:
    def __init__(self, out_dir: str, config: RDEConfig, tokenizer):
        self.out_dir = out_dir
        self.cfg = config
        self.tok = tokenizer
        os.makedirs(out_dir, exist_ok=True)
        self._writer: ShardWriter | None = None
        self._shard_idx = -1
        self.shards: list[dict] = []

    def _shard_path(self, idx: int) -> str:
        return os.path.join(self.out_dir, f"shard_{idx:05d}.rds")

    def _open_new_shard(self):
        self._shard_idx += 1
        self._writer = ShardWriter(
            self._shard_path(self._shard_idx),
            version=self.cfg.rds_version,
            tok_version=getattr(self.tok, "version", self.cfg.tokenizer_version),
            vocab_size=self.tok.vocab_size,
            seq_len=self.cfg.seq_len,
            dtype_flag=self.cfg.dtype_flag,
        )

    def add_chunk(self, token_ids, meta: dict):
        if self._writer is None:
            self._open_new_shard()
        # shard size limit cross ho raha ho to rotate karo
        if self._writer.data_bytes >= self.cfg.shard_max_bytes and self._writer.n_chunks:
            self._finalize_shard()
            self._open_new_shard()
        self._writer.add_chunk(token_ids, meta)

    def _finalize_shard(self):
        w = self._writer
        path = w.path
        n_chunks = w.n_chunks
        w.close()
        self.shards.append({"file": os.path.basename(path), "chunks": n_chunks})
        self._writer = None

    def finalize(self, stats_summary: dict, lock: dict | None = None) -> dict:
        if self._writer is not None and self._writer.n_chunks:
            self._finalize_shard()

        # har shard ka checksum manifest me record karo (verify shortcut)
        from .rds import RDSReader
        for sh in self.shards:
            with RDSReader(os.path.join(self.out_dir, sh["file"])) as r:
                sh["checksum_ok"] = r.verify_checksum()
                sh["tokens"] = sum(r._index[i][1] for i in range(len(r)))

        manifest = {
            "format": "RDS",
            "rds_version": self.cfg.rds_version,
            "tokenizer_version": getattr(self.tok, "version",
                                         self.cfg.tokenizer_version),
            "vocab_size": self.tok.vocab_size,
            "seq_len": self.cfg.seq_len,
            "dtype": "uint16" if self.cfg.dtype_flag == 0 else "uint32",
            "n_shards": len(self.shards),
            "shards": self.shards,
            "lock": lock or {},
            "stats": stats_summary,
        }
        with open(os.path.join(self.out_dir, "manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return manifest
