"""Training-time reader — Modules 13 (streaming), 14 (mmap), 16 (random access).

`RDSDataset` ek manifest folder ko kholता hai aur saare shards ke chunks ko ek
flat, randomly-addressable dataset ki tarah expose karta hai — sab mmap se, RAM
me kuch load kiye bina. Features:

  ✅ Memory-mapped (100GB+ dataset bhi chalega)
  ✅ Lazy loading (sirf maanga hua chunk padho)
  ✅ Deterministic shuffling (seed => resume par same order)
  ✅ Multi-worker sharding (worker_id / num_workers se data baant lo)

Framework-agnostic: har item token-ids (array) deta hai. PyTorch me isko
`torch.utils.data.Dataset` me wrap karke DataLoader(num_workers=…) laga do.
"""

from __future__ import annotations

import json
import os

from .rds import RDSReader


class RDSDataset:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.seq_len = self.manifest["seq_len"]
        self._readers: list[RDSReader] = []
        self._flat: list[tuple[int, int]] = []       # global idx -> (shard, local idx)
        for si, sh in enumerate(self.manifest["shards"]):
            r = RDSReader(os.path.join(data_dir, sh["file"]))
            self._readers.append(r)
            for li in range(len(r)):
                self._flat.append((si, li))

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, idx: int):
        shard, local = self._flat[idx]
        return self._readers[shard][local]           # array of token ids (lazy, mmap)

    def meta(self, idx: int) -> dict:
        shard, local = self._flat[idx]
        return self._readers[shard].metadata(local)

    def indices(self, *, shuffle: bool = True, seed: int = 0,
                worker_id: int = 0, num_workers: int = 1):
        """Iteration order do — deterministic (resume-safe) + worker-partitioned."""
        order = list(range(len(self)))
        if shuffle:
            import random
            random.Random(seed).shuffle(order)         # same seed => same order
        return order[worker_id::num_workers]           # har worker apna slice

    def iter_chunks(self, **kw):
        for idx in self.indices(**kw):
            yield self[idx]

    def verify(self) -> bool:
        return all(r.verify_checksum() for r in self._readers)

    def close(self):
        for r in self._readers:
            r.close()
