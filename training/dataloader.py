"""Training data layer — RDSDataset ko PyTorch DataLoader me laata hai.

Har RDS chunk `seq_len` tokens ka hai. Causal LM training ke liye:
    input  = tokens[:-1]
    target = tokens[1:]          (agla token predict karo)

mmap se lazy padhta hai (RAM safe). Train/val split deterministic hai (seed se),
aur curriculum on ho to easy->hard order.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from dataset import RDSDataset


class RDSTorchDataset(Dataset):
    def __init__(self, rds: RDSDataset, indices: list[int]):
        self.rds = rds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        arr = self.rds[idx]                          # token ids (mmap, lazy)
        ids = torch.tensor(arr, dtype=torch.long)
        mask_arr = self.rds.get_mask(idx)
        if mask_arr is not None:
            mask = torch.tensor(mask_arr, dtype=torch.long)
            tgt = ids[1:].clone()
            tgt[mask[1:] == 0] = -100
            return ids[:-1], tgt
        return ids[:-1], ids[1:]                     # input, target (shifted)


class ResumableBatchSampler(Sampler):
    """Epoch order is seed-derived; cursor counts consumed, not prefetched batches.

    Exact continuation assumes an unchanged dataset/split and deterministic items.
    Worker-local random transforms and distributed sharding are not restored.
    """
    def __init__(self, size, batch_size, shuffle, seed):
        self.size, self.batch_size = size, batch_size
        self.shuffle, self.seed = shuffle, seed
        self.epoch, self.cursor = 0, 0
        if size < batch_size:
            raise ValueError("training dataset must contain at least one full batch")

    def __len__(self):
        return self.size // self.batch_size

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.size, generator=generator).tolist()
        else:
            order = range(self.size)
        for batch in range(self.cursor, len(self)):
            start = batch * self.batch_size
            yield list(order[start:start + self.batch_size])

    def consumed(self):
        self.cursor += 1
        if self.cursor == len(self):
            self.epoch += 1
            self.cursor = 0

    def state_dict(self):
        return dict(size=self.size, batch_size=self.batch_size, shuffle=self.shuffle,
                    seed=self.seed, epoch=self.epoch, cursor=self.cursor)

    def load_state_dict(self, state):
        for key in ("size", "batch_size", "shuffle", "seed"):
            if state[key] != getattr(self, key):
                raise ValueError(f"incompatible train loader {key} on resume")
        if state["epoch"] < 0 or not 0 <= state["cursor"] < len(self):
            raise ValueError("invalid train loader cursor on resume")
        self.epoch, self.cursor = state["epoch"], state["cursor"]


def _dataset_identity(rds, indices):
    """Cheap identity guard, not a full checksum verification of shard contents.

    RDS footers contain content hashes. File size/mtime also detect usual in-place
    edits, but deliberately forged footers or timestamps are outside this guard.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(rds.manifest, sort_keys=True).encode())
    for index in indices:
        digest.update(int(index).to_bytes(8, "little", signed=False))
    for shard in rds.manifest.get("shards", []):
        for name in (shard.get("file"), shard.get("mask_file")):
            if not name:
                continue
            path = os.path.join(rds.data_dir, name)
            digest.update(name.encode())
            if not os.path.exists(path):
                digest.update(b"missing")
                continue
            stat = os.stat(path)
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
            with open(path, "rb") as f:
                f.seek(max(0, stat.st_size - 36))
                digest.update(f.read())
    return digest.hexdigest()


def make_dataloaders(config):
    """Train + val DataLoaders (+ manifest). Curriculum ordering optional."""
    train_rds = RDSDataset(config.data_dir)
    n = len(train_rds)

    if config.val_data_dir:
        val_rds = RDSDataset(config.val_data_dir)
        train_idx = list(range(n))
        val_idx = list(range(len(val_rds)))
    else:
        order = train_rds.indices(shuffle=True, seed=config.seed)   # deterministic
        n_val = max(1, int(n * config.val_fraction))
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        val_rds = train_rds

    # Curriculum: easy -> hard ordering (no shuffle so order is preserved).
    if config.curriculum:
        from .curriculum import curriculum_order
        train_idx = curriculum_order(train_rds, train_idx)
        shuffle = False
    else:
        shuffle = True

    pin = torch.cuda.is_available()
    sampler = ResumableBatchSampler(len(train_idx), config.micro_batch_size,
                                    shuffle, config.seed)
    train_loader = DataLoader(
        RDSTorchDataset(train_rds, train_idx), batch_sampler=sampler,
        num_workers=config.num_workers, pin_memory=pin,
        generator=torch.Generator().manual_seed(config.seed))
    train_loader.resume_data_identity = _dataset_identity(train_rds, train_idx)
    val_loader = DataLoader(
        RDSTorchDataset(val_rds, val_idx), batch_size=config.micro_batch_size,
        shuffle=False, num_workers=config.num_workers, drop_last=False, pin_memory=pin,
        generator=torch.Generator().manual_seed(config.seed + 1))
    val_loader.resume_data_identity = _dataset_identity(val_rds, val_idx)
    return train_loader, val_loader, train_rds.manifest
