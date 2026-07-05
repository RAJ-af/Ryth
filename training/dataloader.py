"""Training data layer — RDSDataset ko PyTorch DataLoader me laata hai.

Har RDS chunk `seq_len` tokens ka hai. Causal LM training ke liye:
    input  = tokens[:-1]
    target = tokens[1:]          (agla token predict karo)

mmap se lazy padhta hai (RAM safe). Train/val split deterministic hai (seed se),
aur curriculum on ho to easy->hard order.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import RDSDataset


class RDSTorchDataset(Dataset):
    def __init__(self, rds: RDSDataset, indices: list[int]):
        self.rds = rds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        arr = self.rds[self.indices[i]]              # token ids (mmap, lazy)
        ids = torch.tensor(arr, dtype=torch.long)
        return ids[:-1], ids[1:]                     # input, target (shifted)


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
    train_loader = DataLoader(
        RDSTorchDataset(train_rds, train_idx), batch_size=config.micro_batch_size,
        shuffle=shuffle, num_workers=config.num_workers, drop_last=True, pin_memory=pin)
    val_loader = DataLoader(
        RDSTorchDataset(val_rds, val_idx), batch_size=config.micro_batch_size,
        shuffle=False, num_workers=config.num_workers, drop_last=False, pin_memory=pin)
    return train_loader, val_loader, train_rds.manifest
