"""Checkpoint metadata + save/load.

Checkpoint me sirf weights nahi hone chahiye — reproducibility ke liye ye bhi:
    Model Version · Tokenizer Hash · Dataset Version · RDS Version
    Git Commit · PyTorch Version · full config

Bahut projects ye baad me add karte hain; Ryth shuru se karta hai taaki purane
checkpoints hamesha traceable rahein.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict

import torch


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def build_checkpoint_metadata(config, *, tokenizer_hash: str | None = None,
                              dataset_version: str | None = None,
                              rds_version: int | None = None) -> dict:
    """Reproducibility metadata for a checkpoint."""
    return {
        "model_version": config.model_version,
        "architecture_version": config.architecture_version,
        "checkpoint_version": config.checkpoint_version,
        "tokenizer_hash": tokenizer_hash,
        "dataset_version": dataset_version,
        "rds_version": rds_version,
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "config": asdict(config),
    }


def save_checkpoint(path: str, model, config, *, step: int = 0,
                    optimizer_state: dict | None = None,
                    tokenizer_hash: str | None = None,
                    dataset_version: str | None = None,
                    rds_version: int | None = None) -> str:
    """Save model weights + full reproducibility metadata to `path`."""
    ckpt = {
        "model": model.state_dict(),
        "step": step,
        "metadata": build_checkpoint_metadata(
            config, tokenizer_hash=tokenizer_hash,
            dataset_version=dataset_version, rds_version=rds_version),
    }
    if optimizer_state is not None:
        ckpt["optimizer"] = optimizer_state
    torch.save(ckpt, path)
    return path


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """Load a checkpoint dict (model / step / metadata / optimizer)."""
    return torch.load(path, map_location=map_location, weights_only=False)
