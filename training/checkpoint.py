"""Checkpoint manager — save / load / auto-resume + experiment tracking.

Ek checkpoint me sab kuch hota hai jisse training bit-for-bit resume ho: model,
optimizer, step, best_val, RNG states, config, aur experiment metadata (git commit,
tokenizer hash, dataset version, model version, torch version).

`keep_last` se sirf recent N checkpoints disk par rehte hain (space bachao).
"latest.pt" hamesha newest ki taraf point karta hai (auto-resume ke liye).
"""

from __future__ import annotations

import glob
import os
import subprocess

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


def experiment_metadata(config, *, model_version=None, tokenizer_hash=None,
                        dataset_version=None, rds_version=None) -> dict:
    """Reproducibility trace attached to every checkpoint."""
    return {
        "run_name": config.run_name,
        "model_preset": config.model_preset,
        "model_version": model_version,
        "tokenizer_hash": tokenizer_hash,
        "dataset_version": dataset_version,
        "rds_version": rds_version,
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "seed": config.seed,
    }


class CheckpointManager:
    def __init__(self, out_dir: str, keep_last: int = 3):
        self.out_dir = out_dir
        self.keep_last = keep_last
        os.makedirs(out_dir, exist_ok=True)

    def save(self, *, model, optimizer, step, best_val, config,
             metadata: dict | None = None, tag: str | None = None) -> str:
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val": best_val,
            "config": vars(config),
            "metadata": metadata or {},
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
        }
        name = tag or f"step_{step:07d}"
        path = os.path.join(self.out_dir, f"{name}.pt")
        torch.save(state, path)
        torch.save(state, os.path.join(self.out_dir, "latest.pt"))
        if tag is None:
            self._rotate()
        return path

    def _rotate(self):
        ckpts = sorted(glob.glob(os.path.join(self.out_dir, "step_*.pt")))
        for old in ckpts[:-self.keep_last] if self.keep_last > 0 else []:
            try:
                os.remove(old)
            except OSError:
                pass

    def resolve(self, resume: str | None) -> str | None:
        if resume == "latest":
            latest = os.path.join(self.out_dir, "latest.pt")
            return latest if os.path.exists(latest) else None
        return resume if resume and os.path.exists(resume) else None

    def load(self, path, *, model, optimizer=None, map_location="cpu") -> dict:
        state = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        rng = state.get("torch_rng")
        if rng is not None:
            torch.set_rng_state(rng.to("cpu", torch.uint8) if hasattr(rng, "to") else rng)
        if state.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng"]])
        return state
