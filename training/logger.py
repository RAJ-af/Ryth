"""Logger — console + JSONL (+ optional TensorBoard).

JSONL har run ka permanent, machine-readable record deta hai. Console line human
ke liye. TensorBoard optional (config.tensorboard + torch.utils.tensorboard).
"""

from __future__ import annotations

import json
import os


class Logger:
    def __init__(self, out_dir: str, tensorboard: bool = False):
        os.makedirs(out_dir, exist_ok=True)
        self._path = os.path.join(out_dir, "metrics.jsonl")
        self._f = open(self._path, "a", encoding="utf-8")
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(out_dir, "tb"))
            except Exception as e:                       # noqa: BLE001
                print(f"[logger] tensorboard unavailable: {e}")

    def log(self, step: int, metrics: dict, kind: str = "train"):
        row = {"step": step, "kind": kind, **metrics}
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()
        if self.tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(f"{kind}/{k}", v, step)
        msg = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in metrics.items())
        print(f"[{kind:5s} {step:>7}] {msg}")

    def close(self):
        self._f.close()
        if self.tb:
            self.tb.close()
