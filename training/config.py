"""Training configuration for the Ryth Training Engine.

Ek single dataclass jo poore training run ko control karti hai. Isi ko badal ke
30M se 1B tak experiments chalao — training code same rehta hai.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # --- data ---
    data_dir: str = "rds_out"           # RDSDataset folder (manifest.json wala)
    val_data_dir: str | None = None     # optional alag validation set
    val_fraction: float = 0.02          # agar val_data_dir None -> train se hold-out

    # --- model ---
    model_preset: str = "ryth_30m"      # RythConfig.<preset>
    seq_len: int = 1024                 # dataset seq_len se match hona chahiye

    # --- optimizer ---
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # --- scheduler ---
    scheduler: str = "cosine"           # warmup + cosine
    warmup_steps: int = 100
    max_steps: int = 2000
    min_lr_ratio: float = 0.1           # cosine floor = lr * ratio

    # --- batch / gradient ---
    micro_batch_size: int = 8           # per forward
    grad_accum_steps: int = 4           # effective batch = micro * accum
    grad_clip: float = 1.0              # 0 disables

    # --- precision / memory ---
    dtype: str = "bf16"                 # "bf16" | "fp16" | "fp32"
    grad_checkpointing: bool = False

    # --- eval / logging / checkpoint ---
    eval_every: int = 200
    eval_steps: int = 50
    log_every: int = 10
    save_every: int = 500
    keep_last: int = 3                  # kitne recent checkpoints rakhne hain
    out_dir: str = "runs/ryth"
    tensorboard: bool = False

    # --- resume / early stopping ---
    resume: str | None = None           # checkpoint path | "latest" | None
    early_stop_patience: int = 0        # 0 = off

    # --- curriculum ---
    curriculum: bool = False            # RDE difficulty easy->hard ordering

    # --- determinism / device ---
    seed: int = 1337
    deterministic: bool = False
    device: str = "auto"                # "auto" | "cuda" | "cpu"
    num_workers: int = 2
    run_name: str = "ryth-run"

    def __post_init__(self):
        assert self.dtype in ("bf16", "fp16", "fp32"), f"bad dtype {self.dtype}"
        assert self.micro_batch_size >= 1 and self.grad_accum_steps >= 1
        assert self.warmup_steps <= self.max_steps

    @property
    def effective_batch(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch * self.seq_len
