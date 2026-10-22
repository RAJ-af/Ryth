"""sft.train — Supervised Fine-Tuning (SFT) training engine wrapper.

Initialises from a base model weights checkpoint:
  torch.load(path) -> {"model": state_dict, "preset": str, "vocab_size": int, "seq_len": int}

Fine-tunes with:
  - Loss mask applied to non-assistant tokens (via target ignore_index = -100)
  - Default learning rate 5e-5, 3 epochs, 5% warmup, cosine decay
  - Reports validation loss on assistant tokens only
  - Saves best.pt checkpoint
"""

from __future__ import annotations

import os
import torch

from model import RythConfig, RythForCausalLM
from training.config import TrainConfig
from training.trainer import Trainer


def load_sft_weights(weights_path: str, device: str = "cpu", target_dtype: str = "fp32"):
    """Load weights from file, returning model and metadata dict."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid weights file at {weights_path}: expected dict")

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    preset = ckpt.get("preset", "ryth_30m")
    vocab_size = ckpt.get("vocab_size")
    seq_len = ckpt.get("seq_len", 1024)

    if vocab_size is None:
        if "tok_emb.weight" in state_dict:
            vocab_size = state_dict["tok_emb.weight"].shape[0]
        elif "wte.weight" in state_dict:
            vocab_size = state_dict["wte.weight"].shape[0]
        else:
            vocab_size = 24000

    mcfg = getattr(RythConfig, preset)(vocab_size=vocab_size)
    if seq_len > mcfg.max_seq_len:
        mcfg.max_seq_len = seq_len

    model = RythForCausalLM(mcfg)
    model.load_state_dict(state_dict)

    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(target_dtype, torch.float32)

    model = model.to(dtype=torch_dtype)

    meta = {
        "preset": preset,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
    }
    return model, meta


def train_sft(
    data_dir: str,
    weights_path: str,
    out_dir: str = "runs/sft",
    epochs: int = 3,
    lr: float = 5e-5,
    micro_batch_size: int = 4,
    grad_accum_steps: int = 2,
    dtype: str = "fp32",
    device: str = "auto",
    seed: int = 1337,
    eval_every: int = 50,
    save_every: int = 100,
    log_every: int = 10,
    seq_len: int | None = None,
) -> float:
    """Run SFT training loop on pretrained model."""
    os.makedirs(out_dir, exist_ok=True)

    train_data_dir = os.path.join(data_dir, "train") if os.path.exists(os.path.join(data_dir, "train")) else data_dir
    val_data_dir = os.path.join(data_dir, "val") if os.path.exists(os.path.join(data_dir, "val")) else None

    model, w_meta = load_sft_weights(weights_path, device=device, target_dtype=dtype)
    model_seq_len = seq_len or w_meta["seq_len"]

    tmp_cfg = TrainConfig(
        data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seq_len=model_seq_len,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        dtype=dtype,
        device=device,
        seed=seed,
    )

    from training.dataloader import make_dataloaders
    train_loader, val_loader, manifest = make_dataloaders(tmp_cfg)

    steps_per_epoch = max(1, len(train_loader) // grad_accum_steps)
    max_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(0.05 * max_steps))

    config = TrainConfig(
        data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        model_preset=w_meta["preset"],
        seq_len=model_seq_len,
        lr=lr,
        scheduler="cosine",
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        dtype=dtype,
        out_dir=out_dir,
        eval_every=min(eval_every, max_steps),
        save_every=min(save_every, max_steps),
        log_every=min(log_every, max_steps),
        seed=seed,
        device=device,
    )

    trainer = Trainer(config, model=model)
    best_val_loss = trainer.train()

    best_pt_path = os.path.join(out_dir, "best.pt")
    save_dict = {
        "model": trainer.model.state_dict(),
        "preset": w_meta["preset"],
        "vocab_size": w_meta["vocab_size"],
        "seq_len": model_seq_len,
        "best_val_loss": best_val_loss,
        "step": trainer.step,
    }
    torch.save(save_dict, best_pt_path)
    print(f"[sft train] Saved best.pt to {best_pt_path} (val_loss={best_val_loss:.4f})")

    return best_val_loss
