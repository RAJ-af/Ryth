"""Opt-in FSDP experiment, separate from Ryth's trainer.

Example (run from the repository root):
  torchrun --standalone --nproc-per-node=2 -m experiments.fsdp_train \
      --data-dir /path/to/rds --model-config model.json --steps 100 \
      --checkpoint /shared/run/final.pt

model.json is a JSON object accepted by RythConfig; vocab_size must match the
RDS manifest. All ranks must see the same immutable RDS data and configuration.
This is CUDA/NCCL FULL_SHARD only, >=2 ranks, one GPU per process. Epochs drop
incomplete global batches; there is no validation, resume, or optimizer export.
The optional checkpoint is a synchronous rank-zero *weights-only export*, not a
training checkpoint and not compatible with exact continuation. Never share its
path with another job. FSDP collectives require every rank to participate; a
rank failure can abort/hang the job until torchrun/NCCL times out.

Initialization constructs a complete model in CPU RAM on every rank, then
wraps blocks individually. It is not scalable/meta initialization: budget CPU
RAM per process and GPU transient all-gathers/activations. Weight export gathers
full FP32 weights on rank zero CPU and can OOM; omit --checkpoint if unsafe.
BF16 hardware support is required for bf16. FP16 is deliberately unsupported:
a normal GradScaler is unsafe for sharded overflow handling, and this experiment
does not implement a checkpointable distributed scaler. Gradient accumulation
synchronizes every microbatch instead of no_sync (avoids full-gradient memory).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

from dataset import RDSDataset
from dataset.rds import HEADER_SIZE, INDEX_SIZE
from model import RythConfig, RythForCausalLM
from model.transformer_block import TransformerBlock
from training.dataloader import RDSTorchDataset
from training.scheduler import WarmupCosine, set_lr


@dataclass
class ExperimentConfig:
    data_dir: str
    model_config: str
    steps: int = 100
    batch_size: int = 1
    accumulation: int = 1
    dtype: str = "bf16"
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 0
    grad_clip: float = 1.0
    seed: int = 1337
    checkpoint: str | None = None
    resume: str | None = None

    def validate(self):
        if self.steps < 1 or self.batch_size < 1 or self.accumulation < 1:
            raise ValueError("steps, batch_size, and accumulation must be positive")
        if self.dtype not in ("fp32", "bf16"):
            raise ValueError("only fp32 and bf16 are supported; fp16 needs a sharded scaler")
        if not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be between 0 and steps")
        if (not all(math.isfinite(x) for x in (self.lr, self.weight_decay, self.grad_clip))
                or not (self.lr > 0 and self.weight_decay >= 0 and self.grad_clip >= 0)):
            raise ValueError("invalid optimizer or gradient clipping settings")
        if self.resume is not None:
            raise ValueError("resume is unsupported; checkpoints are weights-only exports")


def make_sampler(dataset, rank: int, world_size: int, seed: int, batch_size: int):
    """Identical shuffled permutation split without padding; remainder is dropped."""
    if world_size < 2 or not 0 <= rank < world_size or batch_size < 1:
        raise ValueError("invalid distributed sampler parameters")
    if len(dataset) < world_size * batch_size:
        raise ValueError("RDS must contain at least one complete global batch")
    return DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                              shuffle=True, seed=seed, drop_last=True)


def validate_data(rds, model_config):
    if rds.seq_len < 2 or rds.seq_len - 1 > model_config.max_seq_len:
        raise ValueError("RDS sequence length must be >=2 and fit the model context")
    if rds.manifest["vocab_size"] != model_config.vocab_size:
        raise ValueError("RDS and model vocabulary sizes differ")
    for shard, reader, mask in zip(rds.manifest["shards"], rds._readers, rds._mask_readers):
        if reader.seq_len != rds.seq_len or reader.vocab_size != model_config.vocab_size:
            raise ValueError("RDS shard header does not match manifest")
        if shard.get("mask_file") and mask is None:
            raise ValueError("missing RDS mask shard")
        if mask is not None and (mask.seq_len != rds.seq_len or len(mask) != len(reader)):
            raise ValueError("RDS mask shape mismatch")
        for current in (reader, mask):
            if current is None:
                continue
            # A valid-length index entry must also point into the data section;
            # otherwise mmap slicing can silently read index/metadata as tokens.
            if (current.dtype_flag not in (0, 1) or current.data_offset < HEADER_SIZE
                    or not current.data_offset <= current.index_offset <= current.meta_offset
                    <= current.footer_offset <= len(current._mm) - 36
                    or current.index_offset + len(current) * INDEX_SIZE > current.meta_offset):
                raise ValueError("invalid RDS shard layout")
            if any(length != rds.seq_len or off < current.data_offset
                   or off + length * current.itemsize > current.index_offset
                   for off, length in current._index):
                raise ValueError("RDS chunk shape or data offset mismatch")
    if model_config.use_gradient_checkpointing:
        raise ValueError("model activation checkpointing is not supported by this experiment")


def _batches(loader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def export_weights(model, path, step, model_config, cfg, manifest):
    # Both state_dict() and the failure broadcast are collective. A rank-0 I/O
    # failure is communicated before any rank leaves; atomic replace avoids a
    # partially written final file. Process death/OOM cannot be recovered here.
    settings = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, settings):
        state = model.state_dict()
    error = [None]
    if dist.get_rank() == 0:
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": state, "step": step, "model_config": asdict(model_config),
                        "experiment_config": asdict(cfg), "rds_manifest": manifest,
                        "format": "fsdp-weights-only-v1"}, tmp)
            os.replace(tmp, path)
        except Exception as exc:
            error[0] = f"weight export failed: {exc}"
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError as exc:
                if error[0] is None:
                    error[0] = f"weight export cleanup failed: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def train(cfg: ExperimentConfig):
    cfg.validate()
    if not all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise RuntimeError("launch using torchrun with at least two CUDA ranks")
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world < 2 or not torch.cuda.is_available() or local >= torch.cuda.device_count():
        raise RuntimeError("FSDP experiment requires >=2 ranks and one CUDA GPU per local rank")
    torch.cuda.set_device(local)
    if cfg.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requires a CUDA GPU with native bf16 support")
    device = torch.device("cuda", local)
    dist.init_process_group("nccl")
    rds = None
    try:
        with open(cfg.model_config, encoding="utf-8") as f:
            model_config = RythConfig(**json.load(f))
        rds = RDSDataset(cfg.data_dir)
        validate_data(rds, model_config)
        dataset = RDSTorchDataset(rds, range(len(rds)))
        sampler = make_sampler(dataset, rank, world, cfg.seed, cfg.batch_size)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler,
                            drop_last=True, num_workers=0, pin_memory=True)
        batches = _batches(loader, sampler)
        # Same CPU seed makes each rank's initial full model identical.
        torch.manual_seed(cfg.seed)
        base = RythForCausalLM(model_config)
        total_params = sum(p.numel() for p in base.parameters())
        # Record grouping before FSDP changes original parameter shapes to shards.
        decay_ids = {id(p) for p in base.parameters() if p.dim() >= 2}
        mp = (MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                             buffer_dtype=torch.float32) if cfg.dtype == "bf16" else None)
        model = FSDP(base, auto_wrap_policy=partial(transformer_auto_wrap_policy,
                     transformer_layer_cls={TransformerBlock}),
                     sharding_strategy=ShardingStrategy.FULL_SHARD,
                     mixed_precision=mp, device_id=device, use_orig_params=True)
        groups = [{"params": [p for p in model.parameters() if (id(p) in decay_ids) == decay],
                   "weight_decay": cfg.weight_decay if decay else 0.0}
                  for decay in (True, False)]
        optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)
        scheduler = WarmupCosine(cfg.lr, cfg.warmup_steps, cfg.steps)
        # Distinct stochastic layers per rank, after identical initialization.
        torch.cuda.manual_seed(cfg.seed + rank)
        model.train()
        for step in range(cfg.steps):
            optimizer.zero_grad(set_to_none=True)
            microbatches = [next(batches) for _ in range(cfg.accumulation)]
            # Normalize by global number of unmasked tokens, including across
            # accumulation. FSDP averages gradients across ranks.
            valid = sum(int((y != -100).sum()) for _, y in microbatches)
            bad = any(bool(((x < 0) | (x >= model_config.vocab_size)).any()) or
                      bool(((y != -100) & ((y < 0) | (y >= model_config.vocab_size))).any())
                      for x, y in microbatches)
            counts = torch.tensor([valid, int(bad)], device=device, dtype=torch.long)
            dist.all_reduce(counts)
            if counts[1].item() or not counts[0].item():
                raise ValueError("invalid token IDs or no unmasked targets in global batch")
            scale = world / counts[0].item()
            for x, y in microbatches:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.dtype == "bf16"):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                           y.reshape(-1), ignore_index=-100, reduction="sum")
                (loss * scale).backward()
            norm = model.clip_grad_norm_(cfg.grad_clip if cfg.grad_clip else float("inf"))
            if not torch.isfinite(norm):
                raise RuntimeError("non-finite distributed gradient norm; aborting")
            set_lr(optimizer, scheduler.lr_at(step))
            optimizer.step()
            if rank == 0:
                print(f"step={step + 1} grad_norm={norm.item():.4f}", flush=True)
        # Report local (not all-gathered) parameter and Adam moment storage.
        local_params = sum(p.numel() for p in model.parameters())
        local_moments = sum(state["exp_avg"].numel() for state in optimizer.state.values()
                            if "exp_avg" in state)
        print(f"rank={rank} local_params={local_params} local_adam_moments={local_moments} "
              f"total_params={total_params}", flush=True)
        if cfg.checkpoint:
            export_weights(model, cfg.checkpoint, cfg.steps, model_config, cfg, rds.manifest)
    finally:
        if rds is not None:
            rds.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()
    try:
        train(ExperimentConfig(**vars(args)))
    except (ValueError, RuntimeError, OSError, TypeError, AssertionError) as exc:
        parser.exit(1, f"FSDP experiment error: {exc}\n")


if __name__ == "__main__":
    main()
