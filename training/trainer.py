"""Trainer — the main training loop for Ryth.

Ties together: model (model core) + data (RDS) + optimizer + scheduler + precision
+ gradient + loss + evaluator + logger + checkpoint + callbacks.

Features: gradient accumulation, bf16/fp16 autocast, gradient clipping, NaN
detection & skip, gradient checkpointing, deterministic seeding, auto-resume,
validation + perplexity, early stopping, experiment tracking. CPU & GPU.
"""

from __future__ import annotations

import math
import random

import torch

try:
    import numpy as np
except ImportError:
    np = None

from model import RythConfig, RythForCausalLM

from .callbacks import CallbackList, EarlyStopping
from .checkpoint import CheckpointManager, experiment_metadata
from .dataloader import make_dataloaders
from .evaluator import evaluate
from .gradient import GradientManager, clip_grad_norm, grads_are_finite
from .logger import Logger
from .loss import lm_cross_entropy
from .metrics import MetricsTracker
from .optimizer import build_optimizer
from .precision import autocast_context, make_grad_scaler
from .scheduler import build_scheduler, set_lr


def pick_device(want: str) -> str:
    if want != "auto":
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(self, config, model=None):
        """`model` optional — agar diya jaaye (pre-built RythForCausalLM) to wahi
        use hota hai (fine-tuning / custom sizes); warna preset se banta hai."""
        self.cfg = config
        self.device = pick_device(config.device)
        self._set_seed()

        # --- data ---
        self.train_loader, self.val_loader, self.manifest = make_dataloaders(config)
        vocab = self.manifest["vocab_size"]

        # --- model ---
        if model is None:
            # preset ka max_seq_len rakho, par training seq_len bada ho to extend karo
            mcfg = getattr(RythConfig, config.model_preset)(
                vocab_size=vocab,
                use_gradient_checkpointing=config.grad_checkpointing)
            if config.seq_len > mcfg.max_seq_len:
                mcfg.max_seq_len = config.seq_len
            model = RythForCausalLM(mcfg)
        self.model = model.to(self.device)
        self.model_version = self.model.config.model_version

        # --- optim / sched / precision / grad ---
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(config)
        self.autocast = autocast_context(self.device, config.dtype)
        self.scaler = make_grad_scaler(self.device, config.dtype)
        self.grad = GradientManager(config.grad_clip)

        # --- infra ---
        self.logger = Logger(config.out_dir, config.tensorboard)
        self.ckpt = CheckpointManager(config.out_dir, config.keep_last)
        self.metrics = MetricsTracker()
        self.callbacks = CallbackList()
        if config.early_stop_patience:
            self.callbacks.add(EarlyStopping(config.early_stop_patience))

        lock = self.manifest.get("lock", {})
        self.exp_meta = experiment_metadata(
            config, model_version=self.model_version,
            tokenizer_hash=lock.get("tokenizer_hash"),
            dataset_version=lock.get("dataset_version"),
            rds_version=self.manifest.get("rds_version"))

        # Checkpoints cover completed updates, not callback/early-stop or metrics
        # history; stochastic worker transforms and changed datasets cannot resume
        # bit-for-bit even when the sampler cursor can be restored.
        self.step = 0
        self.best_val = math.inf
        if config.resume:
            self._resume(config.resume)

        n = self.model.num_params()
        print(f"[trainer] device={self.device} dtype={config.dtype} "
              f"params={n/1e6:.1f}M vocab={vocab} preset={config.model_preset} "
              f"eff_batch={config.effective_batch}")

    # ------------------------------------------------------------------ #
    def _set_seed(self):
        random.seed(self.cfg.seed)
        if np is not None:
            np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
        if self.cfg.deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _data_identity(self):
        return {"train": getattr(self.train_loader, "resume_data_identity", None),
                "val": getattr(self.val_loader, "resume_data_identity", None),
                "manifest": self.manifest}

    def _validate_resume(self, state):
        # Fields that change update semantics or the data order. Output/logging
        # settings and worker count may change without invalidating continuation.
        fields = ("data_dir", "val_data_dir", "val_fraction", "model_preset", "seq_len",
                  "optimizer", "lr", "weight_decay", "beta1", "beta2", "eps",
                  "scheduler", "warmup_steps", "min_lr_ratio",
                  "micro_batch_size", "grad_accum_steps", "grad_clip", "dtype",
                  "grad_checkpointing", "curriculum", "seed", "deterministic")
        if self.cfg.optimizer == "galore":
            fields += ("galore_rank", "galore_update_interval", "galore_scale")
        saved = state.get("config", {})
        for key in fields:
            if key in saved and saved[key] != getattr(self.cfg, key):
                raise ValueError(f"incompatible checkpoint configuration: {key}")
        if "max_steps" in saved and saved["max_steps"] != self.cfg.max_steps:
            print("[trainer] max_steps changed; resumed learning-rate schedule differs")
        if state.get("data_identity") is not None:
            if state["data_identity"] != self._data_identity():
                raise ValueError("incompatible checkpoint dataset identity")
        else:
            print("[trainer] checkpoint has no dataset identity; data cannot be verified")
        loader_state = state.get("train_loader_state")
        if loader_state is not None:
            # Validate before applying any checkpoint state.
            sampler = self.train_loader.batch_sampler
            current = sampler.state_dict()
            sampler.load_state_dict(loader_state)
            sampler.load_state_dict(current)

    def _resume(self, resume):
        path = self.ckpt.resolve(resume)
        if not path:
            print(f"[trainer] resume '{resume}' not found — fresh start")
            return
        state = self.ckpt.load(path, model=self.model, optimizer=self.optimizer,
                               scaler=self.scaler, map_location=self.device,
                               validate=self._validate_resume)
        loader_state = state.get("train_loader_state")
        if loader_state is not None:
            self.train_loader.batch_sampler.load_state_dict(loader_state)
        else:
            print("[trainer] checkpoint has no train loader cursor; restarting data order")
        self.step = state.get("step", 0)
        self.best_val = state.get("best_val", math.inf)
        print(f"[trainer] resumed from {path} @ step {self.step}")

    def _infinite(self, loader):
        while True:
            for batch in loader:
                # DataLoader can prefetch ahead; advance only on actual consumption.
                loader.batch_sampler.consumed()
                yield batch

    def _save(self, tag=None):
        self.ckpt.save(model=self.model, optimizer=self.optimizer, step=self.step,
                       best_val=self.best_val, config=self.cfg,
                       metadata=self.exp_meta, tag=tag, scaler=self.scaler,
                       train_loader_state=self.train_loader.batch_sampler.state_dict(),
                       data_identity=self._data_identity())

    def _optimizer_step(self):
        # unscale_ records fp16 overflows for scaler.update(), even on a skip.
        self.scaler.unscale_(self.optimizer)
        finite = grads_are_finite(self.model)
        if finite:
            norm = clip_grad_norm(self.model, self.cfg.grad_clip)
            self.grad.last_grad_norm = norm
            finite = math.isfinite(norm) and grads_are_finite(self.model)
        if not finite:
            self.grad.nan_skips += 1
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()
            return False
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return True

    def _discard_partial_gradients(self, had_backward):
        # A nonfinite later loss can follow an overflowing scaled backward. Record
        # that overflow and reset GradScaler's per-optimizer stage before retrying.
        if had_backward and self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
            if grads_are_finite(self.model):
                self.scaler.update(new_scale=self.scaler.get_scale())
            else:
                self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ #
    def train(self):
        cfg = self.cfg
        self.model.train()
        data = self._infinite(self.train_loader)
        self.callbacks.on_train_begin(self)
        self.metrics.reset()
        consecutive_skips = 0
        window_targets = 0

        while self.step < cfg.max_steps:
            lr = self.scheduler.lr_at(self.step)
            set_lr(self.optimizer, lr)
            self.grad.zero_grad(self.optimizer)

            # ---- gradient accumulation ----
            accum_loss, valid = 0.0, True
            input_tokens = target_tokens = 0
            had_backward = False
            for _ in range(cfg.grad_accum_steps):
                inp, tgt = next(data)
                input_tokens += inp.numel()
                target_tokens += int((tgt != -100).sum().item())
                inp, tgt = inp.to(self.device), tgt.to(self.device)
                with self.autocast():
                    logits, _ = self.model(inp)
                    loss = lm_cross_entropy(logits, tgt) / cfg.grad_accum_steps
                if not self.grad.backward(loss, self.scaler):
                    valid = False
                    break
                had_backward = True
                accum_loss += loss.item()

            # Never apply a partially accumulated update. Invalid attempts consume
            # data but not scheduler/update steps. Bound retries on bad datasets.
            if not valid:
                self._discard_partial_gradients(had_backward)
            if not valid or not self._optimizer_step():
                self.optimizer.zero_grad(set_to_none=True)
                consecutive_skips += 1
                if consecutive_skips >= 100:
                    raise RuntimeError("100 consecutive invalid training updates")
                continue
            consecutive_skips = 0
            self.step += 1
            self.metrics.update(accum_loss, input_tokens)
            window_targets += target_tokens

            # ---- logging ----
            if self.step % cfg.log_every == 0:
                snap = self.metrics.snapshot(lr, self.grad.last_grad_norm)
                # Successful updates only; throughput counts actual causal inputs.
                snap["input_tokens"] = self.metrics.tokens
                snap["target_tokens"] = window_targets
                self.logger.log(self.step, snap)
                self.callbacks.on_step_end(self, self.step, snap)
                self.metrics.reset()
                window_targets = 0

            # ---- eval + best-checkpoint + early stop ----
            if self.step % cfg.eval_every == 0:
                em = evaluate(self.model, self.val_loader, self.device,
                              self.autocast, cfg.eval_steps)
                self.logger.log(self.step, em, kind="eval")
                self.callbacks.on_eval_end(self, self.step, em)
                if em["val_loss"] < self.best_val:
                    self.best_val = em["val_loss"]
                    self._save(tag="best")
                if self.callbacks.should_stop:
                    print(f"[trainer] early stop @ step {self.step}")
                    break

            # ---- periodic checkpoint ----
            if self.step % cfg.save_every == 0:
                self._save()

        self._save(tag="final")
        self.callbacks.on_train_end(self)
        self.logger.close()
        print(f"[trainer] done @ step {self.step}  best_val={self.best_val:.4f}")
        return self.best_val
