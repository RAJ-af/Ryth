"""Trainer — the main training loop for Ryth.

Ties together: model (model core) + data (RDS) + optimizer + scheduler + precision
+ gradient + loss + evaluator + logger + checkpoint + callbacks.

Features: gradient accumulation, bf16/fp16 autocast, gradient clipping, NaN
detection & skip, gradient checkpointing, deterministic seeding, auto-resume,
validation + perplexity, early stopping, experiment tracking. CPU & GPU.
"""

from __future__ import annotations

import math

import torch

from model import RythConfig, RythForCausalLM

from .callbacks import CallbackList, EarlyStopping
from .checkpoint import CheckpointManager, experiment_metadata
from .dataloader import make_dataloaders
from .evaluator import evaluate
from .gradient import GradientManager
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

        # --- state ---
        self.step = 0
        self.best_val = math.inf
        if config.resume:
            self._resume(config.resume)

        n = self.model.num_params()
        print(f"[trainer] device={self.device} dtype={config.dtype} "
              f"params={n/1e6:.1f}M vocab={vocab} preset={config.model_preset} "
              f"eff_batch={config.effective_batch} tokens/step={config.tokens_per_step}")

    # ------------------------------------------------------------------ #
    def _set_seed(self):
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

    def _resume(self, resume):
        path = self.ckpt.resolve(resume)
        if not path:
            print(f"[trainer] resume '{resume}' not found — fresh start")
            return
        state = self.ckpt.load(path, model=self.model, optimizer=self.optimizer,
                               map_location=self.device)
        self.step = state.get("step", 0)
        self.best_val = state.get("best_val", math.inf)
        print(f"[trainer] resumed from {path} @ step {self.step}")

    def _infinite(self, loader):
        while True:
            for batch in loader:
                yield batch

    def _save(self, tag=None):
        self.ckpt.save(model=self.model, optimizer=self.optimizer, step=self.step,
                       best_val=self.best_val, config=self.cfg,
                       metadata=self.exp_meta, tag=tag)

    # ------------------------------------------------------------------ #
    def train(self):
        cfg = self.cfg
        self.model.train()
        data = self._infinite(self.train_loader)
        self.callbacks.on_train_begin(self)
        self.metrics.reset()

        while self.step < cfg.max_steps:
            lr = self.scheduler.lr_at(self.step)
            set_lr(self.optimizer, lr)
            self.grad.zero_grad(self.optimizer)

            # ---- gradient accumulation ----
            accum_loss, n_ok = 0.0, 0
            for _ in range(cfg.grad_accum_steps):
                inp, tgt = next(data)
                inp, tgt = inp.to(self.device), tgt.to(self.device)
                with self.autocast():
                    logits, _ = self.model(inp)
                    loss = lm_cross_entropy(logits, tgt) / cfg.grad_accum_steps
                if self.grad.backward(loss, self.scaler):
                    accum_loss += loss.item()
                    n_ok += 1

            # ---- clip + step (skip if all micro-steps were NaN) ----
            if n_ok > 0:
                self.grad.step(self.model, self.optimizer, self.scaler)
            else:
                self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            self.metrics.update(accum_loss, cfg.tokens_per_step)

            # ---- logging ----
            if self.step % cfg.log_every == 0:
                snap = self.metrics.snapshot(lr, self.grad.last_grad_norm)
                self.logger.log(self.step, snap)
                self.callbacks.on_step_end(self, self.step, snap)
                self.metrics.reset()

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
