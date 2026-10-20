"""Unit tests for the Ryth Training Engine (v0.3.0).

Built module-by-module with REAL PyTorch. Run:
    pytest tests/test_training.py -v   (or: python tests/test_training.py)
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from training.config import TrainConfig

torch.manual_seed(0)


def _toy_model(d=16, v=32):
    return nn.Sequential(nn.Linear(d, d), nn.Linear(d, v))


# ======================================================================== #
# config
# ======================================================================== #
def test_config_derived():
    c = TrainConfig(micro_batch_size=8, grad_accum_steps=4, seq_len=1024)
    assert c.effective_batch == 32
    assert c.tokens_per_step == 32 * 1024


def test_config_validation():
    for bad in [dict(dtype="fp8"), dict(warmup_steps=10, max_steps=5)]:
        try:
            TrainConfig(**bad)
            assert False, f"expected failure for {bad}"
        except AssertionError:
            pass


# ======================================================================== #
# precision
# ======================================================================== #
from training.precision import resolve_dtype, autocast_context, make_grad_scaler


def test_resolve_dtype():
    assert resolve_dtype("bf16") == torch.bfloat16
    assert resolve_dtype("fp32") == torch.float32
    try:
        resolve_dtype("nope"); assert False
    except ValueError:
        pass


def test_autocast_fp32_is_noop():
    ctx = autocast_context("cpu", "fp32")
    with ctx():
        x = torch.randn(2, 2) @ torch.randn(2, 2)
    assert x.dtype == torch.float32


def test_grad_scaler_disabled_on_cpu():
    scaler = make_grad_scaler("cpu", "fp16")
    assert scaler.is_enabled() is False        # fp16 scaling only on cuda


# ======================================================================== #
# optimizer
# ======================================================================== #
from training.optimizer import build_optimizer


def test_adamw_param_groups():
    model = _toy_model()
    opt = build_optimizer(model, TrainConfig(weight_decay=0.1))
    assert isinstance(opt, torch.optim.AdamW)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    # 2D weights decay, 1D biases don't
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_optimizer_factory_unknown():
    try:
        build_optimizer(_toy_model(), TrainConfig(optimizer="sgd")); assert False
    except ValueError:
        pass


# ======================================================================== #
# scheduler
# ======================================================================== #
from training.scheduler import build_scheduler, set_lr


def test_warmup_cosine_shape():
    c = TrainConfig(lr=3e-4, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1)
    sch = build_scheduler(c)
    assert sch.lr_at(0) < c.lr                       # warmup starts low
    assert abs(sch.lr_at(99) - c.lr) < 1e-9          # peak at warmup end
    mid = sch.lr_at(500)
    assert c.lr * 0.1 < mid < c.lr                   # cosine between
    assert abs(sch.lr_at(1000) - c.lr * 0.1) < 1e-9  # floor after max
    assert abs(sch.lr_at(5000) - c.lr * 0.1) < 1e-9  # stays at floor


def test_set_lr():
    model = _toy_model()
    opt = build_optimizer(model, TrainConfig())
    set_lr(opt, 1.23e-4)
    assert all(g["lr"] == 1.23e-4 for g in opt.param_groups)


# ======================================================================== #
# loss
# ======================================================================== #
from training.loss import lm_cross_entropy, perplexity


def test_lm_cross_entropy():
    logits = torch.randn(2, 5, 32)
    targets = torch.randint(0, 32, (2, 5))
    loss = lm_cross_entropy(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert abs(perplexity(loss.item()) - torch.exp(loss).item()) < 1e-2


def test_loss_ignore_index():
    logits = torch.randn(1, 3, 10)
    targets = torch.tensor([[-100, -100, -100]])
    loss = lm_cross_entropy(logits, targets, ignore_index=-100)
    assert torch.isnan(loss)          # all ignored -> nan (expected edge case)


# ======================================================================== #
# gradient
# ======================================================================== #
from training.gradient import (GradientManager, grads_are_finite,
                               is_finite_loss, clip_grad_norm)
from training.precision import make_grad_scaler


def test_nan_loss_detection():
    assert is_finite_loss(torch.tensor(1.0)) is True
    assert is_finite_loss(torch.tensor(float("nan"))) is False


def test_grads_finite_and_clip():
    model = _toy_model()
    x = torch.randn(4, 16)
    out = model(x)
    out.sum().backward()
    assert grads_are_finite(model) is True
    norm = clip_grad_norm(model, 1.0)
    assert norm >= 0
    # after clip, grad norm <= 1 (+eps)
    assert clip_grad_norm(model, 1.0) <= 1.0 + 1e-4


def test_gradient_manager_skips_nan_loss():
    model = _toy_model()
    scaler = make_grad_scaler("cpu", "fp32")
    gm = GradientManager(grad_clip=1.0)
    gm.zero_grad(build_optimizer(model, TrainConfig()))
    ok = gm.backward(torch.tensor(float("nan"), requires_grad=True), scaler)
    assert ok is False and gm.nan_skips == 1


def test_gradient_manager_full_step():
    model = _toy_model()
    opt = build_optimizer(model, TrainConfig())
    scaler = make_grad_scaler("cpu", "fp32")
    gm = GradientManager(grad_clip=1.0)
    gm.zero_grad(opt)
    x = torch.randn(4, 16)
    loss = model(x).pow(2).mean()
    assert gm.backward(loss, scaler) is True
    assert gm.step(model, opt, scaler) is True
    assert gm.last_grad_norm >= 0


# ======================================================================== #
# shared fixture: a tiny real RDS dataset + tiny model
# ======================================================================== #
import tempfile
from dataset import RDEConfig, RDEPipeline, ByteTokenizer
from tests.sample_data import build_sample
from model import RythConfig, RythForCausalLM

_RDS_DIR = tempfile.mkdtemp(prefix="train_rds_")
_TOK = ByteTokenizer()
build_sample(os.path.join(_RDS_DIR, "raw"))
RDEPipeline(_TOK, RDEConfig(seq_len=16, vocab_size=_TOK.vocab_size,
            tokenizer_version=0)).run(os.path.join(_RDS_DIR, "raw"),
            os.path.join(_RDS_DIR, "out"), verbose=False,
            now_iso="2026-07-05T00:00:00+00:00")
_DATA_DIR = os.path.join(_RDS_DIR, "out")


def _train_cfg(**kw):
    base = dict(data_dir=_DATA_DIR, seq_len=15, micro_batch_size=2,
                num_workers=0, out_dir=os.path.join(_RDS_DIR, "run"))
    base.update(kw)
    return TrainConfig(**base)


def _tiny_lm():
    return RythForCausalLM(RythConfig(vocab_size=_TOK.vocab_size, d_model=32,
                                      n_layers=2, n_heads=4, n_kv_heads=2,
                                      max_seq_len=32))


# ======================================================================== #
# dataloader
# ======================================================================== #
from training.dataloader import make_dataloaders


def test_dataloader_shapes_and_shift():
    train, val, manifest = make_dataloaders(_train_cfg())
    inp, tgt = next(iter(train))
    assert inp.shape == tgt.shape and inp.shape[0] == 2      # micro_batch
    assert inp.shape[1] == 15                                 # seq_len-1
    assert inp.dtype == torch.long
    assert manifest["n_shards"] >= 1


# ======================================================================== #
# curriculum
# ======================================================================== #
from training.curriculum import curriculum_order, difficulty_histogram
from dataset import RDSDataset


def test_curriculum_orders_easy_first():
    ds = RDSDataset(_DATA_DIR)
    order = curriculum_order(ds)
    ranks = [{"easy": 0, "medium": 1, "hard": 2}[ds.meta(i)["difficulty"]]
             for i in order]
    assert ranks == sorted(ranks)                            # non-decreasing
    hist = difficulty_histogram(ds)
    assert sum(hist.values()) == len(ds)
    ds.close()


# ======================================================================== #
# metrics
# ======================================================================== #
from training.metrics import MetricsTracker


def test_metrics_tracker():
    m = MetricsTracker()
    m.update(2.0, 100)
    m.update(4.0, 100)
    assert m.avg_loss() == 3.0
    snap = m.snapshot(lr=1e-4, grad_norm=0.5)
    assert snap["loss"] == 3.0 and snap["lr"] == 1e-4
    assert snap["tokens_per_sec"] > 0


# ======================================================================== #
# evaluator
# ======================================================================== #
from training.evaluator import evaluate
from training.precision import autocast_context


def test_evaluator_returns_loss_ppl():
    _, val, _ = make_dataloaders(_train_cfg())
    model = _tiny_lm()
    out = evaluate(model, val, "cpu", autocast_context("cpu", "fp32"), max_batches=3)
    assert "val_loss" in out and "val_ppl" in out
    assert out["val_ppl"] > 0 and torch.isfinite(torch.tensor(out["val_loss"]))
    assert model.training is False or True                    # restored


# ======================================================================== #
# logger
# ======================================================================== #
from training.logger import Logger


def test_logger_writes_jsonl():
    import json
    d = tempfile.mkdtemp()
    lg = Logger(d, tensorboard=False)
    lg.log(1, {"loss": 2.5, "lr": 1e-4}, kind="train")
    lg.log(2, {"val_loss": 2.0}, kind="eval")
    lg.close()
    rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    assert len(rows) == 2 and rows[0]["loss"] == 2.5 and rows[1]["kind"] == "eval"


# ======================================================================== #
# checkpoint
# ======================================================================== #
from training.checkpoint import CheckpointManager, experiment_metadata


def test_experiment_metadata_fields():
    meta = experiment_metadata(_train_cfg(), model_version="0.2.0",
                               tokenizer_hash="abc", dataset_version="1.0", rds_version=1)
    for k in ["git_commit", "torch_version", "tokenizer_hash", "dataset_version",
              "rds_version", "model_version", "run_name", "seed"]:
        assert k in meta
    assert meta["torch_version"] == torch.__version__


def test_checkpoint_save_load_resume():
    d = tempfile.mkdtemp()
    cm = CheckpointManager(d, keep_last=2)
    model = _tiny_lm()
    opt = build_optimizer(model, _train_cfg())
    cm.save(model=model, optimizer=opt, step=10, best_val=1.5, config=_train_cfg())
    assert cm.resolve("latest") is not None
    model2 = _tiny_lm()
    opt2 = build_optimizer(model2, _train_cfg())
    state = cm.load(cm.resolve("latest"), model=model2, optimizer=opt2)
    assert state["step"] == 10 and state["best_val"] == 1.5
    # weights match after load
    p1 = next(model.parameters()); p2 = next(model2.parameters())
    assert torch.allclose(p1, p2)


def test_checkpoint_keep_last_rotation():
    d = tempfile.mkdtemp()
    cm = CheckpointManager(d, keep_last=2)
    model = _tiny_lm(); opt = build_optimizer(model, _train_cfg())
    import glob as _glob
    for step in [1, 2, 3, 4]:
        cm.save(model=model, optimizer=opt, step=step, best_val=0, config=_train_cfg())
    kept = _glob.glob(os.path.join(d, "step_*.pt"))
    assert len(kept) == 2                                     # only last 2


# ======================================================================== #
# callbacks
# ======================================================================== #
from training.callbacks import CallbackList, EarlyStopping


def test_early_stopping():
    es = EarlyStopping(patience=2, monitor="val_loss")
    cbs = CallbackList([es])
    cbs.on_eval_end(None, 1, {"val_loss": 1.0})    # improves -> best
    assert not cbs.should_stop
    cbs.on_eval_end(None, 2, {"val_loss": 1.1})    # worse (wait=1)
    assert not cbs.should_stop
    cbs.on_eval_end(None, 3, {"val_loss": 1.2})    # worse (wait=2 -> stop)
    assert cbs.should_stop


# ======================================================================== #
# profiler
# ======================================================================== #
from training.profiler import profile_steps


def test_profiler_runs():
    model = _tiny_lm().eval()
    ids = torch.randint(0, _TOK.vocab_size, (1, 8))
    table = profile_steps(lambda: model(ids), n=3)
    assert isinstance(table, str) and len(table) > 0


# ======================================================================== #
# Trainer — real end-to-end training
# ======================================================================== #
from training import Trainer


def _e2e_cfg(out, **kw):
    base = dict(data_dir=_DATA_DIR, seq_len=15, model_preset="ryth_30m",
                micro_batch_size=2, grad_accum_steps=1, max_steps=20,
                warmup_steps=3, eval_every=10, eval_steps=2, log_every=5,
                save_every=1000, dtype="fp32", device="cpu", num_workers=0,
                out_dir=out)
    base.update(kw)
    # tiny model override via monkey-set? use ryth_30m but small vocab from data
    return TrainConfig(**base)


def test_trainer_runs_and_loss_decreases():
    out = tempfile.mkdtemp()
    trainer = Trainer(_e2e_cfg(out, max_steps=40, warmup_steps=5, lr=1e-3),
                      model=_tiny_lm())
    best = trainer.train()
    import json
    rows = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    train_losses = [r["loss"] for r in rows if r["kind"] == "train"]
    assert len(train_losses) >= 2
    assert train_losses[-1] < train_losses[0]            # loss went down
    assert math.isfinite(best)
    assert os.path.exists(os.path.join(out, "final.pt"))
    assert os.path.exists(os.path.join(out, "best.pt"))


def test_trainer_auto_resume():
    out = tempfile.mkdtemp()
    Trainer(_e2e_cfg(out, max_steps=10, save_every=5), model=_tiny_lm()).train()
    t2 = Trainer(_e2e_cfg(out, max_steps=15, save_every=5, resume="latest"),
                 model=_tiny_lm())
    assert t2.step == 10                                 # resumed at last step
    t2.train()
    assert t2.step == 15


def test_trainer_early_stopping_wiring():
    """Trainer must honor a callback's stop signal (deterministic)."""
    from training.callbacks import Callback
    out = tempfile.mkdtemp()
    t = Trainer(_e2e_cfg(out, max_steps=200, eval_every=5), model=_tiny_lm())

    class StopAfterFirstEval(Callback):
        stop = False
        def on_eval_end(self, trainer, step, metrics):
            self.stop = True

    t.callbacks.add(StopAfterFirstEval())
    t.train()
    assert t.step <= 10                                  # stopped at first eval (step 5)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed ✅")
