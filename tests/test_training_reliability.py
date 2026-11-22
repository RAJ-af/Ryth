"""Regression tests for update atomicity, resume and dense model sizing."""
import copy
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import RythConfig, RythForCausalLM
from training.callbacks import Callback
from training.checkpoint import CheckpointManager
from training.config import TrainConfig
from training.dataloader import ResumableBatchSampler, _dataset_identity
from training.trainer import Trainer
import training.trainer as trainer_module


class Items(Dataset):
    def __len__(self):
        return 11

    def __getitem__(self, index):
        return torch.tensor([index % 7, (index + 1) % 7]), torch.tensor([(index + 1) % 7, (index + 2) % 7])


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_version="test")
        self.embed = nn.Embedding(7, 8)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(8, 7)
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return self.head(self.dropout(self.embed(x))), None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def setup_trainer(monkeypatch, path, model, **kw):
    def loaders(cfg):
        sampler = ResumableBatchSampler(11, cfg.micro_batch_size, True, cfg.seed)
        train = DataLoader(Items(), batch_sampler=sampler, num_workers=cfg.num_workers,
                           generator=torch.Generator().manual_seed(cfg.seed))
        val = DataLoader(Items(), batch_size=2, generator=torch.Generator().manual_seed(123))
        return train, val, {"vocab_size": 7}

    monkeypatch.setattr(trainer_module, "make_dataloaders", loaders)
    args = dict(out_dir=str(path), device="cpu", dtype="fp32", seq_len=2,
                micro_batch_size=2, grad_accum_steps=2, max_steps=6, warmup_steps=0,
                eval_every=100, save_every=100, log_every=1, num_workers=0)
    args.update(kw)
    return Trainer(TrainConfig(**args), model=model)


def test_invalid_partial_accumulation_skips_update(monkeypatch, tmp_path):
    initial = TinyLM()
    initial.dropout.p = 0
    trainer = setup_trainer(monkeypatch, tmp_path / "skipped", copy.deepcopy(initial), max_steps=1)
    original = trainer_module.lm_cross_entropy
    calls = []

    def loss(logits, targets):
        calls.append(trainer.step)
        if len(calls) == 2:
            return logits.sum() * float("nan")
        return original(logits, targets)

    monkeypatch.setattr(trainer_module, "lm_cross_entropy", loss)
    trainer.train()
    assert calls == [0, 0, 0, 0]
    assert trainer.step == 1 and trainer.grad.nan_skips == 1
    assert all(state["step"].item() == 1 for state in trainer.optimizer.state.values())
    monkeypatch.setattr(trainer_module, "lm_cross_entropy", original)
    reference = setup_trainer(monkeypatch, tmp_path / "reference", copy.deepcopy(initial), max_steps=1)
    reference.train_loader.batch_sampler.cursor = 2
    reference.train()
    assert all(torch.equal(a, b) for a, b in zip(trainer.model.parameters(), reference.model.parameters()))


def test_permanently_invalid_loss_stops(monkeypatch, tmp_path):
    trainer = setup_trainer(monkeypatch, tmp_path, TinyLM(), max_steps=1)
    monkeypatch.setattr(trainer_module, "lm_cross_entropy", lambda logits, tgt: logits.sum() * float("nan"))
    with pytest.raises(RuntimeError, match="100 consecutive"):
        trainer.train()
    assert trainer.step == 0 and not trainer.optimizer.state
    trainer.logger.close()


def test_overflow_skips_update_and_backs_off_scaler(monkeypatch, tmp_path):
    trainer = setup_trainer(monkeypatch, tmp_path, TinyLM(), max_steps=1)
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
    calls = []
    def overflow_once(grad):
        calls.append(1)
        return torch.full_like(grad, float("inf")) if len(calls) == 1 else grad
    trainer.model.head.weight.register_hook(overflow_once)
    trainer.train()
    assert len(calls) == 4 and trainer.step == 1
    assert trainer.grad.nan_skips == 1 and trainer.scaler.get_scale() == 8.0
    assert all(state["step"].item() == 1 for state in trainer.optimizer.state.values())


@pytest.mark.parametrize("workers", [0, 2])
def test_interrupted_resume_matches_uninterrupted(monkeypatch, tmp_path, workers):
    torch.manual_seed(9)
    initial = TinyLM()
    whole = setup_trainer(monkeypatch, tmp_path / "whole", copy.deepcopy(initial), num_workers=workers)
    whole.train()

    interrupted = setup_trainer(monkeypatch, tmp_path / "part", copy.deepcopy(initial), num_workers=workers)
    class Interrupt(Callback):
        def on_step_end(self, trainer, step, metrics):
            if step == 3:
                trainer._save()
                raise RuntimeError("interrupt")
    interrupted.callbacks.add(Interrupt())
    with pytest.raises(RuntimeError, match="interrupt"):
        interrupted.train()
    interrupted.logger.close()
    resumed = setup_trainer(monkeypatch, tmp_path / "part", copy.deepcopy(initial),
                            num_workers=workers, resume="latest")
    resumed.train()
    assert resumed.step == whole.step == 6
    assert all(torch.equal(a, b) for a, b in zip(whole.model.parameters(), resumed.model.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(whole.model.seen, interrupted.model.seen + resumed.model.seen))
    assert len(whole.model.seen) == len(interrupted.model.seen) + len(resumed.model.seen)
    for a, b in zip(whole.optimizer.state.values(), resumed.optimizer.state.values()):
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_checkpoint_scaler_rng_and_legacy(tmp_path):
    model = nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", init_scale=32)
    # Initialize the lazy scale and save a non-default state.
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    scaler.step(opt)
    scaler.update()
    manager = CheckpointManager(str(tmp_path))
    random.seed(3)
    torch.manual_seed(4)
    manager.save(model=model, optimizer=opt, scaler=scaler, step=1, best_val=0,
                 config=SimpleNamespace())
    expected = (random.random(), torch.rand(3))
    restored = torch.amp.GradScaler("cpu", init_scale=1)
    manager.load(manager.resolve("latest"), model=model, optimizer=opt, scaler=restored)
    assert restored.get_scale() == 32
    assert random.random() == expected[0] and torch.equal(torch.rand(3), expected[1])
    manager.save(model=model, optimizer=opt, step=1, best_val=0, config=SimpleNamespace())
    legacy = torch.load(manager.resolve("latest"), weights_only=False)
    for key in ("scaler", "train_loader_state", "python_rng", "numpy_rng"):
        del legacy[key]
    torch.save(legacy, tmp_path / "legacy.pt")
    manager.load(tmp_path / "legacy.pt", model=model, scaler=restored)


def test_partial_nonfinite_loss_records_prior_scaled_overflow(monkeypatch, tmp_path):
    trainer = setup_trainer(monkeypatch, tmp_path, TinyLM(), max_steps=1)
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
    calls = []
    original = trainer_module.lm_cross_entropy

    def loss(logits, targets):
        calls.append(1)
        return logits.sum() * float("nan") if len(calls) == 2 else original(logits, targets)

    hook_calls = []
    def overflow_once(grad):
        hook_calls.append(1)
        return torch.full_like(grad, float("inf")) if len(hook_calls) == 1 else grad

    trainer.model.head.weight.register_hook(overflow_once)
    monkeypatch.setattr(trainer_module, "lm_cross_entropy", loss)
    trainer.train()
    assert len(calls) == 4 and trainer.step == 1
    assert trainer.grad.nan_skips == 1 and trainer.scaler.get_scale() == 8.0
    assert all(s["step"].item() == 1 for s in trainer.optimizer.state.values())


def test_metrics_count_shifted_inputs_and_unmasked_targets(monkeypatch, tmp_path):
    def masked_item(self, index):
        return torch.tensor([1, 2, 3]), torch.tensor([2, -100, 4])
    monkeypatch.setattr(Items, "__getitem__", masked_item)
    trainer = setup_trainer(monkeypatch, tmp_path, TinyLM(), seq_len=99, max_steps=1)
    snapshots = []
    class Capture(Callback):
        def on_step_end(self, trainer, step, metrics):
            snapshots.append(metrics.copy())
    trainer.callbacks.add(Capture())
    trainer.train()
    assert snapshots[0]["input_tokens"] == 12
    assert snapshots[0]["target_tokens"] == 8


@pytest.mark.parametrize("change", [
    {"optimizer": "galore"}, {"grad_accum_steps": 1}, {"val_fraction": 0.3},
    {"data_dir": "different"},
])
def test_resume_rejects_incompatible_configuration(monkeypatch, tmp_path, change):
    trainer = setup_trainer(monkeypatch, tmp_path, TinyLM())
    trainer._save()
    trainer.logger.close()
    with pytest.raises(ValueError, match="incompatible checkpoint configuration"):
        setup_trainer(monkeypatch, tmp_path, TinyLM(), resume="latest", **change)


def test_resume_rejects_changed_manifest_before_loading(monkeypatch, tmp_path):
    trainer = setup_trainer(monkeypatch, tmp_path, TinyLM())
    trainer._save()
    trainer.logger.close()
    trainer.manifest["vocab_size"] = 8
    original = copy.deepcopy(trainer.model.state_dict())
    with pytest.raises(ValueError, match="dataset identity"):
        trainer._resume("latest")
    assert all(torch.equal(v, original[k]) for k, v in trainer.model.state_dict().items())


def test_dataset_identity_detects_changed_shard_and_split(tmp_path):
    shard = tmp_path / "shard.rds"
    shard.write_bytes(b"one" * 20)
    rds = SimpleNamespace(data_dir=str(tmp_path), manifest={"shards": [{"file": "shard.rds"}]})
    identity = _dataset_identity(rds, [0, 1, 2])
    assert identity != _dataset_identity(rds, [0, 2, 1])
    shard.write_bytes(b"two" * 20)
    assert identity != _dataset_identity(rds, [0, 1, 2])


def test_sampler_resume_at_epoch_boundary():
    sampler = ResumableBatchSampler(5, 2, True, 3)
    first_epoch = list(sampler)
    sampler.consumed()
    restored = ResumableBatchSampler(5, 2, True, 3)
    restored.load_state_dict(sampler.state_dict())
    assert list(restored) == first_epoch[1:]
    sampler.consumed()
    restored.load_state_dict(sampler.state_dict())
    assert restored.epoch == 1 and restored.cursor == 0
    assert list(restored) == list(sampler)
    with pytest.raises(ValueError, match="batch_size"):
        ResumableBatchSampler(5, 1, True, 3).load_state_dict(sampler.state_dict())


def test_galore_trainer_resume(monkeypatch, tmp_path):
    torch.manual_seed(7)
    initial = TinyLM()
    options = dict(optimizer="galore", galore_rank=2, galore_update_interval=2)
    whole = setup_trainer(monkeypatch, tmp_path / "whole", copy.deepcopy(initial), **options)
    whole.train()
    part = setup_trainer(monkeypatch, tmp_path / "part", copy.deepcopy(initial), **options)
    class Interrupt(Callback):
        def on_step_end(self, trainer, step, metrics):
            if step == 3:
                trainer._save()
                raise RuntimeError("interrupt")
    part.callbacks.add(Interrupt())
    with pytest.raises(RuntimeError, match="interrupt"):
        part.train()
    part.logger.close()
    resumed = setup_trainer(monkeypatch, tmp_path / "part", copy.deepcopy(initial),
                            resume="latest", **options)
    resumed.train()
    assert all(torch.equal(a, b) for a, b in zip(whole.model.parameters(), resumed.model.parameters()))


def test_5b_size_matches_real_architecture_on_meta():
    cfg = RythConfig.ryth_5b()
    assert cfg.estimate_params() == 4_930_440_192
    # Meta tensors validate the actual module shapes without allocating 5B weights.
    with torch.device("meta"):
        model = RythForCausalLM(cfg)
    assert model.num_params() == cfg.estimate_params()
    assert 4.9e9 < cfg.estimate_params() < 5.1e9
