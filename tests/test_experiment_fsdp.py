"""CPU guards/sampling and optional real two-GPU torchrun smoke tests."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from dataset import RDSDataset
from dataset.rds import ShardWriter
from experiments.fsdp_train import ExperimentConfig, make_sampler, train, validate_data
from model import RythConfig, RythForCausalLM
from training.dataloader import RDSTorchDataset


def make_rds(path, n=13, masked=False):
    path.mkdir()
    writer = ShardWriter(str(path / "data.rds"), version=1, tok_version=1,
                         vocab_size=32, seq_len=6, dtype_flag=0)
    mask = (ShardWriter(str(path / "mask.rds"), version=1, tok_version=1,
                        vocab_size=2, seq_len=6, dtype_flag=0) if masked else None)
    for i in range(n):
        writer.add_chunk([(i + j) % 32 for j in range(6)], {})
        if mask:
            mask.add_chunk([1, 0, 1, 1, 0, 1], {})
    writer.close()
    if mask:
        mask.close()
    shard = {"file": "data.rds"}
    if mask:
        shard["mask_file"] = "mask.rds"
    (path / "manifest.json").write_text(json.dumps({"seq_len": 6, "vocab_size": 32,
                                                     "shards": [shard]}))


def tiny_config():
    return RythConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=2,
                      n_kv_heads=1, ffn_hidden=32, max_seq_len=8)


@pytest.mark.parametrize("change, message", [
    ({"steps": 0}, "positive"), ({"batch_size": 0}, "positive"),
    ({"accumulation": 0}, "positive"), ({"dtype": "fp16"}, "sharded scaler"),
    ({"warmup_steps": 101}, "warmup"), ({"resume": "latest"}, "resume"),
    ({"grad_clip": -1}, "invalid"), ({"lr": float("inf")}, "invalid"),
    ({"weight_decay": float("inf")}, "invalid"), ({"grad_clip": float("inf")}, "invalid"),
])
def test_config_guards(change, message):
    cfg = ExperimentConfig(data_dir="x", model_config="y", **change)
    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_requires_torchrun(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    with pytest.raises(RuntimeError, match="torchrun"):
        train(ExperimentConfig("x", "y", dtype="fp32"))


def test_sampler_disjoint_deterministic_and_full_batches():
    dataset = list(range(25))
    samplers = [make_sampler(dataset, rank, 3, 42, 2) for rank in range(3)]
    orders = [list(s) for s in samplers]
    assert [len(x) for x in orders] == [8, 8, 8]
    assert len(set(sum(orders, []))) == 24
    assert list(samplers[0]) == orders[0]
    for sampler in samplers:
        sampler.set_epoch(1)
    new = [list(s) for s in samplers]
    assert new != orders and len(set(sum(new, []))) == 24
    assert all(len(list(DataLoader(dataset, batch_size=2, sampler=s, drop_last=True))) == 4
               for s in samplers)
    with pytest.raises(ValueError, match="complete global batch"):
        make_sampler([1, 2, 3], 0, 2, 0, 2)


def test_real_rds_and_mask_guards(tmp_path):
    make_rds(tmp_path / "data", masked=True)
    rds = RDSDataset(str(tmp_path / "data"))
    try:
        validate_data(rds, tiny_config())
        x, y = RDSTorchDataset(rds, [0])[0]
        assert x.tolist() == [0, 1, 2, 3, 4]
        assert y.tolist() == [-100, 2, 3, -100, 5]
        reader, mask = rds._readers[0], rds._mask_readers[0]
        original = reader._index[0]
        reader._index[0] = (reader.index_offset, rds.seq_len)
        with pytest.raises(ValueError, match="offset"):
            validate_data(rds, tiny_config())
        reader._index[0] = original
        mask.seq_len = rds.seq_len + 1
        with pytest.raises(ValueError, match="mask shape"):
            validate_data(rds, tiny_config())
        mask.seq_len = rds.seq_len
        with pytest.raises(ValueError, match="vocabulary"):
            validate_data(rds, RythConfig(vocab_size=31, d_model=16, n_layers=2,
                                         n_heads=2, n_kv_heads=1, max_seq_len=8))
    finally:
        rds.close()
    (tmp_path / "data" / "mask.rds").unlink()
    rds = RDSDataset(str(tmp_path / "data"))
    try:
        with pytest.raises(ValueError, match="missing RDS mask"):
            validate_data(rds, tiny_config())
    finally:
        rds.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA GPUs")
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_torchrun_training_and_collective_export(tmp_path, dtype):
    if dtype == "bf16" and not all(torch.cuda.get_device_capability(i)[0] >= 8 for i in (0, 1)):
        pytest.skip("requires native BF16 on both GPUs")
    data = tmp_path / "data"
    make_rds(data, n=12, masked=True)
    from dataclasses import asdict
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(asdict(tiny_config())))
    checkpoint = tmp_path / "weights.pt"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nproc-per-node=2", "-m", "experiments.fsdp_train",
           "--data-dir", str(data), "--model-config", str(config_path),
           "--steps", "2", "--accumulation", "2", "--dtype", dtype,
           "--checkpoint", str(checkpoint)]
    root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(cmd, cwd=root, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"},
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    shards = re.findall(r"rank=(\d+) local_params=(\d+) local_adam_moments=(\d+) total_params=(\d+)",
                        result.stdout)
    assert len(shards) == 2, result.stdout
    values = [tuple(map(int, shard)) for shard in shards]
    total = values[0][3]
    assert all(p < total and m < total and t == total for _, p, m, t in values)
    assert sum(p for _, p, _, _ in values) == total
    assert sum(m for _, _, m, _ in values) == total
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["step"] == 2 and state["format"] == "fsdp-weights-only-v1"
    assert "optimizer" not in state
    torch.manual_seed(1337)
    initial = RythForCausalLM(tiny_config())
    initial_state = initial.state_dict()
    assert set(state["model"]) == set(initial_state)
    assert any(not torch.equal(state["model"][k], initial_state[k]) for k in initial_state)
    initial.load_state_dict(state["model"])
    assert not list(tmp_path.glob("*.tmp.*"))
