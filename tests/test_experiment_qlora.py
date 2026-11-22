import importlib.util
import sys

import pytest
import torch
from torch import nn

from experiments import qlora
from model import RythConfig, RythForCausalLM


def tiny_model(**kwargs):
    config = RythConfig(vocab_size=64, d_model=64, n_layers=1, n_heads=4,
                        n_kv_heads=2, ffn_hidden=128, max_seq_len=16, **kwargs)
    return RythForCausalLM(config)


@pytest.mark.parametrize("options", [
    {"rank": 0}, {"rank": True}, {"alpha": float('nan')}, {"alpha": "x"},
    {"dropout": 1}, {"dropout": float('nan')}, {"targets": "q_proj"},
    {"targets": []}, {"targets": [""]}, {"targets": ["missing"]},
])
def test_invalid_options_leave_model_unchanged(options):
    model = tiny_model()
    original = model.decoder.layers[0].attn.q_proj
    with pytest.raises(ValueError):
        qlora.prepare_qlora(model, **options)
    assert model.decoder.layers[0].attn.q_proj is original
    assert all(p.requires_grad for p in model.parameters())


def test_cpu_rejected_and_tied_weight_preserved():
    model = tiny_model()
    with pytest.raises(ValueError, match="CUDA"):
        qlora.prepare_qlora(model)
    assert model.lm_head.weight is model.decoder.embed.weight
    assert all(p.requires_grad for p in model.parameters())


def test_shared_linear_weights_rejected():
    model = tiny_model()
    attn = model.decoder.layers[0].attn
    attn.o_proj.weight = nn.Parameter(attn.q_proj.weight.detach())
    with pytest.raises(ValueError, match="shared weight"):
        qlora.prepare_qlora(model)
    assert attn.o_proj.weight.data_ptr() == attn.q_proj.weight.data_ptr()


def test_shared_linear_module_rejected():
    model = tiny_model()
    attn = model.decoder.layers[0].attn
    attn.o_proj = attn.q_proj
    with pytest.raises(ValueError, match="shared module"):
        qlora.prepare_qlora(model)


def test_missing_dependency_and_failed_conversion_leave_model_unchanged(monkeypatch):
    model = tiny_model()
    original = model.decoder.layers[0].attn.q_proj
    monkeypatch.setattr(qlora, "_require_cuda", lambda *args: None)
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    with pytest.raises(ImportError, match="pip install bitsandbytes"):
        qlora.prepare_qlora(model)
    assert model.decoder.layers[0].attn.q_proj is original
    assert all(p.requires_grad for p in model.parameters())

    monkeypatch.setattr(qlora, "_bitsandbytes", lambda: object())
    calls = []

    def fail_second(_, layer):
        calls.append(layer)
        if len(calls) == 2:
            raise RuntimeError("conversion failed")
        return nn.Linear(layer.in_features, layer.out_features)

    monkeypatch.setattr(qlora, "_make_4bit", fail_second)
    with pytest.raises(RuntimeError, match="conversion failed"):
        qlora.prepare_qlora(model)
    assert model.decoder.layers[0].attn.q_proj is original
    assert all(p.requires_grad for p in model.parameters())


def test_adapter_roundtrip_and_invalid_file_is_atomic(tmp_path):
    model = nn.Sequential(qlora.QLoRALinear(nn.Linear(8, 8), 2, 4, 0))
    with torch.no_grad():
        model[0].lora_B.fill_(0.25)
    path = tmp_path / "adapter.pt"
    qlora.save_qlora_adapter(model, path)
    target = nn.Sequential(qlora.QLoRALinear(nn.Linear(8, 8), 2, 4, 0))
    qlora.load_qlora_adapter(target, path)
    assert torch.equal(target[0].lora_A, model[0].lora_A)
    assert torch.equal(target[0].lora_B, model[0].lora_B)
    record = torch.load(path, weights_only=True)
    record["layers"]["0"]["B"][0, 0] = float('nan')
    torch.save(record, path)
    before = target[0].lora_A.detach().clone()
    with pytest.raises(ValueError, match="invalid adapter tensor"):
        qlora.load_qlora_adapter(target, path)
    assert torch.equal(before, target[0].lora_A)

    record["layers"]["0"]["B"] = torch.full((8, 2), 1e100, dtype=torch.float64)
    torch.save(record, path)
    before_b = target[0].lora_B.detach().clone()
    with pytest.raises(ValueError, match="invalid adapter tensor"):
        qlora.load_qlora_adapter(target, path)
    assert torch.equal(before, target[0].lora_A)
    assert torch.equal(before_b, target[0].lora_B)


cuda_bnb = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("bitsandbytes") is None,
    reason="requires CUDA and bitsandbytes",
)


@cuda_bnb
@pytest.mark.parametrize("tie_embeddings", [True, False])
def test_cuda_nf4_training_and_adapter_reload(tmp_path, tie_embeddings):
    import bitsandbytes as bnb

    torch.manual_seed(7)
    model = tiny_model(tie_embeddings=tie_embeddings).cuda()
    base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    source_weight = model.decoder.layers[0].attn.q_proj.weight.detach().clone()
    qlora.prepare_qlora(model, rank=4, alpha=8)
    attn = model.decoder.layers[0].attn
    assert (model.lm_head.weight is model.decoder.embed.weight) == tie_embeddings
    assert isinstance(attn.q_proj, qlora.QLoRALinear)
    assert isinstance(attn.k_proj, bnb.nn.Linear4bit)
    weight = attn.q_proj.base.weight
    assert weight.dtype == torch.uint8 and weight.quant_state.quant_type == "nf4"
    restored = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
    assert restored.shape == source_weight.shape
    assert (restored - source_weight).abs().mean() < source_weight.abs().mean() * 0.3
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {
        f"decoder.layers.0.attn.{name}.lora_{matrix}"
        for name in ("q_proj", "v_proj") for matrix in ("A", "B")
    }
    tokens = torch.randint(0, 64, (2, 8), device="cuda")
    logits, _ = model(tokens)
    logits.float().square().mean().backward()
    assert attn.q_proj.lora_B.grad is not None
    assert attn.q_proj.lora_B.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    optimizer.step()
    model.eval()
    expected = model(tokens)[0].detach()
    path = tmp_path / "adapter.pt"
    qlora.save_qlora_adapter(model, path)
    fresh = tiny_model(tie_embeddings=tie_embeddings)
    fresh.load_state_dict(base)
    fresh.cuda()
    qlora.prepare_qlora(fresh, rank=4, alpha=8)
    qlora.load_qlora_adapter(fresh, path)
    fresh.eval()
    torch.testing.assert_close(fresh(tokens)[0], expected)


@cuda_bnb
def test_cuda_conversion_failure_is_atomic(monkeypatch):
    model = tiny_model().cuda()
    original = model.decoder.layers[0].attn.q_proj
    make = qlora._make_4bit
    count = 0

    def fail_second(bnb, layer):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("conversion failed")
        return make(bnb, layer)

    monkeypatch.setattr(qlora, "_make_4bit", fail_second)
    with pytest.raises(RuntimeError, match="conversion failed"):
        qlora.prepare_qlora(model)
    assert model.decoder.layers[0].attn.q_proj is original
    assert all(p.requires_grad for p in model.parameters())
