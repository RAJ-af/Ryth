"""Focused CPU tests: pytest -q tests/test_experiment_adapters_distillation.py."""
import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RythConfig, RythForCausalLM
from experiments.adapters import LoRALinear, inject_lora, load_adapter, merge_lora, save_adapter
from experiments.distillation import distillation_loss, distillation_step


def make_model(vocab_size=23, **kw):
    return RythForCausalLM(RythConfig(vocab_size=vocab_size, d_model=16,
                                      n_layers=1, n_heads=2, n_kv_heads=1,
                                      ffn_hidden=32, max_seq_len=16, **kw))


def test_lora_training_persistence_and_merge(tmp_path):
    torch.manual_seed(3)
    model = make_model().eval()
    base = copy.deepcopy(model)
    ids = torch.randint(0, 23, (2, 5))
    original = model(ids)[0].detach()
    inject_lora(model, rank=2, alpha=4)
    assert model.decoder.embed.weight is model.lm_head.weight
    torch.testing.assert_close(model(ids)[0], original)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 4 and all(n.endswith(("lora_A", "lora_B")) for n in trainable)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.5)
    F.cross_entropy(model(ids)[0].reshape(-1, 23), ids.reshape(-1)).backward()
    optimizer.step()
    assert any(torch.count_nonzero(m.lora_B).item() for m in model.modules() if isinstance(m, LoRALinear))
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters() if n in before)
    adapted = model(ids)[0].detach()
    assert not torch.equal(adapted, original)
    path = tmp_path / "adapter.pt"
    save_adapter(model, path)
    load_adapter(base, path)
    torch.testing.assert_close(base(ids)[0], adapted)
    assert base.decoder.embed.weight is base.lm_head.weight
    merge_lora(base)
    assert not any(isinstance(m, LoRALinear) for m in base.modules())
    assert base.decoder.embed.weight is base.lm_head.weight
    torch.testing.assert_close(base(ids)[0], adapted, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="no LoRA"):
        merge_lora(base)


def test_lora_validation_and_dropout(tmp_path):
    model = make_model()
    with pytest.raises(ValueError, match="targets"):
        inject_lora(model, targets=("does_not_exist",))
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError):
        inject_lora(model, rank=0)
    assert all(p.requires_grad for p in model.parameters())
    inject_lora(model, rank=2, dropout=0.2, targets=("w_gate",))
    with pytest.raises(ValueError, match="already"):
        inject_lora(model)
    with pytest.raises(ValueError, match="eval"):
        merge_lora(model)
    path = tmp_path / "adapter.pt"
    save_adapter(model, path)
    other = make_model()
    inject_lora(other, rank=3, targets=("w_gate",))
    with pytest.raises(ValueError, match="configuration"):
        load_adapter(other, path)
    data = torch.load(path, weights_only=True)
    next(iter(data["layers"].values()))["A"] = torch.zeros(1, 1)
    torch.save(data, path)
    untouched = make_model()
    with pytest.raises(ValueError, match="tensors"):
        load_adapter(untouched, path)
    assert all(p.requires_grad for p in untouched.parameters())
    assert not any(isinstance(m, LoRALinear) for m in untouched.modules())


def test_reject_shared_projection_weights():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(4, 4)
            self.proj = nn.Linear(4, 4, bias=False)
            self.proj.weight = self.embed.weight
    model = Shared()
    with pytest.raises(ValueError, match="shared weight"):
        inject_lora(model, targets=("proj",))
    assert model.proj.weight is model.embed.weight and model.embed.weight.requires_grad


def test_distillation_loss_temperature_and_mask():
    torch.manual_seed(12)
    student = torch.randn(2, 3, 7, requires_grad=True)
    teacher = torch.randn(2, 3, 7, requires_grad=True)
    targets = torch.tensor([[1, -100, 2], [3, 4, -100]])
    mask = targets != -100
    t, alpha = 3.0, 0.7
    actual = distillation_loss(student, targets, teacher, temperature=t, alpha=alpha)
    expected = (1-alpha)*F.cross_entropy(student[mask], targets[mask]) + alpha*t*t*F.kl_div(
        F.log_softmax(student[mask]/t, dim=-1), F.softmax(teacher[mask]/t, dim=-1), reduction="batchmean")
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[~mask]) == 0
    torch.testing.assert_close(distillation_loss(student, targets, alpha=0), F.cross_entropy(student[mask], targets[mask]))
    with pytest.raises(ValueError, match="identical"):
        distillation_loss(student, targets, teacher[..., :-1])
    with pytest.raises(ValueError, match="no non-ignored"):
        distillation_loss(student, torch.full_like(targets, -100), teacher)
    with pytest.raises(ValueError, match="temperature"):
        distillation_loss(student, targets, teacher, temperature=0)


def test_distillation_step_freezes_teacher_updates_student_and_skips_teacher():
    torch.manual_seed(4)
    student, teacher = make_model(), make_model()
    ids = torch.randint(0, 23, (2, 4))
    targets = torch.randint(0, 23, (2, 4))
    before_teacher = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    before_student = {n: p.detach().clone() for n, p in student.named_parameters()}
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    loss = distillation_step(student, optimizer, ids, targets, teacher=teacher)
    assert loss > 0 and not teacher.training
    assert all(not p.requires_grad and p.grad is None and torch.equal(p, before_teacher[n]) for n, p in teacher.named_parameters())
    assert any(not torch.equal(p, before_student[n]) for n, p in student.named_parameters())
    def no_forward(*args, **kwargs):
        raise AssertionError("teacher should not run")
    teacher.forward = no_forward
    assert distillation_step(student, optimizer, ids, targets, teacher=teacher, alpha=0) > 0
    with pytest.raises(ValueError, match="share"):
        distillation_step(student, optimizer, ids, targets, teacher=student)
    with pytest.raises(ValueError, match="identical"):
        distillation_step(student, optimizer, ids, targets, teacher=make_model(vocab_size=24))
    cached = torch.randn(2, 4, 23)
    assert distillation_step(student, optimizer, ids, targets, teacher_logits=cached) > 0
