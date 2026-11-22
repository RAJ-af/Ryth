"""Focused tests for the experimental projected optimizer."""
import copy
import io

import pytest
import torch
from torch import nn

from model import RythConfig, RythForCausalLM
from training.config import TrainConfig
from training.galore import GaLoreAdamW
from training.optimizer import build_optimizer


def test_default_is_existing_adamw():
    model = nn.Linear(4, 3)
    config = TrainConfig()
    opt = build_optimizer(model, config)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]['weight_decay'] == config.weight_decay
    assert opt.param_groups[1]['weight_decay'] == 0


@pytest.mark.parametrize('shape', [(6, 4), (4, 6)])
def test_projected_adamw_update_and_decay(shape):
    torch.manual_seed(3)
    p = nn.Parameter(torch.randn(*shape, dtype=torch.float64))
    original = p.detach().clone()
    grad = torch.randn_like(p)
    p.grad = grad.clone()
    opt = GaLoreAdamW([p], lr=0.03, rank=2, scale=0.7, weight_decay=0.2)
    opt.step()
    state = opt.state[p]
    basis = state['basis']
    low = basis.T @ grad if shape[0] >= shape[1] else grad @ basis.T
    expected_low = low / (low.abs() + 1e-8)
    expected = basis @ expected_low if shape[0] >= shape[1] else expected_low @ basis
    torch.testing.assert_close(p, original * (1 - 0.03 * 0.2) - 0.03 * 0.7 * expected)
    assert state['exp_avg'].shape == low.shape
    assert torch.linalg.matrix_rank(expected, atol=1e-10) <= 2


def test_full_rank_fallback_matches_torch_adamw():
    torch.manual_seed(7)
    p = nn.Parameter(torch.randn(3, 4, dtype=torch.float64))
    q = nn.Parameter(p.detach().clone())
    a = GaLoreAdamW([p], lr=0.02, rank=3, weight_decay=0.3, betas=(0.8, 0.9))
    b = torch.optim.AdamW([q], lr=0.02, weight_decay=0.3, betas=(0.8, 0.9))
    for _ in range(5):
        grad = torch.randn_like(p)
        p.grad = grad.clone()
        q.grad = grad.clone()
        a.step()
        b.step()
        torch.testing.assert_close(p, q)
    assert 'basis' not in a.state[p]


def test_refresh_resets_moments_and_skips_missing_gradients():
    p = nn.Parameter(torch.zeros(5, 4))
    opt = GaLoreAdamW([p], lr=0.1, rank=2, update_interval=2)
    for step in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
        assert opt.state[p]['moment_step'] == (1, 2, 1)[step]
    p.grad = None
    opt.step()
    assert opt.state[p]['step'] == 3


def test_resume_through_refresh_and_parameter_groups():
    torch.manual_seed(2)
    model = nn.Linear(6, 4)
    config = TrainConfig(optimizer='galore', galore_rank=2, galore_update_interval=3)
    opt = build_optimizer(model, config)
    grads = [[torch.randn_like(p) for p in model.parameters()] for _ in range(7)]

    def step(m, o, gs):
        for p, g in zip(m.parameters(), gs):
            p.grad = g.clone()
        o.step()

    for gs in grads[:2]:
        step(model, opt, gs)
    buffer = io.BytesIO()
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    resumed = copy.deepcopy(model)
    resumed.load_state_dict(saved['model'])
    resumed_opt = build_optimizer(resumed, config)
    resumed_opt.load_state_dict(saved['optimizer'])
    assert resumed_opt.param_groups[0]['weight_decay'] == config.weight_decay
    assert resumed_opt.param_groups[1]['weight_decay'] == 0
    for gs in grads[2:]:
        step(model, opt, gs)
        step(resumed, resumed_opt, gs)
        for p, q in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_half_precision_state_resume_preserves_fp32():
    p = nn.Parameter(torch.randn(5, 4, dtype=torch.float16))
    opt = GaLoreAdamW([p], rank=2)
    p.grad = torch.randn_like(p)
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    q = nn.Parameter(p.detach().clone())
    resumed = GaLoreAdamW([q], rank=2)
    resumed.load_state_dict(saved)
    for key in ('basis', 'exp_avg', 'exp_avg_sq'):
        assert resumed.state[q][key].dtype == torch.float32
        torch.testing.assert_close(resumed.state[q][key], opt.state[p][key], rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [
    {'galore_rank': 0}, {'galore_rank': -1}, {'galore_rank': 1.5},
    {'galore_update_interval': 0}, {'galore_scale': 0}, {'galore_scale': float('nan')},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(optimizer='galore', **kwargs)


def test_tiny_ryth_finite_training():
    torch.manual_seed(10)
    model = RythForCausalLM(RythConfig(vocab_size=32, d_model=16, n_layers=1,
        n_heads=4, n_kv_heads=2, ffn_hidden=32, max_seq_len=16))
    opt = build_optimizer(model, TrainConfig(optimizer='galore', galore_rank=2,
                                              galore_update_interval=2, lr=1e-3))
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        ids = torch.randint(0, 32, (2, 8))
        logits, _ = model(ids)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 32), ids.reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any('basis' in s for s in opt.state.values())
