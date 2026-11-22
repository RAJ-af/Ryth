"""CPU checks for the opt-in growth and sparse MoE experiments."""
from dataclasses import asdict

import pytest
import torch

from experiments.growth import grow_depth
from experiments.moe import SparseMoE, replace_feedforwards
from model import RythConfig, RythForCausalLM


def tiny(**overrides):
    values = dict(vocab_size=31, d_model=8, n_layers=2, n_heads=2,
                  n_kv_heads=1, ffn_hidden=12, max_seq_len=16,
                  use_flash_attention=False)
    values.update(overrides)
    return RythConfig(**values)


@pytest.mark.parametrize("tied", [True, False])
def test_growth_copies_original_and_updates_config(tied):
    torch.manual_seed(12)
    model = RythForCausalLM(tiny(tie_embeddings=tied)).double().eval()
    ids = torch.randint(0, 31, (2, 4))
    original_logits, _ = model(ids)
    grown = grow_depth(model, 5)
    assert len(model.decoder.layers) == 2
    assert model.config.n_layers == 2
    assert grown.config.n_layers == grown.decoder.config.n_layers == 5
    assert asdict(grown.config) == {**asdict(model.config), "n_layers": 5}
    assert grown.config is grown.decoder.config and grown.config is not model.config
    assert (grown.lm_head.weight is grown.decoder.embed.weight) == tied
    assert grown.lm_head.weight is not model.lm_head.weight
    assert not grown.training and next(grown.parameters()).dtype == torch.float64
    for i in range(5):
        source = model.decoder.layers[i % 2]
        target = grown.decoder.layers[i]
        assert target.layer_idx == i
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            assert torch.equal(source_param, target_param)
            assert source_param is not target_param
    assert grown.decoder.layers[0].attn_norm.weight is not grown.decoder.layers[2].attn_norm.weight
    assert torch.equal(model(ids)[0], original_logits)
    logits, cache = grown(ids, use_cache=True)
    assert logits.shape == (2, 4, 31) and len(cache) == 5
    # Standard dense checkpoints can be reconstructed with the updated config.
    restored = RythForCausalLM(grown.config).double().eval()
    restored.load_state_dict(grown.state_dict())
    assert torch.allclose(restored(ids)[0], logits)


def test_growth_validation_and_optimizer_independence():
    torch.manual_seed(13)
    model = RythForCausalLM(tiny())
    for invalid in [0, 2, -1, 3.0, True]:
        with pytest.raises(ValueError):
            grow_depth(model, invalid)
    with pytest.raises(TypeError):
        grow_depth(object(), 3)
    optimizer = torch.optim.AdamW(model.parameters())
    grown = grow_depth(model, 3)
    old_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) not in old_ids for p in grown.parameters())
    model.config.n_layers = 4
    with pytest.raises(ValueError, match="depth"):
        grow_depth(model, 5)
    model.config.n_layers = 2
    model.config.init_scheme = "invalid"
    with pytest.raises(AssertionError):
        grow_depth(model, 3)


@pytest.mark.parametrize("kwargs", [
    dict(d_model=0, ffn_dim=4, num_experts=2),
    dict(d_model=4, ffn_dim=0, num_experts=2),
    dict(d_model=4, ffn_dim=4, num_experts=0),
    dict(d_model=4, ffn_dim=4, num_experts=2, top_k=3),
    dict(d_model=4, ffn_dim=4, num_experts=2, top_k=True),
    dict(d_model=4, ffn_dim=4, num_experts=2, dropout=1.1),
])
def test_moe_config_validation(kwargs):
    with pytest.raises(ValueError):
        SparseMoE(**kwargs)


def test_sparse_dispatch_exact_top_k_output_and_gradients():
    torch.manual_seed(21)
    moe = SparseMoE(4, 7, 4, top_k=2)
    # Expert 3 never wins; 0/1/2 receive different subsets.
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[2., 0, 0, 0], [0, 2., 0, 0],
                                              [-2., -2., 0, 0], [0., 0., -10., 0.]]))
    x = torch.tensor([[2., 0., 1., 1.], [0., 2., 1., 1.],
                      [-2., -2., 1., 1.]], requires_grad=True)
    calls = [[] for _ in moe.experts]
    handles = [expert.register_forward_hook(
        lambda module, inputs, output, i=i: calls[i].append(inputs[0].detach().clone()))
        for i, expert in enumerate(moe.experts)]
    out, aux = moe.forward_with_aux(x)
    for handle in handles:
        handle.remove()
    probs = moe.router(x).float().softmax(-1)
    chosen, indices = probs.topk(2, -1)
    weights = chosen / chosen.sum(-1, keepdim=True)
    expected = torch.stack([
        sum(moe.experts[indices[t, j]](x[t]) * weights[t, j] for j in range(2))
        for t in range(x.shape[0])])
    assert out.shape == x.shape and torch.allclose(out, expected, atol=1e-7)
    assert sum(sum(batch.shape[0] for batch in per_expert) for per_expert in calls) == 6
    assert calls[3] == []
    for i in range(3):
        expected_tokens = x.detach()[torch.any(indices == i, dim=-1)]
        assert len(calls[i]) == 1 and torch.equal(calls[i][0], expected_tokens)
    fraction = torch.bincount(indices.flatten(), minlength=4).float() / 6
    assert torch.allclose(aux, 4 * (probs.mean(0) * fraction).sum())
    (out.square().sum() + 0.2 * aux).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0
    for expert in moe.experts[:3]:
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters())
    assert all(p.grad is None for p in moe.experts[3].parameters())


def test_top1_balance_gradient_empty_and_invalid_shape():
    torch.manual_seed(22)
    moe = SparseMoE(4, 8, 3, top_k=1)
    x = torch.randn(2, 3, 4, requires_grad=True)
    out, aux = moe.forward_with_aux(x)
    assert out.shape == x.shape and torch.isfinite(aux)
    aux.backward()
    assert moe.router.weight.grad.abs().sum() > 0
    empty = torch.empty(2, 0, 4, requires_grad=True)
    handles = [expert.register_forward_hook(lambda *args: pytest.fail("empty dispatched"))
               for expert in moe.experts]
    result, loss = moe.forward_with_aux(empty)
    for handle in handles:
        handle.remove()
    assert result.shape == empty.shape and loss.item() == 0
    (result.sum() + loss).backward()
    assert empty.grad is not None
    with pytest.raises(ValueError):
        moe(torch.randn(2, 5))


def test_model_replacement_and_checkpoint_roundtrip():
    torch.manual_seed(23)
    model = RythForCausalLM(tiny()).eval()
    dense = [layer.ffn.state_dict() for layer in model.decoder.layers]
    assert replace_feedforwards(model, 3, top_k=2) is model
    for layer, state in zip(model.decoder.layers, dense):
        assert isinstance(layer.ffn, SparseMoE) and not layer.ffn.training
        for expert in layer.ffn.experts:
            assert all(torch.equal(v, expert.state_dict()[k]) for k, v in state.items())
    ids = torch.randint(0, 31, (2, 4))
    logits, cache = model(ids, use_cache=True)
    assert logits.shape == (2, 4, 31) and len(cache) == 2
    aux = sum(layer.ffn.aux_loss for layer in model.decoder.layers)
    (logits.square().mean() + 0.1 * aux).backward()
    assert all(layer.ffn.router.weight.grad is not None for layer in model.decoder.layers)
    restored = RythForCausalLM(model.config).eval()
    with pytest.raises(RuntimeError):
        restored.load_state_dict(model.state_dict())
    replace_feedforwards(restored, 3, top_k=2)
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(restored(ids)[0], logits)
    with pytest.raises(ValueError):
        replace_feedforwards(model, 3)
    checkpointed = RythForCausalLM(tiny(use_gradient_checkpointing=True))
    with pytest.raises(ValueError, match="checkpointing"):
        replace_feedforwards(checkpointed, 3)
