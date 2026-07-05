"""Unit tests for the Ryth model core (v0.2.0).

Built module-by-module: each module's tests must pass before the next is added.
Run:  pytest tests/test_model.py -v   (or: python tests/test_model.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import RythConfig

torch.manual_seed(0)


def tiny_config(**kw):
    """Small config for fast tests."""
    base = dict(vocab_size=128, d_model=64, n_layers=2, n_heads=4,
                n_kv_heads=2, max_seq_len=64)
    base.update(kw)
    return RythConfig(**base)


# ======================================================================== #
# Module 1 — config
# ======================================================================== #
def test_config_derived():
    c = tiny_config()
    assert c.head_dim == 16          # 64 / 4
    assert c.n_rep == 2              # 4 / 2
    assert c.ffn_dim % c.ffn_multiple_of == 0


def test_config_versions_and_flags():
    c = RythConfig()
    assert c.model_version == "0.2.0"
    assert c.architecture_version == 1
    assert c.checkpoint_version == 1
    assert c.attention_backend == "gqa"
    assert c.use_flash_attention is True
    assert c.use_qk_norm is False
    assert c.use_gradient_checkpointing is False
    assert c.init_scheme == "llama"


def test_config_validation():
    for bad in [dict(d_model=64, n_heads=6),           # not divisible
                dict(n_heads=8, n_kv_heads=3),          # not divisible
                dict(init_scheme="nope"),               # bad scheme
                dict(attention_backend="xyz")]:         # bad backend
        try:
            RythConfig(**bad)
            assert False, f"expected failure for {bad}"
        except AssertionError:
            pass


def test_preset_30m_is_512_8_8():
    c = RythConfig.ryth_30m()
    assert (c.d_model, c.n_layers, c.n_heads) == (512, 8, 8)
    assert c.n_kv_heads == 2


def test_presets_scale():
    sizes = {n: getattr(RythConfig, n)(vocab_size=32000).estimate_params()
             for n in ["ryth_30m", "ryth_125m", "ryth_350m", "ryth_1b"]}
    # strictly increasing param counts
    vals = list(sizes.values())
    assert vals == sorted(vals) and len(set(vals)) == 4


# ======================================================================== #
# Module 2 — init (weight initialization schemes)
# ======================================================================== #
import torch.nn as nn
from model.init import apply_init, INIT_FUNCS


class _ToyModel(nn.Module):
    """Minimal module with the naming conventions init keys off of."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 32)
        self.q_proj = nn.Linear(32, 32, bias=False)
        self.o_proj = nn.Linear(32, 32, bias=False)   # residual projection
        self.w_down = nn.Linear(64, 32, bias=False)   # residual projection


def test_init_schemes_run():
    for scheme in ["xavier", "llama", "deepseek"]:
        cfg = tiny_config(init_scheme=scheme)
        m = _ToyModel()
        apply_init(m, cfg)
        assert torch.isfinite(m.q_proj.weight).all()


def test_llama_residual_scaling():
    cfg = tiny_config(init_scheme="llama", n_layers=8)
    m = _ToyModel()
    apply_init(m, cfg)
    # o_proj / w_down should have SMALLER std than q_proj (residual scaling)
    assert m.o_proj.weight.std().item() < m.q_proj.weight.std().item()


def test_ryth_init_not_implemented():
    cfg = tiny_config(init_scheme="ryth")
    try:
        apply_init(_ToyModel(), cfg)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


# ======================================================================== #
# Module 3 — embedding
# ======================================================================== #
from model import TokenEmbedding


def test_embedding_shape():
    emb = TokenEmbedding(50, 24)
    ids = torch.randint(0, 50, (3, 6))
    out = emb(ids)
    assert out.shape == (3, 6, 24)


def test_embedding_weight_property():
    emb = TokenEmbedding(50, 24)
    assert emb.weight.shape == (50, 24)                    # for weight tying
    assert emb.weight is emb.embedding.weight


# ======================================================================== #
# Module 4 — RMSNorm
# ======================================================================== #
from model import RMSNorm


def test_rmsnorm_shape_and_unit_rms():
    norm = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    out = norm(x)
    assert out.shape == x.shape
    rms = out.pow(2).mean(-1).sqrt()               # unit weight => RMS ~ 1
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_dtype_preserved():
    norm = RMSNorm(8)
    x = torch.randn(2, 8, dtype=torch.float32)
    assert norm(x).dtype == torch.float32


# ======================================================================== #
# Module 5 — RoPE
# ======================================================================== #
from model import RotaryEmbedding, apply_rotary, rotate_half


def test_rope_tables_shape():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    cos, sin = rope(10, torch.device("cpu"))
    assert cos.shape == (10, 16) and sin.shape == (10, 16)


def test_rope_preserves_norm():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    cos, sin = rope(10, torch.device("cpu"))
    q = torch.randn(1, 2, 10, 16)
    k = torch.randn(1, 2, 10, 16)
    q2, k2 = apply_rotary(q, k, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-4)   # rotation


def test_rope_offset_matches_slice():
    """cos/sin at offset must equal the same rows of a longer table."""
    rope = RotaryEmbedding(head_dim=8, max_seq_len=32)
    full_cos, full_sin = rope(10, torch.device("cpu"))
    off_cos, off_sin = rope(3, torch.device("cpu"), offset=7)
    assert torch.allclose(off_cos, full_cos[7:10], atol=1e-5)
    assert torch.allclose(off_sin, full_sin[7:10], atol=1e-5)


# ======================================================================== #
# Module 6 — SwiGLU feed-forward
# ======================================================================== #
from model import SwiGLU


def test_swiglu_shape():
    ff = SwiGLU(32, 88)
    x = torch.randn(2, 7, 32)
    assert ff(x).shape == (2, 7, 32)


def test_swiglu_three_projections():
    ff = SwiGLU(16, 40)
    assert ff.w_gate.weight.shape == (40, 16)
    assert ff.w_up.weight.shape == (40, 16)
    assert ff.w_down.weight.shape == (16, 40)


# ======================================================================== #
# Module 7 — attention factory (GQA backend, KV cache, causality, QK-norm)
# ======================================================================== #
from model.attention import build_attention, GQAttention, repeat_kv
from model import RotaryEmbedding as _Rope


def _rope_for(cfg, T, offset=0):
    rope = _Rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    return rope(T, torch.device("cpu"), offset=offset)


def test_factory_selects_gqa():
    attn = build_attention(tiny_config(attention_backend="gqa"))
    assert isinstance(attn, GQAttention)


def test_factory_mla_not_implemented():
    try:
        build_attention(tiny_config(attention_backend="mla"))
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_repeat_kv():
    x = torch.randn(2, 2, 5, 8)
    y = repeat_kv(x, 3)
    assert y.shape == (2, 6, 5, 8)
    assert torch.allclose(y[:, 0], x[:, 0]) and torch.allclose(y[:, 2], x[:, 0])


def test_attention_shape_and_causality():
    cfg = tiny_config()
    attn = build_attention(cfg).eval()
    T = 8
    cos, sin = _rope_for(cfg, T)
    x = torch.randn(1, T, cfg.d_model)
    out1, _ = attn(x, cos, sin)
    assert out1.shape == (1, T, cfg.d_model)
    # future token change must not affect earlier outputs
    x2 = x.clone(); x2[:, -1] += 5.0
    out2, _ = attn(x2, cos, sin)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)


def test_attention_kv_cache_equivalence():
    """Incremental (KV-cache) decoding == full forward."""
    cfg = tiny_config()
    attn = build_attention(cfg).eval()
    T = 6
    x = torch.randn(1, T, cfg.d_model)
    cos, sin = _rope_for(cfg, T)
    with torch.no_grad():
        full, _ = attn(x, cos, sin)
        past, steps = None, []
        for t in range(T):
            c, s = _rope_for(cfg, 1, offset=t)
            o, past = attn(x[:, t:t + 1], c, s, past_kv=past, use_cache=True)
            steps.append(o)
        inc = torch.cat(steps, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max().item()


def test_qk_norm_flag_runs():
    cfg = tiny_config(use_qk_norm=True)
    attn = build_attention(cfg).eval()
    assert hasattr(attn, "q_norm")
    cos, sin = _rope_for(cfg, 5)
    out, _ = attn(torch.randn(1, 5, cfg.d_model), cos, sin)
    assert out.shape == (1, 5, cfg.d_model)


def test_manual_and_flash_match():
    """SDPA path and manual fallback must agree."""
    x = torch.randn(1, 7, 64)
    cfg_flash = tiny_config(use_flash_attention=True)
    cfg_manual = tiny_config(use_flash_attention=False)
    a_flash = build_attention(cfg_flash).eval()
    a_manual = build_attention(cfg_manual).eval()
    a_manual.load_state_dict(a_flash.state_dict())     # same weights
    cos, sin = _rope_for(cfg_flash, 7)
    with torch.no_grad():
        o1, _ = a_flash(x, cos, sin)
        o2, _ = a_manual(x, cos, sin)
    assert torch.allclose(o1, o2, atol=1e-4)


# ======================================================================== #
# Module 8 — transformer block + hooks + grad checkpointing
# ======================================================================== #
from model import TransformerBlock, RythForCausalLM


def test_block_shape_and_cache():
    cfg = tiny_config()
    block = TransformerBlock(cfg, layer_idx=0).eval()
    cos, sin = _rope_for(cfg, 5)
    x = torch.randn(1, 5, cfg.d_model)
    out, present = block(x, cos, sin)
    assert out.shape == x.shape and present is None
    out2, present2 = block(x, cos, sin, use_cache=True)
    assert present2 is not None and present2[0].shape[2] == 5


def test_hooks_fire():
    cfg = tiny_config()
    model = RythForCausalLM(cfg).eval()
    seen = []
    model.register_hook("after_attention", lambda t, **c: seen.append(c["layer"]))
    model(torch.randint(0, cfg.vocab_size, (1, 4)))
    assert seen == [0, 1]                       # one call per layer, in order
    model.clear_hooks()
    seen.clear()
    model(torch.randint(0, cfg.vocab_size, (1, 4)))
    assert seen == []                           # cleared


def test_grad_checkpointing_runs():
    cfg = tiny_config(use_gradient_checkpointing=True)
    model = RythForCausalLM(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(ids)
    logits.sum().backward()                     # must produce grads
    assert model.decoder.embed.weight.grad is not None


# ======================================================================== #
# Module 9 — full causal LM (decoder + lm_head)
# ======================================================================== #
def test_forward_shape():
    cfg = tiny_config()
    model = RythForCausalLM(cfg).eval()
    logits, cache = model(torch.randint(0, cfg.vocab_size, (2, 10)))
    assert logits.shape == (2, 10, cfg.vocab_size) and cache is None


def test_weight_tying():
    m_tied = RythForCausalLM(tiny_config(tie_embeddings=True))
    assert m_tied.lm_head.weight.data_ptr() == m_tied.decoder.embed.weight.data_ptr()
    m_untied = RythForCausalLM(tiny_config(tie_embeddings=False))
    assert m_untied.lm_head.weight.data_ptr() != m_untied.decoder.embed.weight.data_ptr()


def test_model_kv_cache_matches_full():
    cfg = tiny_config()
    model = RythForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        full, _ = model(ids)
        past, steps = None, []
        for t in range(ids.size(1)):
            o, past = model(ids[:, t:t + 1], past_kvs=past, use_cache=True)
            steps.append(o)
        inc = torch.cat(steps, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max().item()


def test_causal_lm_causality():
    """Full model: changing a future token can't change earlier logits."""
    cfg = tiny_config()
    model = RythForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        l1, _ = model(ids)
        ids2 = ids.clone(); ids2[:, -1] = (ids2[:, -1] + 1) % cfg.vocab_size
        l2, _ = model(ids2)
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-4)


# ======================================================================== #
# Module 10 — metrics
# ======================================================================== #
from model import model_metrics, format_metrics


def test_metrics_params_match():
    cfg = tiny_config()
    model = RythForCausalLM(cfg)
    m = model_metrics(model, cfg, seq_len=32, batch=2)
    assert m["parameters"] == model.num_params()
    assert m["estimated_gflops"] > 0
    assert m["context_length"] == cfg.max_seq_len
    assert isinstance(format_metrics(m), str)


def test_kv_cache_scales_with_seq():
    cfg = tiny_config()
    model = RythForCausalLM(cfg)
    small = model_metrics(model, cfg, seq_len=16)["kv_cache_bytes"]
    big = model_metrics(model, cfg, seq_len=64)["kv_cache_bytes"]
    assert big == 4 * small                         # linear in seq_len


# ======================================================================== #
# Module 11 — checkpoint metadata
# ======================================================================== #
import tempfile
from model import save_checkpoint, load_checkpoint, build_checkpoint_metadata


def test_checkpoint_metadata_fields():
    cfg = tiny_config()
    meta = build_checkpoint_metadata(cfg, tokenizer_hash="abc123",
                                     dataset_version="1.0.0", rds_version=1)
    for key in ["model_version", "architecture_version", "checkpoint_version",
                "tokenizer_hash", "dataset_version", "rds_version",
                "git_commit", "torch_version", "config"]:
        assert key in meta
    assert meta["model_version"] == "0.2.0"
    assert meta["tokenizer_hash"] == "abc123"


def test_checkpoint_save_load_roundtrip():
    cfg = tiny_config()
    model = RythForCausalLM(cfg)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "ckpt.pt")
    save_checkpoint(path, model, cfg, step=5, tokenizer_hash="h", rds_version=1)
    ck = load_checkpoint(path)
    assert ck["step"] == 5 and ck["metadata"]["rds_version"] == 1
    model2 = RythForCausalLM(cfg)
    model2.load_state_dict(ck["model"])             # weights load cleanly


# ======================================================================== #
# Module 12 — generation
# ======================================================================== #
from model import generate


def test_generate_lengths_and_cache():
    cfg = tiny_config()
    model = RythForCausalLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out = generate(model, prompt, max_new_tokens=10, temperature=0.8, top_k=5)
    assert out.shape == (1, 14)
    assert torch.equal(out[:, :4], prompt)          # prompt preserved


def test_generate_greedy_deterministic():
    cfg = tiny_config()
    model = RythForCausalLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    a = generate(model, prompt, max_new_tokens=8, temperature=0.0)
    b = generate(model, prompt, max_new_tokens=8, temperature=0.0)
    assert torch.equal(a, b)                        # greedy => deterministic


# ======================================================================== #
# Module 13 — robustness / edge cases (adversarial)
# ======================================================================== #
def test_qk_norm_kv_cache_equivalence():
    cfg = tiny_config(use_qk_norm=True)
    model = RythForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        full, _ = model(ids)
        past, steps = None, []
        for t in range(10):
            o, past = model(ids[:, t:t + 1], past_kvs=past, use_cache=True)
            steps.append(o)
        inc = torch.cat(steps, dim=1)
    assert torch.allclose(full, inc, atol=1e-4)


def test_batch_kv_cache_equivalence():
    cfg = tiny_config()
    model = RythForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (4, 8))
    with torch.no_grad():
        full, _ = model(ids)
        past, steps = None, []
        for t in range(8):
            o, past = model(ids[:, t:t + 1], past_kvs=past, use_cache=True)
            steps.append(o)
        inc = torch.cat(steps, dim=1)
    assert torch.allclose(full, inc, atol=1e-4)


def test_all_init_schemes_forward_finite():
    for scheme in ["xavier", "llama", "deepseek"]:
        model = RythForCausalLM(tiny_config(init_scheme=scheme)).eval()
        out, _ = model(torch.randint(0, tiny_config().vocab_size, (1, 5)))
        assert torch.isfinite(out).all()


def test_seq_len_one():
    cfg = tiny_config()
    model = RythForCausalLM(cfg).eval()
    out, _ = model(torch.randint(0, cfg.vocab_size, (1, 1)))
    assert out.shape == (1, 1, cfg.vocab_size)


def test_generate_zero_tokens():
    cfg = tiny_config()
    model = RythForCausalLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    assert torch.equal(generate(model, prompt, max_new_tokens=0), prompt)


def test_untied_lm_head_is_initialized():
    """Untied lm_head weight must be scheme-initialized, not left at defaults."""
    for scheme in ["llama", "xavier", "deepseek"]:
        cfg = tiny_config(tie_embeddings=False, init_scheme=scheme)
        model = RythForCausalLM(cfg)
        w = model.lm_head.weight
        assert w.data_ptr() != model.decoder.embed.weight.data_ptr()   # untied
        assert torch.isfinite(w).all() and w.abs().sum() > 0           # not zeros
        # std should be in a sane init range (not left as empty()/garbage)
        assert 1e-4 < w.std().item() < 1.0, (scheme, w.std().item())


def test_all_params_receive_grad():
    cfg = tiny_config()
    model = RythForCausalLM(cfg).train()
    out, _ = model(torch.randint(0, cfg.vocab_size, (2, 8)))
    out.sum().backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert missing == [], f"params without grad: {missing}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed ✅")
