# Model Core (v0.2.0)

A decoder-only transformer LLM (Llama/GPT family) written **from scratch in pure
PyTorch**. Scale from 30M → 1B by changing only `RythConfig`. Located in the
`model/` package.

Requires PyTorch: `pip install -e ".[model]"`.

## Architecture

```
input_ids [B,T]
    │  TokenEmbedding
    ▼
  ┌─ N × TransformerBlock (Pre-Norm) ───────────────────┐
  │    x = x + Attention(RMSNorm(x))    ← GQA + RoPE     │
  │    x = x + SwiGLU(RMSNorm(x))                         │
  └──────────────────────────────────────────────────────┘
    │  RMSNorm (final)
    ▼  LMHead (weight-tied)
logits [B, T, vocab_size]
```

## Modules

| File | Component |
|------|-----------|
| `config.py` | `RythConfig` — versions, feature flags, presets |
| `init.py` | weight init: xavier / llama / deepseek / ryth(future) |
| `embedding.py` | token embedding (exposes `.weight` for tying) |
| `rope.py` | rotary positional embeddings (cache-offset aware) |
| `rmsnorm.py` | RMSNorm (float32-stable) |
| `feedforward.py` | SwiGLU FFN |
| `attention/` | **factory** → `gqa.py` (default), `mla.py` (future stub), `base.py` |
| `hooks.py` | before/after attention & ffn hook points |
| `transformer_block.py` | pre-norm block + hooks + grad-checkpoint flag |
| `decoder.py` | `RythDecoder` + `RythForCausalLM` |
| `lm_head.py` | vocab projection (weight tying) |
| `metrics.py` | params, FLOPs, KV-cache size, activation memory |
| `checkpoint.py` | checkpoint metadata + save/load |
| `generate.py` | autoregressive sampling (KV-cache) |

## Quick use

```python
import torch
from model import RythConfig, RythForCausalLM, generate

cfg = RythConfig.ryth_30m(vocab_size=32000)     # or ryth_125m / ryth_350m / ryth_1b
model = RythForCausalLM(cfg)

ids = torch.randint(0, cfg.vocab_size, (1, 16))
logits, _ = model(ids)                          # [1, 16, vocab_size]
print(model.num_params())

out = generate(model, ids, max_new_tokens=32, temperature=0.8, top_k=40)
```

### KV-cache decoding
```python
logits, past = model(ids, use_cache=True)                 # prefill
next_tok = logits[:, -1:].argmax(-1)
logits, past = model(next_tok, past_kvs=past, use_cache=True)   # 1 token/step
```

## Attention factory

Attention is pluggable. `config.attention_backend` selects the implementation:

```python
RythConfig(attention_backend="gqa")   # default — Grouped-Query Attention
RythConfig(attention_backend="mla")   # future — raises NotImplementedError today
```

Add a backend: implement `BaseAttention` in `model/attention/`, register it in
`factory.py`. No block/decoder changes needed.

## Feature flags

| Flag | Default | Effect |
|------|---------|--------|
| `use_flash_attention` | `True` | use SDPA / flash kernels (fallback to manual) |
| `use_qk_norm` | `False` | RMSNorm on query/key (training stability) |
| `use_gradient_checkpointing` | `False` | trade compute for activation memory (training) |
| `attention_backend` | `"gqa"` | attention implementation |

## Initialization schemes

`config.init_scheme` ∈ `{xavier, llama, deepseek, ryth}` (default `llama`,
residual projections scaled by `1/√(2·n_layers)`). `ryth` is reserved for a future
custom scheme (`NotImplementedError` today).

## Versions

`config.model_version="0.2.0"`, `architecture_version=1`, `checkpoint_version=1`.
Checkpoints also record tokenizer hash, dataset version, RDS version, git commit,
and PyTorch version (`model/checkpoint.py`) — for reproducibility and forward
compatibility.

## Hooks (research/debugging)

```python
model.register_hook("after_attention", lambda t, **ctx: print(ctx["layer"], t.std()))
# ... run a forward pass ...
model.clear_hooks()
```
Points: `before_attention`, `after_attention`, `before_ffn`, `after_ffn`.

## Presets

| Preset | d_model | layers | heads (q/kv) | max_seq |
|--------|---------|--------|--------------|---------|
| `ryth_30m` | 512 | 8 | 8 / 2 | 1024 |
| `ryth_125m` | 768 | 12 | 12 / 4 | 2048 |
| `ryth_350m` | 1024 | 24 | 16 / 4 | 2048 |
| `ryth_1b` | 2048 | 22 | 16 / 8 | 4096 |

> Param counts depend on `vocab_size` (tied embeddings dominate small models — e.g.
> `ryth_30m` is ~40M with a 32k vocab). Names are approximate scale labels.

## Metrics & benchmarks

```bash
python -m benchmarks.forward --preset ryth_30m --seq_len 256 --device cpu   # or cuda
python -m benchmarks.memory  --preset ryth_30m --device cuda
python -m benchmarks.speed   --preset ryth_30m --device cuda
```

`benchmarks/forward.py` prints: Parameters, Trainable Params, Estimated FLOPs,
KV Cache Size, Peak Activation Memory, Context Length.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/test_model.py -v      # 37 tests
```
Covers RoPE norm-preservation & offset, GQA + `repeat_kv`, **KV-cache == full
forward**, causality, weight tying, flash-vs-manual parity, QK-norm, hooks,
gradient checkpointing, metrics, checkpoint round-trip, and generation.

> Training is **not** part of this milestone — see [ROADMAP.md](../ROADMAP.md).
