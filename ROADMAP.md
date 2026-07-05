# Roadmap

Ryth is built one solid foundation layer at a time. The data format stays stable
(and versioned) across model sizes — scaling up mostly means changing config, not
rewriting the stack.

## Phases

### Phase 1 — Scratch Tokenizer ✅ (v0.1.0)
From-scratch byte-level BPE tokenizer: training, encode/decode, special tokens,
save/load, byte fallback. Pure standard library.

### Phase 2 — Ryth Data Engine (RDE) ✅ (v0.1.0)
Full data pipeline → RDS binary format: cleaning, validation, quality scoring,
language detection, FIM, smart curriculum, packing, dedup, sharding, checksums,
manifest lock, reproducibility, and a memory-mapped streaming reader.

### Phase 3 — Model Core ✅ (v0.2.0)
Decoder-only transformer from scratch (pure PyTorch): RythConfig, RoPE, RMSNorm,
SwiGLU, Grouped-Query Attention with KV-cache and a FlashAttention (SDPA) path.
Pluggable **attention factory** (GQA + MLA-future stub), init schemes
(xavier/llama/deepseek), feature flags, hooks, metrics, checkpoint metadata, and
generation. Presets 30M → 1B. Unit tests + benchmarks.

### Phase 4 — Training Engine ✅ (v0.3.0)
Pure-PyTorch training over RDS datasets: AdamW, warmup+cosine schedule, gradient
accumulation & clipping, bf16/fp16, gradient checkpointing, NaN detection,
auto-resume, checkpoint manager, JSON+TensorBoard logging, validation +
perplexity, early stopping, curriculum learning, and experiment tracking. CPU & GPU.

### Phase 5 — Ryth Corpus ✅ (Corpus v1.0)
A standalone corpus engineering system (`corpus/` package): permissive-only
sourcing, cleaning + secret redaction, exact + near-duplicate dedup, quality
scoring, deterministic leakage-free splits, configurable training-task generation,
and export to raw/JSONL/Parquet/RDS — reproducible, pure standard library, and
built without modifying the tokenizer/RDE/model/training engine.

### Phase 6 — 30M Prototype 🔜 (next)
Train the first small model end-to-end on a Ryth Corpus build to validate the full
pipeline (corpus → tokenizer → RDE → model → training → generation).

### Phase 7 — 300M ⏳
Scale the model and dataset; refine curriculum and data mixture.

### Phase 8 — 1B ⏳
A larger coding-focused model built on the same foundation.

## RDE v2 (post first-training-run)

Deferred data-engine features, to be added after a successful training run so we
don't over-engineer ahead of need:

- Repository Graph (README → src → tests → docs relations)
- Import Graph (dependency understanding)
- AST Cache (for code search / editing tasks)
- Dataset Diff (compare dataset versions / changed shards)
- Token Frequency Cache

## Guiding principles

- **From scratch, auditable, pure standard library** for core components.
- **Reproducible by default** (manifest lock + deterministic builds).
- **Low-hardware friendly** (uint16, mmap, sharding, lazy loading).
- **Backward-compatible format evolution** (versioned RDS).
- **Don't over-engineer** ahead of a working end-to-end pipeline.

_Timelines are intentionally omitted; this is a from-scratch research project._
