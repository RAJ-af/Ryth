# Architecture

Ryth is built as four independent, composable pillars: the **tokenizer**, the
**Ryth Data Engine (RDE)**, the **model core**, and the **training engine**. Each
has a small, well-defined interface, so a layer can be understood — or replaced —
without touching the others.

| Pillar | Package | Since | Depends on |
|--------|---------|-------|------------|
| Tokenizer | `tokenizer/` | v0.1.0 | standard library only |
| Ryth Data Engine (RDE) | `dataset/` | v0.1.0 | standard library only |
| Model Core | `model/` | v0.2.0 | PyTorch |
| Training Engine | `training/` | v0.3.0 | PyTorch |

The tokenizer and RDE share the tokenizer's `encode`/`decode` interface. The
training engine consumes RDE's `RDSDataset` and drives the model core — the model
never imports the trainer, and the trainer treats the model as a plain
`nn.Module`, so custom models can be trained too.

## High-level flow

```
   raw code repositories
            │
            ▼
   ┌───────────────────────────────────────────────────────────┐
   │                    Ryth Data Engine (RDE)                  │
   │                                                            │
   │  Cleaner ─► Validator ─► Quality ─► Repository ─► Language  │
   │     │                                                      │
   │     ▼                                                      │
   │  Encoder ◄──────── (scratch BPE tokenizer plugs in here)   │
   │     │                                                      │
   │     ▼                                                      │
   │  FIM Builder ─► Smart Curriculum ─► Chunk Builder ─► Dedup  │
   │     │                                                      │
   │     ▼                                                      │
   │  Sharding ─► Manifest Lock ─► Validation Report            │
   └───────────────────────────┬───────────────────────────────┘
                               │
                               ▼
                        RDS shards (binary)
                               │
                               ▼
                RDSDataset  (mmap, random access, streaming)
                               │
                               ▼
   ┌───────────────────────────────────────────────────────────┐
   │                    Training Engine                         │
   │                                                            │
   │  dataloader ─► loss ─► gradient (accum / clip / NaN-skip)   │
   │     │                                                      │
   │     ▼                                                      │
   │  AdamW ─► warmup+cosine ─► mixed precision (bf16/fp16)      │
   │     │                                                      │
   │     ▼                                                      │
   │  evaluator (loss+ppl) ─► logger ─► checkpoint (+resume)     │
   │     │                              ─► callbacks (early stop)│
   │     ▼                                                      │
   │  drives ▼                                                  │
   │  ┌─────────────────────────────────────────────────────┐  │
   │  │  Model Core — decoder-only transformer               │ │
   │  │  RoPE · RMSNorm · SwiGLU · GQA + KV-cache (30M→1B)    │ │
   │  └─────────────────────────────────────────────────────┘  │
   └───────────────────────────────────────────────────────────┘
```

## Design principles

1. **From scratch, pure standard library.** The tokenizer and RDE core have no
   required third-party dependencies. This keeps the project auditable and
   portable, and it runs on modest hardware.

2. **The data format is stable, but versioned.** RDS carries a format `version`
   in its header, and the reader dispatches on it (`SUPPORTED_VERSIONS`). New
   formats (v2, v3) can be added without breaking old shards. We never assume the
   format will *never* change — we assume it *will*, and stay backward-compatible.

3. **Reproducibility is a feature.** Every dataset build writes a **Manifest
   Lock** (dataset version, tokenizer hash, config snapshot, seed, git commit,
   Python version). Given the same inputs + lock, a build is deterministic.

4. **Low-hardware first.** Tokens are stored as `uint16` (half the bytes of
   `uint32`), shards are memory-mapped (100GB+ datasets never load into RAM),
   chunks are loaded lazily, and shuffling is deterministic so training can
   resume with the same order.

## Component responsibilities

| Component | Responsibility |
|-----------|----------------|
| **Tokenizer** (`tokenizer/`) | text ↔ token ids; trained from scratch (BPE) |
| **Cleaner** | drop vendor/binary/empty/generated/duplicate files, validate UTF-8 |
| **Validator** | syntax / encoding / language / file-type / size / extension checks |
| **Quality Analyzer** | 0–100 score from readability, complexity, comments, type hints, tests |
| **Language Detector** | Python, Markdown, JSON, YAML, TOML, Dockerfile, Bash |
| **Encoder** | text → token ids (with optional BOS/EOS) |
| **FIM Builder** | fill-in-the-middle examples from functions |
| **Smart Curriculum** | difficulty from AST depth, cyclomatic complexity, imports, classes, async |
| **Chunk Builder** | pack tokens into fixed `seq_len` chunks |
| **Dedup** | drop duplicate chunks (sha256, or xxhash if installed) |
| **Sharding** | rotate shards at a size cap; write manifest + checksums |
| **RDS** (`rds.py`) | binary read/write, mmap, random access, checksum, versioning |
| **RDSDataset** | training-time reader: streaming, lazy, deterministic shuffle, multi-worker |
| **Stats / Report** | packing statistics + JSON/HTML validation report |
| **Manifest Lock** | reproducibility metadata |
| **Model Core** (`model/`) | decoder-only transformer: RoPE, RMSNorm, SwiGLU, GQA + KV-cache, generation |
| **Training Engine** (`training/`) | AdamW, warmup+cosine, mixed precision, grad accum/clip, checkpoint/resume, curriculum, eval |

## Package layout

- `tokenizer/` — standalone; depends on nothing in `dataset/`.
- `dataset/` — depends only on the standard library. The only place it touches
  the tokenizer is `tokenizer_adapter.load_bpe_tokenizer()`, imported on demand,
  so RDE can run with the built-in `ByteTokenizer` fallback alone.
- `model/` — pure PyTorch; depends on nothing else in Ryth. A `RythConfig` +
  `RythForCausalLM` you can import and run on its own.
- `training/` — pure PyTorch; consumes `dataset/`'s `RDSDataset` and trains any
  `nn.Module` (the model core by default). It imports the model only to build one
  from a preset when you don't pass your own.

See [dataset_engine.md](dataset_engine.md) and [tokenizer.md](tokenizer.md) for
data-side detail, [model.md](model.md) and [training.md](training.md) for the
PyTorch side, and [rds_format.md](rds_format.md) for the binary layout.
