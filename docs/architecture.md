# Architecture

Ryth's Foundation Release has two independent, composable pillars: the
**tokenizer** and the **Ryth Data Engine (RDE)**. They share a tiny interface —
the tokenizer's `encode`/`decode` — and are otherwise decoupled.

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

## Package layout

- `tokenizer/` — standalone; depends on nothing in `dataset/`.
- `dataset/` — depends only on the standard library. The only place it touches
  the tokenizer is `tokenizer_adapter.load_bpe_tokenizer()`, imported on demand,
  so RDE can run with the built-in `ByteTokenizer` fallback alone.

See [dataset_engine.md](dataset_engine.md) and [tokenizer.md](tokenizer.md) for
component-level detail, and [rds_format.md](rds_format.md) for the binary layout.
