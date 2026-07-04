# Changelog

All notable changes to Ryth are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] — Foundation Release — 2026-07-03

The first public release. Ships the two foundational pillars of Ryth: a
from-scratch tokenizer and a complete data engine. **Pure Python standard
library — no required runtime dependencies.**

### Tokenizer — scratch Byte-Level BPE
- From-scratch Byte-Pair Encoding tokenizer (`tokenizer/bpe.py`), no external
  libraries.
- Byte-level with full **byte fallback** — any language/script is representable,
  **no "unknown" token**.
- BPE **training** from a corpus (learns merges from the most frequent pairs).
- GPT-2-style regex **pre-tokenization** (keeps merges within word boundaries).
- **Encode** / **decode** with exact round-trip for in-vocab ids.
- **Special tokens** (FIM + chat sentinels), treated atomically.
- **Save / load** to a JSON file (merges + special tokens).
- Corpus helpers for JSONL datasets and plain-text/code globs.
- CLI: `ryth-tokenizer train | encode | decode`.

### Ryth Data Engine (RDE v1.1)
- **Cleaner** — removes vendor/build folders, binaries, empty files, generated
  code, non-UTF-8 files, and exact duplicates.
- **Validator** — syntax, encoding, language, file-type, size, extension checks
  (Python files are `compile()`-checked; code is never executed).
- **Quality Analyzer** — 0–100 score (readability, complexity, comments, type
  hints, tests, README/License).
- **Language Detector** — Python, Markdown, JSON, YAML, TOML, Dockerfile, Bash.
- **Encoder** — text → token ids with optional BOS/EOS.
- **FIM Builder** — deterministic fill-in-the-middle examples from functions.
- **Smart Curriculum** — difficulty from AST depth, cyclomatic complexity,
  imports, classes, async (not size alone); orders easy → hard.
- **Chunk Builder** — token packing into fixed `seq_len` chunks.
- **Deduplication** — chunk-level (sha256, or xxhash if installed).
- **RDS binary format** — `uint16` tokens, memory-mapped, O(1) random access,
  per-chunk JSON metadata, sha256 footer checksum.
- **Sharding** — automatic shard rotation at a size cap.
- **Manifest Lock** — reproducibility metadata (dataset version, tokenizer hash,
  model version, creation time, git commit, python version, config, seed).
- **Version-aware reader** — header `version` dispatch (`_PARSERS`) for
  backward-compatible format evolution.
- **RDSDataset** — streaming, lazy, deterministic-shuffle, multi-worker reader.
- **Packing statistics** + **validation report** (JSON + HTML).
- **Replayability** — `verify_lock()` and deterministic builds.
- **Dual input layouts** — both `root/<repo>/<files>` (multi-repo) and
  `root/<files>` (flat) are supported; mixed layouts process each file once.
- **No silent empty datasets** — if zero usable files are discovered/kept, the
  build raises `EmptyDatasetError` (CLI exit code 2) with a descriptive message.
- **`--debug` flag** — prints every pipeline stage (discovery → cleaner →
  validator → language → encoder → chunker).
- CLI: `ryth-rde build | inspect | verify | stats | manifest`.

### Packaging & docs
- Installable via `pip install -e .` with console scripts `ryth-tokenizer` and
  `ryth-rde`.
- Documentation: `README.md` + `docs/` (architecture, tokenizer, dataset engine,
  RDS format, quickstart, FAQ).
- Runnable examples in `examples/`, reference configs in `configs/`.
- Test suite (`tests/`): 15 RDE tests + 8 tokenizer tests, pure standard library.

### Not included (see [ROADMAP.md](ROADMAP.md))
- Training engine, model architecture, and trained checkpoints (Phase 3+).

[0.1.0]: https://github.com/RAJ-af/Ryth/releases/tag/v0.1.0
