# Changelog

All notable changes to Ryth are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.4.0] — Extensible BPE Tokenizer (v2) — 2026-07-07

Extensible Byte-Pair Encoding (BPE) tokenizer supporting vocabulary extension, reserved special token slots, script-aware pre-tokenization, Unicode NFC normalization, and reproducible sha256 checksums.

- **Tokenizer Version 2** — Bumped tokenizer version to `2`. **Breaking change**: Default `--reserved 256` special token slots shift special token IDs relative to v0.3.x tokenizers, making old model checkpoints trained with v0.3.x tokenizers incompatible.
- **Reserved Special Tokens** — Added `--reserved N` (default 256) to `ryth-tokenizer train`, creating `<|reserved_0|>`...`<|reserved_{N-1}|>` slots at fixed sequential IDs before chat/FIM special tokens.
- **Incremental Vocab Extension (`extend`)** — Added `ryth-tokenizer extend --base tok/tokenizer.json --files '<glob>' --add_vocab N --out tok_v2` and `BPETokenizer.extend()`. All existing merges and token IDs remain 100% identical; new merges are learned only from the new corpus and appended above existing IDs.
- **Unicode NFC Normalization** — Applied `unicodedata.normalize("NFC", ...)` during `train` and `encode`. `decode()` does not normalize and joins raw byte tokens, ensuring exact roundtrip `decode(encode(x)) == NFC(x)`.
- **Script-Aware Devanagari Pre-tokenization** — Enhanced regex pre-tokenizer to keep Devanagari words intact with combining marks (matras, virama, nukta, anusvara) and ZWJ/ZWNJ (`\u200d`/`\u200c`), while excluding danda (`।`) and double danda (`॥`) into separate punctuation tokens.
- **JSON Checksum Verification** — Included `version` and `sha256` fields in `tokenizer.json`. `BPETokenizer.load()` automatically verifies the content hash (excluding the `sha256` key itself) and raises a `ValueError` on tampering, while maintaining backward compatibility for older v1 `tokenizer.json` files.
- **Tokenizer Metrics Script** — Added `scripts/tokenizer_report.py` to evaluate and report tokens-per-word and bytes-per-token metrics for English, Hindi, Hinglish, and Python code.

## [0.3.2] — Fast BPE Tokenizer Trainer — 2026-07-06

Optimized Byte-Pair Encoding (BPE) tokenizer training in pure Python standard library (~300x faster).

- **Optimized BPE Training Engine** — Replaced full-corpus rescans with unique-word deduplication, incremental pair-count updates, inverted index mapping, and lazy heap selection with deterministic tie-breaking.
- **Multiprocessing Pre-tokenization** — Added optional parallel word counting (`--workers N`, defaulting to `os.cpu_count()`) for regex pre-tokenization while preserving bit-identical merge sequence ordering.
- **Corpus Truncation Flag** — Added `--max_bytes N` to `ryth-tokenizer train` to cap corpus text size.
- **Benchmark Script** — Added `scripts/bench_tokenizer.py` to benchmark tokenizer training duration and throughput on 2 MB and 10 MB code corpora.
- **Reference Naive Trainer** — Maintained `BPETokenizer._train_naive` for differential testing and parity verification in test suite.

## [0.3.1] — 30M Prototype End-to-End Tools — 2026-07-06

End-to-end data preparation, CPU smoke testing, and generation scripts for the 30M prototype.

- **`scripts/prepare_30m_data.sh`** — Automated pipeline that builds the corpus, trains the BPE tokenizer (vocab 16000, saved to `tok/`), and constructs the RDS dataset (seq_len 1024). Automatically fetches permissively licensed source code or falls back to a sample fixture if offline.
- **`scripts/smoke_test.sh`** — CPU smoke test script wrapping `ryth-train` for 20 steps to confirm training pipeline integrity and loss decrease.
- **`scripts/generate.py`** — Inference generation CLI loading checkpoint (`final.pt`) + tokenizer (`tokenizer.json`) with configurable prompt, temperature, top-k, max new tokens, and auto device selection.

## [Unreleased]

- **RDE Plain-Text Mode (`build-text`)** — Added plain-text mode to the Ryth Data Engine (`dataset/text_pipeline.py` and `ryth-rde build-text <input_dir> <out_dir>`) for building RDS datasets from prose / non-code `.txt` files. Bypasses code-specific cleaners/validators/quality filters while keeping NFC normalization, exact duplicate removal by SHA-256 hash, and dropping documents shorter than 50 characters. Documents are split by `<|endoftext|>`, deterministically partitioned into train and validation splits by hash (`--val_frac`), upsampled/downsampled by source file stem via `--mix`, and packed with `<|endoftext|>` sentinel IDs. Manifests record token counts per source.

## [Unreleased] — End-to-end notebook (official entry point) — 2026-07-06

The canonical "Run All" notebook for Ryth — one file that takes a raw code
corpus (or a synthetic fallback) all the way to a trained model + eval +
artifacts, using only the public library. No core code changes.

- **`notebooks/ryth_end_to_end.ipynb`** — 11 stages: env setup → configuration
  → corpus → tokenizer → RDS → model → training → evaluation → export →
  resume state → summary. **Idempotent** (every stage detects existing assets
  and skips; training auto-resumes from `latest.pt`), **GPU-aware** (`fp16`
  on T4 Turing, `bf16` on Ampere+; override via `FORCE_DTYPE`).
- **One configuration cell** — flip `SMOKE = False` for a real run, point
  `RAW_DIR` / `GITHUB_SOURCES` / `HF_SOURCES` at your data, and choose
  `MODEL_PRESET = "ryth_30m" | "ryth_125m" | "ryth_350m" | "ryth_1b"`.
- Runs unmodified on **Kaggle** (T4 / L4 / A100), **Colab**, **local**, or a
  cluster — only the env-setup cell's `pip install` branch differs.
- Each run writes a `summary.json` + training curve (`training_curve.png`)
  alongside the tokenizer, RDS shards, and checkpoints.

### Notes
- The Kaggle-specific sibling (`ryth_kaggle_train.ipynb`) is kept as a
  reference; the new notebook supersedes it.
- Run artifacts (RDS shards, checkpoints, plots) are git-ignored in
  `notebooks/ryth_run/`.

## [Corpus 1.0] — Ryth Corpus — 2026-07-05

A standalone **corpus engineering system** (`corpus/` package) that turns raw code
repositories into a world-class, license-clean, deduplicated, quality-scored,
task-formatted training dataset for 30M → 1B models. Pure standard library core;
**does not modify** the tokenizer, RDE, model core, or training engine.

### Corpus pipeline (`corpus/` package)
- **`CorpusConfig`** — one config for languages, task ratios, license policy,
  quality threshold, dedup thresholds, splits, size caps (configurable ratios).
- **Sources** (`sources/`) — declarative registry: `github` (public zip via
  `urllib`, no token), `huggingface` (optional `datasets`), `http` (docs), `local`
  (offline). Downloaders (`download/`) materialize sources into a uniform staging dir.
- **Licenses** (`licenses/`) — SPDX detection from LICENSE files + headers;
  permissive-only policy (MIT/Apache-2.0/BSD/ISC/MPL-2.0/…), GPL/unknown rejected
  unless explicitly enabled. The source's license hint is re-verified, never trusted.
- **Cleaning** (`cleaners/`) — removes vendor/build/cache/venv folders, binaries,
  minified bundles, lock files, generated code, corrupted encodings, oversized
  files; strips **Jupyter notebook outputs**; detects + **redacts secrets/API keys**.
- **Filters** (`filters/`) — license/size/language filters + deterministic
  language-ratio balancing.
- **Dedup** (`dedup/`) — exact (sha256) + **near-duplicate (MinHash + LSH)**, at
  file and repository level.
- **Quality** (`quality/`) — 0–100 repo score from syntax validity, documentation,
  tests, project structure, comments, complexity, maintainability, duplicate ratio;
  configurable weights + minimum threshold.
- **Splits** (`split/`) — deterministic, **leakage-free** repo-level train/val/test.
- **Metadata** (`metadata/`) — `FileRecord`/`RepoRecord` (+ JSONL store): repository,
  license, language, path, hash, size, quality score, source, split, timestamp.
- **Tasks** (`tasks/`) — next-token, FIM, completion, editing, docstring→code,
  README→code, code→explanation, bug-fixing, refactoring, unit-test generation, with
  configurable ratios.
- **Exporters** (`exporters/`) — raw folders, JSONL, Parquet (optional `pyarrow`),
  and **Ryth RDS via the existing RDE (unchanged)**.
- **Reports** (`report.py`) — language/license/duplicate/quality/rankings/size/
  task-distribution, as **HTML + JSON**.
- **CLI** — `ryth-corpus download | clean | score | build | stats | export`.
- **Reproducible** — no wall-clock, no RNG; all selection derives from content
  hashes + `seed`; timestamps passed in by the caller.
- **33 unit tests** (offline, synthetic repos) covering every module.

## [0.3.0] — Training Engine — 2026-07-05

Pure-PyTorch training engine for the Ryth model core, over RDS datasets. Modular
and production-quality. Requires PyTorch (`pip install -e ".[train]"`).

### Training engine (`training/` package)
- **`TrainConfig`** — single config controlling the whole run.
- **AdamW optimizer** factory with decay/no-decay param groups (fused on CUDA).
- **Warmup + cosine** scheduler factory.
- **Gradient**: accumulation, clipping, and **NaN/Inf detection & skip**.
- **Mixed precision**: bf16 / fp16 autocast + GradScaler (fp16).
- **Gradient checkpointing** (via model core flag).
- **Loss**: causal-LM cross entropy + perplexity.
- **Dataloader**: RDSDataset → next-token batches, deterministic train/val split.
- **Curriculum learning**: RDE difficulty metadata → easy→hard ordering.
- **Evaluator**: validation loss + perplexity.
- **Logger**: console + JSONL + optional TensorBoard.
- **Checkpoint manager**: save/load, **auto-resume** (`latest.pt`), `keep_last`
  rotation, `best.pt`/`final.pt`, and **experiment metadata** (git commit,
  tokenizer hash, dataset version, model version, torch version, seed).
- **Callbacks**: extensible callback system + `EarlyStopping`.
- **Profiler**: `torch.profiler` wrapper.
- **Trainer**: the loop; also accepts an injected model (fine-tuning / custom sizes).
- **Metrics**: throughput + running averages.
- **Benchmark** (`training/benchmark.py`) and **CLI** (`ryth-train`).
- **Determinism**: seed + optional deterministic mode. CPU & GPU.
- **28 unit tests** (real PyTorch), incl. end-to-end training (loss decreases),
  auto-resume, and early-stop wiring.

### Kaggle end-to-end (tooling)
- **`notebooks/ryth_kaggle_train.ipynb`** — one notebook that runs the whole stack
  on a free Kaggle **T4** GPU: corpus → tokenizer → RDS → 30M model → training →
  checkpoint → **resume** → **code generation**. Self-contained (synthetic smoke
  corpus) and uses only the public library — no core changes.
- **`scripts/kaggle_train.py`** — the same pipeline as one command
  (`--smoke` for a fast test, `--raw <dir>` for a real run).
- Auto-selects **fp16 on T4** (Turing has no native bf16) and **bf16 on Ampere+**.

### Notes
- Model core (v0.2.0) unchanged. Next: train the 30M prototype (ROADMAP Phase 5).

## [0.2.0] — Model Core — 2026-07-04

The transformer model core, from scratch in pure PyTorch. Requires PyTorch
(`pip install -e ".[model]"`); the tokenizer + data engine remain dependency-free.

### Model core (`model/` package)
- **`RythConfig`** — single config with presets (30M/125M/350M/1B), version fields
  (`model_version`, `architecture_version`, `checkpoint_version`), and feature
  flags (`use_flash_attention`, `use_qk_norm`, `use_gradient_checkpointing`,
  `attention_backend`).
- **RoPE** — rotary positional embeddings, configurable theta, KV-cache offset aware.
- **RMSNorm** — float32-stable normalization.
- **SwiGLU** — gated feed-forward.
- **Attention factory** — `model/attention/`: `BaseAttention` interface, **GQA**
  (Grouped-Query Attention) with KV-cache + SDPA/FlashAttention + manual fallback,
  and an **MLA** stub (`NotImplementedError`) reserved for the future.
- **Initialization module** — xavier / llama / deepseek schemes (llama default,
  residual projections scaled by `1/√(2·n_layers)`); `ryth` reserved for future.
- **Hooks** — `before_attention` / `after_attention` / `before_ffn` / `after_ffn`
  per block, for research and debugging.
- **Metrics** — parameters, trainable params, estimated FLOPs, KV-cache size,
  activation memory, context length.
- **Checkpoint metadata** — model/tokenizer/dataset/RDS versions, git commit,
  PyTorch version, full config (`model/checkpoint.py`).
- **Decoder + causal LM** — `RythDecoder`, `RythForCausalLM` with weight tying.
- **Generation** — autoregressive sampling with KV-cache (greedy / temperature / top-k).
- **Benchmarks** — `benchmarks/forward.py`, `memory.py`, `speed.py` (CPU / GPU).
- **Tests** — 37 unit tests (KV-cache == full-forward equivalence, causality,
  weight tying, flash-vs-manual parity, hooks, gradient checkpointing, and more).

### Not included (see [ROADMAP.md](ROADMAP.md))
- Training loop (Phase 4 — Training Engine).

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
