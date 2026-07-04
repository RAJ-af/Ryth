# Ryth Data Engine (RDE)

RDE turns raw code repositories into compact, reproducible, memory-mapped binary
shards (the RDS format) ready for training. It is a full pipeline, not just an
encoder.

## Input layouts

RDE accepts either layout:

- **Multi-repo:** `root/<repo>/<files...>` — each top-level folder is a repo.
- **Flat:** `root/<files...>` — `root` itself is treated as one repo.

If both loose files and subdirectories exist under `root`, subdirectories are
processed as repos and the loose files as a `root` repo (each file exactly once).
If **zero** usable files are discovered, `run()` raises `EmptyDatasetError` and
the CLI exits with a descriptive message — an empty dataset is never written.

## Pipeline

```
 root/<repo>/<files>   (or flat: root/<files>)
        │
        ▼
   ┌─────────┐   ┌───────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
   │ Cleaner │─► │ Validator │─► │ Quality │─► │Repository│─► │ Language │
   └─────────┘   └───────────┘   └─────────┘   └──────────┘   └──────────┘
        │
        ▼
   ┌─────────┐   ┌─────────────┐   ┌───────────────┐   ┌───────────┐
   │ Encoder │─► │ FIM Builder │─► │Smart Curriculum│─►│Chunk Build │
   └─────────┘   └─────────────┘   └───────────────┘   └───────────┘
        │
        ▼
   ┌───────┐   ┌──────────┐   ┌───────────────┐   ┌─────────────────┐
   │ Dedup │─► │ Sharding │─► │ Manifest Lock │─► │ Validation Report│
   └───────┘   └──────────┘   └───────────────┘   └─────────────────┘
```

Run it in one call:

```python
from dataset import RDEConfig, RDEPipeline
from dataset.tokenizer_adapter import load_bpe_tokenizer

tok = load_bpe_tokenizer("tok/tokenizer.json")       # or ByteTokenizer()
cfg = RDEConfig(seq_len=1024, vocab_size=tok.vocab_size)
manifest = RDEPipeline(tok, cfg).run("raw_repos", "rds_out")
```

## Stages

### Cleaner (`cleaner.py`)
Drops junk before anything expensive happens:
- vendor/build folders (`node_modules`, `venv`, `dist`, …)
- binary files (null bytes / high non-text ratio)
- empty files
- generated code (`@generated`, `DO NOT EDIT`, …)
- non-UTF-8 files
- exact-duplicate files (content SHA-256)

### Validator (`validator.py`)
Flags files that fail: **size**, **extension** allow-list, **language** (must be
known), **encoding**, and **syntax** (Python files are `compile()`-checked; if a
file doesn't compile, it's dropped). No code is executed.

### Quality Analyzer (`quality.py`)
Scores each file **0–100** by blending: readability (line length/long lines),
complexity (branches per function), comments ratio, type-hint coverage, presence
of tests, and repo-level README/License. Also records AST signals (functions,
classes, imports, depth, cyclomatic, async) used downstream.

### Language Detector (`language.py`)
Extension + content heuristics → `python`, `markdown`, `json`, `yaml`, `toml`,
`dockerfile`, `bash`, … (JSON is confirmed by parsing).

### Encoder (`encoder.py`)
`text → token ids` via the tokenizer, optionally wrapped with BOS/EOS.

### FIM Builder (`fim.py`)
Extracts functions via AST and builds **fill-in-the-middle** examples in PSM
form:

```
<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}
```

Split points are seeded per file path, so FIM generation is **deterministic**
(replay-safe). Controlled by `fim_enabled`, `fim_rate`, `fim_min_chars`.

### Smart Curriculum (`curriculum.py`)
Assigns **easy / medium / hard** using a multi-signal complexity score (AST
depth, cyclomatic complexity, imports, classes, async, functions, size) — **not**
file size alone — then orders records easy→hard so a model can learn gradually.

### Chunk Builder — packing (`chunker.py`)
Concatenates encoded documents and cuts them into fixed `seq_len` chunks
("packing"), so there is almost no wasted padding. Optional sliding-window
`overlap`. The final short remainder is padded and its pad count recorded.

### Dedup (`dedup.py`)
Drops duplicate **chunks** by hashing token ids (SHA-256 by default; uses
`xxhash` automatically if installed).

### Sharding (`sharding.py`)
Writes RDS shards and rotates to a new shard when `shard_max_bytes` is reached.
Emits `manifest.json` with per-shard chunk counts and checksums.

### Manifest Lock (`lock.py`)
Reproducibility record embedded in the manifest: `dataset_version`,
`tokenizer_hash`, `model_version`, `creation_time`, `git_commit`,
`python_version`, `seed`, and a full config snapshot. `verify_lock()` reports any
mismatch between a lock and a current config/tokenizer.

### Packing statistics + Validation Report (`stats.py`, `report.py`)
Reports padding %, **packing efficiency %**, average context usage, compression
ratio, language/difficulty splits, duplicate %, and drop reasons — as
`report.json` and a human-readable `report.html`.

## RDS format
Compact binary shards: `uint16` tokens, memory-mapped, O(1) random access,
per-chunk metadata, checksums, and a version field. Full spec in
[rds_format.md](rds_format.md).

## Memory mapping & streaming (`dataset.py`)
`RDSDataset` opens a manifest folder and exposes all shards' chunks as one flat,
randomly-addressable dataset via **mmap** — nothing loads into RAM:

```python
from dataset import RDSDataset
ds = RDSDataset("rds_out")
ds.verify()                 # all shard checksums
chunk = ds[82939]           # O(1) lazy random access
order = ds.indices(seed=1234, worker_id=0, num_workers=8)  # deterministic + sharded
```

- **Streaming / lazy:** only the requested chunk is read.
- **Deterministic shuffle:** same seed → same order → training resume is exact.
- **Multi-worker:** `worker_id`/`num_workers` partition the data disjointly.

## Replayability
Because file iteration is sorted, curriculum sort is stable, and FIM is seeded, a
build is deterministic. Re-running with the same config + seed + inputs yields
byte-identical shards (same checksums). The Manifest Lock records exactly what
produced a dataset.

## Configuration
All behavior is controlled by `RDEConfig` (`config.py`). Common fields:
`seq_len`, `overlap`, `vocab_size`, `shard_max_bytes`, `min_quality`,
`dedup_files`, `dedup_chunks`, `fim_enabled`, `fim_rate`, `dataset_version`,
`model_version`, `seed`. See [configs/](../configs) for reference values.

## CLI
```bash
ryth-rde build   raw_repos rds_out --tokenizer tok/tokenizer.json --seq_len 1024
ryth-rde verify  rds_out
ryth-rde inspect rds_out --chunk 0 --decode --tokenizer tok/tokenizer.json
ryth-rde stats   rds_out
ryth-rde manifest rds_out
```
