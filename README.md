<div align="center">

# Ryth

**A coding-first LLM, built from scratch.**

*v0.3.0: scratch BPE tokenizer + Ryth Data Engine + transformer model core + pure-PyTorch training engine.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](CHANGELOG.md)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](ROADMAP.md)

</div>

---

## Introduction

**Ryth** is an effort to build a coding-focused large language model **entirely
from scratch** — no black-box frameworks for the core pieces. Every foundational
component is written in clean, readable Python so it can be understood, audited,
and extended.

As of **v0.3.0**, all four foundational pillars are in place — the full
data → tokenizer → model → training path is buildable from source:

1. **A scratch Byte-Level BPE Tokenizer** — pure Python, no external libraries.
2. **The Ryth Data Engine (RDE)** — a complete data pipeline that turns raw code
   repositories into a compact, reproducible, memory-mapped binary format (RDS)
   ready for training.
3. **The Model Core** — a decoder-only transformer (RoPE, RMSNorm, SwiGLU,
   Grouped-Query Attention, KV-cache) in pure PyTorch, with presets from 30M → 1B.
4. **The Training Engine** — a pure-PyTorch trainer over RDS datasets: AdamW,
   warmup+cosine, mixed precision, gradient accumulation/clipping, checkpointing
   with auto-resume, curriculum learning, and experiment tracking.

> Pillars 1–2 are **pure standard library** (no runtime deps). Pillars 3–4 need
> PyTorch (`pip install -e ".[model]"` / `".[train]"`). Next on the
> [roadmap](ROADMAP.md): training the first **30M prototype** end-to-end.

## Vision

A capable coding model you can build, understand, and run without depending on
heavyweight frameworks — designed to be reproducible and friendly to modest
hardware. Start small (a prototype), scale deliberately, and keep the data format
stable across model sizes so only the config changes.

## Features

**Tokenizer**
- Byte-level BPE — **any** language/script is representable, **no "unknown" token**
- From-scratch BPE training (learns merges from your corpus)
- Special tokens (FIM + chat sentinels), save/load to JSON
- GPT-2-style pre-tokenization (keeps merges from crossing word boundaries)

**Ryth Data Engine (RDE v1.1)**
- Cleaner, Validator, Quality Analyzer, Language Detector
- Encoder → Chunk Builder (packing) → Smart Curriculum (easy→hard)
- **FIM Builder** (fill-in-the-middle examples from functions)
- **RDS binary format**: `uint16` tokens, memory-mapped, O(1) random access
- **Sharding**, **checksums** (corruption detection), **deterministic shuffle**
- **Manifest Lock** for exact reproducibility; **version-aware** reader
- **Packing statistics** + auto **validation report** (JSON + HTML)
- **Pure standard library** — no required runtime dependencies

**Model Core (v0.2.0 — needs PyTorch)**
- Decoder-only transformer: **RoPE**, **RMSNorm**, **SwiGLU**
- **Grouped-Query Attention** with KV-cache + FlashAttention (SDPA) path
- Pluggable **attention factory** (GQA now, MLA reserved), init schemes, feature flags
- Research **hooks**, parameter/FLOP **metrics**, **checkpoint metadata**
- Autoregressive **generation** (greedy / temperature / top-k); presets 30M → 1B

**Training Engine (v0.3.0 — needs PyTorch)**
- **AdamW** (decay/no-decay groups) + **warmup+cosine** schedule
- **Gradient** accumulation, clipping, and **NaN/Inf detection & skip**
- **Mixed precision** (bf16/fp16) + **gradient checkpointing**
- **Checkpoint manager** with **auto-resume**, rotation, `best`/`final`
- **Curriculum learning** (RDE difficulty), **validation + perplexity**
- **Experiment tracking** (git commit, tokenizer hash, dataset/model version)
- JSON + TensorBoard logging; **CPU & GPU**; `ryth-train` CLI

## Architecture

```
                          ┌──────────────────────────┐
   raw code repos ──────► │      Ryth Data Engine     │
                          │                           │
   Cleaner → Validator → Quality → Language → Encoder │
        → FIM Builder → Smart Curriculum → Chunk Builder
        → Dedup → Sharding → Manifest Lock → Report    │
                          └────────────┬──────────────┘
                                       │  writes
                                       ▼
                          ┌──────────────────────────┐
                          │        RDS shards         │
                          │  header · data(uint16)    │
                          │  index · metadata · footer│
                          └────────────┬──────────────┘
                                       │  mmap, random access
                                       ▼
                            RDSDataset (training-ready)
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │      Training Engine      │
                          │  dataloader → loss → grad │
                          │  AdamW → warmup+cosine    │
                          │  eval/ppl → checkpoint    │
                          └────────────┬──────────────┘
                                       │  trains
                                       ▼
                          ┌──────────────────────────┐
                          │        Model Core         │
                          │  RoPE · RMSNorm · SwiGLU  │
                          │  GQA + KV-cache (30M→1B)  │
                          └──────────────────────────┘

   Scratch BPE tokenizer plugs in at the "Encoder" stage.
```

## Installation

Requires **Python 3.9+**. The tokenizer + data engine have **no dependencies**;
the model core and training engine need **PyTorch**.

```bash
git clone https://github.com/RAJ-af/Ryth.git
cd Ryth
pip install -e .            # core: tokenizer + RDE (pure standard library)
# optional extras:
pip install -e ".[fast]"    # + xxhash (faster dedup)
pip install -e ".[model]"   # + torch (model core)
pip install -e ".[train]"   # + torch (training engine)
pip install -e ".[dev]"     # + pytest + torch (run the full test suite)
```

This installs three command-line tools: **`ryth-tokenizer`**, **`ryth-rde`**,
and **`ryth-train`**.

## Quick Start

```bash
# 1. Put permissively-licensed code under raw_repos/
#    Layout can be raw_repos/<repo>/<files...>  OR  raw_repos/<files...> (flat).
#    (or generate a sample fixture — see docs/quickstart.md)

# 2. Train a tokenizer on your code
ryth-tokenizer train --files 'raw_repos/**/*.py' --vocab 8000 --out tok

# 3. Build an RDS dataset (uses your tokenizer; add --debug to trace every stage)
ryth-rde build raw_repos rds_out --tokenizer tok/tokenizer.json --seq_len 1024

# 4. Verify + inspect the result
ryth-rde verify  rds_out
ryth-rde inspect rds_out --chunk 0 --decode --tokenizer tok/tokenizer.json
ryth-rde stats   rds_out

# 5. Train a model on the dataset (needs PyTorch: pip install -e ".[train]")
ryth-train --data_dir rds_out --model_preset ryth_30m --max_steps 2000 --dtype bf16
ryth-train --data_dir rds_out --resume latest        # auto-resume
```

Full walkthrough: **[docs/quickstart.md](docs/quickstart.md)**.

## Folder structure

```
Ryth/
├── README.md            LICENSE  CONTRIBUTING.md  CODE_OF_CONDUCT.md
├── SECURITY.md          CHANGELOG.md  ROADMAP.md   RELEASE_CHECKLIST.md
├── pyproject.toml       requirements.txt  .gitignore
├── tokenizer/           # scratch byte-level BPE tokenizer (package)
│   ├── bpe.py           #   the tokenizer
│   ├── train.py         #   corpus helpers + training
│   └── cli.py           #   ryth-tokenizer
├── dataset/             # Ryth Data Engine (RDE) package
│   ├── pipeline.py      #   orchestrates all stages
│   ├── cleaner.py validator.py quality.py language.py repository.py
│   ├── encoder.py fim.py curriculum.py chunker.py dedup.py
│   ├── rds.py           #   RDS binary reader/writer
│   ├── sharding.py dataset.py lock.py report.py stats.py
│   └── cli.py           #   ryth-rde
├── model/               # decoder-only transformer (needs PyTorch)
│   ├── config.py        #   RythConfig + presets (30M → 1B)
│   ├── attention/       #   base · gqa · mla-stub · factory
│   ├── rope.py rmsnorm.py feedforward.py embedding.py lm_head.py
│   ├── decoder.py       #   RythDecoder / RythForCausalLM
│   ├── init.py hooks.py metrics.py checkpoint.py generate.py
│   └── ...
├── training/            # Ryth Training Engine (needs PyTorch)
│   ├── config.py        #   TrainConfig — single source of truth
│   ├── trainer.py       #   the training loop
│   ├── optimizer.py scheduler.py loss.py gradient.py precision.py
│   ├── dataloader.py curriculum.py evaluator.py metrics.py
│   ├── logger.py checkpoint.py callbacks.py profiler.py benchmark.py
│   └── cli.py           #   ryth-train
├── docs/                # architecture, tokenizer, dataset engine, RDS, model, training, quickstart, faq
├── examples/            # runnable example scripts
├── tests/               # pytest suite (core is pure stdlib; model/training need torch)
├── scripts/             # convenience shell scripts
└── configs/             # reference config values
```

## Examples

Runnable scripts in [`examples/`](examples/):

| Script | What it does |
|--------|--------------|
| `example_train_tokenizer.py` | Train a BPE tokenizer on sample code |
| `example_encode_dataset.py`  | Build an RDS dataset with the pipeline |
| `example_inspect_dataset.py` | Inspect chunks + metadata |
| `example_verify_dataset.py`  | Verify shard checksums |
| `example_decode_tokens.py`   | Encode ↔ decode roundtrip |

```bash
python examples/example_train_tokenizer.py
```

## Dataset format (RDS)

RDS is a compact, self-describing binary shard format:

```
[HEADER 64B]  magic · version · tokenizer_version · vocab_size · seq_len
              dtype · n_chunks · data/index/metadata/footer offsets
[DATA]        token ids, back-to-back (uint16 → half the storage)
[INDEX]       per-chunk (offset, length) → O(1) random access
[METADATA]    JSON: per-chunk repo/language/quality/difficulty/path
[FOOTER]      sha256 checksum → corruption detection
```

Memory-mapped, version-aware, and shardable. Details:
**[docs/rds_format.md](docs/rds_format.md)**.

## Tokenizer overview

A **byte-level** Byte-Pair Encoding tokenizer written from scratch. Because it
operates on UTF-8 bytes, it can represent **any** text (any language/script) with
no unknown tokens (byte fallback). BPE then learns merges for the most frequent
patterns in your corpus, compressing common code sequences into single tokens.
Details: **[docs/tokenizer.md](docs/tokenizer.md)**.

## Model core (v0.2.0)

A decoder-only transformer built from scratch in pure PyTorch (`model/` package,
`pip install -e ".[model]"`): RoPE, RMSNorm, SwiGLU, **Grouped-Query Attention**
with KV-cache and a FlashAttention (SDPA) path. Extensible by design — a pluggable
**attention factory** (GQA now, MLA reserved), configurable init schemes, feature
flags, research hooks, metrics, and checkpoint metadata. Scale 30M → 1B via
`RythConfig` presets.

```python
import torch
from model import RythConfig, RythForCausalLM, generate
model = RythForCausalLM(RythConfig.ryth_30m(vocab_size=32000))
logits, _ = model(torch.randint(0, 32000, (1, 16)))
```

Details: **[docs/model.md](docs/model.md)**.

## Training engine (v0.3.0)

Pure-PyTorch training over RDS datasets (`training/` package,
`pip install -e ".[train]"`): AdamW, warmup+cosine schedule, gradient
accumulation & clipping, bf16/fp16, gradient checkpointing, NaN detection,
auto-resume, checkpoint manager, JSON+TensorBoard logging, validation +
perplexity, early stopping, curriculum learning (RDE difficulty), and experiment
tracking (git commit, tokenizer hash, dataset/model version). CPU & GPU.

```python
from training import TrainConfig, Trainer
Trainer(TrainConfig(data_dir="rds_out", model_preset="ryth_30m",
                    dtype="bf16", max_steps=2000)).train()
```
```bash
ryth-train --data_dir rds_out --model_preset ryth_30m --max_steps 2000 --dtype bf16
ryth-train --data_dir rds_out --resume latest      # auto-resume
```

Details: **[docs/training.md](docs/training.md)**.

## CLI Reference

Ryth installs two console commands. Below is the exact `--help` output for every
subcommand (copied verbatim from the installed CLI).

### `ryth-tokenizer`

```text
$ ryth-tokenizer --help
usage: ryth-tokenizer [-h] {train,encode,decode} ...

Ryth scratch BPE tokenizer CLI.

positional arguments:
  {train,encode,decode}
    train               train a BPE tokenizer on a corpus
    encode              encode text to token ids
    decode              decode token ids to text

options:
  -h, --help            show this help message and exit
```

```text
$ ryth-tokenizer train --help
usage: ryth-tokenizer train [-h] [--data DATA] [--files FILES] [--vocab VOCAB]
                            [--out OUT]

options:
  -h, --help     show this help message and exit
  --data DATA    jsonl corpus folder
  --files FILES  glob for plain text/code files (e.g. 'src/**/*.py')
  --vocab VOCAB  target vocab size
  --out OUT      output folder
```

```text
$ ryth-tokenizer encode --help
usage: ryth-tokenizer encode [-h] --tokenizer TOKENIZER [--text TEXT]
                             [--input INPUT]

options:
  -h, --help            show this help message and exit
  --tokenizer TOKENIZER
                        tokenizer.json path
  --text TEXT           text to encode
  --input INPUT         file to encode ('-' for stdin)
```

```text
$ ryth-tokenizer decode --help
usage: ryth-tokenizer decode [-h] --tokenizer TOKENIZER --ids IDS

options:
  -h, --help            show this help message and exit
  --tokenizer TOKENIZER
                        tokenizer.json path
  --ids IDS             space/comma separated token ids
```

### `ryth-rde`

```text
$ ryth-rde --help
usage: ryth-rde [-h] {build,inspect,verify,stats,manifest} ...

Ryth Data Engine CLI.

positional arguments:
  {build,inspect,verify,stats,manifest}
    build               build an RDS dataset from raw repos
    inspect             inspect a dataset / chunk
    verify              verify shard checksums
    stats               print dataset statistics
    manifest            print manifest + reproducibility lock

options:
  -h, --help            show this help message and exit
```

```text
$ ryth-rde build --help
usage: ryth-rde build [-h] [--tokenizer TOKENIZER] [--seq_len SEQ_LEN]
                      [--shard_mb SHARD_MB] [--debug]
                      root out

positional arguments:
  root                  input folder: <root>/<repo>/<files...> or
                        <root>/<files...>
  out                   output dataset folder

options:
  -h, --help            show this help message and exit
  --tokenizer TOKENIZER
                        BPE tokenizer.json (default: byte-level)
  --seq_len SEQ_LEN     chunk length in tokens (default: 1024)
  --shard_mb SHARD_MB   shard size (MB)
  --debug               print every pipeline stage (discovery, cleaner,
                        validator, language, encoder, chunker)
```

```text
$ ryth-rde inspect --help
usage: ryth-rde inspect [-h] [--chunk CHUNK] [--decode]
                        [--tokenizer TOKENIZER]
                        data_dir

positional arguments:
  data_dir

options:
  -h, --help            show this help message and exit
  --chunk CHUNK
  --decode
  --tokenizer TOKENIZER
```

```text
$ ryth-rde verify --help
usage: ryth-rde verify [-h] data_dir

positional arguments:
  data_dir

options:
  -h, --help  show this help message and exit
```

```text
$ ryth-rde stats --help
usage: ryth-rde stats [-h] data_dir

positional arguments:
  data_dir

options:
  -h, --help  show this help message and exit
```

```text
$ ryth-rde manifest --help
usage: ryth-rde manifest [-h] [--full] data_dir

positional arguments:
  data_dir

options:
  -h, --help  show this help message and exit
  --full      include full stats block
```

## Development roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| 1 | Scratch Tokenizer | ✅ Done (v0.1.0) |
| 2 | Ryth Data Engine (RDE) | ✅ Done (v0.1.0) |
| 3 | Model Core (transformer) | ✅ Done (v0.2.0) |
| 4 | Training Engine | ✅ Done (v0.3.0) |
| 5 | 30M Prototype | 🔜 Next |
| 6 | 300M / 1B | ⏳ Planned |

See [ROADMAP.md](ROADMAP.md).

## License

[MIT](LICENSE) © 2026 RAJ-af and the Ryth contributors.
