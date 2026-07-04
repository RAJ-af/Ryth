<div align="center">

# Ryth

**A coding-first LLM, built from scratch.**

*Foundation Release (v0.1.0): a scratch byte-level BPE tokenizer + the Ryth Data Engine.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](ROADMAP.md)

</div>

---

## Introduction

**Ryth** is an effort to build a coding-focused large language model **entirely
from scratch** — no black-box frameworks for the core pieces. Every foundational
component is written in clean, readable Python so it can be understood, audited,
and extended.

This **Foundation Release (v0.1.0)** ships the first two pillars:

1. **A scratch Byte-Level BPE Tokenizer** — pure Python, no external libraries.
2. **The Ryth Data Engine (RDE)** — a complete data pipeline that turns raw code
   repositories into a compact, reproducible, memory-mapped binary format (RDS)
   ready for training.

> The model and training engine are on the [roadmap](ROADMAP.md) (Phase 3+) and
> are **not** part of this release.

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

   Scratch BPE tokenizer plugs in at the "Encoder" stage.
```

## Installation

Requires **Python 3.9+**. Core install has **no dependencies**.

```bash
git clone https://github.com/RAJ-af/Ryth.git
cd Ryth
pip install -e .            # core (pure standard library)
# optional extras:
pip install -e ".[fast]"    # + xxhash (faster dedup)
pip install -e ".[dev]"     # + pytest (run tests)
```

This installs two command-line tools: **`ryth-tokenizer`** and **`ryth-rde`**.

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
├── docs/                # architecture, tokenizer, dataset engine, RDS, quickstart, faq
├── examples/            # runnable example scripts
├── tests/               # pytest suite (pure standard library)
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
| 4 | Training Engine | 🔜 Next |
| 5 | 30M Prototype | ⏳ Planned |
| 6 | 300M / 1B | ⏳ Planned |

See [ROADMAP.md](ROADMAP.md).

## License

[MIT](LICENSE) © 2026 RAJ-af and the Ryth contributors.
