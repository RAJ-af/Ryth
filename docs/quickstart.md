# Quick Start

A step-by-step walkthrough from clone to a verified RDS dataset. Every command is
documented. Requires **Python 3.9+**.

## 1. Clone

```bash
git clone https://github.com/RAJ-af/Ryth.git
cd Ryth
```

## 2. Install

Core install has **no dependencies** (pure standard library):

```bash
pip install -e .            # core
pip install -e ".[dev]"     # + pytest (to run the test suite)
pip install -e ".[fast]"    # + xxhash (faster chunk dedup, optional)
```

This adds two CLI tools to your environment: `ryth-tokenizer` and `ryth-rde`.

Verify the install:

```bash
ryth-tokenizer --help
ryth-rde --help
```

## 3. Get some code to train on

RDE accepts **either** input layout:

```
raw_repos/<repo_name>/<files...>     # multi-repo: each top-level folder is a repo
raw_repos/<files...>                 # flat: raw_repos itself is one repo
```

```bash
mkdir -p raw_repos
# Add permissively-licensed (MIT/Apache/BSD) Python code, e.g.:
git clone --depth 1 https://github.com/<owner>/<permissive-repo> raw_repos/<name>
# ...or just drop .py files directly into raw_repos/ (flat layout).
```

> If no usable files are discovered, `ryth-rde build` **exits with a descriptive
> error** (it never writes an empty dataset). Use `--debug` to print every
> pipeline stage.

> **Only use permissively-licensed code** for training data — it is your
> responsibility to respect licenses.

Just exploring? Generate the built-in sample fixture:

```bash
python -c "from tests.sample_data import build_sample; build_sample('raw_repos')"
```

## 4. Train a tokenizer

Train the scratch BPE tokenizer on your code files:

```bash
ryth-tokenizer train --files 'raw_repos/**/*.py' --vocab 8000 --out tok
```

- `--files` — glob of files to train on (or `--data <jsonl_dir>` for JSONL corpora)
- `--vocab` — target vocabulary size (must be ≥ 256)
- `--out`   — output folder; writes `tok/tokenizer.json`

## 5. Encode the dataset (build RDS)

Run the full RDE pipeline to produce memory-mapped RDS shards:

```bash
ryth-rde build raw_repos rds_out \
    --tokenizer tok/tokenizer.json \
    --seq_len 1024 \
    --shard_mb 256
```

- `raw_repos` — input root, `rds_out` — output folder
- `--tokenizer` — your trained tokenizer (omit to use the byte-level fallback)
- `--seq_len` — chunk length in tokens
- `--shard_mb` — shard size cap in MB (rotate to a new shard when exceeded)

Outputs: `rds_out/shard_*.rds`, `manifest.json`, `report.json`, `report.html`.

## 6. Inspect the RDS dataset

```bash
ryth-rde inspect rds_out                       # summary
ryth-rde inspect rds_out --chunk 0 \
    --decode --tokenizer tok/tokenizer.json     # decode one chunk to text
```

## 7. Verify the dataset

Check every shard's sha256 checksum:

```bash
ryth-rde verify rds_out
# -> checksum verify: ALL OK (N shards)
```

Print statistics (packing efficiency, languages, difficulty split, …):

```bash
ryth-rde stats    rds_out
ryth-rde manifest rds_out          # manifest + reproducibility lock
```

Open `rds_out/report.html` in a browser for the full validation report.

## 8. Run the examples / tests

```bash
python examples/example_train_tokenizer.py
python examples/example_encode_dataset.py
python examples/example_inspect_dataset.py
python examples/example_verify_dataset.py
python examples/example_decode_tokens.py

pytest -q          # or: python tests/test_rde.py
```

## Using it from Python

```python
from tokenizer import BPETokenizer, DEFAULT_SPECIAL_TOKENS
from dataset import RDEConfig, RDEPipeline, RDSDataset
from dataset.tokenizer_adapter import load_bpe_tokenizer

tok = load_bpe_tokenizer("tok/tokenizer.json")
cfg = RDEConfig(seq_len=1024, vocab_size=tok.vocab_size)
manifest = RDEPipeline(tok, cfg).run("raw_repos", "rds_out")

ds = RDSDataset("rds_out")
print(len(ds), "chunks; checksums:", ds.verify())
```

Next: read [dataset_engine.md](dataset_engine.md) and [tokenizer.md](tokenizer.md).
