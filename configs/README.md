# Configs

These JSON files are **reference configurations** — they mirror the fields and
defaults of `dataset.config.RDEConfig`. RDE does not auto-load them; they exist so
you can see all tunable values in one place and apply a known-good set explicitly.

## Files

| File | Purpose |
|------|---------|
| `rde_default.json`   | Mirrors the built-in `RDEConfig` defaults |
| `rde_prototype.json` | Small/prototype settings for quick experiments |

## Applying a config in Python

```python
import json
from dataset import RDEConfig, RDEPipeline
from dataset.tokenizer_adapter import load_bpe_tokenizer

with open("configs/rde_prototype.json") as f:
    params = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

tok = load_bpe_tokenizer("tok/tokenizer.json")
params["vocab_size"] = tok.vocab_size          # keep vocab in sync with the tokenizer
cfg = RDEConfig(**params)

RDEPipeline(tok, cfg).run("raw_repos", "rds_out")
```

## Field reference

See `dataset/config.py` for the authoritative list and inline documentation of
every field.
