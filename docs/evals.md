# Evals — measuring Ryth checkpoints

Offline-first quality harness: pass@k (HumanEval/MBPP-style) + held-out
perplexity. Runs on CPU (Termux-friendly).

## Quickstart

```bash
pip install -e ".[dev]"
ryth-eval ppl  --ckpt runs/x/best.pt --tokenizer tok/tokenizer.json \
              --files python=val_py.txt --files c=val_c.txt
ryth-eval humaneval --ckpt best.pt --tokenizer tok/tokenizer.json \
              --problems_file humaneval.jsonl --n_samples 20
ryth-eval mbpp     --ckpt best.pt --tokenizer tok/tokenizer.json \
              --problems_file mbpp.jsonl --n_samples 20
```

Results land in `results/*.json` — track them across runs (spec §5).

## Getting real benchmark files

```bash
python -c "from evals.datasets import download_humaneval; download_humaneval('bench')"
python -c "from evals.datasets import download_mbpp; download_mbpp('bench')"
```

HumanEval is MIT, MBPP is CC-BY-4.0. A random-weight model scores ~0% —
that is the expected baseline (spec acceptance criterion).

## Security note

The pass@k harness EXECUTES generated code locally in a subprocess with a
timeout (no network jail). Only run it on machines you trust; prefer a
container in CI.
