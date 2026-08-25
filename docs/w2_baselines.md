# W2 / M2 — Eval baselines (recorded 2026-08-25)

Setup: random-weight `ryth_30m` (24.3M params, seed 1234), near-byte BPE
vocab 260, seq_len 256, CPU. Limit 40 problems/task, n_samples=1,
max_new_tokens=32. Random weights ⇒ scores 0 expected — ye RUN proof hai,
quality proof nahi.

| Benchmark | Metric | Value | File |
|---|---|---|---|
| HumanEval (40) | pass@1 | 0.0 | results/w2_humaneval_baseline.json |
| MBPP (40) | pass@1 | 0.0 | results/w2_mbpp_baseline.json |
| Held-out Python ppl | perplexity | 212.51 | results/w2_ppl_baseline.json |

Sweep command (deterministic — same script/seed => same numbers):

```bash
python3 scripts/w2_baselines.py --results results --bench bench --limit 40
```

Compare karne ke liye: `ryth-eval report results`.

## Post-training rerun (30M checkpoint, FULL settings)

```bash
# w1-prep artifacts ke saath (Kaggle ya local):
ryth-eval humaneval --ckpt runs/ryth-kaggle/best.pt \
  --tokenizer tok/tokenizer.json --problems_file bench/humaneval.jsonl.gz \
  --n_samples 20 --max_new_tokens 256 --ks 1,5,10
ryth-eval mbpp --ckpt runs/ryth-kaggle/best.pt \
  --tokenizer tok/tokenizer.json --problems_file bench/mbpp.jsonl \
  --n_samples 20 --max_new_tokens 256 --ks 1,5,10
ryth-eval ppl --ckpt runs/ryth-kaggle/best.pt \
  --tokenizer tok/tokenizer.json \
  --files python=val_src_py.txt --files c=val_src_c.txt
ryth-eval report results --out results/table.md
```

Expectations (capability ladder memory): 30M base model pass@1 ~0–2%,
ppl clearly ln(24576)≈10.1 se neeche. Honest framing — ye prototype hai,
production-grade coding assistant nahi.

## Provenance

- HumanEval MIT (openai/human-eval), MBPP CC-BY-4.0 (HF datasets-server
  google-research-datasets/mbpp, full config: 964 rows) — bench/README.md.
- val_python.txt HumanEval content se bana tha (104,214 chars, provisional);
  W1 corpus ke aane par val_src se replace hoga. ⚠ pass@k generated code
  LOCALLY execute karta hai — trusted machine par hi chalao (docs/evals.md).
