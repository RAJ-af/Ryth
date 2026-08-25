# bench/ — real benchmark files (committed for reproducible offline runs)

| File | Source | License |
|---|---|---|
| `humaneval.jsonl.gz` | openai/human-eval (data/HumanEval.jsonl.gz) | MIT |
| `mbpp.jsonl` | google-research-datasets/mbpp via HF datasets-server (full config) | CC-BY-4.0 |

Fetch ke liye: `evals.datasets.download_humaneval` / `download_mbpp` (network
opt-in). Note: GitHub ka original `mbpp.jsonl` 404 ho chuka hai, isliye
downloader HF datasets-server rows API se original schema (`text`, `test_list`,
`code`) paginate karke laata hai.

Tests sirf counts verify karte hain — kabhi download nahi karte.

NOTE (M2): `bench/val_python.txt` provisional held-out set hai (HumanEval
prompts+canonical solutions se bana). Post-W1 real corpus val split aayega
(w1-prep `val_src/`) — phir wahi use hoga, ye rakhna sirf history ke liye.
