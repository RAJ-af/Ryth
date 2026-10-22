# sft — Supervised Fine-Tuning Pipeline & Tools (phase-6 §6)

Complete SFT pipeline: synthetic instruction data generation, chat dataset RDS building with parallel loss masking, SFT model training, interactive CLI chat interface, and stdlib data generator.

## Overview

1. **`scripts/gen_sft_data.py`** — Stdlib-only tool (`urllib`, `json`) to generate chat training data via Gemini or OpenAI-compatible APIs using `data/sft/topics_starter.txt`.
2. **`ryth-sft generate`** — Seed-based generation pipeline using teacher models.
3. **`ryth-sft build <data_dir> <out_dir> --tokenizer tok.json --seq_len 1024 --val_frac 0.05`** — Converts JSONL chat pairs into packed RDS shards with parallel loss masks (loss calculated strictly on assistant tokens and `<|end|>` end-of-turn). Reports counts per file name (`en`, `hi`, `hinglish`).
4. **`ryth-sft train`** — Fine-tunes base model weights (`{"model": state_dict, "preset": str, "vocab_size": int, "seq_len": int}`) with loss masking, lr 5e-5, 3 epochs, 5% warmup, cosine decay, val loss reported on assistant tokens only, and saves `best.pt`.
5. **`scripts/chat.py --weights W --tokenizer T`** — Interactive chat CLI with `--temperature`, `--top_k`, `--max_new_tokens`, `--lang_hint`, and auto device selection.

## 1. SFT Data Generation (`scripts/gen_sft_data.py`)

Generate multilingual SFT dataset from starter topics (`data/sft/topics_starter.txt`):

```bash
# Set provider credentials
export PROVIDER=openai_compat  # or gemini
export API_KEY=sk-...
export MODEL=google/gemini-2.0-flash-001

# Generate English SFT data
python3 scripts/gen_sft_data.py --lang en --target 100 --out data/sft/en.jsonl

# Generate Hindi (Devanagari) SFT data
python3 scripts/gen_sft_data.py --lang hi --target 100 --out data/sft/hi.jsonl

# Generate Hinglish (Latin alphabet) SFT data
python3 scripts/gen_sft_data.py --lang hinglish --target 100 --out data/sft/hinglish.jsonl
```

## 2. Dataset Building & Loss Masking (`ryth-sft build`)

Pack JSONL chat files into RDS shards with parallel loss masks:

```bash
ryth-sft build data/sft/ sft_rds/ --tokenizer tok/tokenizer.json --seq_len 1024 --val_frac 0.05
```

Reports counts per file name:
```
[build] en.jsonl: read=100 valid=100 malformed=0 tokens=15000
[build] hi.jsonl: read=100 valid=100 malformed=0 tokens=14000
[build] hinglish.jsonl: read=100 valid=100 malformed=0 tokens=14500
[build] Complete: train_chunks=42 val_chunks=3 -> sft_rds/
```

- **Loss Masking Rule**: Loss mask is 1 ONLY on assistant content tokens and assistant end-of-turn (`<|end|>`). User/system tokens and `<|assistant|>` sentinel are masked with 0 (`ignore_index = -100`).

## 3. SFT Training (`ryth-sft train`)

Fine-tune pretrained base weights on SFT RDS dataset:

```bash
ryth-sft train \
  --data_dir sft_rds/ \
  --weights pretrained_30m.pt \
  --out_dir runs/sft_30m/ \
  --epochs 3 \
  --lr 5e-5 \
  --micro_batch_size 4 \
  --grad_accum_steps 2 \
  --dtype fp32 \
  --device auto
```

Saves `runs/sft_30m/best.pt` containing `{"model": state_dict, "preset": preset, "vocab_size": vocab_size, "seq_len": seq_len, "best_val_loss": loss}`.

## 4. Interactive Chat (`scripts/chat.py`)

Start interactive chat loop:

```bash
python3 scripts/chat.py --weights runs/sft_30m/best.pt --tokenizer tok/tokenizer.json --lang_hint hi
```

Flags:
- `--weights` / `-w`: Path to fine-tuned model checkpoint (`.pt`)
- `--tokenizer` / `-t`: Path to tokenizer JSON (`tokenizer.json`)
- `--temperature`: Sampling temperature (default 0.7)
- `--top_k`: Top-k sampling threshold (default 50)
- `--max_new_tokens`: Maximum new tokens to generate (default 512)
- `--lang_hint`: Language directive (`en`, `hi`, `hinglish`)
- `--device`: Target device (`auto`, `cuda`, `cpu`)

## Status (M3)

- ✅ Pipeline code + offline dry-run (`--dry-run`)
- ⛔ Real generation GATED: owner se `RYTH_TEACHER_API_KEY` +
  `RYTH_TEACHER_MODEL` chahiye (proxy base-url bhi tab)

## Offline smoke (aaj chalega)

```bash
python3 -m sft.cli generate --src <koi-chhota-.py-dir> \
    --out /tmp/sft_smoke.jsonl --dry-run
```

## Real run (key aane par)

```bash
export RYTH_TEACHER_API_KEY=sk-...        # owner proxy key
export RYTH_TEACHER_MODEL=<backend-name>  # Nemotron-class
ryth-sft generate --src corpus_out --target 5000 \
  --out data/sft_v1.jsonl --tokenizer tok/tokenizer.json \
  --base-url https://<proxy-endpoint>/v1   # agar direct OpenRouter nahi
```

Acceptance (spec §6): ~5–10k examples, pass_rate ≥ 0.90 (stats JSON me;
CLI warn karta hai agar neeche), owner spot-check sample.

Run behavior jo API-spend/resilience ke liye matter karta hai:

- `--target N` = jitne examples **PASS** hote hi teacher calls band
  (spend cap — poora seed-pool jalata nahi).
- Har PASS row turant flush hoti hai + `<out>.stats.json` sidecar —
  mid-run crash par bhi likha hua data bacha rehta hai (sidecar me
  `"crashed": true` + `n_written`).
- Missing key/model jaisi SETUP galti clean SystemExit hai (traceback
  nahi); per-seed transient teacher errors soft hain — reason stats ke
  `filter_reasons.teacher_error` me ginta hai.
- Dedup case-preserving hai (`Add` ≠ `add`) — sirf whitespace
  normalize hota hai.

## Output format

Har row: `task`, `messages` (persona-system/user/assistant), `text`
(W2 chat-template rendered), optional `token_ids` (sirf --tokenizer ke
saath — provisional tokenizer bake karne se bacho), `meta`. Trainer
`text`/`token_ids` seedha consume karega.

Teacher meta-directives stored turns me leak NAHI hote (tested) — persona
`SYSTEM_PROMPT` hi dataset me jaata hai.

## Security

⚠ `test_gen` validator generated asserts LOCAL subprocess me execute karta
hai (timeout 10s, no network jail) — trusted machine par hi chalao.
