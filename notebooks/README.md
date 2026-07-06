# Notebooks

End-to-end, runnable notebooks for Ryth. Both walk through the full pipeline
(`corpus → tokenizer → RDS → model → training → eval → export`) using only the
public library — no core code is modified.

## `ryth_end_to_end.ipynb` — **official** end-to-end entry point

The **canonical** "Run All" notebook. One file takes you from raw code (or a
synthetic corpus) to a trained model, eval results, and sample generations.

```
Corpus Engine ─► Tokenizer ─► RDE (RDS) ─► Model ─► Training ─► Eval ─► Export
```

- **Idempotent**: every stage detects existing outputs and **skips** completed
  work; `final.pt` exists ⇒ training is skipped, and `latest.pt` resumes a
  previous run.
- **Single configuration cell** — flip `SMOKE = False` for a real run; switch
  `MODEL_PRESET` to scale `ryth_30m → ryth_125m → ryth_350m → ryth_1b` (only
  the config changes).
- **GPU-aware**: auto-picks `fp16` on Kaggle T4 (Turing, no native bf16) and
  `bf16` on Ampere+. Override with `FORCE_DTYPE`.

### Run it on Kaggle (best place to run a real job)

1. **New Notebook** → upload / copy `ryth_end_to_end.ipynb`.
2. **Settings → Accelerator → GPU T4** (×1 or ×2). Detected automatically.
3. Get Ryth into the kernel (either is fine):
   - **Settings → Internet → On** — the env-setup cell installs it from GitHub, **or**
   - Attach the Ryth repo as a **Kaggle dataset** (Add Data) — the same cell
     finds it under `/kaggle/input/…`.
4. **Run All.** Defaults are a fast, synthetic **smoke test**.

### Make it a real run

In the **Configuration** cell:

- `SMOKE = False`
- `RAW_DIR = "/kaggle/input/<your-code-dataset>"` — a folder of
  **permissively-licensed** code (MIT/Apache/BSD).
- Raise `VOCAB_SIZE` (16k–32k), `SEQ_LEN` (512–1024), `MAX_STEPS`
  (tens of thousands).
- Kaggle sessions are time-limited (12 h); checkpoints persist in
  `/kaggle/working`, so restart the session and re-run to **resume** from
  `latest.pt`.

Equally happy on Colab, a local box, or a cluster — same notebook, same
library, same outputs.

## `ryth_kaggle_train.ipynb` — Kaggle-specific 30M training variant

A leaner, Kaggle-only sibling focused on training the 30M model. Same pipeline,
fewer surface cells (older version kept for reference).

### Run it on Kaggle
1. **New Notebook** → upload / copy `ryth_kaggle_train.ipynb`.
2. **Settings → Accelerator → GPU T4** (×1 or ×2). Auto-detected.
3. Get Ryth into the kernel (either one):
   - **Settings → Internet → On** — the first cell `pip install`s it from GitHub, **or**
   - Attach the Ryth repo as a **Kaggle dataset** (Add Data) — the first cell finds it
     under `/kaggle/input/…` automatically.
4. **Run All**. Defaults are a fast, synthetic **smoke test** that exercises every
   stage in ~1–2 minutes on a T4.

### Make it a real run
In the **Configuration** cell:
- `SMOKE = False`
- `RAW_DIR = "/kaggle/input/<your-code-dataset>"` — a folder of
  **permissively-licensed** code (MIT/Apache/BSD).
- Raise `VOCAB` (16k–32k), `SEQ_LEN` (512–1024), `STEPS` (tens of thousands).
- Kaggle sessions are time-limited — checkpoints persist in `/kaggle/working`, so
  restart the session and re-run with `resume="latest"` to continue.

### Precision on T4
T4 is a Turing GPU (sm_75) with **no native bf16**, so the notebook selects
**fp16** automatically. Ampere+ GPUs (A100/L4) get **bf16**. Override with
`FORCE_DTYPE`.

## Equivalent script

For the Kaggle variant, the same pipeline as a single command (see [`../scripts/kaggle_train.py`](../scripts/kaggle_train.py)):

```bash
python scripts/kaggle_train.py --smoke                 # synthetic smoke test
python scripts/kaggle_train.py --raw /path/to/code \
    --steps 5000 --seq_len 512 --vocab 16000           # real run
```

Both paths use the same public APIs documented in
[`../docs/training.md`](../docs/training.md).
