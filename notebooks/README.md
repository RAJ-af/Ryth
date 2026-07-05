# Notebooks

End-to-end, runnable notebooks for Ryth.

## `ryth_kaggle_train.ipynb` — train the 30M model on Kaggle (T4)

A complete, self-contained walkthrough that runs the **whole stack** using only
the public library (no core changes):

```
corpus → tokenizer → RDS dataset → 30M model → training
       → checkpoints → resume → code generation
```

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

The same pipeline as a single command (see [`../scripts/kaggle_train.py`](../scripts/kaggle_train.py)):

```bash
python scripts/kaggle_train.py --smoke                 # synthetic smoke test
python scripts/kaggle_train.py --raw /path/to/code \
    --steps 5000 --seq_len 512 --vocab 16000           # real run
```

Both paths use the same public APIs documented in
[`../docs/training.md`](../docs/training.md).
