# W1 Runbook — 30M pretraining on Kaggle (owner clicks only)

Spec §4 ka production flow: real C+Python corpus (≥600M tokens) → 24k scratch
BPE → RDS@1024 → `ryth_30m` fp16 T4 training with auto-resume. Saara heavy
compute Kaggle par; ye doc exact click-path deta hai.

## 0. One-time setup

- Kaggle account → Settings → **Internet + GPU (T4)** verify notebooks ke liye enabled.
- Repo GitHub par ho (public, ya Kaggle me import-able).
- Local sanity (optional): `python3 -m pytest tests/ -q` — sab green hon.

## 1. CPU prep session (~4–8h, free quota)

1. Kaggle → New Notebook → File → Import → `notebooks/ryth_kaggle_train.ipynb`.
2. Settings: **Accelerator = None**, **Internet = On**.
3. Section A run karo (cells A1, A2). Har stage idempotent hai — session beech
   me mare to bas Run All dobara; `_DONE` markers complete stages skip karte
   hain (`corpus_out/stage/<source>/_DONE`, `rds_w1/part_*/_DONE`).
4. A1 pehle **probe** chalata hai (columns + license histogram) — isse license
   policy confirm hoti hai (allowlist ya unknown-keep; neeche Troubleshooting).

## 2. Package prep outputs

1. Notebook right panel → **Output** → "Save Version" → private dataset banao,
   naam **`w1-prep`** (outputs: `corpus_out/`, `tok/`, `rds_w1/`, `val_src/`).
2. Expected sizes: corpus_out ~2.4 GB text, rds_w1 ~2–3 GB shards, val_src tiny.

## 3. GPU train session (~2–4h GPU quota)

1. Same notebook → Settings: **Accelerator = GPU T4**, Internet On.
2. Right panel → Input → attach dataset `w1-prep` (path `/kaggle/input/w1-prep`).
3. Section B run karo (B1, B2). B1 prepared tok/RDS ko cache paths par copy
   karta hai — tokenizer/RDS rebuild SKIP hota hai, GPU sirf train karta hai.
4. **Resume:** 12h session limit ya disconnect aaye → naya session, wahi steps
   dobara. `runs/ryth-kaggle/latest.pt` se auto-resume hota hai (same WORK path:
   `/kaggle/working/run`). NOTE: `/kaggle/working` outputs bhi Save Version me
   persist karo warna checkpoints kho jayenge.
5. Throughput line `[batch] effective tokens/step = 262,144` + tokens/sec log —
   ye deliverable hai (loss curve PNG/log ke saath).

## 4. Acceptance (M1)

- [ ] val loss untrained baseline (~ln 24576 ≈ 10.1) se clearly neeche (**< 3.0**)
- [ ] Python samples mostly syntactically valid; C structurally sane (cell B2)
- [ ] `rds_w1/final/manifest.json` + `tokenizer.json.meta.json` preserved
- [ ] Eval recorded: `ryth-eval ppl --ckpt best.pt --tokenizer tok/tokenizer.json ...`
      numbers `results/` me (docs/evals.md quickstart)

## Budget notes (spec §11)

- Weekly GPU quota ~30h; 30M run ≈ 2–4h → multiple resumes easily fit.
- Corpus target ≥600M tokens (~2.4 GB chars). Agar `_SUMMARY.json` SHORT bole:
  probe output dekho kaunsa subset short pada, `--total-gb` badhao, ya
  `configs/w1_sources.json` me curated GitHub entries add karo (kind=github,
  downloader pehle se wired hai — zero code change).

## Troubleshooting

- **`load_dataset` script-dataset reject ho** (`Dataset scripts are no longer
  supported` — newer `datasets` versions): The Stack-dedup legacy loading-script
  format hai. Fallback: parquet mirror —
  `load_dataset("bigcode/the-stack-dedup", revision="refs/convert/parquet", data_dir="data/python", streaming=True)`
  ya `bigcode/starcoderdata` (native parquet, `content`+`license` columns,
  streaming works). Probe (§1 step 4) pehle hi ye bata degi. Downloader me
  location/revision change config-level fix hai.
- **Session beech me mari**: wahi section dobara Run karo — `_DONE` markers +
  `latest.pt` resume sab handle karte hain (same `/kaggle/working` path).
- **fp16 NaN loss T4 par**: `--dtype bf16` try karo (T4 support karta hai), ya
  `--grad_ckpt` ke saath fp32 fallback — speed thodi giregi, stability milegi.
- **Tokenizer ETA bahut lambi** (>3h): `--sample-mb 30` kar do (quality ka
  negligible effect @24k vocab) — probe line batayegi naya ETA.
- **License column mila hi nahi** (probe me `unknown` dominant): policy
  fallback = The Stack-dedup ka apna dedup+opt-out filtering documented rakho,
  manifest me `license_policy: "stack-dedup-default"` note karo. Owner decision
  point — spec §7 multi-source rule ke under honest disclosure.
