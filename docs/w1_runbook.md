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
3. **Run All** — bas. Notebook Run-All-safe hai: A1 prep karta hai, B1 wahan
   clean-exit dega ("w1-prep attach karo") — CPU session me wo EXPECTED hai,
   walkthrough sections khud skip hote hain (`WALKTHROUGH=False`).
   Har stage idempotent hai — session beech me mare to bas Run All dobara;
   `_DONE` markers complete stages skip karte hain
   (`corpus_out/stage/<source>/_DONE`, `rds_w1/part_*/_DONE`).
4. A1 pehle **probe** chalata hai (columns + license histogram) — isse license
   policy confirm hoti hai (allowlist ya unknown-keep; neeche Troubleshooting).

## 2. Package prep outputs

1. Notebook right panel → **Output** → "Save Version" → private dataset banao,
   naam **`w1-prep`** (outputs: `corpus_out/`, `tok/`, `rds_w1/`, `val_src/`).
   NOTE: Section A sab kuch `/kaggle/working/prep` me likhta hai — wahi
   folder Save Version me persist hota hai (`/kaggle/work` scratch hai,
   use me kabhi mat likhna).
2. Expected sizes: corpus_out ~2.4 GB text, rds_w1 ~2–3 GB shards, val_src tiny.

## 3. GPU train session (~2–4h GPU quota)

1. Same notebook → Settings: **Accelerator = GPU T4**, Internet On.
2. Right panel → Input → attach dataset `w1-prep` (path `/kaggle/input/w1-prep`).
3. **Run All** — bas. A1 attached `w1-prep` dekh ke khud SKIP hota hai (poora
   prep dobara NAHI hota); B1 tok/RDS ko cache paths par copy karke train
   karta hai, B2 samples deta hai. Walkthrough sections phir bhi skip.
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

- **GATED dataset error** (`401 authentication` / `accept gate at the hub`):
  `bigcode/the-stack-dedup` AUR `bigcode/starcoderdata` dono `gated: auto`
  hain — bina HF token nahi khulenge. Isi liye primary source ab
  **`codeparrot/github-code` @ `refs/convert/parquet`** hai (UNGATED;
  2026-08-25 ko anonymous streaming VERIFY kiya: Python-all/C-all dirs,
  columns `code/language/license/path/repo_name/size`). Probe + downloader
  dono isi config se chalte hain; agar kabhi primary gate ho jaye to config
  entry me `"fallbacks": [{"location": ..., "subpath": ..., "revision": ...}]`
  dal do — downloader khud agla source try karega.
- **`load_dataset` script-dataset reject** (`Dataset scripts are no longer
  supported`): github-code legacy script-format hai, isliye hum `ref:
  refs/convert/parquet` (auto-converted parquet branch) use karte hain —
  naye `datasets` versions ke saath compatible.
- **Session beech me mari**: wahi section dobara Run karo — `_DONE` markers +
  `latest.pt` resume sab handle karte hain (same `/kaggle/working` path).
- **fp16 NaN loss T4 par**: `--grad_ckpt` ke saath fp32 fallback karo
  (T4/sm_75 me native bf16 NAHI hai — bf16 flag error ya bahut slow emulation
  dega). Ampere+ accelerator mila ho tabhi bf16 try karo. Speed giregi,
  stability milegi.
- **Tokenizer ETA bahut lambi** (>3h): `--sample-mb 30` kar do (quality ka
  negligible effect @24k vocab) — probe line batayegi naya ETA.
- **License column mila hi nahi** (probe me `unknown` dominant): policy
  fallback = The Stack-dedup ka apna dedup+opt-out filtering documented rakho,
  manifest me `license_policy: "stack-dedup-default"` note karo. Owner decision
  point — spec §7 multi-source rule ke under honest disclosure.
