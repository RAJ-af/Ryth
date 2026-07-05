# Ryth Corpus v1.0

A **corpus engineering system** that turns raw code repositories into a
world-class, license-clean, deduplicated, quality-scored, task-formatted training
dataset for Ryth models (30M → 1B). It is a **separate package** (`corpus/`) — it
does not modify the tokenizer, RDE, model core, or training engine.

Pure Python standard library for the core. Optional extras: `pyarrow` (Parquet),
`datasets` (Hugging Face sources).

## Pipeline

```
sources ─► download ─► clean ─► language ─► license filter ─► exact dedup
        ─► near dedup ─► repo dedup ─► quality score/threshold ─► split
        ─► (optional) language-ratio balance ─► metadata + tasks + export + report
```

Every stage annotates a `FileRecord`; dropped files carry a `drop_reason`, so the
final report accounts for exactly what was removed and why.

## Modules

| Package | Role |
|---------|------|
| `corpus/config.py` | `CorpusConfig` — one config for the whole build |
| `corpus/sources/` | declarative source registry (github / hf / http / local) |
| `corpus/download/` | downloaders → local staging (`local` is offline) |
| `corpus/licenses/` | SPDX detection + permissive/copyleft policy |
| `corpus/languages.py` | language detection (priority coding languages) |
| `corpus/cleaners/` | vendor/binary/minified/lockfile/generated/notebook/secret cleaning |
| `corpus/filters/` | license/size/language filters + ratio balancing |
| `corpus/dedup/` | exact (sha256) + near-dup (MinHash/LSH), file + repo |
| `corpus/quality/` | per-repo signals + 0–100 score |
| `corpus/split/` | deterministic, leakage-free repo-level train/val/test |
| `corpus/metadata/` | `FileRecord` / `RepoRecord` + JSONL store |
| `corpus/tasks/` | task builders (FIM, bug-fix, docstring→code, …) + mixer |
| `corpus/exporters/` | raw / JSONL / Parquet / **RDS (via RDE, unchanged)** |
| `corpus/report.py` | stats + HTML/JSON reports |
| `corpus/pipeline.py` | `CorpusPipeline` orchestrator |
| `corpus/cli.py` | `ryth-corpus` |

## Sources & licenses

Sources are declarative (`Source`): `github` (public zip via `urllib`, no token),
`huggingface` (needs `datasets`), `http` (docs/markdown), `local` (a folder on
disk — always offline). Only **permissive** licenses are accepted by default
(MIT, Apache-2.0, BSD, ISC, MPL-2.0, Unlicense, 0BSD). GPL/unknown are rejected
unless `allow_copyleft` / `allow_unknown_licenses` are set. The license hint on a
source is never trusted blindly — it is re-detected from the downloaded
`LICENSE`/SPDX headers.

## Cleaning

Automatically removes: vendor/build/cache/venv folders, binaries, minified
bundles, lock files, generated code, corrupted (non-UTF-8) files, oversized files,
and Jupyter **notebook outputs** (code kept). **Secrets/API keys** (AWS keys,
tokens, private-key blocks, `api_key = "…"`) are detected and **redacted** (or the
file dropped, if configured).

## Quality scoring

Each repository gets a 0–100 score from weighted signals: **syntax validity,
documentation, tests, project structure, comments, complexity, maintainability,
duplicate ratio**. `min_quality` drops low-scoring repos. Weights are configurable.

## Deterministic splits

Splits are assigned at the **repository** level from `hash(seed:repo)`, so every
file of a repo lands in the same split — **no leakage** between train / validation
/ test. `verify_no_leakage()` asserts this.

## Training tasks

`build_task_dataset()` produces configurable-ratio task examples: **next-token,
FIM, completion, editing, docstring→code, README→code, code→explanation,
bug-fixing, refactoring, unit-test generation**. FIM sentinels match the
tokenizer's special tokens. Ratios are enforced deterministically.

## Export

- **raw** — `<out>/<split>/<repo>/<path>` (feeds the tokenizer & RDE directly)
- **jsonl** — records (metadata + content), split-wise; and task examples
- **parquet** — columnar (needs `pyarrow`)
- **rds** — Ryth RDS shards, one dataset per split, **via the existing RDE**
  (the RDE implementation is not modified)

## CLI

```bash
ryth-corpus download --input raw_repos --stage stage        # or --source-file sources.json
ryth-corpus clean    --input raw_repos --out clean_out
ryth-corpus score    --input raw_repos
ryth-corpus build    --input raw_repos --out corpus_out --tasks --min-quality 40
ryth-corpus stats    --records corpus_out/records.jsonl --html corpus_out/report.html
ryth-corpus export   --records corpus_out/records.jsonl --format rds --out rds_out --seq-len 1024
```

`build` writes `records.jsonl`, `repos.jsonl`, `report.json`, `report.html`
(and `tasks.jsonl` with `--tasks`). Each top-level subdirectory of `--input` is
treated as a repository.

## Python API

```python
from corpus import CorpusConfig, CorpusPipeline, Source, SourceList, build_task_dataset
from corpus.exporters import export_rds

cfg = CorpusConfig(min_quality=40, near_dedup=True,
                   language_ratios={"python": 0.5, "rust": 0.5},
                   task_ratios={"next_token": 0.6, "fim": 0.2, "bug_fixing": 0.2})
sources = SourceList([Source("local:my", "local", "raw_repos/my_project")])

pipe = CorpusPipeline(cfg)
result = pipe.build(sources.enabled(), stage_dir="stage",
                    created_at="2026-07-05T00:00:00Z")   # timestamps passed in (deterministic)

examples = build_task_dataset(result.records, cfg)
export_rds(result.records, "rds_out", seq_len=1024)      # uses the existing RDE
```

## Reports

`report.json` / `report.html` include: language distribution, license
distribution, duplicate statistics, quality histogram, repository rankings,
dataset size, and task distribution.

## Reproducibility

The library reads **no wall-clock and no RNG** — all selection (dedup, split,
balancing, task sampling) is derived from content hashes and the config `seed`.
Timestamps are passed in by the caller. Same inputs + config ⇒ same corpus.

## Tests

```bash
pytest tests/test_corpus.py -q      # 33 tests (offline), + 1 parquet test (skipped without pyarrow)
```
