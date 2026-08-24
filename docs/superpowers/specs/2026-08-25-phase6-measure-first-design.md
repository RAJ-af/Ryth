# Ryth Phase 6 — "Measure-First" Program (Design)

**Date:** 2026-08-25
**Status:** Approved design (brainstormed with project owner)
**Scope:** Architectural — adds two new packages (`evals/`, `sft/`), a Kaggle
pretraining run, and an ongoing experiment track. No changes to existing core
pillars (tokenizer / RDE / model / training engine).

---

## 1. Goals & Non-goals

**Goals**

1. Train the first real Ryth model (30M) end-to-end on a real C+Python corpus
   on Kaggle — proving the full pipeline at production scale.
2. Establish quality *measurement* from day one: perplexity + HumanEval/MBPP
   pass@k harness, so every future model size has a score.
3. Start the SFT-data flywheel: use a large teacher model (Nemotron-class,
   via OpenRouter-compatible API) to generate high-quality instruction data.
4. Give the owner structured room to run research experiments (memory /
   reservoir-style ideas) without derailing the main ladder.

**Non-goals (this phase)**

- No 5B-scale pretraining (infeasible on Kaggle; needs dedicated hardware).
- No multi-GPU/DDP support yet.
- No RLHF/DPO. No serving/inference server.
- No new languages beyond C + Python.

## 2. Locked decisions (from brainstorming session)

| Decision | Choice | Rationale |
|---|---|---|
| First milestone | **30M prototype** (roadmap Phase 6) | Cheapest full-pipeline proof (~1 GPU-session) |
| Language scope | **C + Python**, staged curriculum | Owner's quality-first, learn-one-language-at-a-time instinct; RDE Smart Curriculum orders easy→hard |
| Data sources | **Multi-source**: HF code datasets (license-filtered) + curated GitHub repos, all through ryth-corpus cleaning/dedup | "Har jagah se high-quality" — no single point of failure |
| Program shape | **Measure-first**: W1 pretrain ∥ W2 evals ∥ W3 SFT-data | Quality matters ⇒ measurement exists before scale-up |
| Compute | Kaggle only (owner is on Termux/proot; no local GPU) | T4 fp16, resume across sessions |
| Experiments | Dedicated track after each milestone; first candidate = kNN-Memory Head | Owner's core motive is learning/experimenting |

## 3. Verification baseline (2026-08-25 audit)

Full local audit performed before this phase:

- Test suite: **135 passed / 0 failed** (was: corpus package unimportable).
- End-to-end CPU smoke (`scripts/kaggle_train.py`): corpus → tokenizer → RDS →
  30M train steps → checkpoint → generation — all green.

Bugs found and fixed during the audit:

1. `corpus/cleaners/secrets.py` was missing from git entirely — `.gitignore`'s
   `secrets.*` rule silently ignored the source module. Module rewritten
   (AWS/GitHub/Slack tokens, private-key blocks, bearer headers, credential
   assignments detected + redacted); `.gitignore` narrowed to explicit
   credential extensions.
2. `RDSReader` held open file handles → unpicklable under Python 3.14+
   multiprocessing (`forkserver` default). Fixed with lazy-reopen
   (`__getstate__`/`__setstate__`) + regression test.

These fixes are prerequisites for Phase 6 and are part of this change set.

## 4. Workstream 1 — 30M pretraining on Kaggle

**Corpus build** (Kaggle CPU session or local):
- Sources: The Stack-dedup subsets (C, Python) filtered to permissive licenses
  (MIT/Apache/BSD/ISC/MPL-2.0) + curated GitHub repos via `ryth-corpus`.
- Target: ≥ 600M clean tokens after dedup/quality filtering (~2.5 GB text).

**Tokenizer:** fresh scratch BPE, vocab ≈ 24k, trained on a stratified
C+Python sample of the corpus. Byte-level fallback keeps any script
representable.

**Data:** RDS shards, `seq_len=1024`, uint16 dtype.

**Model/training:** `ryth_30m` preset (d=512, L=8, H=8, n_kv=2; ≈32M params
at 16k vocab, ≈36M at 24k — verified via `estimate_params`), fp16 on T4,
AdamW lr ≈ 6e-4 with warmup+cosine,
effective batch ≈ 260k tokens/step, ~2–4 h GPU total. Auto-resume across
sessions via `latest.pt`.

**Deliverables:** `best.pt`/`final.pt`, validation-loss curve, throughput
report, sample generations (C + Python prompts), dataset manifest lock.

**Acceptance:** val loss clearly below untrained baseline; generated snippets
are syntactically valid Python/C most of the time; reproducible manifest.

## 5. Workstream 2 — `evals/` package (new)

Pure-Python + PyTorch, CPU-friendly (runs on owner's Termux box).

| Unit | Purpose |
|---|---|
| `evals/chat_template.py` | Defines the Ryth chat format using the tokenizer's existing special-token system (`<|user|>`, `<|assistant|>`, `<|end|>` sentinels); encode/decode helpers |
| `evals/humaneval.py` | HumanEval pass@k runner: load checkpoint → sampled generation with stop conditions → sandboxed exec → k∈{1,5,10} |
| `evals/mbpp.py` | MBPP runner, same harness |
| `evals/ppl.py` | Held-out perplexity, reportable per language (C vs Python) |
| `evals/cli.py` | `ryth-eval --ckpt best.pt --task humaneval --n_samples 20` |

Design rules: reuse `model.generate`; no core modifications; every task
returns a JSON result file so scores are tracked across runs.

**Acceptance:** harness runs a random-weight 30M model end-to-end on CPU
(score will be ~0% — that is expected and still valuable as the baseline);
results JSON written.

## 6. Workstream 3 — `sft/` package (teacher-generated instruction data)

New package producing chat-format SFT corpora using a Nemotron-class teacher.

| Unit | Purpose |
|---|---|
| `sft/teacher.py` | OpenRouter-compatible API client; configurable `base_url` (works with owner's proxy or direct OpenRouter); retry/rate-limit handling |
| `sft/tasks/` | Generators reusing ryth-corpus task definitions: instruction→code, bug-fix, docstring→code, explain-code, test-gen |
| `sft/filter.py` | Quality gate: rule filters (compiles/runs, length, dedup) + optional teacher self-verification |
| `sft/cli.py` | `ryth-sft generate --corpus corpus_out --target 10000 --out sft_v1.jsonl` |

Output format: JSONL conversations already wrapped in the W2 chat template
tokens, ready for the future SFT trainer.

**Blocked-on (from owner):** API endpoint + key at implementation time.

**Acceptance:** v1 dataset of ~5–10k examples where ≥ 90% pass rule filters;
spot-check sample reviewed by owner.

## 7. Experiment Track (research budget)

One bounded experiment per milestone gap; results recorded in
`docs/experiments/<name>.md`. First candidate:

**kNN-Memory Head ("bio-memory" experiment).** After W1 completes, build an
external memory index over training-token hidden states; at inference,
retrieve nearest neighbours and blend their next-token distributions with the
model's (kNN-LM style λ-mix). This adds retrieval memory *without any
retraining* — the closest small-scale analogue to the owner's
"brain that stores memory without training" idea. Success metric: measurable
validation-perplexity reduction on held-out C+Python. Pure PyTorch, CPU
feasible at 30M scale.

Backlog (post-M3): reservoir-style fixed-random projection layer;
curriculum ablation (C-first vs mixed); FIM-rate sweep.

## 8. Agent fleet usage

Owner's Claude Code proxy exposes multiple backends. Working split:
main loop = architecture + core implementation + review integration;
parallel subagents = self-contained units (data-prep scripts, eval-runner
implementation, SFT scaffolding, code-review passes) dispatched per
superpowers subagent-driven development. All heavy GPU runs stay manual
(owner triggers Kaggle notebooks/scripts prepared by the agents).

## 9. Milestones

| # | Scope | Acceptance proof |
|---|---|---|
| M0 | Audit fixes committed; `evals/` skeleton; Kaggle smoke notebook green | pytest green incl. new tests; notebook Run-All passes |
| M1 | Real corpus ≥ 600M tokens; **30M trained** | ppl curve + samples + manifest |
| M2 | Eval baseline scores recorded (HumanEval/MBPP/ppl) | results JSONs in repo |
| M3 | SFT dataset v1 (5–10k filtered examples) + kNN-memory experiment write-up | filter-pass rate ≥ 90%; experiment doc |
| M3+ | SFT trainer integration; ladder to 125M/350M | separate spec |

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Scratch pure-Python BPE/RDE slow at GB scale | Benchmark early on 100MB slice; subsample tokenizer training; chunked processing |
| Kaggle internet/session limits | Attach data as Kaggle dataset; idempotent stages + auto-resume (already built) |
| Weekly GPU quota exhaustion | Keep runs resumable; prefer CPU sessions for prep |
| Teacher API unavailable/expensive | Rule-filter-only mode degrades gracefully; cache raw responses |
| Small-model expectations mismatch | Milestone acceptance criteria stated explicitly (§4/§9); capability ladder documented in README later |

## 11. Testing strategy

- New packages follow repo convention: pytest suites, pure-stdlib parts where
  possible, deterministic seeds.
- `evals/`: unit-test chat-template roundtrip + pass@k math with a stub
  generator; integration-test on random weights.
- `sft/`: unit-test client against a fake HTTP server + filter rules on
  fixtures; no live API calls in tests.
- Regression tests added alongside every bug fix (pattern set by today's audit).
