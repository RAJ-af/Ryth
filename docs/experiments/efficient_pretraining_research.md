# Efficient pretraining research: 5B from scratch

**Status:** research and proposed protocols; some [experimental prototypes](../../experiments/README.md) are implemented but not benchmark results. **Evidence checked:** 2026-11-22. Numerical gates are proposed decision rules, not literature guarantees. No Ryth 5B training result or hardware measurement is claimed here.

## Decision in one minute

- Pehle **data quality, real throughput and a small scaling study** measure karo. A 5B model that fits in VRAM is not necessarily affordable to train to useful quality. If useful tokens, calendar time or money are insufficient, a well-trained smaller dense model is the honest alternative.
- Start with the existing dense GQA/SwiGLU/RoPE architecture and AdamW. Use supported mixed precision and verify the actual SDPA kernel. Activation checkpointing trades time for memory; sharding, 8-bit states and new optimizers are later experiments, not free speedups.
- Chinchilla's roughly 20 training tokens per parameter is a **starting comparison**, not a universal optimum or quality target. For 5B it is ~100B tokens; deployment cost, data distribution, repetition, model and context change the optimum. Quality needs evaluation, not just a token count.
- Priority: **P0** feasibility + leakage-safe evaluation → **P1** data/scaling → **P2** kernel/batch/checkpoint → **P3** optimizer/sharding → **P4** architecture changes. Promote only on matched-quality, end-to-end cost, with predeclared gates.

## 1. What Ryth already has (and does not)

| Area | Existing anchor | Consequence for this study |
|---|---|---|
| Model | [`model/config.py`](../../model/config.py): dense decoder, tied embeddings, GQA, RoPE, SwiGLU; experimental `ryth_5b` preset | Count changes with vocabulary and tying; preset is not a validated 5B run. MLA is a stub. |
| Attention | [`model/attention/gqa.py`](../../model/attention/gqa.py) repeats K/V to query-head count before PyTorch SDPA; flag selects SDPA, not a guaranteed flash kernel | Profile actual backend, copies and peak memory; fewer KV heads do not guarantee training speedup. |
| Train | [`training/trainer.py`](../../training/trainer.py), [`training/optimizer.py`](../../training/optimizer.py): single-device AdamW or experimental GaLore-style optimizer; separate experimental FSDP path | FSDP and GaLore-style implementations require GPU validation and are not equivalent to full production 5B training. No 8-bit optimizer is implemented. For a prebuilt model, configure activation checkpointing in its model config. |
| Data | [`docs/dataset_engine.md`](../dataset_engine.md): packed RDS, file/chunk dedup, provenance; [`training/dataloader.py`](../../training/dataloader.py) supports separate validation RDS | Default random **chunk** holdout can share source documents/repos and near-duplicates. Split by source/repo before packing and audit cross-split overlaps. |
| Tokens | Dataloader shifts each stored chunk to `ids[:-1]` / `ids[1:]`; masks may exclude targets. `TrainConfig.tokens_per_step` uses configured `seq_len` | Count actual non-ignored target positions across workers, not the configured product; report both input positions and loss-bearing targets. |
| Eval and reproducibility | [`docs/evals.md`](../evals.md), [`docs/w2_baselines.md`](../w2_baselines.md), [`bench/README.md`](../../bench/README.md) | W2 numbers are random-weight smoke results, not trained quality. Provisional Python PPL text derives from HumanEval: never treat it as independent HumanEval validation. |
| Resume | [`training/checkpoint.py`](../../training/checkpoint.py) stores model, optimizer, step, RNG, scaler and sampler cursor; compatibility guards check config and dataset identity | Exact continuation still requires unchanged inputs and deterministic kernels/items; callbacks and metric history are not restored. Test uninterrupted vs resumed runs and reserve storage. FSDP experiment only exports weights and cannot resume. |

Ryth's [W1 runbook](../w1_runbook.md) is a **30M** code-model plan, not a 5B budget. T4 has no native BF16; use FP16 with loss scaling or FP32 and measure stability. Do not extrapolate its time estimate to 5B.

## 2. Evidence versus hypotheses

| Primary evidence | Finding / limit | Ryth hypothesis (unverified) |
|---|---|---|
| [Chinchilla, Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556) | Within their studied compute/data regime, model size and tokens should grow roughly together; ~20 tokens/parameter is a useful heuristic, not a law for every dataset. | A 5B run near 100B tokens may be a reasonable baseline; fit local curves first. |
| [Data-constrained scaling, Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264) | In their experiments up to ~4 epochs had little loss penalty versus unique data; more repetition eventually has diminishing returns. Not a license to repeat any tiny corpus indefinitely. | Measure unique and repeated tokens separately and stop repetition when held-out returns flatten. |
| [Deduplication, Lee et al. 2021](https://arxiv.org/abs/2107.06499); [DataComp-LM, Li et al. 2024](https://arxiv.org/abs/2406.11794) | Dedup can reduce memorization and train/test overlap; filtering and mixture substantially affect quality. DataComp reports results for its own data and evaluation, not Ryth. | Reproducible license-aware filtering and overlap controls may beat more low-quality tokens per GPU-hour. |
| [FlashAttention, Dao et al. 2022](https://arxiv.org/abs/2205.14135) | IO-aware **exact** attention avoids materializing the full attention matrix; quadratic attention arithmetic remains. Published speedups depend on workload/hardware. | SDPA flash, if actually dispatched, may save memory and time at Ryth's shapes. |
| [GQA, Ainslie et al. 2023](https://arxiv.org/abs/2305.13245) | Shared KV heads mainly reduce inference KV-cache/bandwidth; the paper studies uptraining MHA checkpoints, not scratch Ryth training. | Different KV ratios may trade quality and decode memory; measure pretraining separately. |
| [Activation checkpointing, Chen et al. 2016](https://arxiv.org/abs/1604.06174); [ZeRO, Rajbhandari et al. 2019](https://arxiv.org/abs/1910.02054) | Recompute reduces activation storage; sharding reduces replicated training state. Both add work/communication and gains depend on implementation. | Use the least costly memory intervention that fits; sharding might lose on slow interconnect. |
| [8-bit optimizers, Dettmers et al. 2021](https://arxiv.org/abs/2110.02861); [GaLore, Zhao et al. 2024](https://arxiv.org/abs/2403.03507) | Papers report memory savings in their tested regimes; GaLore changes update dynamics despite full-parameter learning. Neither proves parity on Ryth's corpus or a fast 5B run. | Try only after AdamW reference; quantify convergence, checkpoint/resume and wall-clock, not just VRAM. |

Sources above are primary research; abstract-level evidence is used for their stated results, not as proof of a particular implementation's behavior.

## 3. Budget and feasibility before rental

### Compute and time

For a dense decoder, rough training compute `C ≈ 6 × N × D` FLOPs, where `N` is trainable parameters and `D` is processed training tokens. It omits or approximates attention's sequence-length cost, logits, optimizer, activation recomputation and communication. Use it for a first estimate, then calibrate against measured step time. For 5B:

| Processed tokens D | Rough FLOPs | At aggregate **100,000 useful target tokens/s** |
|---:|---:|---:|
| 10B | 3 × 10^20 | 27.8 h |
| 100B | 3 × 10^21 | 277.8 h / 11.6 days |
| 1T | 3 × 10^22 | 2,778 h / 115.7 days |

100k tokens/s is **only arithmetic**, not a Ryth measurement or an attainable promise. Divide D by the measured aggregate end-to-end valid-target rate, not GPU count times a quoted peak. Add preprocessing, evaluation, checkpointing, warmup, downtime/restarts and reserve; quote GPU-hours = elapsed hours × number of GPUs and total cost = GPU-hours × hourly rate + storage/CPU/network. Record energy if available. Recompute for actual hardware and sequence length. If fitting 5B makes batch tiny or offloads heavily, speed may become prohibitive.

### Memory accounting

For 5B unique parameters, one FP32 copy is **20 GB decimal** (~18.6 GiB); tied weights count once. With Ryth's usual FP32 model parameters under autocast, a first AdamW estimate is FP32 weights 20 GB + FP32 gradients 20 GB + two FP32 moments 40 GB = **80 GB decimal** (~74.5 GiB), before activations, temporary buffers, logits, CUDA context, fragmentation and checkpoints. Actual dtype/state choices must be inspected after an optimizer step; other mixed-precision implementations may hold separate master weights. BF16 autocast alone does not halve all this state.

Add activations (depend on layers, microbatch, sequence, kernel, checkpoint policy), attention intermediates, `B × T × vocab` logits and FP32 loss conversion in [`training/loss.py`](../../training/loss.py), workspace, communication buffers and memory fragmentation. Save host RAM/disk budget too: optimizer-inclusive checkpoints are much larger than weights-only, and `CheckpointManager.save` writes both tagged and latest files. Record `max_memory_allocated` **and** `max_memory_reserved` after warmup/full optimizer step, plus host RAM and disk peak. OOM margin target: at least 10% free device memory under the production shape; this is a planning gate, not a guarantee.

For `p` data-parallel ranks, idealized ZeRO-1 shards optimizer state, ZeRO-2 also gradients, ZeRO-3 also parameters; activation memory is not automatically divided by `p`, and all-gathers, transients and communication add overhead. Measure, do not divide total observed memory by `p`. CPU/NVMe offload can fit a model but adds host memory, PCIe/storage traffic and potentially severe latency.

## 4. Protocols — prioritized, falsifiable

Common controls: pin git SHA, config, hardware/GPU count, driver, PyTorch/CUDA, precision, tokenizer hash, data manifests, licenses and source versions, split/overlap method, seed, sequence length, valid-token budget, effective batch in **valid tokens**, LR schedule/warmup, checkpoint and eval cadence. Use paired seeds where feasible (3 for pilot quality comparisons), fixed validation and final held-out test; never tune on final test. Record confidence intervals via paired per-document bootstrap for loss and per-problem bootstrap for task metrics; report individual seeds too. Pre-register gates below, save all arms including failures, and retune learning rate with equal search budgets when changing optimizer/batch/model. Matched-tokens answers sample efficiency; matched wall-clock/GPU-hours answers practical efficiency. Neither substitutes for the other.

### P0 — Feasibility, throughput and correctness

1. Build separate train/validation/test at repository or source-document level **before packing**, and deduplicate exact and near overlaps across splits and against benchmarks. Record exclusions and license/opt-out policy. RDS chunk dedup alone is not a near-duplicate or benchmark-contamination audit. Freeze held-out sets before tuning.
2. Establish hardware inventory: GPU model/count, VRAM, interconnect, CPU/RAM, storage bandwidth, device price and quota. On target device run e.g. `python -m training.benchmark --preset ryth_30m --seq_len 1024 --batch 4 --dtype fp16 --device cuda --steps 100` (BF16 only on supported hardware). This is synthetic, uses two warmup steps, one microbatch, no real input/eval/checkpoints and its output counts input positions; it is **not** end-to-end throughput. Repeat after longer warmup with synchronized timings and actual RDS.
3. Measure train-step p50/p95, aggregate valid-target tokens/s, GPU utilization, allocated/reserved peak, data wait, CPU/RAM, eval and checkpoint wall time, failed/skipped steps, throughput including all overhead. For distributed experiments sum targets across ranks once; don't count repeated/ignored targets as unique data. Check actual SDPA kernel via profiler, and full forward/backward/optimizer correctness. Scale test shapes as memory allows; do not extrapolate linearly across model sizes.
4. Resume drill: compare uninterrupted vs interrupted training at identical token positions, including data order, scaler, scheduler and optimizer. Track replayed/skipped tokens. If current machinery cannot do this, record it as an implementation gap; do not claim exact replay.

**Gate:** no launch without a measured 5B-shaped full step and multi-hour stability test, independent holdouts, budget including overhead, and successful resume or explicitly accepted replay risk. Run is blocked if estimated cost/time exceeds owner budget; no claimed improvement from synthetic benchmarks alone.

### P1 — Data and scaling, highest expected leverage

- Use existing `ryth_30m`, `ryth_125m`, `ryth_350m` and, if feasible, `ryth_1b` with the same tokenizer and held-out suite. Measure actual parameter counts, not preset names. For each size select e.g. 5, 10, 20, 40 **processed** tokens/parameter; include a longer point only if budget permits. Train from scratch, fixed sequence and source mix, matched scheduler shape in tokens; report learning curves, compute and wall time. Fit a simple held-out loss-vs-N/D curve with uncertainty; reserve one pilot point for a prediction check. Extrapolation from <1B to 5B is uncertain and must not be sold as a forecast.
- Compare fixed-token data arms: (A) original mix, (B) license-reviewed filtered + exact/near dedup, (C) B with a predeclared source mix adjustment. Fix tokenizer, ordering policy, FIM rate and total processed valid targets; log source tokens, unique documents and repetition/epoch distribution, rejected counts and preprocessing cost. If filtering changes unique corpus size, explicitly report repeated-token fraction; compare both equal processed tokens and equal compute. Audit benchmark contamination on each arm.
- Evaluate held-out next-token cross entropy and perplexity by source/language, plus code task pass@1 with fixed prompt/template/sampling and execution sandbox; include relevant non-code/language evaluations if general quality is intended. Keep code benchmark tasks out of training, development and PPL sets. PPL comparisons require identical tokenizer, context and target mask; for different tokenizers add comparable byte-normalized loss and downstream metrics. Execution of generated code requires isolation; see [`docs/evals.md`](../evals.md).
- **Proposed gate:** advance an arm only if paired bootstrap 95% interval favors lower aggregate held-out loss, no priority slice regresses >1% relative loss, no contamination/license failure, and measured end-to-end GPU-hours to the same loss do not increase. Tune thresholds before seeing results; if intervals overlap, call it inconclusive. Lower PPL alone is not proof of coding or general capability.

### P2 — Attention, batch and activation memory

- Fix model, data sequence, total valid tokens and optimizer. First compare current SDPA flag on/off with causal masking and dropout held fixed. Test logits, gradients and causality within dtype-appropriate tolerances; then profile actual dispatched backend, `repeat_kv` allocations, end-to-end tokens/s and peak memory at lengths 512/1024/2048 (and target context). Some devices/dtypes/shapes fall back; manual attention may OOM. Flash is exact up to floating-point differences, not a change to context semantics.
- Compare checkpoint off/on with same effective valid-target batch and LR; sweep feasible microbatch/accumulation pairs (e.g. 1×8, 2×4, 4×2) at fixed global batch. Checkpoint recomputation can permit larger microbatches, but compare full-step rate including recompute. Accumulation improves fit, not intrinsic compute; different batch/packing/partial batches can change training dynamics. For custom models enable `RythConfig.use_gradient_checkpointing`, not only the trainer setting.
- **Gate:** numerical/causality checks pass and no loss divergence; retain a change if it improves end-to-end valid tokens/s by ≥5% at matched held-out loss (≤1% relative degradation), or enables a previously OOM workload within budget. Repeat timings ≥3 times; report uncertainty and actual kernel. These are proposed cutoffs.
- Local/sliding-window or sparse attention is a **later, new implementation**, not flash: compare at equal compute and token budgets, test long-range retrieval and downstream quality. It changes receptive field and may not be faster if kernels are inefficient.

### P3 — Optimizer, state sharding and offload

- Reference is Ryth fused-when-available AdamW with current betas/weight decay and cosine schedule. First compare checkpointed AdamW against a future 8-bit-state AdamW implementation, then optionally GaLore; use same initialization/data, equal LR search allowance and seeds. Measure post-step dtype/state size, loss spikes, gradient norms, NaN/skipped steps, quality at equal tokens **and** time to reference loss, peak memory, checkpoint size, save/load and restart. The papers' results do not establish this implementation's safety.
- On ≥2 GPUs only after single-device reference, compare replicated DDP vs future ZeRO/FSDP-style optimizer-only, gradient and parameter sharding as available. Same global valid-token batch, data partition, precision and update schedule; compare GPU-hours, aggregate throughput, scaling efficiency, peak per-rank memory and communication. Test checkpoint portability and resume. DDP/FSDP/ZeRO integration is **not** present in the current trainer. Offload is a separate arm with CPU RAM, PCIe/NVMe bytes and wall time measured.
- **Gate:** require ≥15% measured peak-memory reduction or a previously impossible fit, ≤1% relative held-out loss regression with uncertainty reported, no stability/resume failure, and no worse cost to reach reference loss. A smaller memory number alone is insufficient. Weak interconnect, optimizer retuning and sharded all-gather spikes are main confounders.

### P4 — Candidate 5B architectures (unverified designs)

Keep vocab 32,000, tied embeddings, GQA/RoPE/SwiGLU, head dimension 128 and `ffn_multiple_of=256`. Approximate counts from `RythConfig.estimate_params()`; instantiate and count unique parameters before use. Examples, **not existing presets**:

| Candidate | Layers | d_model | Q / KV heads | FFN dim | Approx. tied params |
|---|---:|---:|---:|---:|---:|
| A: deeper/narrower | 36 | 3584 | 28 / 7 | 9728 | ~5.04B |
| B: wider/shallower | 32 | 3840 | 30 / 6 | 10240 | ~5.03B |

Change width/depth and KV ratio together here, so this comparison cannot attribute the outcome to either alone; follow with one-factor KV-head ablations at a smaller scale. Real count depends on vocab, untied head and config; compare parameters, `6ND`, actual FLOPs, peak memory, throughput and inference KV cache. At fixed hardware, matrix shapes and depth affect utilization/latency; equal parameter counts do not imply equal speed. Train scaled-down analogues with paired seeds, same data, context, tokens and equal tuning; evaluate loss, slices, code tasks and generation latency at matched compute and matched wall time. Promote a 5B candidate only after a multi-hour 5B-shaped stability/throughput test and a budgeted full-run plan. **Proposed gate:** lower loss with a 95% paired interval excluding zero, no priority-slice regression >1%, and lower cost to reference loss. If neither wins reliably, keep the simpler established baseline.

MoE, MLA, recurrent/state-space or low-rank attention are research alternatives, not drop-in Ryth features: total vs active parameters, routing imbalance, communication, kernel support, long-context quality and scratch-training stability make comparisons difficult. A 5B-total MoE is not equivalent to a 5B-active dense model. Defer until a dense baseline exists; compare equal **total and active** parameter counts separately, training FLOPs, memory and actual wall time. Similarly, LoRA over frozen random weights is not equivalent to full from-scratch 5B pretraining.

## 5. Launch checklist and reporting template

- [ ] Target quality and task mix declared; independent held-out sets and contamination/license audit frozen.
- [ ] Data manifest lists unique/processed/non-pad tokens, sources, repeat distribution, tokenizer and split hashes; compute estimated from actual N and measured throughput.
- [ ] Memory accounting includes parameters, gradients, states, activations, logits, workspace, fragmentation, checkpoints and host/offload; production shape fits with margin.
- [ ] Kernel dispatch and numerical tests recorded; interrupted/resumed run audited; total GPU-hours, dollars and elapsed-time budget approved with reserve.
- [ ] Stop conditions: persistent nonfinite loss/gradients, data leak/license issue, OOM or projected budget overrun; preserve logs/checkpoints and report failure, not a guessed success.

For each arm record: hypothesis; baseline/one change; configs and hashes; N, context and global batch; hardware/software; processed/unique/valid tokens; actual kernel; mean and interval for held-out loss/PPL and slice/task metrics; seeds; valid tokens/s, peak allocated/reserved VRAM, GPU-hours, elapsed time, dollars, checkpoint/host/storage overhead; failures; acceptance decision and confounders. A high-quality 5B is an empirical outcome, not something guaranteed by any recipe above.
