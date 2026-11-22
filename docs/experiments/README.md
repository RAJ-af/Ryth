# Ryth 5B: research aur experiments

**Short answer:** low VRAM mein 5B *fit* karna aur high-quality 5B ko scratch se *jaldi train* karna alag problems hain. Memory-saving papers promising hain, lekin speed aur quality Ryth par measure nahi hui. Neeche research aur proposed experiments hain—trained model ya completed experiments nahi.

## Kahaan se shuru karein?

1. [Experiment catalog](ryth_5b_experiment_catalog.md) — prioritized experiments, controls, metrics, safety gates aur paper links. Search IDs `R5-01` to `R5-07`.
2. [Efficient pretraining research](efficient_pretraining_research.md) — compute/VRAM/time math, scaling, attention, optimizers, sharding aur candidate 5B architectures.
3. [Data and coding research](data_and_coding_research.md) — dataset rights, dedup, contamination, tokenizer, FIM, coding eval aur SFT.
4. [Existing kNN-memory design](knn_memory_head.md) — training-free retrieval idea; abhi design stage, trained checkpoint aur leakage-safe evaluation zaroori.

## Experiment search index

| ID | Search words | Scope | Priority |
|---|---|---|---|
| [R5-01](ryth_5b_experiment_catalog.md#r5-01--knn-memory-head-p1-inference-only) | kNN, retrieval, external memory, RAG-like | Inference; scratch training ka replacement nahi | P1 |
| [R5-02](ryth_5b_experiment_catalog.md#r5-02--low-rank-optimizer-state-for-scratch-p1) | GaLore, low-rank optimizer, 24 GB, scratch | Full-parameter scratch optimization | P1 |
| [R5-03](ryth_5b_experiment_catalog.md#r5-03--post-training-quantization-p1-inference-only) | GPTQ, int8, 4-bit, quantization | Inference only | P1 |
| [R5-04](ryth_5b_experiment_catalog.md#r5-04--loraqlora-adaptation-p2-fine-tuning-only) | LoRA, QLoRA, adapters, NF4 | Pretrained model fine-tuning | P2 |
| [R5-05](ryth_5b_experiment_catalog.md#r5-05--distillation-p2-scratch-student-or-adaptation-not-free-compute) | teacher, student, logits, distillation | Scratch student or adaptation; teacher cost included | P2 |
| [R5-06](ryth_5b_experiment_catalog.md#r5-06--progressive-depth-growth-p2-scratch-pretraining) | progressive growth, stacking, depth | Scratch architecture experiment | P2 |
| [R5-07](ryth_5b_experiment_catalog.md#r5-07--sparse-moe-feed-forward-p3-scratch-architecture) | MoE, experts, sparse routing, active parameters | Scratch architecture experiment | P3 |

Also search the companion documents for **Chinchilla/scaling laws**, **FlashAttention/SDPA**, **activation checkpointing**, **ZeRO/FSDP**, **8-bit optimizers**, **data quality**, **dedup**, **Hindi tokenization**, **FIM**, **HumanEval/MBPP**, **synthetic SFT** and **benchmark contamination**.

## Pehle kya karna hai?

- **P0:** independent repo-level train/validation/test, provenance and licenses, safe generated-code evaluation, reliable resume, valid-token accounting and real GPU throughput.
- **P1:** train a real small dense baseline; compare data and tokenizer choices, then investigate GaLore, attention and checkpointing on matched quality and measured cost. GaLore's paper reports a specialized 7B scratch-training feasibility setup on a 24 GB GPU; this is not a speed guarantee or proof that Ryth currently supports it.
- **P2/P3:** move only promising experiments to 300M/1B, then authorize a 5B run after a full-step GPU memory/throughput test, adequate data, resume drill and approved budget. Existing code has no distributed 5B trainer or validated 5B preset.

**Evidence:** the committed T4 run is a 12-step small smoke test; W2 benchmark results are random-weight wiring checks, not trained-model quality. All experiment gains remain hypotheses until independently run and measured. Hardware price, GPU-hours and completion time depend on the actual GPU and token budget.
