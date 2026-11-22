# Experimental implementations

Install the project with PyTorch (`pip install '.[model]'`) and run commands from the repository root. These are research prototypes, not validated 5B training recipes or optimized kernels. CPU regression tests exercise small models; CUDA, multi-GPU FSDP, bitsandbytes NF4, 5B training, throughput and model quality have not been validated. Run `python -m pytest -q tests/test_experiment_*.py tests/test_galore.py tests/test_training_reliability.py` for the focused tests.

## Training and architecture prototypes

- **GaLore-style projected AdamW:** use `ryth-train --data_dir rds_out --optimizer galore --galore_rank 128 --galore_update_interval 200` with a prepared RDS dataset. This is experimental and not an 8-bit optimizer. The normal trainer defaults to AdamW. See [training documentation](../docs/training.md) for other options and resumption requirements.
- **LoRA:** load a pretrained float model, call `experiments.adapters.inject_lora(model, rank=8, alpha=16)` before building an optimizer over trainable parameters, and use `save_adapter`/`load_adapter` or `merge_lora` in eval mode. Adapter files do not include the base model; use the same base and tokenizer.
- **QLoRA:** follow the complete example in [qlora.py](qlora.py). It requires CUDA and compatible optional `bitsandbytes`, an existing float base loaded on GPU, and enough peak memory for both float and replacement layers. Preserve the exact base checkpoint and tokenizer; adapter files do not identify them. NF4 lifecycle is not GPU-validated here.
- **Distillation:** `experiments.distillation.distillation_step(student, optimizer, input_ids, targets, teacher=teacher)` runs an actual optimizer step. Teacher and student must use compatible vocabularies and aligned tokens; teacher cost is extra and the teacher is frozen.
- **Depth growth:** `experiments.growth.grow_depth(model, n_layers)` returns an independent model with copied blocks and a new config. It changes outputs, invalidates caches, and requires a new optimizer and scheduler.
- **Sparse MoE:** `experiments.moe.replace_feedforwards(model, num_experts=4, top_k=2)` converts dense FFNs before optimizer creation. The per-layer `ffn.aux_loss` is available after forward; the caller must add a suitable auxiliary loss to its training objective. This is not an integrated distributed MoE trainer.
- **FSDP:** `torchrun --standalone --nproc-per-node=2 -m experiments.fsdp_train --data-dir rds_out --model-config model.json --steps 100` requires two CUDA GPUs, NCCL and a JSON object accepted by `RythConfig` with the RDS vocabulary. It has no resume or validation, and optional `--checkpoint` is a weights-only export that gathers full FP32 weights on rank 0 and can OOM. Each rank constructs a full model in CPU RAM, the sampler/index use O(dataset size) CPU memory per rank, and failures may hang collectives. Data layout checks do not verify checksums or binary mask values. Tied-weight export and multi-GPU behavior remain unverified.

## Data and tokenizer audits

```sh
python -m experiments.data_audit --train train.jsonl --validation val.jsonl --tokenizer tokenizer.json
python -m experiments.tokenizer_benchmark --tokenizer tokenizer.json --input val.jsonl
```

JSONL rows need a string `text` or `content`; tokenizer benchmarking additionally requires `language`. The data audit checks exact full-document overlap and reports self-reported provenance; it does not detect near duplicates, benchmark contamination, or establish license rights. Use held-out representative samples.

## CPU retrieval and INT8

The following inference prototypes do not alter model weights on disk.

## kNN memory head

Prepare **separate** train and held-out JSON files, each a list of documents, e.g. `[[1, 2, 3, 4], [5, 6, 7]]`. Token IDs must use the checkpoint tokenizer and be within its vocabulary. Document boundaries reset context; each window has at most `seq-len` inputs and predicts each next token once. No pairs are created across documents. Keys are post-final-RMSNorm decoder outputs at position t; values are token t+1. Windows reset context (not a sliding full-context score).

```sh
python -m experiments.retrieval --checkpoint model.pt --datastore train.json --eval val.json --memory memory.pt --seq-len 128 --max-entries 10000 --k 8 --temperature 1 --weight 0.25
# Reuse the index with the exact same checkpoint and datastore file:
python -m experiments.retrieval --checkpoint model.pt --datastore train.json --eval val.json --memory memory.pt --load-memory --seq-len 128
```

The checkpoint must be a `model.checkpoint.save_checkpoint` file containing `metadata.config`; for training checkpoints without a model config, supply `--config model-config.json` with the full RythConfig fields. Load only trusted checkpoints. The CLI emits token-weighted base and interpolated held-out perplexities, using exact squared-L2 search with fp16 key storage and fp32 distances, distance-softmax temperature, and probability interpolation. Index building is deterministic, taking the first `max-entries` pairs, not a representative random sample. Search chunks bound temporary distance matrices; dense query-by-vocabulary probabilities and the output itself still cost memory. Index storage is approximately `entries * (2 * hidden_size + 4)` bytes. This is a small flat index, not ANN or a production-scale index.

**Leakage protocol:** construct the index exclusively from training documents; exclude validation/test documents, their copies, overlapping chunks, near-duplicates, and any future/evaluation-derived content. Separate files alone do not establish independence. The CLI rejects identical paths/hardlinks and the API rejects exact whole-document matches by token hash, but neither detects partial, re-chunked, or near-duplicate overlap. Independently verify provenance and deduplicate before reporting results. The stored index hashes the CLI checkpoint and datastore file, not tokenizer identity; keep tokenizer and preprocessing fixed. When using the Python API, the caller must ensure model/checkpoint and data provenance match, including after loading an index. No success claim is implied by a run or a hyperparameter sweep.

Python entry points: `KNNMemory.build(model, datastore_documents, max_entries=...)`, `memory.save/load`, `memory.query(hidden)`, `memory.mix(logits, hidden)`, and `evaluate(model, memory, eval_documents)`. Use eval mode and the same decoder for build and query. The API accepts separate document iterables, not a mixed split.

## Weight-only INT8

```sh
python -m experiments.quantization --checkpoint untied-model.pt --eval val.json --output model-int8.pt --seq-len 128
```

`quantize_model(model)` copies an **untied** model and replaces `nn.Linear` projections and the untied LM head with signed int8 tensors plus float32 per-output-row symmetric scales. All shared/tied parameters, including Ryth's default tied embedding/head, are explicitly rejected rather than silently broken. Embeddings, norms, biases, and other non-linear weights remain floating point. At each matmul the int8 weight is dequantized into the activation dtype; this is actual compact persistent storage but **not** integer arithmetic, and can be slower with temporary full-size weights. `tensor_bytes` reports unique parameter/buffer storage, including scales, not allocator overhead, activations, peak memory, checkpoint disk size, or latency. CLI prints mean/max absolute logit drift on supplied data and these byte counts; it does not claim accuracy or speed improvements. Save/load using `save_quantized` / `load_quantized`; this custom state format is not directly compatible with a normal Ryth model. It is for inference, not quantization-aware training.

Run focused tests: `python -m pytest -q tests/test_experiment_retrieval_quantization.py`.
