# Training Engine (v0.3.0)

Pure-PyTorch training for the Ryth model core over RDS datasets. Modular and
production-quality. Located in the `training/` package.

Install: `pip install -e ".[train]"` (needs PyTorch).

## Pipeline

```
RDS dataset ─► dataloader ─► model (core) ─► loss ─► gradient (accum/clip/NaN)
            ─► optimizer (AdamW) ─► scheduler (warmup+cosine)
            ─► metrics/logger ─► evaluator (loss+ppl) ─► checkpoint (+resume)
            ─► callbacks (early stop)
```

## Modules

| File | Role |
|------|------|
| `config.py` | `TrainConfig` — single source of truth |
| `precision.py` | bf16/fp16 autocast + GradScaler |
| `gradient.py` | accumulation, clipping, NaN/Inf detection & skip |
| `optimizer.py` | AdamW factory (decay/no-decay groups, fused on CUDA) |
| `scheduler.py` | warmup + cosine factory |
| `loss.py` | causal-LM cross entropy + perplexity |
| `dataloader.py` | RDSDataset → next-token batches, deterministic split |
| `curriculum.py` | RDE difficulty → easy→hard ordering |
| `metrics.py` | throughput + running averages |
| `evaluator.py` | validation loss + perplexity |
| `logger.py` | console + JSONL + optional TensorBoard |
| `checkpoint.py` | save/load/auto-resume, `keep_last`, experiment metadata |
| `callbacks.py` | callback system + `EarlyStopping` |
| `profiler.py` | `torch.profiler` wrapper |
| `trainer.py` | the loop tying it all together |
| `benchmark.py` | training throughput benchmark |
| `cli.py` | `ryth-train` |

## Features

✅ AdamW ✅ warmup+cosine ✅ gradient accumulation ✅ gradient clipping
✅ bf16/fp16 ✅ gradient checkpointing ✅ auto-resume ✅ checkpoint manager
✅ JSON + TensorBoard logging ✅ validation + perplexity ✅ seed & deterministic
✅ experiment tracking (git commit, tokenizer hash, dataset/model version)
✅ curriculum learning ✅ CPU & GPU

## Quick start

```python
from training import TrainConfig, Trainer

Trainer(TrainConfig(
    data_dir="rds_out",          # RDE output (manifest.json)
    model_preset="ryth_30m",     # ryth_30m | ryth_125m | ryth_350m | ryth_1b
    seq_len=1024, max_steps=2000,
    micro_batch_size=8, grad_accum_steps=4,   # effective batch 32
    dtype="bf16",
    curriculum=True,             # easy→hard (uses RDE difficulty metadata)
)).train()
```

CLI:
```bash
ryth-train --data_dir rds_out --model_preset ryth_30m --max_steps 2000 \
    --micro_batch_size 8 --grad_accum_steps 4 --dtype bf16
ryth-train --data_dir rds_out --resume latest      # auto-resume
```

### Fine-tune / custom model
Pass a pre-built model to the trainer instead of a preset:

```python
from model import RythConfig, RythForCausalLM
model = RythForCausalLM(RythConfig(vocab_size=..., d_model=256, n_layers=4))
Trainer(TrainConfig(data_dir="rds_out"), model=model).train()
```

## Checkpoints & resume

Each checkpoint stores model + optimizer + step + best_val + RNG states + config +
**experiment metadata** (git commit, tokenizer hash, dataset version, model
version, torch version). `keep_last` rotates old checkpoints; `latest.pt` enables
`--resume latest`. `best.pt` is the best-validation model; `final.pt` at the end.

## Benchmark

```bash
python -m training.benchmark --preset ryth_30m --seq_len 256 --batch 4 --device cuda
```
Reports step time and steps/sec + tokens/sec (synthetic data, real train steps).

## Tests

```bash
pip install -e ".[dev]"
pytest tests/test_training.py -v      # 28 tests
```
Covers config, precision, gradient (clip + NaN skip), optimizer/scheduler
factories, loss, dataloader, curriculum, metrics, evaluator, logger, checkpoint
(save/load/resume/rotation), callbacks, profiler, and **real end-to-end training**
(loss decreases, auto-resume, early-stop wiring).

> Model core is unchanged (v0.2.0). Next: train the 30M prototype (ROADMAP Phase 5).
