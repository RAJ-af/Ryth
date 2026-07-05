#!/usr/bin/env python
"""End-to-end Ryth training driver — built for Kaggle T4 GPUs.

Runs the full pipeline in one command, using ONLY the public Ryth library
(no core changes):

    corpus  ->  tokenizer  ->  RDS dataset  ->  30M model  ->  training
            ->  checkpoints ->  resume       ->  code generation

By default it builds a small *synthetic* corpus and runs a short **smoke test**
so it works out-of-the-box on Kaggle with no dataset attached and no internet.
Point `--raw` at real permissively-licensed code (e.g. a Kaggle input dataset)
and bump `--steps` for a real run.

Examples
--------
    # 60-step smoke test on whatever device is available
    python scripts/kaggle_train.py --smoke

    # train on your own code, longer run
    python scripts/kaggle_train.py --raw /kaggle/input/my-code --steps 5000 \
        --seq_len 512 --vocab 16000 --micro_batch 16

The same stages are shown cell-by-cell in notebooks/ryth_kaggle_train.ipynb.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# --- make the repo importable when run from a clone (scripts/..) ------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ─────────────────────────────────────────────────────────────────────────────
# Environment: device + dtype
# ─────────────────────────────────────────────────────────────────────────────
def pick_device_dtype(force_dtype: str | None = None):
    """Return (device, dtype). T4 (Turing, sm_75) has NO native bf16 — use fp16.
    Ampere+ (sm_80+) gets bf16. CPU gets fp32."""
    import torch

    if not torch.cuda.is_available():
        return "cpu", (force_dtype or "fp32")
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    auto = "bf16" if major >= 8 else "fp16"          # T4 -> fp16, A100 -> bf16
    dtype = force_dtype or auto
    print(f"[env] GPU={name}  capability=sm_{major}{minor}  dtype={dtype}")
    return "cuda", dtype


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — corpus
# ─────────────────────────────────────────────────────────────────────────────
_FUNC_TEMPLATES = [
    "def scale_{i}(x):\n    \"\"\"Multiply x by {i}.\"\"\"\n    return x * {i}\n",
    "def add_{i}(a, b):\n    total = a + b + {i}\n    return total\n",
    "class Box{i}:\n    def __init__(self, v):\n        self.v = v + {i}\n\n"
    "    def get(self):\n        return self.v\n",
    "def clamp_{i}(x, lo={i}, hi={hi}):\n    if x < lo:\n        return lo\n"
    "    if x > hi:\n        return hi\n    return x\n",
    "def fib_{i}(n):\n    a, b = 0, {i}\n    for _ in range(n):\n"
    "        a, b = b, a + b\n    return a\n",
    "def filter_{i}(items):\n    out = []\n    for it in items:\n"
    "        if it % {i} == 0:\n            out.append(it)\n    return out\n",
]


def make_synthetic_corpus(root: str, n_files: int = 120) -> str:
    """Write many small, *varied* python files so the RDE produces plenty of
    unique chunks (dedup won't collapse them). Also drops in the library's own
    sample repo, which exercises every RDE filter."""
    os.makedirs(root, exist_ok=True)
    for i in range(n_files):
        parts, tmpl = [], _FUNC_TEMPLATES[i % len(_FUNC_TEMPLATES)]
        # a few functions per file, each seeded with unique numbers
        for k in range(4):
            j = i * 7 + k * 13 + 1
            parts.append(tmpl.format(i=j, hi=j + 50))
        body = f'"""Synthetic module {i} for Ryth smoke training."""\n\n' + "\n".join(parts)
        d = os.path.join(root, f"pkg_{i // 20}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"mod_{i}.py"), "w", encoding="utf-8") as f:
            f.write(body)
    # add the real sample fixture too (good/tests/complex/generated/broken/...)
    try:
        from tests.sample_data import build_sample
        build_sample(root)
    except Exception as e:                       # pragma: no cover
        print(f"[corpus] sample fixture skipped: {e}")
    n = len(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    print(f"[corpus] {n} python files under {root}")
    return root


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def build_or_load_tokenizer(raw: str, out_dir: str, vocab: int, rebuild: bool):
    """Train a scratch BPE tokenizer on the corpus (or load a cached one)."""
    from dataset import load_bpe_tokenizer
    from tokenizer.train import train_tokenizer, iter_text_files

    path = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(path) and not rebuild:
        print(f"[tokenizer] loading cached {path}")
        return load_bpe_tokenizer(path)

    texts = list(iter_text_files(os.path.join(raw, "**", "*.py")))
    if not texts:
        raise SystemExit(f"[tokenizer] no .py files under {raw!r}")
    print(f"[tokenizer] training on {len(texts)} files -> vocab≈{vocab} ...")
    train_tokenizer(texts, vocab_size=vocab, out_dir=out_dir, verbose=True)
    return load_bpe_tokenizer(path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — RDS dataset
# ─────────────────────────────────────────────────────────────────────────────
def build_or_load_rds(tok, raw: str, out_dir: str, seq_len: int, rebuild: bool):
    from dataset import RDEConfig, RDEPipeline, RDSDataset

    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path) and not rebuild:
        ds = RDSDataset(out_dir)
        print(f"[rds] loaded {len(ds)} chunks (seq_len={ds.seq_len})")
        return ds

    cfg = RDEConfig(seq_len=seq_len, vocab_size=tok.vocab_size,
                    tokenizer_version=getattr(tok, "version", 1),
                    shard_max_bytes=256 * 1024 ** 2)
    RDEPipeline(tok, cfg).run(raw, out_dir, verbose=True)
    ds = RDSDataset(out_dir)
    print(f"[rds] built {len(ds)} chunks (seq_len={ds.seq_len})")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(preset: str, vocab: int, seq_len: int, grad_ckpt: bool):
    from model import RythConfig, RythForCausalLM

    mcfg = getattr(RythConfig, preset)(vocab_size=vocab,
                                       use_gradient_checkpointing=grad_ckpt)
    if seq_len > mcfg.max_seq_len:
        mcfg.max_seq_len = seq_len
    model = RythForCausalLM(mcfg)
    print(f"[model] {preset}: {model.num_params()/1e6:.1f}M params "
          f"(d_model={mcfg.d_model}, n_layers={mcfg.n_layers}, "
          f"n_heads={mcfg.n_heads}, n_kv={mcfg.n_kv_heads}, vocab={vocab})")
    return model, mcfg


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_sample(ckpt_path: str, tok, preset: str, seq_len: int,
                    device: str, prompt: str, max_new_tokens: int = 60):
    import torch
    from model import generate

    vocab = tok.vocab_size
    model, _ = build_model(preset, vocab, seq_len, grad_ckpt=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    ids = tok.encode(prompt) or [getattr(tok, "bos_id", 0)]
    ids = ids[-seq_len:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, x, max_new_tokens=max_new_tokens,
                   temperature=0.8, top_k=40, eos_id=getattr(tok, "eos_id", None))
    text = tok.decode(out[0].tolist())
    print("\n" + "=" * 60 + f"\n[generate] prompt: {prompt!r}\n" + "-" * 60)
    print(text)
    print("=" * 60)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description="Ryth end-to-end Kaggle trainer")
    p.add_argument("--work", default=os.environ.get("RYTH_WORK", _default_work()),
                   help="working dir for corpus/tokenizer/rds/checkpoints")
    p.add_argument("--raw", default=None,
                   help="folder of real code (default: build a synthetic corpus)")
    p.add_argument("--preset", default="ryth_30m")
    p.add_argument("--vocab", type=int, default=2000)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--micro_batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dtype", default=None, help="force bf16|fp16|fp32 (else auto)")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--smoke", action="store_true",
                   help="tiny fast config that exercises the whole pipeline")
    p.add_argument("--resume_demo", action="store_true",
                   help="after training, resume from latest and run a few more steps")
    p.add_argument("--rebuild", action="store_true",
                   help="rebuild tokenizer + RDS even if cached")
    p.add_argument("--prompt", default="def add(a, b):\n")
    args = p.parse_args(argv)

    if args.smoke:
        args.vocab, args.seq_len = 2000, 128
        args.micro_batch, args.grad_accum = 8, 2
        args.steps, args.warmup = 60, 10
        args.resume_demo = True

    from training import TrainConfig, Trainer

    work = args.work
    raw = args.raw or os.path.join(work, "raw_repos")
    tok_dir = os.path.join(work, "tok")
    rds_dir = os.path.join(work, "rds_out")
    run_dir = os.path.join(work, "runs", "ryth-kaggle")
    os.makedirs(work, exist_ok=True)

    device, dtype = pick_device_dtype(args.dtype)

    # 1) corpus (synthetic unless --raw given)
    if args.raw is None:
        make_synthetic_corpus(raw)
    else:
        print(f"[corpus] using real code at {raw}")

    # 2) tokenizer
    tok = build_or_load_tokenizer(raw, tok_dir, args.vocab, args.rebuild)
    print(f"[tokenizer] vocab_size={tok.vocab_size}")

    # 3) RDS dataset
    ds = build_or_load_rds(tok, raw, rds_dir, args.seq_len, args.rebuild)
    n_train_est = len(ds) - max(1, int(len(ds) * 0.02))
    if n_train_est < args.micro_batch:
        raise SystemExit(
            f"[rds] only ~{n_train_est} train chunks but micro_batch="
            f"{args.micro_batch}; DataLoader drop_last would starve the loop. "
            f"Lower --micro_batch or add more/ bigger corpus.")

    # 4) model (30M preset), instantiated once and handed to the trainer
    model, _ = build_model(args.preset, tok.vocab_size, args.seq_len, args.grad_ckpt)

    # 5) train (smoke)
    cfg = TrainConfig(
        data_dir=rds_dir, model_preset=args.preset, seq_len=args.seq_len,
        lr=args.lr, warmup_steps=args.warmup, max_steps=args.steps,
        micro_batch_size=args.micro_batch, grad_accum_steps=args.grad_accum,
        dtype=dtype, grad_checkpointing=args.grad_ckpt,
        eval_every=max(10, args.steps // 3), eval_steps=5, log_every=5,
        save_every=max(10, args.steps // 2), out_dir=run_dir,
        num_workers=args.num_workers, device=device, run_name="kaggle-smoke")
    print(f"\n[train] {args.steps} steps | eff_batch={cfg.effective_batch} "
          f"| tokens/step={cfg.tokens_per_step}\n")
    Trainer(cfg, model=model).train()

    # 6) checkpoints
    cks = sorted(glob.glob(os.path.join(run_dir, "*.pt")))
    print("\n[checkpoints]")
    for c in cks:
        print(f"  {os.path.basename(c):18s} {os.path.getsize(c)/1e6:6.1f} MB")

    # 7) resume + continue
    if args.resume_demo:
        cfg2 = TrainConfig(
            data_dir=rds_dir, model_preset=args.preset, seq_len=args.seq_len,
            lr=args.lr, warmup_steps=args.warmup, max_steps=args.steps + 20,
            micro_batch_size=args.micro_batch, grad_accum_steps=args.grad_accum,
            dtype=dtype, grad_checkpointing=args.grad_ckpt,
            eval_every=max(10, args.steps // 3), eval_steps=5, log_every=5,
            save_every=max(10, args.steps // 2), out_dir=run_dir,
            num_workers=args.num_workers, device=device,
            resume="latest", run_name="kaggle-resume")
        print(f"\n[resume] continuing from latest.pt -> {cfg2.max_steps} steps\n")
        Trainer(cfg2).train()

    # 8) generate code from the best checkpoint
    best = os.path.join(run_dir, "best.pt")
    ckpt = best if os.path.exists(best) else os.path.join(run_dir, "final.pt")
    generate_sample(ckpt, tok, args.preset, args.seq_len, device, args.prompt)

    print("\n[done] end-to-end pipeline complete ✅")


def _default_work() -> str:
    for base in ("/kaggle/working", os.getcwd()):
        if os.path.isdir(base):
            return os.path.join(base, "ryth_work")
    return "ryth_work"


if __name__ == "__main__":
    main()
