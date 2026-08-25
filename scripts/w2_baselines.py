"""W2/M2 baseline sweep — random-weight 30M par real benchmarks, ek command.

Random weights => pass@k 0 aur ppl ~ln(vocab) hota hai HI — point ye hai ki
poora harness REAL files par end-to-end chale aur committed results JSON
banaye jiske against trained checkpoints compare honge (spec §10 M2).

Budgets jaan-boojh ke chhote hain (CPU minutes, ghanton nahi). Post-training
full settings docs/w2_baselines.md me documented hain.
"""

from __future__ import annotations

import argparse
import json
import os


def build_val_python(bench_dir: str, out_path: str) -> int:
    """Provisional held-out Python text: HumanEval prompt+solution, sorted."""
    from evals.datasets import load_problems

    probs = sorted(load_problems(os.path.join(bench_dir, "humaneval.jsonl.gz")),
                   key=lambda p: p.task_id)
    parts = [f"{p.prompt}\n{p.canonical_solution}\n"
             for p in probs if p.canonical_solution]
    text = "\n\n".join(parts)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return len(text)


def _near_byte_tokenizer():
    from tokenizer.bpe import BPETokenizer

    tok = BPETokenizer()
    tok.train(["hello world"], vocab_size=260, verbose=False)   # M0 recipe
    return tok


def _random_model(vocab: int, seq_len: int):
    # NOTE: preset classmethod apne max_seq_len override nahi karne deta
    # (config.py:104) — isliye attrs post-construction set hote hain.
    import torch
    from model import RythConfig, RythForCausalLM

    torch.manual_seed(1234)                          # deterministic baseline
    cfg = RythConfig.ryth_30m()
    cfg.vocab_size = vocab
    cfg.max_seq_len = seq_len
    return RythForCausalLM(cfg).eval()               # eval-mode: future dropout-safe


def run_all(results_dir: str, limit: int = 40, max_new_tokens: int = 32,
            bench_dir: str = "bench",
            val_max_chars: int | None = None) -> dict:
    """val_max_chars: sirf smoke tests ke liye — held-out text ka head slice.
    Committed baseline (Task 5) defaults par chalta hai, ye knob None hi."""
    from evals import mbpp
    from evals.datasets import load_problems
    from evals.humaneval import evaluate as he_eval
    from evals.mbpp import evaluate as mp_eval
    from evals.ppl import evaluate_files

    os.makedirs(results_dir, exist_ok=True)
    tok = _near_byte_tokenizer()
    model = _random_model(tok.vocab_size, seq_len=256)
    quiet = lambda *a, **k: None                      # noqa: E731

    val_py = os.path.join(results_dir, "val_python.txt")
    n_chars = build_val_python(bench_dir, val_py)
    if val_max_chars and n_chars > val_max_chars:
        with open(val_py, encoding="utf-8") as f:
            text = f.read(val_max_chars)
        with open(val_py, "w", encoding="utf-8") as f:
            f.write(text)
        n_chars = len(text)
    print(f"[val] python held-out chars={n_chars:,}")

    # baseline provenance — post-training rerun isi se compare hoga (W2 review)
    prov = {"tokenizer": "near-byte-260", "model_init_seed": 1234,
            "checkpoint": "random-init", "seq_len": 256, "limit": limit}

    he = he_eval(load_problems(os.path.join(bench_dir, "humaneval.jsonl.gz"))[:limit],
                 model=model, tok=tok, n_samples=1, ks=(1,),
                 max_new_tokens=max_new_tokens, progress=quiet)
    he.setdefault("meta", {}).update(prov)
    with open(os.path.join(results_dir, "w2_humaneval_baseline.json"),
              "w", encoding="utf-8") as f:
        json.dump(he, f, indent=2)

    mp = mp_eval(mbpp.load_mbpp(os.path.join(bench_dir, "mbpp.jsonl"))[:limit],
                 model=model, tok=tok, n_samples=1, ks=(1,),
                 max_new_tokens=max_new_tokens, progress=quiet)
    mp.setdefault("meta", {}).update(prov)
    with open(os.path.join(results_dir, "w2_mbpp_baseline.json"),
              "w", encoding="utf-8") as f:
        json.dump(mp, f, indent=2)

    ppl = evaluate_files(model, tok, {"python": val_py}, seq_len=256)
    with open(os.path.join(results_dir, "w2_ppl_baseline.json"),
              "w", encoding="utf-8") as f:
        json.dump({"meta": {"task": "ppl",
                            "val_python_chars": n_chars, **prov},
                   "perplexity": ppl}, f, indent=2)

    print("[baselines]", json.dumps(
        {"pass@1_he": he["pass_at_k"]["pass@1"],
         "pass@1_mp": mp["pass_at_k"]["pass@1"],
         "ppl_python": round(ppl["python"], 2)}))
    return {"humaneval": he, "mbpp": mp, "ppl": ppl}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--bench", default="bench")
    p.add_argument("--limit", type=int, default=40,
                   help="per-task problem slice (CPU budget)")
    p.add_argument("--max-new-tokens", type=int, default=32)
    a = p.parse_args(argv)
    run_all(a.results, limit=a.limit, max_new_tokens=a.max_new_tokens,
            bench_dir=a.bench)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
