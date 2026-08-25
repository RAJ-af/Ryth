# Experiment 1 — kNN-Memory Head ("bio-memory")

Status: **DESIGN — implementation gated on W1 checkpoint** (spec §8 kehta hai
"After W1 completes"; pehle train hue bina hidden-states hi nahi).
Spec anchor: `docs/superpowers/specs/2026-08-25-phase6-measure-first-design.md` §8.
Success metric: held-out C+Python validation-perplexity me **measurable
reduction**, zero retraining.

---

## 1. Motivation

Owner ka core idea: *"dimaag memory store karta hai bina khud ko dobara
train kiye."* Har nayi cheez dekhne par gradient update nahi hoti — ek
store banta hai, aur recall par wahi purane patterns naye decision me mix
hote hain.

Iska LLM analogue = **kNN-LM style memory head**: model ko chhedo mat,
uske training-time internal representations se ek external index banao, aur
inference par nearest-neighbour next-token distribution ko model ki apni
distribution ke saath blend kar do. 30M-param model jo 600M tokens dekh
chuka hai, uske paas "yaad" hoti hai jo weights me compress NAHI hui —
rare idioms, exact API spellings, project-specific C macros — sab index me
byte-exact yaad rehte hain.

Ye experiment-track ka pehla candidate isliye hai kyunki:
1. **Zero retraining** — W1 checkpoint as-is use hoga (Kaggle quota bachta hai).
2. Pure PyTorch + CPU-feasible at this scale (numbers §5 me).
3. Clean falsifiable metric — ppl giri ya nahi, koi subjective judgement nahi.

## 2. Background (kNN-LM, ek paragraph)

Khandelwal et al. 2020 ("Generalization through Memorization"): base LM ke
har context position par, us layer ki hidden state `f(x)` ko key banao aur
*next* token id value rakho. Inference par current context ki key se k
nearest neighbours nikaalo; unke values ka distance-weighted sum ek empirical
distribution `p_knn` deta hai:

```
p_knn(y | x) ∝ Σ_{(k_i, v_i) ∈ N_k} 1[v_i = y] · exp(−d(f(x), k_i) / τ)
p(y | x)     = λ · p_knn(y | x) + (1 − λ) · p_model(y | x)
```

Unka finding: ye interpolation rare/low-frequency tokens par sabse zyada
jeetata hai — exactly wahan jahan 30M model sabse kamzor hoga.

## 3. Design — Ryth codebase par concrete

**Key tensor:** `RythDecoder.forward()` jo return karta hai wo
post-final-RMSNorm `[B, T, d_model]` hidden hai (`model/decoder.py:49`) —
LM-head ka direct input. kNN-LM bhi output-softmax se pehle wali layer use
karta hai, to keys semantically "wahi jagah jahan prediction banti hai".
Layer-choice ablation baad me `model/hooks.py` (`after_ffn`) se ho sakta
hai — core me zero change.

**Value:** position `t` par key = hidden[t], value = `input_ids[t+1]`.
Sequence ka aakhri token skip (uska next nahi pata).

**Index entry:** fp16 key (d_model=512 → 1 KB) + int32 value. Flat array +
chunked matmul search (§5 tak CPU ok). Koi FAISS nahi — stdlib+torch only,
repo convention.

**Blend point:** evals/ppl.py ke forward loop me logits ke baad log-softmax
se p_model nikaal kar λ-mix; generation (model/generate.py) me same wrapper.
Ek hi class dono serve karegi:

```python
class KNNMemoryHead:            # experiments/knn/ package (planned)
    def __init__(self, keys_fp16, values_int32, k=128, tau=10.0, lam=0.25)
    def build(...)              # RDSDataset chunks x decoder hiddens -> arrays
    def query(hidden_btd)       # -> p_knn [B, T, vocab] (sparse-friendly)
    def mix(logits_btv)         # -> adjusted logits (log-space λ-mix)
```

## 4. Integration points (sab existing APIs)

| Kaam | Existing entry | Kyun chalega |
|---|---|---|
| Hiddens nikaalo | `RythDecoder.forward(ids)[0]` | post-norm hidden already exposed |
| Training tokens | `RDSDataset(data_dir).iter_chunks()` | RDS parts + manifest already built by w1_pack_rds |
| Val ppl | `evals.ppl.perplexity(model, tok, text)` | forward-loop me `mix()` plug |
| Generation check | `model.generate` loop | same `mix()` before sampling |

Core pillars (tokenizer/dataset/model/training) me **zero diff** — poora
experiment `experiments/knn/` me rahega, per phase-6 rule "no changes to
existing core".

## 5. Scale math — honest numbers

ryth_30m: d_model=512, vocab≈24k. Entry = 512·2B (fp16) + 4B ≈ **1 KB**.

| Indexed tokens | Raw size | Phone/Termux (~8GB free) | Kaggle disk (~20GB) |
|---|---|---|---|
| 600M (poora corpus) | ~600 GB | ❌ | ❌ |
| 50M | ~50 GB | ❌ | borderline ❌ |
| 10M | ~10 GB | ❌ | ✅ |
| 5M | ~5 GB | ❌ | ✅ |
| 2M + PCA→128-dim | ~0.3 GB | ✅ | ✅ |

To plan teeno tiers:
- **T1 (first result):** subsampled 5–10M entries on Kaggle, flat fp16 search.
- **T2:** PCA projection 512→128 (fit on 100k sample) → phone-scale 2–5M entries.
- **T3 (agar ppl girti hai):** IVF-style coarse centroids for sublinear search.

Subsampling strategy: uniform stride pehle (simple, unbiased), phir
frequency-aware (rare-token positions prefer) agar T1 me signal dikhe.
Search cost flat: 10M × 512-dim fp16 matmul ≈ 10 GFLOP/query-batch — CPU par
batched windows ke saath theek, single-token generate me latency badhegi
(T2/T3 isi liye).

## 6. Evaluation protocol

1. Checkpoint: W1 ka final ryth_30m (jab owner Kaggle run complete kare).
2. Index source: **train split RDS only** — val split index me kabhi nahi
   (leakage = fake win, protocol me hard assert).
3. Metric: `evals.ppl.evaluate_files` — val_python.txt (+ C val jab banega),
   with-memory vs without-memory, same tokenizer, same seq_len, seed-fixed.
4. Sweep: k ∈ {32, 128, 512}, λ ∈ {0.05, 0.1, 0.25, 0.5}, τ ∈ {5, 10, 20}.
   Best config report karo, poora grid results JSON me.
5. Success (spec §8): measurable val-ppl reduction. Chhota model + 30×
   compute-heavy retrieval me bhi na gire to wo bhi ek honest recorded result
   hai (backlog ablation ka input).
6. Sanity checks: random-weight checkpoint par p_kNN ppl ko IMPROVE karne ki
   koshish bhi karo — real checkpoint par hi fayda hona chahiye (warna
   implementation bug dhundo).

## 7. Risks

| Risk | Mitigation |
|---|---|
| Train/val leakage | Index build sirf train-split paths se; assert in build() |
| CPU search too slow at T1 | Batched windows; T2 PCA fallback documented up front |
| fp16 distance precision | Distances fp32 me compute (keys fp16 storage only) |
| SFT ke baad distribution shift | Index pretraining corpus se hi bana rahega; writeup me explicitly note |
| False positive from dedup overlap | corpus dedup already applied pre-pack; val file provenance bench/README pattern follow karega |

## 8. Implementation plan (post-W1, ~chhote tasks)

1. `experiments/knn/index.py` — build(): iter_chunks × decoder hiddens →
   memmap-able .npz-style pair (fp16 keys, int32 values) + manifest json
   (checkpoint sha, tokenizer version, split provenance). Resumable per-chunk.
2. `experiments/knn/search.py` — flat chunked top-k (torch.topk on fp32
   distances), pure torch.
3. `experiments/knn/head.py` — KNNMemoryHead.query/mix + unit tests with a
   tiny hand-built index (exactness test: known neighbour recover ho).
4. `scripts/knn_eval.py` — sweep runner → `results/knn_*.json` (W2 report
   format reuse), docs/experiments me result table append.

Har task TDD, full suite green, alag commits — W1 checkpoint aa te hi start.

---

*Owner note:* idea tumhara tha — formalization me maine kNN-LM se joda.
Agar tumhara mental model alag tha (e.g. write-time memory, ya retrieval
conditioning instead of output-mixing), bolo — design doc badalne me 10 min
hain, code se pehle sahi hai.
