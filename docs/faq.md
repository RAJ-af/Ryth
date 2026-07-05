# FAQ

### What is Ryth?

An effort to build a **coding-first LLM from scratch**. As of **v0.3.0** it ships
four pillars: a scratch byte-level BPE **tokenizer**, the **Ryth Data Engine
(RDE)** (raw code → training-ready binary shards), a decoder-only transformer
**model core**, and a pure-PyTorch **training engine**.

### Is the model / training engine included?

Yes. The **model core** landed in v0.2.0 (`model/` package) and the **training
engine** in v0.3.0 (`training/` package) — both pure PyTorch. The full
data → tokenizer → model → training path is buildable from source. Next on the
[roadmap](../ROADMAP.md): actually training the first **30M prototype**
end-to-end (Phase 5).

### What are the dependencies?

The tokenizer and RDE core are **pure Python standard library** — no required
runtime dependencies. The model core and training engine need **PyTorch**
(`pip install -e ".[model]"` / `".[train]"`). `xxhash` (faster dedup), `pytest`
(tests), and `tensorboard` (optional logging) are extras.

### Which Python versions are supported?

Python **3.9+**.

### Does the tokenizer support non-English / Indian languages?

Yes — it is **byte-level**, so any UTF-8 text (Hindi, Tamil, Telugu, Urdu,
Punjabi, etc.) encodes and decodes losslessly with **no unknown tokens**. Note
that scripts not seen during training tokenize less efficiently (more tokens per
character) until you include them in the training corpus. Efficiency comes from
*data*; the architecture is language-agnostic.

### How do I make the tokenizer efficient for a new language?

Add text in that language (and code with comments in it) to the tokenizer
training corpus, and consider a larger `--vocab`. The BPE trainer will learn
merges for that script.

### Why store tokens as uint16?

For vocabularies ≤ 65536, `uint16` uses 2 bytes per token instead of 4 — halving
storage and I/O. RDE automatically switches to `uint32` for larger vocabularies
(via the header `dtype_flag`).

### Can RDE handle datasets bigger than RAM?

Yes. RDS shards are **memory-mapped** and read lazily — only the chunks you
access are paged in. 100GB+ datasets work on machines with modest RAM.

### Is dataset building reproducible?

Yes. File iteration is sorted, curriculum sorting is stable, and FIM generation
is seeded — so the same inputs + config + seed produce **byte-identical** shards.
Each build records a **Manifest Lock** (versions, tokenizer hash, config, seed,
git commit) so you always know how a dataset was produced.

### Will the RDS format change? Will my old datasets break?

The format may evolve — and that's fine. RDS carries a `version` in its header,
and the reader dispatches on it. New versions add a parser; **old shards keep
working**. A reader older than a shard's version raises a clear "upgrade" error.

### What files does RDE drop, and why?

Vendor/build folders, binary files, empty files, generated code, non-UTF-8 files,
duplicates (Cleaner), plus files failing size/extension/language/encoding/syntax
checks (Validator), and optionally low-quality files (`min_quality`). Run
`ryth-rde stats <dir>` to see drop reasons.

### Does the Validator execute my code?

No. It uses `compile(..., "exec")` to check Python **syntax only** — it does not
run the code. Still, treat untrusted repositories with care and process them in
an isolated environment. See [SECURITY.md](../SECURITY.md).

### What is FIM and why is it here?

**Fill-In-the-Middle**: training examples where the model must predict a missing
middle span given a prefix and suffix — valuable for code completion. RDE's FIM
Builder generates these from functions using sentinel tokens
(`<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>`) that match the
tokenizer's special tokens.

### How is "difficulty" decided for the curriculum?

By a multi-signal complexity score — AST depth, cyclomatic complexity, imports,
classes, async usage, function count, and size — **not** file size alone. Records
are ordered easy → medium → hard.

### What is the model architecture?

A **decoder-only transformer** in pure PyTorch: **RoPE** positional embeddings,
**RMSNorm**, **SwiGLU** feed-forward, and **Grouped-Query Attention** with a
KV-cache and a FlashAttention (SDPA) path. It's pre-norm with weight tying, and an
attention **factory** leaves room for future variants (e.g. MLA). Sizes are
`RythConfig` presets from **30M → 1B**. See [model.md](model.md).

### Do I need a GPU to train?

No — the training engine runs on **CPU and GPU**. CPU is fine for the tiny
prototypes and the test suite; a GPU (with bf16/fp16) is strongly recommended for
the 30M+ presets. See [training.md](training.md).

### Can I train my own model, or only the presets?

Either. Pass `model_preset="ryth_30m"` (…`125m`/`350m`/`1b`) to `TrainConfig`, or
build any `nn.Module` yourself and hand it to `Trainer(config, model=...)` — handy
for fine-tuning or custom sizes.

### Can I resume an interrupted training run?

Yes. Each run writes `latest.pt` (model + optimizer + step + RNG states + config +
experiment metadata). Re-run with `--resume latest` (or `resume="latest"`) to pick
up exactly where it stopped. `keep_last` rotates old checkpoints; `best.pt` and
`final.pt` are kept too.

### How do I report a bug or contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) and open an issue at
https://github.com/RAJ-af/Ryth/issues. For security issues, see
[SECURITY.md](../SECURITY.md).
