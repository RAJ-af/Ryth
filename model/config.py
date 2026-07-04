"""Ryth model configuration.

Ek single dataclass jo poore model ki shape + behaviour control karti hai. Isi ko
badal ke aap 30M se 1B tak scale karte ho, aur feature-flags se experiments on/off
karte ho — koi architecture code chhune ki zaroorat nahi.
"""

from __future__ import annotations

from dataclasses import dataclass

# Valid choices (config validate karti hai)
INIT_SCHEMES = ("xavier", "llama", "deepseek", "ryth")
ATTENTION_BACKENDS = ("gqa", "mla")


@dataclass
class RythConfig:
    # --- vocabulary ---
    vocab_size: int = 32000

    # --- core dims ---
    d_model: int = 768            # hidden size
    n_layers: int = 12            # number of transformer blocks
    n_heads: int = 12             # query heads
    n_kv_heads: int = 4           # key/value heads (GQA). n_heads % n_kv_heads == 0

    # --- feed-forward (SwiGLU) ---
    ffn_hidden: int | None = None  # None => auto (~8/3 * d_model, rounded)
    ffn_multiple_of: int = 256     # ffn hidden ko is multiple par round karo

    # --- positions ---
    max_seq_len: int = 2048
    rope_theta: float = 10000.0    # RoPE base frequency (bada => lambe context)

    # --- normalization ---
    norm_eps: float = 1e-5

    # --- regularization ---
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    # --- weight tying / init ---
    tie_embeddings: bool = True    # lm_head weight = embedding weight
    init_std: float = 0.02
    init_scheme: str = "llama"     # xavier | llama | deepseek | ryth

    # --- feature flags (future experiments ke liye — hardcode nahi) ---
    use_flash_attention: bool = True     # SDPA / flash kernels agar available ho
    use_qk_norm: bool = False            # query/key ko RMSNorm karo (stability)
    use_gradient_checkpointing: bool = False
    attention_backend: str = "gqa"       # gqa | mla (mla future)

    # --- versions (future checkpoint / arch compatibility ke liye) ---
    model_version: str = "0.2.0"
    architecture_version: int = 1
    checkpoint_version: int = 1

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must divide by n_kv_heads"
        assert self.head_dim % 2 == 0, "RoPE ke liye head_dim even hona chahiye"
        assert self.init_scheme in INIT_SCHEMES, \
            f"init_scheme must be one of {INIT_SCHEMES}"
        assert self.attention_backend in ATTENTION_BACKENDS, \
            f"attention_backend must be one of {ATTENTION_BACKENDS}"

    # ---- derived ----
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """Har KV head ko kitni query heads share karti hain (GQA group size)."""
        return self.n_heads // self.n_kv_heads

    @property
    def ffn_dim(self) -> int:
        if self.ffn_hidden is not None:
            return self.ffn_hidden
        hidden = int(2 * (4 * self.d_model) / 3)       # SwiGLU convention (~8/3 d_model)
        m = self.ffn_multiple_of
        return m * ((hidden + m - 1) // m)             # round up to multiple_of

    def estimate_params(self) -> int:
        """Roughly total parameter count (tied embeddings => not double counted)."""
        d, L, V = self.d_model, self.n_layers, self.vocab_size
        emb = V * d
        attn = L * (d * self.n_heads * self.head_dim +
                    2 * d * self.n_kv_heads * self.head_dim +
                    self.n_heads * self.head_dim * d)
        ffn = L * 3 * d * self.ffn_dim
        norms = L * 2 * d + d
        head = 0 if self.tie_embeddings else V * d
        return emb + attn + ffn + norms + head

    # ---------------------------------------------------------------- #
    # Ready-made presets (scale 30M -> 1B). Sirf preset badlo, baaki same.
    # ---------------------------------------------------------------- #
    @classmethod
    def ryth_30m(cls, **kw) -> "RythConfig":
        # hidden 512 (ecosystem-common, future-compatible), 8 layers, 8 heads.
        return cls(d_model=512, n_layers=8, n_heads=8, n_kv_heads=2,
                   max_seq_len=1024, **kw)

    @classmethod
    def ryth_125m(cls, **kw) -> "RythConfig":
        return cls(d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, **kw)

    @classmethod
    def ryth_350m(cls, **kw) -> "RythConfig":
        return cls(d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4, **kw)

    @classmethod
    def ryth_1b(cls, **kw) -> "RythConfig":
        return cls(d_model=2048, n_layers=22, n_heads=16, n_kv_heads=8,
                   max_seq_len=4096, **kw)
