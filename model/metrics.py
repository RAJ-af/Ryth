"""Model metrics — research me bahut kaam aate hain.

Har forward test / benchmark ke baad ye print hote hain:
    Parameters · Trainable Params · Estimated FLOPs · KV Cache Size
    Peak Activation Memory · Context Length

FLOPs / activation estimates *approximate* hain (documented formulae) — exact
profiling ke liye torch profiler use karo; ye quick research-time numbers hain.
"""

from __future__ import annotations


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(config, seq_len: int, batch: int = 1) -> int:
    """Approx forward FLOPs. matmuls ~ 2*N*tokens, plus attention O(T^2) term."""
    tokens = batch * seq_len
    d, L = config.d_model, config.n_layers
    # non-embedding params (projections + ffn), per the config estimate
    per_layer_attn = (d * config.n_heads * config.head_dim
                      + 2 * d * config.n_kv_heads * config.head_dim
                      + config.n_heads * config.head_dim * d)
    per_layer_ffn = 3 * d * config.ffn_dim
    n_nonembed = L * (per_layer_attn + per_layer_ffn)
    matmul_flops = 2 * n_nonembed * tokens
    # attention scores (QK^T) + weighted sum (AV): ~ 2 * 2 * d * T^2 per layer
    attn_flops = 2 * 2 * L * d * (seq_len ** 2) * batch
    return matmul_flops + attn_flops


def kv_cache_bytes(config, seq_len: int, batch: int = 1,
                   dtype_bytes: int = 2) -> int:
    """KV-cache size: 2 (k+v) * layers * n_kv_heads * head_dim * seq * batch * bytes."""
    return (2 * config.n_layers * config.n_kv_heads * config.head_dim
            * seq_len * batch * dtype_bytes)


def activation_bytes(config, seq_len: int, batch: int = 1,
                     dtype_bytes: int = 2) -> int:
    """Rough peak activation memory for training (stored for backward)."""
    d, L = config.d_model, config.n_layers
    stream = L * batch * seq_len * (4 * d + config.ffn_dim)      # residual + ffn interm.
    scores = L * batch * config.n_heads * (seq_len ** 2)         # attention maps
    return (stream + scores) * dtype_bytes


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def model_metrics(model, config, seq_len: int | None = None, batch: int = 1,
                  dtype_bytes: int = 2) -> dict:
    seq_len = seq_len or config.max_seq_len
    total, trainable = count_params(model)
    return {
        "parameters": total,
        "trainable_params": trainable,
        "context_length": config.max_seq_len,
        "seq_len": seq_len,
        "batch": batch,
        "estimated_gflops": round(estimate_flops(config, seq_len, batch) / 1e9, 2),
        "kv_cache_bytes": kv_cache_bytes(config, seq_len, batch, dtype_bytes),
        "activation_bytes": activation_bytes(config, seq_len, batch, dtype_bytes),
    }


def format_metrics(m: dict) -> str:
    return (
        f"  Parameters        : {m['parameters']/1e6:.2f}M\n"
        f"  Trainable Params  : {m['trainable_params']/1e6:.2f}M\n"
        f"  Context Length    : {m['context_length']}\n"
        f"  Estimated FLOPs   : {m['estimated_gflops']} GFLOPs "
        f"(seq={m['seq_len']}, batch={m['batch']})\n"
        f"  KV Cache Size     : {_human_bytes(m['kv_cache_bytes'])}\n"
        f"  Peak Activations  : {_human_bytes(m['activation_bytes'])} (rough)"
    )
