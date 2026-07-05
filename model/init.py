"""Weight initialization schemes.

Har model eventually apni init strategy badalta hai, isliye init ko alag rakha hai.
`apply_init(model, config)` config.init_scheme ke hisaab se sahi scheme chun-ke
poore model ki weights initialize karta hai.

Schemes:
  * xavier   — Xavier/Glorot normal on linear weights; embeddings ~ N(0, init_std)
  * llama    — N(0, init_std) everywhere; residual output projections (o_proj,
               w_down) ko 1/sqrt(2*n_layers) se scale (deep-net stability). [default]
  * deepseek — chhoti constant std (0.006) sab par, bina per-layer residual scaling.
  * ryth     — (future) Ryth-specific init — abhi NotImplementedError.
"""

from __future__ import annotations

import math

import torch.nn as nn

_RESIDUAL_SUFFIXES = ("o_proj.weight", "w_down.weight")


def base_std(config) -> float:
    """Per-scheme base std for a generic weight."""
    return 0.006 if config.init_scheme == "deepseek" else config.init_std


def init_output_weight(weight, config) -> None:
    """Initialize a bare output-projection weight (e.g. an untied LM head).

    Untied lm_head.weight is a plain Parameter (not a Linear/Embedding module),
    so the module-walk in apply_init skips it — init it explicitly here so it
    respects config.init_scheme.
    """
    if config.init_scheme == "xavier":
        nn.init.xavier_normal_(weight)
    else:
        nn.init.normal_(weight, mean=0.0, std=base_std(config))


def _normal_linears_and_embeddings(model, std):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


def _scale_residual_projections(model, std):
    for name, p in model.named_parameters():
        if name.endswith(_RESIDUAL_SUFFIXES):
            nn.init.normal_(p, mean=0.0, std=std)


# --------------------------------------------------------------------------- #
def xavier_init(model, config):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=config.init_std)


def llama_init(model, config):
    _normal_linears_and_embeddings(model, config.init_std)
    res_std = config.init_std / math.sqrt(2 * config.n_layers)
    _scale_residual_projections(model, res_std)


def deepseek_init(model, config):
    # DeepSeek-style: small constant std, rely on it instead of per-layer scaling.
    _normal_linears_and_embeddings(model, std=0.006)


def ryth_init(model, config):
    raise NotImplementedError(
        "Ryth custom init is a future experiment (planned for a later version).")


INIT_FUNCS = {
    "xavier": xavier_init,
    "llama": llama_init,
    "deepseek": deepseek_init,
    "ryth": ryth_init,
}


def apply_init(model, config) -> None:
    """config.init_scheme ke hisaab se model initialize karo."""
    INIT_FUNCS[config.init_scheme](model, config)
