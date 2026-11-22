"""Experimental, opt-in sparse top-k SwiGLU feed-forward routing.

Standalone example::

    import torch
    from experiments.moe import SparseMoE
    moe = SparseMoE(d_model=32, ffn_dim=64, num_experts=4, top_k=2)
    y, balance = moe.forward_with_aux(torch.randn(2, 8, 32))
    (y.square().mean() + 0.01 * balance).backward()

For a Ryth model, ``replace_feedforwards(model, num_experts=4, top_k=2)``
mutates its decoder in place. After each forward, add
``0.01 * sum(layer.ffn.aux_loss for layer in model.decoder.layers)`` to the
training loss before backward. The stored loss is overwritten on each forward;
do not use this interface with gradient checkpointing, concurrent forwards or
multiple microbatches without collecting each loss. No existing training loop
is changed to consume it automatically.

Replacement changes parameters: rebuild optimizer/scheduler and distributed
wrappers. RythConfig and normal checkpoint metadata do not describe MoE. Record
num_experts/top_k separately and apply the same replacement to a fresh model
*before* loading its state dict; dense checkpoints and MoE state dicts are not
interchangeable. Config parameter/FLOP estimates are no longer accurate.
This reference implementation dispatches selected tokens only, but makes no
speed or memory savings claims (Python dispatch can be slower than dense FFNs).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.decoder import RythForCausalLM
from model.feedforward import SwiGLU


class SparseMoE(nn.Module):
    """Route each token to exactly top_k experts, with no capacity dropping.

    The auxiliary loss is ``E * sum(mean_probability * assignment_fraction)``;
    assignment fractions count all selected slots and sum to one. Routing
    weights are normalized among selected experts, and full router softmax is
    used for the differentiable balancing loss. ``aux_loss`` is the latest
    forward's loss (or None before the first forward).
    """

    def __init__(self, d_model: int, ffn_dim: int, num_experts: int,
                 top_k: int = 1, dropout: float = 0.0):
        super().__init__()
        for name, value in (("d_model", d_model), ("ffn_dim", ffn_dim),
                            ("num_experts", num_experts), ("top_k", top_k)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if top_k > num_experts:
            raise ValueError("top_k must not exceed num_experts")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(d_model, ffn_dim, dropout) for _ in range(num_experts))
        self.aux_loss = None

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return output and differentiable balancing loss for this call."""
        if x.ndim < 1 or x.shape[-1] != self.d_model:
            raise ValueError("last input dimension must equal d_model")
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        if flat.shape[0] == 0:
            # Preserve a valid backward path without calling any experts.
            aux = logits.sum() * 0.0
            self.aux_loss = aux
            return x + aux.to(x.dtype), aux
        probs = F.softmax(logits.float(), dim=-1)
        selected_probs, indices = probs.topk(self.top_k, dim=-1)
        weights = selected_probs / selected_probs.sum(dim=-1, keepdim=True)
        counts = torch.bincount(indices.reshape(-1), minlength=self.num_experts)
        fraction = counts.to(probs.dtype) / indices.numel()
        aux = self.num_experts * torch.sum(probs.mean(dim=0) * fraction)
        out = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot = torch.where(indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(flat.index_select(0, token_idx))
            weighted = expert_out * weights[token_idx, slot, None].to(expert_out.dtype)
            out = out.index_add(0, token_idx, weighted)
        self.aux_loss = aux
        return out.reshape_as(x), aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x)[0]


def replace_feedforwards(model: RythForCausalLM, num_experts: int,
                         top_k: int = 1) -> RythForCausalLM:
    """Replace every dense FFN in-place with copies as experts; return model.

    This must happen before creating an optimizer. Existing FFN weights are
    copied to all experts; the router is freshly initialized. This helper rejects
    already-converted or gradient-checkpointed blocks, since Ryth's checkpoint
    recomputation cannot safely expose the per-forward auxiliary loss.
    """
    if not isinstance(model, RythForCausalLM):
        raise TypeError("model must be a RythForCausalLM")
    if not model.decoder.layers:
        raise ValueError("model must have at least one decoder layer")
    if any(not isinstance(layer.ffn, SwiGLU) or layer.use_grad_ckpt
           for layer in model.decoder.layers):
        raise ValueError("requires dense SwiGLU blocks without gradient checkpointing")
    replacements = []
    for layer in model.decoder.layers:
        old = layer.ffn
        weight = old.w_gate.weight
        moe = SparseMoE(weight.shape[1], weight.shape[0], num_experts,
                        top_k, old.dropout.p).to(device=weight.device, dtype=weight.dtype)
        moe.experts = nn.ModuleList(copy.deepcopy(old) for _ in range(num_experts))
        moe.train(old.training)
        replacements.append(moe)
    for layer, moe in zip(model.decoder.layers, replacements):
        layer.ffn = moe
    return model
