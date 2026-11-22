"""Experimental GaLore-style projected AdamW, implemented with PyTorch.

Set TrainConfig(optimizer="galore", galore_rank=..., galore_update_interval=...,
 galore_scale=...) to opt in. Two-dimensional parameters with both dimensions
larger than the rank are projected; all others use full-rank AdamW. This is an
experimental variant, not an exact paper reproduction or a cost/quality promise.
SVD can be expensive and projected state plus the basis may not save memory.

On refresh the basis changes, so projected moments and their bias-correction
counter are reset rather than treating old coordinates as if they were new.
Weight decay remains decoupled and acts on the whole parameter, not the update.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


def _validate(rank, update_interval, scale):
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("galore_rank must be a positive integer")
    if (isinstance(update_interval, bool) or not isinstance(update_interval, int)
            or update_interval < 1):
        raise ValueError("galore_update_interval must be a positive integer")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("galore_scale must be finite and positive")


def _work_dtype(param):
    return torch.float64 if param.dtype == torch.float64 else torch.float32


class GaLoreAdamW(Optimizer):
    """AdamW in a periodically refreshed, one-sided low-rank gradient basis."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, rank=128, update_interval=200, scale=1.0):
        _validate(rank, update_interval, scale)
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and non-negative")
        if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("betas must be in [0, 1)")
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("eps must be finite and non-negative")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay, rank=rank,
                                     update_interval=update_interval, scale=scale))
        for group in self.param_groups:
            _validate(group['rank'], group['update_interval'], group['scale'])
            if not math.isfinite(group['weight_decay']) or group['weight_decay'] < 0:
                raise ValueError("weight_decay must be finite and non-negative")

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict casts floating tensors to the parameter dtype.
        # Preserve fp32 basis/moments for fp16/bf16 parameters without a lossy
        # intermediate cast, including when loading a CPU checkpoint onto a GPU.
        super().load_state_dict(state_dict)
        for saved_group, group in zip(state_dict['param_groups'], self.param_groups):
            for saved_id, param in zip(saved_group['params'], group['params']):
                original = state_dict['state'].get(saved_id, {})
                for key in ('basis', 'exp_avg', 'exp_avg_sq'):
                    if key in original:
                        self.state[param][key] = original[key].to(
                            device=param.device, dtype=_work_dtype(param), copy=True)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            rank, interval, scale = group['rank'], group['update_interval'], group['scale']
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("GaLoreAdamW does not support sparse gradients")
                if not p.is_floating_point():
                    raise RuntimeError("GaLoreAdamW requires real floating-point parameters")
                grad = p.grad.detach().to(dtype=_work_dtype(p))
                state = self.state[p]
                projected = p.ndim == 2 and rank < min(p.shape)
                step = state.get('step', 0)
                if projected:
                    if step % interval == 0:
                        # Full SVD in fp32 (fp64 for fp64 weights). Keep the side
                        # giving the smaller projected moment tensor.
                        u, _, vh = torch.linalg.svd(grad, full_matrices=False)
                        # Clone even an already-contiguous slice so it cannot retain
                        # the full SVD output's backing storage in optimizer state.
                        vectors = u[:, :rank] if p.shape[0] >= p.shape[1] else vh[:rank, :]
                        state['basis'] = vectors.clone(memory_format=torch.contiguous_format)
                        state.pop('exp_avg', None)
                        state.pop('exp_avg_sq', None)
                        state['moment_step'] = 0
                    basis = state['basis']
                    grad = basis.T @ grad if p.shape[0] >= p.shape[1] else grad @ basis.T

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                    state['moment_step'] = 0
                state['step'] = step + 1
                state['moment_step'] += 1
                t = state['moment_step']
                avg, avg_sq = state['exp_avg'], state['exp_avg_sq']
                avg.lerp_(grad, 1 - beta1)
                avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = avg_sq.sqrt().div_(math.sqrt(1 - beta2 ** t)).add_(group['eps'])
                update = avg / denom / (1 - beta1 ** t)
                if projected:
                    update = basis @ update if p.shape[0] >= p.shape[1] else update @ basis
                    update.mul_(scale)
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update.to(dtype=p.dtype), alpha=-group['lr'])
        return loss
