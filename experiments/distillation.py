"""Decoder-logit knowledge distillation for Ryth causal language models.

The two models must use the *same tokenizer and token-ID mapping*, not merely
vocabularies of equal size; this cannot be inferred from logits. Inputs and
already-shifted targets use the same convention as training.dataloader
(``input_ids = tokens[:-1]``, ``targets = tokens[1:]``).

Runnable example from the repository root (requires PyTorch)::

    python -c 'import torch; from model import RythConfig, RythForCausalLM; from experiments.distillation import distillation_step; c=RythConfig(vocab_size=32,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,ffn_hidden=32); s=RythForCausalLM(c); t=RythForCausalLM(c); o=torch.optim.AdamW(s.parameters(),lr=1e-3); x=torch.randint(32,(2,4)); print(distillation_step(s,o,x,x,teacher=t))'

For real data, iterate ``training.dataloader.RDSTorchDataset`` (or
``make_dataloaders``) and load the desired base checkpoints into the models
before creating the optimizer. Set ``alpha=0`` for target-only training without
running the teacher, or supply cached ``teacher_logits`` instead of a teacher.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def distillation_loss(student_logits, targets, teacher_logits=None, *,
                      temperature=2.0, alpha=0.5, ignore_index=-100):
    """Mean CE + T² KL(teacher || student) over non-ignored target positions.

    ``alpha`` is the KL weight; teacher logits, when supplied, are detached.
    Both logits must be [batch, sequence, vocabulary].
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if student_logits.ndim != 3 or targets.shape != student_logits.shape[:2]:
        raise ValueError("expected logits [B, T, V] and targets [B, T]")
    if targets.dtype != torch.long:
        raise ValueError("targets must be torch.long")
    mask = targets != ignore_index
    if not mask.any():
        raise ValueError("batch has no non-ignored target tokens")
    if alpha and teacher_logits is None:
        raise ValueError("teacher logits are required when alpha > 0")
    if teacher_logits is not None and teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must have identical [B, T, V] shapes")
    valid_student = student_logits[mask].float()
    ce = F.cross_entropy(valid_student, targets[mask])
    if alpha == 0:
        return ce
    valid_teacher = teacher_logits.detach()[mask].float()
    kl = F.kl_div(F.log_softmax(valid_student / temperature, dim=-1),
                  F.softmax(valid_teacher / temperature, dim=-1),
                  reduction="batchmean") * temperature ** 2
    return (1 - alpha) * ce + alpha * kl


def distillation_step(student: nn.Module, optimizer, input_ids, targets, *,
                      teacher: nn.Module | None = None, teacher_logits=None,
                      temperature=2.0, alpha=0.5, ignore_index=-100):
    """Run one optimizer step and return the scalar loss.

    A supplied teacher is permanently frozen and put in eval mode. Teacher and
    student must not share parameters. The optimizer must contain only student
    parameters; create it after any adapter injection. Models, tokens and cached
    logits must already be on the appropriate device. With alpha=0 the teacher
    forward pass is skipped. Cached logits must correspond to these exact inputs.
    """
    if teacher is not None and teacher_logits is not None:
        raise ValueError("provide a teacher or cached logits, not both")
    student_ids = {id(p) for p in student.parameters()}
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if not optimizer_ids or not optimizer_ids <= student_ids:
        raise ValueError("optimizer must contain only student parameters")
    if teacher is not None:
        if any(id(p) in student_ids for p in teacher.parameters()):
            raise ValueError("teacher and student cannot share parameters")
        teacher.requires_grad_(False)
        for p in teacher.parameters():
            p.grad = None
        teacher.eval()
        if alpha and teacher_logits is None:
            with torch.no_grad():
                teacher_logits, _ = teacher(input_ids)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    student_logits, _ = student(input_ids)
    loss = distillation_loss(student_logits, targets, teacher_logits,
                             temperature=temperature, alpha=alpha, ignore_index=ignore_index)
    loss.backward()
    optimizer.step()
    return loss.detach().item()
