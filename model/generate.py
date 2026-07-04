"""Autoregressive text generation with the model's KV-cache.

Trained model se output nikalne ke liye. Greedy ya temperature + top-k sampling.
Ye token-ids ke saath kaam karta hai; text ke liye apna tokenizer use karo.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, input_ids, *, max_new_tokens: int = 50,
             temperature: float = 1.0, top_k: int | None = None,
             eos_id: int | None = None):
    """
    input_ids : [B, T] long (prompt)
    returns   : [B, T + n] long (prompt + generated), n <= max_new_tokens
    """
    model.eval()
    device = input_ids.device
    out = input_ids
    past = None
    cur = input_ids
    for _ in range(max_new_tokens):
        logits, past = model(cur, past_kvs=past, use_cache=True)
        logits = logits[:, -1, :]
        if temperature != 1.0:
            logits = logits / max(1e-6, temperature)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        if temperature == 0.0:
            nxt = logits.argmax(dim=-1, keepdim=True)       # greedy
        else:
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        out = torch.cat((out, nxt), dim=1)
        cur = nxt                                            # only new token (cache)
        if eos_id is not None and (nxt == eos_id).all():
            break
    return out
