"""Checkpoint -> model, aur prompt -> completion sampling.

Core packages ko chhue bina: preset/config rebuild wahi pattern jo
scripts/kaggle_train.py use karta hai. Sampling batch-size-1 (per-sequence
EOS stop ke liye — see model.generate docstring).
"""

from __future__ import annotations

import dataclasses

import torch

from .chat_template import register_chat_tokens, render


def find_eos(tok) -> int | None:
    """Registered EOS-jaisa pehla special token id, warna None."""
    for name in ("<|end|>", "<|eos|>", "<|endoftext|>"):
        tid = (getattr(tok, "special_tokens", {}) or {}).get(name)
        if tid is not None:
            return tid
    return None


def truncate_at_stops(text: str, stops: tuple[str, ...]) -> str:
    """Sabse pehle milne wale stop-string par kaato (koi stop na mile toh poora)."""
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def extract_code(text: str) -> str:
    """Markdown fences ho toh andar ka code nikaalo, warna jaisa hai waisa.

    NOTE (ruling): bina-fence output ka LEADING indent jaan-boojh ke rakha
    jaata hai — completions aksar function-body continuation hote hain;
    sirf trailing whitespace kaata jaata hai.
    """
    t = text.rstrip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:]                          # ``` / ```python
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return t


def load_model(ckpt_path: str, vocab_size: int | None = None, *,
               preset: str | None = "ryth_30m", seq_len: int = 1024,
               device: str = "cpu"):
    """Checkpoint wapas model me: stored config se rebuild, phir weights load."""
    from model import RythConfig, RythForCausalLM

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = state.get("config")
    if not isinstance(cfg_dict, dict) or "d_model" not in cfg_dict:
        meta = state.get("metadata") or {}         # save_checkpoint format
        if isinstance(meta.get("config"), dict) and "d_model" in meta["config"]:
            cfg_dict = meta["config"]
    if isinstance(cfg_dict, dict) and "d_model" in cfg_dict:
        # purane saves me extra keys ho sakte hain — sirf jaani-pehchaan fields rakho
        field_names = {f.name for f in dataclasses.fields(RythConfig)}
        mcfg = RythConfig(**{k: v for k, v in cfg_dict.items()
                             if k in field_names})
    else:
        # training checkpoints me "config" = vars(TrainConfig) hota hai
        # (d_model nahi) — architecture ka sach metadata.model_preset me hai
        meta = state.get("metadata") or {}
        name = preset or (meta.get("model_preset")
                          if isinstance(meta, dict) else None)
        if not name:
            raise ValueError(
                "checkpoint me model-config bhi nahi aur preset bhi nahi — "
                "--preset do (ryth_30m|ryth_125m|...) ya checkpoint "
                "metadata.model_preset rakho")
        mcfg = getattr(RythConfig, name)(vocab_size=vocab_size or 32000)
    if seq_len > mcfg.max_seq_len:
        mcfg.max_seq_len = seq_len
    net = RythForCausalLM(mcfg)
    net.load_state_dict(state["model"])
    return net.to(device).eval()


@torch.no_grad()
def sample_completion(model, tok, prompt: str, *, mode: str = "base",
                      messages: list[dict] | None = None, max_new_tokens: int = 256,
                      temperature: float = 0.8, top_k: int | None = 40,
                      stop_strings: tuple[str, ...] = (),
                      eos_token: str = "<|end|>") -> str:
    """Batch-size-1 sampling; sirf NAYE token decode hote hain, stops par kaat ke."""
    from model import generate

    if mode == "chat":
        register_chat_tokens(tok)
        msgs = messages if messages is not None else [{"role": "user",
                                                       "content": prompt}]
        text_prompt = render(msgs, add_generation_prompt=True)
        stops = tuple(set(stop_strings) | {eos_token})
    else:
        text_prompt = prompt
        stops = stop_strings

    ids = tok.encode(text_prompt) or [0]           # empty-prompt guard
    x = torch.tensor([ids], dtype=torch.long,
                     device=next(model.parameters()).device)
    eos_id = find_eos(tok) if mode == "chat" else None
    out = generate(model, x, max_new_tokens=max_new_tokens, temperature=temperature,
                   top_k=top_k, eos_id=eos_id)
    new_ids = out[0, x.size(1):].tolist()
    text = tok.decode(new_ids)
    return truncate_at_stops(text, stops).rstrip()
