"""Ryth chat format — tokenizer ke special-token system par bana.

Format (SFT data aur inference dono isi se banenge):

    <|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>{reply}<|end|>

Pure standard library. Tokenizer object sirf 4 methods expect karta hai jo
scratch BPE pehle se deta hai (see Interfaces in the plan).
"""

from __future__ import annotations

CHAT_TOKENS = ("<|system|>", "<|user|>", "<|assistant|>", "<|end|>")
_ROLES = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def register_chat_tokens(tok) -> dict[str, int]:
    """Chat sentinels tokenizer me add karo (agar pehle se nahi hain)."""
    have = getattr(tok, "special_tokens", {}) or {}
    missing = [t for t in CHAT_TOKENS if t not in have]
    if missing:
        tok.add_special_tokens(missing)
    return {t: tok.special_tokens[t] for t in CHAT_TOKENS}


def render(messages: list[dict], *, add_generation_prompt: bool = False) -> str:
    """Messages -> chat-formatted string. Har turn apne sentinel se bandhta hai."""
    parts: list[str] = []
    for m in messages:
        role = m["role"]
        if role not in _ROLES:
            raise ValueError(f"unknown role {role!r} (expected one of {sorted(_ROLES)})")
        parts.append(f"{_ROLES[role]}{m['content']}<|end|>")
    if add_generation_prompt:
        parts.append("<|assistant|>")
    return "".join(parts)


def extract_assistant(text: str) -> str:
    """Last assistant-turn ka content nikalo (<|end|> tak)."""
    idx = text.rfind("<|assistant|>")
    if idx == -1:
        return ""
    body = text[idx + len("<|assistant|>"):]
    end = body.find("<|end|>")
    return (body[:end] if end != -1 else body).strip()