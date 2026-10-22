#!/usr/bin/env python3
"""Interactive CLI chat interface for Ryth fine-tuned chat models.

Usage:
    python3 scripts/chat.py --weights runs/sft/best.pt --tokenizer tok/tokenizer.json
    python3 scripts/chat.py --weights runs/sft/best.pt --tokenizer tok/tokenizer.json --lang_hint hi
"""

from __future__ import annotations

import argparse
import sys
import torch

from dataset import load_bpe_tokenizer
from evals.chat_template import register_chat_tokens
from evals.generation import load_model, sample_completion
from sft.tasks.prompts import SYSTEM_PROMPT


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Interactive chat with Ryth model.")
    parser.add_argument("--weights", "-w", required=True, help="Path to model weights file (.pt)")
    parser.add_argument("--tokenizer", "-t", required=True, help="Path to tokenizer JSON file")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling threshold")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--lang_hint", default="", help="Language hint: en, hi, or hinglish")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")

    args = parser.parse_args(argv)

    dev = pick_device(args.device)
    print(f"[chat] Loading tokenizer from {args.tokenizer}...")
    tok = load_bpe_tokenizer(args.tokenizer)
    register_chat_tokens(tok)

    print(f"[chat] Loading model from {args.weights} on {dev}...")
    model = load_model(args.weights, vocab_size=tok.vocab_size, device=dev)

    sys_prompt = SYSTEM_PROMPT
    if args.lang_hint:
        hint_map = {
            "en": "Respond in simple, clear English.",
            "hi": "Respond in Devanagari Hindi script (हिंदी).",
            "hinglish": "Respond in Hinglish (Hindi written using Roman/Latin alphabet). Do NOT use Devanagari script.",
        }
        hint = hint_map.get(args.lang_hint.lower(), f"Respond in {args.lang_hint}.")
        sys_prompt += f"\n\n[Language Directive: {hint}]"

    messages = [{"role": "system", "content": sys_prompt}]

    print("=" * 60)
    print("  Ryth Interactive Chat")
    print("  Type 'exit', 'quit', or press Ctrl+C/Ctrl+D to end conversation.")
    if args.lang_hint:
        print(f"  Language hint: {args.lang_hint}")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Exiting chat. Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = sample_completion(
                model,
                tok,
                prompt="",
                mode="chat",
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token="<|end|>",
            )
        except Exception as e:
            print(f"\n[Error generating response: {e}]")
            messages.pop()
            continue

        print(f"\nRyth: {response}")
        messages.append({"role": "assistant", "content": response})

    return 0


if __name__ == "__main__":
    sys.exit(main())
