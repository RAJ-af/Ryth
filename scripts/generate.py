#!/usr/bin/env python3
"""Generate code using a trained Ryth model checkpoint and tokenizer.

Usage:
    python scripts/generate.py \
        --checkpoint runs/ryth/final.pt \
        --tokenizer tok/tokenizer.json \
        --prompt "def fibonacci(n):" \
        --temperature 0.8 \
        --top_k 40 \
        --max_new_tokens 100
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from dataset import load_bpe_tokenizer
from model import RythConfig, RythForCausalLM, generate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ryth-generate",
        description="Generate text/code using a trained Ryth checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/ryth/final.pt",
        help="Path to trained model checkpoint (.pt)"
    )
    parser.add_argument(
        "--tokenizer",
        default="tok/tokenizer.json",
        help="Path to tokenizer.json"
    )
    parser.add_argument(
        "--prompt",
        default="def fibonacci(n):\n",
        help="Prompt string to start generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0.0 = greedy)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k filtering (0/None = off)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}", file=sys.stderr)
        return 1

    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer file not found: {args.tokenizer}", file=sys.stderr)
        return 1

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = load_bpe_tokenizer(args.tokenizer)

    print(f"Loading checkpoint from {args.checkpoint} on {device}...")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Extract model preset or configuration from checkpoint state
    model_preset = "ryth_30m"
    if "metadata" in state and isinstance(state["metadata"], dict):
        model_preset = state["metadata"].get("model_preset", model_preset)
    elif "config" in state and isinstance(state["config"], dict):
        model_preset = state["config"].get("model_preset", model_preset)
    elif "model_preset" in state:
        model_preset = state["model_preset"]

    if hasattr(RythConfig, model_preset):
        preset_fn = getattr(RythConfig, model_preset)
        mcfg = preset_fn(vocab_size=tokenizer.vocab_size)
    else:
        mcfg = RythConfig.ryth_30m(vocab_size=tokenizer.vocab_size)

    model = RythForCausalLM(mcfg)
    if "model" in state:
        model.load_state_dict(state["model"])
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device).eval()

    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        bos = getattr(tokenizer, "bos_id", 0)
        prompt_ids = [bos]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    eos_id = getattr(tokenizer, "eos_id", None)
    top_k = args.top_k if args.top_k > 0 else None

    print("\n--- Generating ---")
    out = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        eos_id=eos_id
    )

    generated_text = tokenizer.decode(out[0].tolist())
    print(generated_text)
    print("------------------\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
