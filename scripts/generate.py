#!/usr/bin/env python3
"""Text/code generation script for Ryth models.

Loads a trained model checkpoint (`final.pt`, `best.pt`, or custom path) along
with a tokenizer, takes a prompt, and prints generated output using autoregressive
sampling (greedy or temperature + top-k).

Usage:
    python scripts/generate.py --checkpoint run_out/final.pt --tokenizer tok/tokenizer.json --prompt "def add(a, b):"
"""

import argparse
import sys
import os

# Ensure repo root is on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
from dataset import load_bpe_tokenizer
from model import RythConfig, RythForCausalLM, generate


def build_parser():
    parser = argparse.ArgumentParser(description="Generate code/text using a trained Ryth model checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--prompt", default=None, help="Input prompt text. If omitted, prompts for stdin or uses default.")
    parser.add_argument("--model_preset", default=None, help="Model preset (e.g. ryth_30m). If omitted, inferred from checkpoint config/metadata.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0 for greedy decoding).")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling cutoff.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--device", default="cpu", help="Device to run inference on (cpu, cuda, mps).")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint file '{args.checkpoint}' not found.", file=sys.stderr)
        return 1

    if not os.path.exists(args.tokenizer):
        print(f"Error: tokenizer file '{args.tokenizer}' not found.", file=sys.stderr)
        return 1

    # 1. Load tokenizer
    tok = load_bpe_tokenizer(args.tokenizer)

    # 2. Load checkpoint
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Resolve model config
    model_preset = args.model_preset
    if not model_preset:
        if isinstance(state, dict) and "config" in state and hasattr(state["config"], "model_preset"):
            model_preset = state["config"].model_preset
        elif isinstance(state, dict) and "metadata" in state and isinstance(state["metadata"], dict) and "model_preset" in state["metadata"]:
            model_preset = state["metadata"]["model_preset"]
        else:
            model_preset = "ryth_30m"

    mcfg = getattr(RythConfig, model_preset)(vocab_size=tok.vocab_size)
    model = RythForCausalLM(mcfg)

    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    # 3. Read prompt
    prompt = args.prompt
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read()
        else:
            prompt = "def hello_world():\n"

    # 4. Tokenize & Generate
    prompt_ids = tok.encode(prompt)
    if not prompt_ids:
        prompt_ids = [getattr(tok, "bos_id", 0)]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    eos_id = getattr(tok, "eos_id", None)
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=eos_id,
    )

    generated_text = tok.decode(output_ids[0].tolist())
    print(generated_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
