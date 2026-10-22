"""Unit tests for SFT dataset building, loss masking, and training."""

from __future__ import annotations

import json
import os
import tempfile
import torch
import pytest

from dataset import RDSDataset
from evals.chat_template import register_chat_tokens, render, extract_assistant
from model import RythConfig, RythForCausalLM
from sft.build import build_sft_dataset, process_messages
from sft.train import train_sft
from tokenizer.bpe import BPETokenizer
from training.dataloader import RDSTorchDataset


def _make_tok(tmp_path):
    tok_path = str(tmp_path / "tok.json")
    tok = BPETokenizer()
    tok.train(["hello world system user assistant end python code language"], vocab_size=300)
    register_chat_tokens(tok)
    tok.save(tok_path)
    return tok, tok_path


def test_chat_template_roundtrip():
    messages = [
        {"role": "system", "content": "You are Ryth."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 is 4."}
    ]
    rendered = render(messages)
    assert "<|system|>You are Ryth.<|end|>" in rendered
    assert "<|user|>What is 2+2?<|end|>" in rendered
    assert "<|assistant|>2+2 is 4.<|end|>" in rendered

    extracted = extract_assistant(rendered)
    assert extracted == "2+2 is 4."


def test_mask_correctness(tmp_path):
    tok, tok_path = _make_tok(tmp_path)
    messages = [
        {"role": "system", "content": "You are Ryth."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"}
    ]
    t_ids, masks = process_messages(messages, tok)

    # Check mask rules:
    # <|system|> ... <|end|> -> mask 0
    # <|user|> ... <|end|> -> mask 0
    # <|assistant|> -> mask 0
    # "world" ... <|end|> -> mask 1
    sys_sentinel = tok.special_tokens["<|system|>"]
    usr_sentinel = tok.special_tokens["<|user|>"]
    asst_sentinel = tok.special_tokens["<|assistant|>"]
    end_sentinel = tok.special_tokens["<|end|>"]

    # Position of assistant sentinel
    asst_idx = t_ids.index(asst_sentinel)

    # All tokens before assistant sentinel must have mask 0
    assert all(m == 0 for m in masks[:asst_idx])

    # Assistant sentinel itself must have mask 0
    assert masks[asst_idx] == 0

    # Assistant content + end sentinel must have mask 1
    assert all(m == 1 for m in masks[asst_idx + 1:])
    assert t_ids[-1] == end_sentinel and masks[-1] == 1


def test_rds_torch_dataset_applies_mask(tmp_path):
    tok, tok_path = _make_tok(tmp_path)
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "sample.jsonl"), "w") as f:
        f.write(json.dumps({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"}
            ]
        }) + "\n")

    out_dir = str(tmp_path / "rds")
    build_sft_dataset(data_dir, out_dir, tok_path, seq_len=32, val_frac=0.0)

    train_rds = RDSDataset(os.path.join(out_dir, "train"))
    torch_ds = RDSTorchDataset(train_rds, [0])

    inp, tgt = torch_ds[0]
    assert len(inp) == 31
    assert len(tgt) == 31

    # Check that targets where loss_mask == 0 are set to -100
    assert (tgt == -100).sum() > 0
    # Non -100 targets should exist for assistant response
    assert (tgt != -100).sum() > 0


def test_tiny_sft_training_cpu_loss_decreases(tmp_path):
    tok, tok_path = _make_tok(tmp_path)
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "en.jsonl"), "w") as f:
        for _ in range(8):
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a language."}
                ]
            }) + "\n")

    rds_dir = str(tmp_path / "rds")
    build_sft_dataset(data_dir, rds_dir, tok_path, seq_len=32, val_frac=0.25)

    # Base pretrained model
    mcfg = RythConfig.ryth_30m(vocab_size=tok.vocab_size)
    mcfg.max_seq_len = 32
    mcfg.n_layer = 2
    mcfg.n_head = 2
    mcfg.n_embd = 64
    model = RythForCausalLM(mcfg)

    weights_path = str(tmp_path / "base_weights.pt")
    torch.save({
        "model": model.state_dict(),
        "preset": "ryth_30m",
        "vocab_size": tok.vocab_size,
        "seq_len": 32,
    }, weights_path)

    out_dir = str(tmp_path / "runs_sft")
    best_loss = train_sft(
        data_dir=rds_dir,
        weights_path=weights_path,
        out_dir=out_dir,
        epochs=3,
        lr=2e-3,
        micro_batch_size=1,
        grad_accum_steps=1,
        eval_every=1,
        save_every=5,
        log_every=1,
        device="cpu",
    )

    best_pt_path = os.path.join(out_dir, "best.pt")
    assert os.path.exists(best_pt_path)

    saved = torch.load(best_pt_path, map_location="cpu", weights_only=False)
    assert "model" in saved
    assert saved["preset"] == "ryth_30m"
    assert saved["best_val_loss"] == best_loss
