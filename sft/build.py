"""sft.build — convert chat JSONL data into packed RDS shards with parallel loss masks.

Reads *.jsonl files from data_dir:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Applies NFC normalization, renders with chat template, tokenizes with special
chat tokens, and generates a parallel loss mask:
  - Mask = 1 ONLY on assistant content tokens and assistant end-of-turn (<|end|>).
  - Mask = 0 on user/system tokens and <|assistant|> sentinel.

Packs or pads into seq_len chunks, splits into train/val according to val_frac,
writes RDS shards plus parallel mask RDS shards, and prints per-file summary counts.
"""

from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path

from dataset.rds import ShardWriter
from evals.chat_template import register_chat_tokens


def validate_jsonl_line(obj: dict) -> tuple[bool, str]:
    """Check if a parsed JSON object meets SFT chat format requirements."""
    if not isinstance(obj, dict):
        return False, "not a JSON object"
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False, "messages must be a list with >= 2 items"
    valid_roles = {"system", "user", "assistant"}
    has_user = False
    has_assistant = False
    for m in msgs:
        if not isinstance(m, dict):
            return False, "message item is not an object"
        role = m.get("role")
        content = m.get("content")
        if role not in valid_roles:
            return False, f"invalid role {role!r}"
        if not isinstance(content, str) or not content.strip():
            return False, f"empty or invalid content for role {role!r}"
        if role == "user":
            has_user = True
        if role == "assistant":
            has_assistant = True
    if not (has_user and has_assistant):
        return False, "must contain both user and assistant messages"
    return True, ""


def process_messages(messages: list[dict], tok) -> tuple[list[int], list[int]]:
    """Convert messages to token_ids and loss_mask using registered chat tokens.

    Returns:
      token_ids: list of token integers
      loss_mask: list of 0/1 integers (1 for assistant tokens & eot, 0 otherwise)
    """
    token_ids: list[int] = []
    loss_mask: list[int] = []

    # Chat tokens dict
    special = tok.special_tokens

    for msg in messages:
        role = msg["role"]
        content = unicodedata.normalize("NFC", msg["content"])
        role_sentinel = f"<|{role}|>"

        # Encode role sentinel, content, and end sentinel
        s_id = special[role_sentinel] if role_sentinel in special else tok.encode(role_sentinel)
        if isinstance(s_id, list):
            s_ids = s_id
        else:
            s_ids = [s_id]

        c_ids = tok.encode(content)

        end_sentinel = "<|end|>"
        e_id = special[end_sentinel] if end_sentinel in special else tok.encode(end_sentinel)
        if isinstance(e_id, list):
            e_ids = e_id
        else:
            e_ids = [e_id]

        turn_tokens = s_ids + c_ids + e_ids

        if role == "assistant":
            # sentinel mask = 0; content & end-of-turn mask = 1
            turn_mask = [0] * len(s_ids) + [1] * (len(c_ids) + len(e_ids))
        else:
            # user and system tokens mask = 0
            turn_mask = [0] * len(turn_tokens)

        token_ids.extend(turn_tokens)
        loss_mask.extend(turn_mask)

    return token_ids, loss_mask


def build_sft_dataset(
    data_dir: str,
    out_dir: str,
    tokenizer_path: str,
    seq_len: int = 1024,
    val_frac: float = 0.05,
    seed: int = 42,
) -> dict:
    """Build SFT RDS dataset with parallel loss mask from data_dir to out_dir."""
    from dataset import load_bpe_tokenizer

    tok = load_bpe_tokenizer(tokenizer_path)
    register_chat_tokens(tok)

    data_path = Path(data_dir)
    if data_path.is_file():
        jsonl_files = [data_path]
    else:
        jsonl_files = sorted(list(data_path.rglob("*.jsonl")))

    if not jsonl_files:
        raise ValueError(f"No *.jsonl files found in {data_dir}")

    file_reports = {}
    all_chunks: list[tuple[list[int], list[int]]] = []

    pad_token_id = tok.special_tokens.get("<|end|>", 0)

    for fpath in jsonl_files:
        fname = fpath.name
        read_count = 0
        valid_count = 0
        malformed_count = 0
        total_tokens = 0

        current_tokens: list[int] = []
        current_masks: list[int] = []

        with open(fpath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                read_count += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    malformed_count += 1
                    continue

                valid, err = validate_jsonl_line(obj)
                if not valid:
                    malformed_count += 1
                    continue

                valid_count += 1
                t_ids, l_masks = process_messages(obj["messages"], tok)
                total_tokens += len(t_ids)

                current_tokens.extend(t_ids)
                current_masks.extend(l_masks)

                # Pack into seq_len chunks
                while len(current_tokens) >= seq_len:
                    chunk_toks = current_tokens[:seq_len]
                    chunk_masks = current_masks[:seq_len]
                    current_tokens = current_tokens[seq_len:]
                    current_masks = current_masks[seq_len:]
                    all_chunks.append((chunk_toks, chunk_masks))

        # Handle leftover padding for remaining tokens in file
        if current_tokens:
            rem = seq_len - len(current_tokens)
            chunk_toks = current_tokens + [pad_token_id] * rem
            chunk_masks = current_masks + [0] * rem
            all_chunks.append((chunk_toks, chunk_masks))

        file_reports[fname] = {
            "file": fname,
            "read": read_count,
            "valid": valid_count,
            "malformed": malformed_count,
            "tokens": total_tokens,
        }

    # Split into train and val chunks
    import random
    rng = random.Random(seed)
    n_total = len(all_chunks)
    n_val = max(1, int(n_total * val_frac)) if (n_total > 1 and val_frac > 0) else 0

    indices = list(range(n_total))
    rng.shuffle(indices)

    val_set_indices = set(indices[:n_val])
    train_chunks = [all_chunks[i] for i in range(n_total) if i not in val_set_indices]
    val_chunks = [all_chunks[i] for i in val_set_indices]

    if not train_chunks and val_chunks:
        train_chunks = val_chunks
        val_chunks = []

    # Function to write split
    def _write_split(split_name: str, chunks: list[tuple[list[int], list[int]]]):
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        if not chunks:
            return

        tok_writer = ShardWriter(
            os.path.join(split_dir, "shard_00000.rds"),
            version=1,
            tok_version=getattr(tok, "version", 1),
            vocab_size=tok.vocab_size,
            seq_len=seq_len,
            dtype_flag=0,  # uint16
        )
        mask_writer = ShardWriter(
            os.path.join(split_dir, "shard_00000_mask.rds"),
            version=1,
            tok_version=getattr(tok, "version", 1),
            vocab_size=tok.vocab_size,
            seq_len=seq_len,
            dtype_flag=0,  # uint16
        )

        for c_toks, c_masks in chunks:
            tok_writer.add_chunk(c_toks, {"kind": "sft_tokens"})
            mask_writer.add_chunk(c_masks, {"kind": "sft_mask"})

        tok_writer.close()
        mask_writer.close()

        manifest = {
            "format": "RDS",
            "rds_version": 1,
            "tokenizer_version": getattr(tok, "version", 1),
            "vocab_size": tok.vocab_size,
            "seq_len": seq_len,
            "dtype": "uint16",
            "n_shards": 1,
            "shards": [
                {
                    "file": "shard_00000.rds",
                    "mask_file": "shard_00000_mask.rds",
                    "chunks": len(chunks),
                }
            ],
            "stats": {
                "total_chunks": len(chunks),
                "total_tokens": len(chunks) * seq_len,
            },
        }
        with open(os.path.join(split_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    _write_split("train", train_chunks)
    _write_split("val", val_chunks)

    # Top-level manifest
    top_manifest = {
        "format": "SFT_RDS",
        "seq_len": seq_len,
        "vocab_size": tok.vocab_size,
        "train_dir": "train",
        "val_dir": "val",
        "file_reports": file_reports,
        "total_chunks": n_total,
        "train_chunks": len(train_chunks),
        "val_chunks": len(val_chunks),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(top_manifest, f, indent=2)

    return top_manifest
