"""RDE Text Pipeline — prose / plain-text dataset builder for RDS.

Supports processing plain-text (*.txt) datasets into RDS shards for training
and validation, bypassing code-specific cleaners/validators/quality filters.

Key features:
1. Input: *.txt files in input_dir where documents are separated by "<|endoftext|>".
2. Streaming/bounded-memory file processing.
3. Filtering: NFC normalization, drops docs shorter than 50 chars, exact-duplicate removal by SHA-256 hash.
4. Source mixing: --mix hi=1.0,en=1.0,stories=0.3 upsamples/downsamples by file stem (e.g. `hi.txt` -> stem `hi`).
5. Split by document hash into train and validation directories.
6. Encodes docs, appends eot_id between documents, packs into seq_len chunks, and writes RDS shards.
7. Manifest written with token counts per source.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import unicodedata

from .config import RDEConfig
from .lock import build_lock
from .sharding import ShardManager
from .stats import StatsCollector


class TextPipeline:
    def __init__(self, tokenizer, config: RDEConfig | None = None,
                 val_frac: float = 0.01, mix: dict[str, float] | None = None):
        self.tok = tokenizer
        self.cfg = config or RDEConfig()
        self.val_frac = float(val_frac)
        self.mix = mix or {}

        # Determine endoftext token id
        if hasattr(self.tok, "eos_id"):
            self.eot_id = self.tok.eos_id
        elif hasattr(self.tok, "special_tokens") and "<|endoftext|>" in self.tok.special_tokens:
            self.eot_id = self.tok.special_tokens["<|endoftext|>"]
        else:
            self.eot_id = 0

    def _get_stem(self, filepath: str) -> str:
        base = os.path.basename(filepath)
        stem, _ = os.path.splitext(base)
        return stem

    def run(self, input_dir: str, out_dir: str, verbose: bool = True,
            now_iso: str | None = None, debug: bool = False) -> dict:
        if not os.path.isdir(input_dir):
            raise RuntimeError(f"input path is not a directory: {input_dir!r}")

        # Gather *.txt files recursively
        txt_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True))
        if not txt_files:
            txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
            txt_files = sorted(list(set(txt_files)))

        if not txt_files:
            raise RuntimeError(f"No .txt files found in input directory: {input_dir!r}")

        # Setup output directories for train and val
        has_val = self.val_frac > 0.0
        train_out = os.path.join(out_dir, "train") if has_val else out_dir
        val_out = os.path.join(out_dir, "val") if has_val else None

        train_shards = ShardManager(train_out, self.cfg, self.tok)
        val_shards = ShardManager(val_out, self.cfg, self.tok) if val_out else None

        train_stats = StatsCollector(self.cfg.seq_len, self.cfg.token_itemsize)
        val_stats = StatsCollector(self.cfg.seq_len, self.cfg.token_itemsize)

        seen_doc_hashes = set()

        # Buffers for chunk packing
        train_buf: list[int] = []
        train_buf_meta: list[dict] = []

        val_buf: list[int] = []
        val_buf_meta: list[dict] = []

        tokens_per_source_train: dict[str, int] = {}
        tokens_per_source_val: dict[str, int] = {}

        seq_len = self.cfg.seq_len
        pad_id = getattr(self.tok, "pad_id", 0)

        def flush_chunks(buf: list[int], buf_meta: list[dict], shards: ShardManager,
                         stats: StatsCollector, is_final: bool = False):
            while len(buf) >= seq_len:
                chunk_ids = buf[:seq_len]
                chunk_meta = dict(buf_meta[0]) if buf_meta else {}
                shards.add_chunk(chunk_ids, chunk_meta)
                stats.total_chunks += 1
                stats.total_tokens += seq_len
                del buf[:seq_len]
                del buf_meta[:seq_len]

            if is_final and buf:
                meta = dict(buf_meta[0]) if buf_meta else {}
                meta["padded"] = seq_len - len(buf)
                chunk_ids = buf + [pad_id] * (seq_len - len(buf))
                shards.add_chunk(chunk_ids, meta)
                stats.total_chunks += 1
                stats.total_tokens += len(buf)  # non-pad tokens count
                buf.clear()
                buf_meta.clear()

        # Process file by file (streaming, bounded memory)
        for filepath in txt_files:
            stem = self._get_stem(filepath)
            ratio = self.mix.get(stem, 1.0)
            if ratio <= 0:
                continue

            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            raw_docs = content.split("<|endoftext|>")
            for raw_doc in raw_docs:
                # 1. NFC normalization
                doc = unicodedata.normalize("NFC", raw_doc)

                # 2. Filter: drop docs shorter than 50 chars
                if len(doc) < 50:
                    continue

                # 3. Exact-duplicate removal by hash
                doc_bytes = doc.encode("utf-8")
                h_obj = hashlib.sha256(doc_bytes)
                h_hex = h_obj.hexdigest()
                if h_hex in seen_doc_hashes:
                    continue
                seen_doc_hashes.add(h_hex)

                # 4. Hash-based split into train / val
                # Convert first 8 bytes of hash to uint64 for deterministic ratio
                h_val = int(h_hex[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
                is_val = has_val and (h_val < self.val_frac)

                # 5. Encoding
                doc_tokens = self.tok.encode(doc)
                if not doc_tokens:
                    continue

                # 6. Apply --mix upsampling/downsampling
                base_repeats = int(math.floor(ratio))
                frac = ratio - base_repeats
                if frac > 0:
                    frac_val = (int(h_hex[16:32], 16) / float(0xFFFFFFFFFFFFFFFF))
                    if frac_val < frac:
                        repeats = base_repeats + 1
                    else:
                        repeats = base_repeats
                else:
                    repeats = base_repeats

                if repeats <= 0:
                    continue

                full_doc_tokens = doc_tokens + [self.eot_id]
                n_toks = len(full_doc_tokens) * repeats

                doc_meta = {
                    "source": stem,
                    "file": os.path.basename(filepath),
                    "hash": h_hex[:12],
                }

                if is_val:
                    tokens_per_source_val[stem] = tokens_per_source_val.get(stem, 0) + n_toks
                    val_stats.total_files_seen += repeats
                    val_stats.files_kept += repeats
                    for _ in range(repeats):
                        val_buf.extend(full_doc_tokens)
                        val_buf_meta.extend([doc_meta] * len(full_doc_tokens))
                    flush_chunks(val_buf, val_buf_meta, val_shards, val_stats, is_final=False)
                else:
                    tokens_per_source_train[stem] = tokens_per_source_train.get(stem, 0) + n_toks
                    train_stats.total_files_seen += repeats
                    train_stats.files_kept += repeats
                    for _ in range(repeats):
                        train_buf.extend(full_doc_tokens)
                        train_buf_meta.extend([doc_meta] * len(full_doc_tokens))
                    flush_chunks(train_buf, train_buf_meta, train_shards, train_stats, is_final=False)

        # Flush remainders
        flush_chunks(train_buf, train_buf_meta, train_shards, train_stats, is_final=True)
        if val_shards and val_out:
            flush_chunks(val_buf, val_buf_meta, val_shards, val_stats, is_final=True)

        # Check if train wrote any chunks
        if train_stats.total_chunks == 0:
            raise RuntimeError("No usable documents found or 0 chunks produced from text input.")

        # Build locks and finalize
        lock_train = build_lock(self.cfg, self.tok,
                                dataset_version=self.cfg.dataset_version,
                                model_version=self.cfg.model_version,
                                source_root=input_dir, now_iso=now_iso)
        train_summary = train_stats.summary()
        train_summary["tokens_per_source"] = tokens_per_source_train
        manifest_train = train_shards.finalize(train_summary, lock=lock_train)

        if val_shards and val_out:
            lock_val = build_lock(self.cfg, self.tok,
                                  dataset_version=self.cfg.dataset_version,
                                  model_version=self.cfg.model_version,
                                  source_root=input_dir, now_iso=now_iso)
            val_summary = val_stats.summary()
            val_summary["tokens_per_source"] = tokens_per_source_val
            manifest_val = val_shards.finalize(val_summary, lock=lock_val)
        else:
            manifest_val = None

        res = {"train": manifest_train}
        if manifest_val:
            res["val"] = manifest_val

        if verbose:
            print(f"TextPipeline complete: {train_stats.total_chunks} train chunks written.")
            if manifest_val:
                print(f"  {val_stats.total_chunks} val chunks written.")

        return res
