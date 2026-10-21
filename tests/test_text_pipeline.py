"""Unit tests for RDE Text Pipeline (ryth-rde build-text).
"""

import os
import shutil
import tempfile
import json
import pytest

from dataset import ByteTokenizer, RDSDataset, TextPipeline
from dataset.cli import main as cli_main


def _create_fake_corpus(dir_path):
    os.makedirs(dir_path, exist_ok=True)

    # Document 1 (> 50 chars)
    doc_en_1 = "This is a sufficiently long story in English to pass the 50 character length requirement easily."
    # Document 2 (> 50 chars)
    doc_en_2 = "Another English prose document for testing the RDE text pipeline mode with multiple docs."
    # Duplicate doc
    doc_en_dup = "This is a sufficiently long story in English to pass the 50 character length requirement easily."
    # Short doc (< 50 chars)
    doc_en_short = "Too short doc."

    content_en = f"{doc_en_1}\n<|endoftext|>\n{doc_en_2}\n<|endoftext|>\n{doc_en_dup}\n<|endoftext|>\n{doc_en_short}"

    with open(os.path.join(dir_path, "en.txt"), "w", encoding="utf-8") as f:
        f.write(content_en)

    # Hindi document
    doc_hi_1 = "यह हिंदी भाषा का एक परीक्षण दस्तावेज़ है जो पचास अक्षरों से अधिक लंबा है।"
    content_hi = f"{doc_hi_1}\n<|endoftext|>\n"
    with open(os.path.join(dir_path, "hi.txt"), "w", encoding="utf-8") as f:
        f.write(content_hi)


def test_text_pipeline_processing_and_decoding():
    tmp = tempfile.mkdtemp()
    try:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        _create_fake_corpus(in_dir)

        tok = ByteTokenizer()
        # seq_len small enough so chunks get built easily
        pipe = TextPipeline(tok, val_frac=0.0, mix={"en": 1.0, "hi": 1.0})
        res = pipe.run(in_dir, out_dir, verbose=False)

        assert "train" in res
        manifest = res["train"]
        assert manifest["stats"]["tokens_per_source"]["en"] > 0
        assert manifest["stats"]["tokens_per_source"]["hi"] > 0

        # Read dataset with RDSDataset
        ds = RDSDataset(out_dir)
        assert len(ds) > 0

        # Decode chunk 0 and check original text
        chunk0 = ds[0]
        decoded = tok.decode(list(chunk0))

        assert "sufficiently long story in English" in decoded or "परीक्षण दस्तावेज़" in decoded
        ds.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_text_pipeline_cli_and_splits():
    tmp = tempfile.mkdtemp()
    try:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        _create_fake_corpus(in_dir)

        # Run via CLI main
        argv = ["build-text", in_dir, out_dir, "--seq_len", "64", "--val_frac", "0.5", "--mix", "en=1.0,hi=2.0"]
        try:
            cli_main(argv)
        except SystemExit as e:
            assert e.code == 0

        assert os.path.exists(os.path.join(out_dir, "train", "manifest.json"))
        assert os.path.exists(os.path.join(out_dir, "val", "manifest.json"))

        ds_train = RDSDataset(os.path.join(out_dir, "train"))
        assert len(ds_train) > 0
        ds_train.close()

        ds_val = RDSDataset(os.path.join(out_dir, "val"))
        assert len(ds_val) > 0
        ds_val.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
