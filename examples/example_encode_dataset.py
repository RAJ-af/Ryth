"""Example: build an RDS dataset from raw repos with the RDE pipeline.

Run:  python examples/example_encode_dataset.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import RDEConfig, RDEPipeline
from dataset.tokenizer_adapter import ByteTokenizer
from tests.sample_data import build_sample


def main():
    tmp = tempfile.mkdtemp(prefix="ryth_rde_")
    root = os.path.join(tmp, "raw_repos")
    out = os.path.join(tmp, "rds_out")
    build_sample(root)

    # Use the built-in byte-level tokenizer for a zero-dependency example.
    # For real training, pass load_bpe_tokenizer("tokenizer.json").
    tok = ByteTokenizer()
    cfg = RDEConfig(seq_len=64, vocab_size=tok.vocab_size, tokenizer_version=0)

    manifest = RDEPipeline(tok, cfg).run(root, out, verbose=True)
    print(f"\nwrote {manifest['n_shards']} shard(s) to {out}")
    print("artifacts:", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
