"""Example: verify RDS shard checksums — and show corruption detection.

Run:  python examples/example_verify_dataset.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import RDEConfig, RDEPipeline, RDSDataset, RDSReader
from dataset.tokenizer_adapter import ByteTokenizer
from tests.sample_data import build_sample


def main():
    tmp = tempfile.mkdtemp(prefix="ryth_verify_")
    root, out = os.path.join(tmp, "raw_repos"), os.path.join(tmp, "rds_out")
    build_sample(root)

    tok = ByteTokenizer()
    cfg = RDEConfig(seq_len=64, vocab_size=tok.vocab_size, tokenizer_version=0)
    RDEPipeline(tok, cfg).run(root, out, verbose=False)

    ds = RDSDataset(out)
    print("all shard checksums valid:", ds.verify())
    ds.close()

    # Now corrupt one byte and show the checksum catches it.
    shard = os.path.join(out, "shard_00000.rds")
    with open(shard, "r+b") as f:
        f.seek(80)                                  # a byte inside the DATA section
        b = f.read(1)
        f.seek(80)
        f.write(bytes([b[0] ^ 0xFF]))               # flip it

    with RDSReader(shard) as r:
        print("after corrupting 1 byte, checksum valid:", r.verify_checksum())


if __name__ == "__main__":
    main()
