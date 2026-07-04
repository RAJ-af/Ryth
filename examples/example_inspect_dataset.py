"""Example: inspect an RDS dataset — chunks, metadata, random access.

Run:  python examples/example_inspect_dataset.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import RDEConfig, RDEPipeline, RDSDataset
from dataset.tokenizer_adapter import ByteTokenizer
from tests.sample_data import build_sample


def main():
    tmp = tempfile.mkdtemp(prefix="ryth_inspect_")
    root, out = os.path.join(tmp, "raw_repos"), os.path.join(tmp, "rds_out")
    build_sample(root)

    tok = ByteTokenizer()
    cfg = RDEConfig(seq_len=64, vocab_size=tok.vocab_size, tokenizer_version=0)
    RDEPipeline(tok, cfg).run(root, out, verbose=False)

    ds = RDSDataset(out)
    print(f"total chunks: {len(ds)} across {ds.manifest['n_shards']} shard(s)")

    # random access into the middle of the dataset (O(1), memory-mapped)
    idx = len(ds) // 2
    chunk = ds[idx]
    meta = ds.meta(idx)
    print(f"\nchunk[{idx}]: {len(chunk)} tokens")
    print(f"  repo={meta['repo']} language={meta['language']} "
          f"quality={meta['quality']} difficulty={meta['difficulty']}")
    print(f"  path={meta['path']}")
    print(f"  first ids: {list(chunk[:16])}")

    # decode it back to text with the byte tokenizer
    print("\ndecoded preview:")
    print("-" * 40)
    print(tok.decode(list(chunk))[:200])
    ds.close()


if __name__ == "__main__":
    main()
