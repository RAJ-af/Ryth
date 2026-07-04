# RDS Binary Format

**RDS** (Ryth DataSet) is the compact, self-describing binary shard format RDE
writes. It is designed for fast, memory-mapped, random-access reading of token
data on modest hardware. Implemented in `dataset/rds.py`.

## File layout

A single shard file (`shard_00000.rds`, …) has five sections, little-endian:

```
┌────────────────────────────────────────────────────────────────┐
│ HEADER (64 bytes, fixed)                                        │
│   magic          "RDS1"        (4 bytes)                        │
│   version        uint16        format version (dispatch key)    │
│   tok_version    uint16        tokenizer version                │
│   vocab_size     uint32                                         │
│   seq_len        uint32        target chunk length              │
│   dtype_flag     uint8         0 = uint16, 1 = uint32           │
│   flags          uint8         reserved                         │
│   n_chunks       uint32                                         │
│   data_offset    uint64                                         │
│   index_offset   uint64                                         │
│   meta_offset    uint64                                         │
│   footer_offset  uint64                                         │
│   (padded to 64 bytes)                                          │
├────────────────────────────────────────────────────────────────┤
│ DATA                                                            │
│   all chunks' token ids concatenated (uint16 or uint32)        │
├────────────────────────────────────────────────────────────────┤
│ INDEX                                                           │
│   n_chunks × ( offset: uint64, length_in_tokens: uint32 )      │
├────────────────────────────────────────────────────────────────┤
│ METADATA                                                        │
│   uint64 length + UTF-8 JSON (per-chunk metadata list)         │
├────────────────────────────────────────────────────────────────┤
│ FOOTER                                                          │
│   sha256 digest of bytes[0 : footer_offset]  (32 bytes)        │
│   magic "RDSE"  (4 bytes)                                       │
└────────────────────────────────────────────────────────────────┘
```

The header stores every section's byte offset, so a reader can `mmap` the file
and jump straight to any chunk without scanning.

## Why these choices

| Choice | Reason |
|--------|--------|
| **uint16 tokens** | Vocab ≤ 65536 fits in 2 bytes → **half the storage & I/O** vs uint32. `dtype_flag` selects uint32 for larger vocabularies. |
| **Index section** | `(offset, length)` per chunk → **O(1) random access** (Module 16). |
| **JSON metadata** | Per-chunk repo/language/quality/difficulty/path — human-readable, forward-compatible. |
| **sha256 footer** | **Corruption detection** — `verify_checksum()` recomputes and compares. |
| **version field** | **Backward compatibility** — see below. |

## Reading

```python
from dataset import RDSReader

with RDSReader("rds_out/shard_00000.rds") as r:
    print(len(r))                 # number of chunks
    ids = r[0]                    # array('H') of token ids (mmap, lazy)
    meta = r.metadata(0)          # dict for chunk 0
    assert r.verify_checksum()    # integrity check
```

For training you normally use `RDSDataset`, which spans all shards in a manifest
folder and adds deterministic shuffling + multi-worker partitioning. See
[dataset_engine.md](dataset_engine.md#memory-mapping--streaming-datasetpy).

## Manifest

Alongside the shards, RDE writes `manifest.json`:

```jsonc
{
  "format": "RDS",
  "rds_version": 1,
  "tokenizer_version": 1,
  "vocab_size": 8000,
  "seq_len": 1024,
  "dtype": "uint16",
  "n_shards": 3,
  "shards": [
    { "file": "shard_00000.rds", "chunks": 5, "tokens": 5120, "checksum_ok": true },
    ...
  ],
  "lock": { /* Manifest Lock — reproducibility metadata */ },
  "stats": { /* statistics incl. packing efficiency */ }
}
```

## Versioning & backward compatibility

The reader parses `magic` + `version`, then dispatches to a version-specific
header parser via the `_PARSERS` registry:

```python
_PARSERS = { 1: RDSReader._parse_v1 }
SUPPORTED_VERSIONS = (1,)
```

To introduce a new layout (v2, v3), add a `_parse_v2` and register it. Old shards
keep loading via their own parser; a reader that encounters a **newer** version
than it supports raises a clear error asking you to upgrade. **We do not assume
the format will never change — we make change safe.**
