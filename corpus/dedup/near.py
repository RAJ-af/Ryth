"""Near-duplicate detection — MinHash + LSH banding (pure Python).

Code me exact dups ke alawa near-dups bhi hote hain (whitespace/rename tweaks).
MinHash signatures se Jaccard similarity estimate karte hain, aur LSH banding se
candidate pairs efficiently nikaalte hain (O(n) buckets, not O(n^2)).

Deterministic: permutations index se derive hoti hain (koi RNG nahi), shingles
blake2b se hash hote hain.
"""

from __future__ import annotations

import hashlib

_PRIME = (1 << 61) - 1        # Mersenne prime for universal hashing


def _shingles(text: str, k: int = 5) -> set:
    text = " ".join(text.split())
    if len(text) <= k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def _base_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
                          "big")


def minhash_signature(text: str, num_perms: int = 64) -> tuple:
    """Return an int tuple of length num_perms (MinHash signature)."""
    sh = _shingles(text)
    if not sh:
        return tuple([0] * num_perms)
    hashes = [_base_hash(s) for s in sh]
    sig = []
    for i in range(num_perms):
        a = 2 * i + 1          # odd multiplier, deterministic
        b = (i * i + 7)
        sig.append(min(((a * h + b) % _PRIME) for h in hashes))
    return tuple(sig)


def jaccard_estimate(sig_a: tuple, sig_b: tuple) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    same = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return same / len(sig_a)


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller root wins
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


class NearDeduper:
    """MinHash + LSH near-duplicate grouping over (id, text) items."""

    def __init__(self, num_perms: int = 64, bands: int = 16, threshold: float = 0.85):
        assert num_perms % bands == 0, "num_perms must divide by bands"
        self.num_perms = num_perms
        self.bands = bands
        self.rows = num_perms // bands
        self.threshold = threshold
        self._sigs: dict = {}

    def add(self, item_id, text: str) -> None:
        self._sigs[item_id] = minhash_signature(text, self.num_perms)

    def _band_key(self, sig, band):
        seg = sig[band * self.rows:(band + 1) * self.rows]
        return (band, hashlib.blake2b(repr(seg).encode(), digest_size=8).hexdigest())

    def duplicate_groups(self) -> list:
        """Return groups (lists of ids) that are mutual near-duplicates.

        Only groups with >1 member are returned. Representative = min id."""
        buckets: dict = {}
        for iid, sig in self._sigs.items():
            for band in range(self.bands):
                buckets.setdefault(self._band_key(sig, band), []).append(iid)

        uf = _UnionFind()
        for iid in self._sigs:
            uf.find(iid)
        for members in buckets.values():
            if len(members) < 2:
                continue
            base = members[0]
            base_sig = self._sigs[base]
            for other in members[1:]:
                if jaccard_estimate(base_sig, self._sigs[other]) >= self.threshold:
                    uf.union(base, other)

        groups: dict = {}
        for iid in self._sigs:
            groups.setdefault(uf.find(iid), []).append(iid)
        return [sorted(g) for g in groups.values() if len(g) > 1]

    def redundant_ids(self) -> set:
        """Ids to drop: every group member except the representative (min)."""
        drop = set()
        for group in self.duplicate_groups():
            drop.update(group[1:])       # keep group[0]
        return drop


def dedupe_near(records: list, num_perms=64, bands=16, threshold=0.85) -> tuple:
    """Return (kept, dropped) using near-duplicate grouping over record content."""
    nd = NearDeduper(num_perms, bands, threshold)
    for i, r in enumerate(records):
        nd.add(i, r.content or "")
    drop_idx = nd.redundant_ids()
    kept, dropped = [], []
    for i, r in enumerate(records):
        if i in drop_idx:
            r.drop_reason = "near_duplicate"
            dropped.append(r)
        else:
            kept.append(r)
    return kept, dropped
