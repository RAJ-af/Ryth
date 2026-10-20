"""Fast BPE trainer — wahi EXACT algorithm jaise `bpe.BPETokenizer.train`,
par incremental pair-counts + lazy heap ke saath (pure Python, koi dependency nahi).

Naive trainer har merge par POORA corpus rescan karta hai -> O(merges x corpus).
Ye implementation:
  * identical regex-pieces ko EK `_Chunk` me dedup karke multiplicity count karti hai
  * optional multiprocessing se pre-tokenization / word-counting parallel karti hai
  * global pair-counts sirf TOUCHED chunks se update hote hain
  * max-count pair lazy heap se nikalti hai (stale entries pop-par discard)

Tie-breaking: naive `max(counts, key=counts.get)` insertion-ordered dict par
chalta hai — matlab equal-count pairs me jo corpus-scan me SABSE PEHLE aaya wo
jeeta. Ye implementation exactly wahi chunti hai: tie-group ke har pair ka
(earliest chunk idx, earliest offset) nikal kar minimum leti hai. Chunk indices
first-occurrence order me assigned hote hain, isliye min-idx = scan-order.
Result: merge sequence naive ke BIT-IDENTICAL (tests/test_fast_bpe.py proof).

Merge hue pairs kabhi dobara nahi ban sakte (coalescing se naya adjacency
sirf naye composite-id se banta hai) — isliye counts.pop(pair) permanent
staleness deta hai, heap entries khud-ba-khud discard ho jaate hain.

Memory note: `where` (pair -> chunk-index set) har distinct pair ka set rakhta
hai — 30MB-class samples par ~1GB ke aas-paas. Production-scale sample chunte
waqt iska dhyan rakho (docs/w1_vocab_decision.md me measured numbers).
"""

from __future__ import annotations

import heapq
import multiprocessing
import os
from typing import Iterable

from tokenizer.bpe import _SPLIT_PATTERN, BPETokenizer


class _Chunk:
    __slots__ = ("ids", "count")

    def __init__(self, ids: list[int], count: int):
        self.ids = ids            # current token-id list (mutate hota hai)
        self.count = count        # expanded-corpus multiplicity


def cap_texts_by_bytes(texts: Iterable[str], max_bytes: int | None = None) -> list[str]:
    """Truncate texts so that total size in UTF-8 bytes does not exceed max_bytes."""
    if max_bytes is None or max_bytes <= 0:
        return list(texts)
    out = []
    curr_bytes = 0
    for text in texts:
        b = text.encode("utf-8")
        if curr_bytes + len(b) <= max_bytes:
            out.append(text)
            curr_bytes += len(b)
        else:
            rem = max_bytes - curr_bytes
            if rem > 0:
                sliced_b = b[:rem]
                sliced_text = sliced_b.decode("utf-8", errors="ignore")
                if sliced_text:
                    out.append(sliced_text)
            break
    return out


def _process_text_batch(batch: list[str]) -> tuple[dict[bytes, int], list[bytes]]:
    counts: dict[bytes, int] = {}
    first_seen: list[bytes] = []
    for text in batch:
        for piece in _SPLIT_PATTERN.findall(text):
            b = piece.encode("utf-8")
            if not b:
                continue
            c = counts.get(b, 0)
            if c == 0:
                first_seen.append(b)
            counts[b] = c + 1
    return counts, first_seen


def _build_chunks(texts, workers: int | None = None) -> list[_Chunk]:
    """Regex pieces -> deduped chunks (first-occurrence order)."""
    if workers is None:
        workers = os.cpu_count() or 1

    if workers <= 1 or len(texts) <= 1:
        seen: dict[bytes, _Chunk] = {}
        chunks: list[_Chunk] = []
        for text in texts:
            for piece in _SPLIT_PATTERN.findall(text):
                b = piece.encode("utf-8")
                if not b:
                    continue
                ch = seen.get(b)
                if ch is None:
                    ch = _Chunk(list(b), 1)
                    seen[b] = ch
                    chunks.append(ch)
                else:
                    ch.count += 1
        return chunks

    n_batches = min(workers, len(texts))
    batch_size = (len(texts) + n_batches - 1) // n_batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    try:
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.map(_process_text_batch, batches)
    except Exception:
        # Fallback to single-process on environment restriction
        return _build_chunks(texts, workers=1)

    seen: dict[bytes, _Chunk] = {}
    chunks: list[_Chunk] = []
    for batch_counts, batch_first_seen in results:
        for b in batch_first_seen:
            ch = seen.get(b)
            if ch is None:
                ch = _Chunk(list(b), batch_counts[b])
                seen[b] = ch
                chunks.append(ch)
            else:
                ch.count += batch_counts[b]
    return chunks


def train_fast(texts, vocab_size, verbose=False, checkpoint_path=None,
               checkpoint_every=2000, workers=None, max_bytes=None):
    """BPETokenizer return karta hai — merges naive train() jaise EXACT.

    checkpoint_path diya ho to har `checkpoint_every` merges par partial
    tokenizer save hota hai.
    """
    assert vocab_size >= 256, "vocab_size kam se kam 256 hona chahiye"
    num_merges = vocab_size - 256

    if max_bytes is not None and max_bytes > 0:
        texts = cap_texts_by_bytes(texts, max_bytes)

    chunks = _build_chunks(texts, workers=workers)

    vocab = {i: bytes([i]) for i in range(256)}
    counts: dict[tuple[int, int], int] = {}
    where: dict[tuple[int, int], set[int]] = {}
    for ci, ch in enumerate(chunks):
        for a, b in zip(ch.ids, ch.ids[1:]):
            p = (a, b)
            counts[p] = counts.get(p, 0) + ch.count
            where.setdefault(p, set()).add(ci)

    # provisional heap key (seq); asli tie-break pop-par _first_pos se
    seq = 0
    heap: list[tuple[int, int, tuple[int, int]]] = []
    for p in counts:
        heap.append((-counts[p], seq, p))
        seq += 1
    heapq.heapify(heap)

    def _touch(p):
        """Har count-change ke baad fresh entry — warna naye/badle pairs
        heap se gayab reh jaate (init par sirf existing pairs the).
        heappush hi use karo — plain append heap-invariant tod deta."""
        c = counts.get(p, 0)
        if c:
            nonlocal seq
            heapq.heappush(heap, (-c, seq, p))
            seq += 1

    merges: dict[tuple[int, int], int] = {}

    def _first_pos(pair) -> tuple[int, int]:
        """(earliest chunk idx, offset) — naive scan-order tie-break key."""
        best = None
        for ci in where.get(pair, ()):
            ids = chunks[ci].ids
            for off in range(len(ids) - 1):
                if ids[off] == pair[0] and ids[off + 1] == pair[1]:
                    cand = (ci, off)
                    if best is None or cand < best:
                        best = cand
                    break
        return best if best is not None else (1 << 60, 0)

    def _choose():
        """Max-count pair; ties me corpus-order pehle wala (EXACT naive)."""
        nonlocal seq
        while heap:
            negc, _, p = heapq.heappop(heap)
            cur = counts.get(p, 0)
            if cur == 0 or -negc != cur:
                continue                       # stale / merged-away entry
            group = [p]
            while heap:
                nxt = heap[0]
                if nxt[0] == negc:             # same count — group member
                    _, _, q = heapq.heappop(heap)
                    cq = counts.get(q, 0)
                    if cq == cur:
                        group.append(q)
                    elif cq:
                        heapq.heappush(heap, (-cq, seq, q))
                        seq += 1
                elif counts.get(nxt[2], 0) != -nxt[0]:
                    heapq.heappop(heap)        # stale mid-range entry — skip,
                    continue                   # warna tie-group adhoora reh jata
                else:
                    break                      # genuine chhota count — rukh jao
            winner = min(group, key=_first_pos) if len(group) > 1 else group[0]
            for q in group:                    # losers valid-count ke saath wapas
                if q is not winner:
                    heapq.heappush(heap, (-cur, seq, q))
                    seq += 1
            return winner, cur
        return None, 0

    m = 0
    while m < num_merges:
        pair, c = _choose()
        if pair is None or c < 2:
            break                              # naive jaisa hi early-stop
        new_id = 256 + m
        touched = sorted(where.pop(pair, ()))

        for ci in touched:
            ch = chunks[ci]
            # is chunk ke purane pair-contributions hatao
            for a, b in zip(ch.ids, ch.ids[1:]):
                p = (a, b)
                nc = counts[p] - ch.count
                if nc:
                    counts[p] = nc
                    _touch(p)                  # ghata — fresh heap entry
                else:
                    del counts[p]
                s = where.get(p)
                if s is not None:
                    s.discard(ci)
                    if not s:
                        del where[p]
            # merge apply (left-to-right greedy, overlap i+=2 — naive jaisa)
            ids, out, i = ch.ids, [], 0
            L = len(ids)
            while i < L:
                if i < L - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ch.ids = out
            # naye pair-contributions jodo
            for a, b in zip(out, out[1:]):
                p = (a, b)
                counts[p] = counts.get(p, 0) + ch.count
                where.setdefault(p, set()).add(ci)
                _touch(p)                      # naya/badha pair -> heap me

        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

        m += 1
        if verbose and m % 100 == 0:
            print(f"  merge {m}/{num_merges}: {pair} -> {new_id} "
                  f"({vocab[new_id]!r}) x{c}", flush=True)
        if checkpoint_path and m % checkpoint_every == 0:
            part = BPETokenizer()
            part.merges = dict(merges)
            part.vocab = dict(vocab)
            part.save(checkpoint_path)

    tok = BPETokenizer()
    tok.merges = merges
    tok.vocab = vocab
    return tok
