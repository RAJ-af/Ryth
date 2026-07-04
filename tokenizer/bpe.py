"""Apna khud ka Byte-Pair Encoding (BPE) tokenizer — SCRATCH se, pure Python.

Koi external library nahi. Ye wahi algorithm hai jo GPT-2 / Code Llama / StarCoder
jaise models use karte hain, bas simple aur padhne-layak banaya gaya hai.

BPE ka idea (bahut simple):
  1. Text ko uske UTF-8 **bytes** me todo -> har byte ek token (0..255).
     (Byte-level hone se koi bhi character represent ho jata hai -> koi "unknown" nahi.)
  2. Poore data me dekho: kaunsa do-token ka **pair** sabse zyada baar aata hai.
  3. Us pair ko ek **naya single token** bana do (merge). Vocab me add karo.
  4. Step 2-3 ko baar baar dohrao jab tak vocab_size pura na ho jaye.
  Jo pairs (jaise "de"+"f" -> "def") aksar aate hain wo apne aap ek token ban jaate hain.
"""

from __future__ import annotations

import json
import re

# GPT-2 style pre-tokenization pattern (stdlib `re` version, ASCII-friendly — code
# ke liye kaafi hai). Ye text ko words/numbers/spaces me todta hai TAKI merges
# word ke aar-paar (jaise space ke across) na hon -> behtar tokens.
# Full-Unicode chahiye to `pip install regex` karke \p{L}\p{N} wala pattern use karo.
_SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def _get_pair_counts(ids, counts):
    """Ek token-list me har adjacent pair (a,b) kitni baar aaya — counts me add karo."""
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _merge(ids, pair, new_id):
    """`ids` me jahan jahan `pair` (do lagatar token) mile, use `new_id` se replace karo."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}   # (a,b) -> new_id  (learning order)
        self.vocab: dict[int, bytes] = {}              # id -> raw bytes
        self.special_tokens: dict[str, int] = {}       # "<|fim_prefix|>" -> id
        self._inv_special: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # TRAINING
    # ------------------------------------------------------------------ #
    def train(self, texts, vocab_size, verbose=False):
        """`texts` (strings ka iterable) se BPE merges seekho jab tak vocab_size na aaye."""
        assert vocab_size >= 256, "vocab_size kam se kam 256 (base bytes) hona chahiye"
        num_merges = vocab_size - 256

        # har text ko regex se pieces me todo, phir har piece ko bytes ki list me.
        # chunks = pieces ki list; har piece token-ids ki list (shuru me raw bytes).
        chunks = []
        for text in texts:
            for piece in _SPLIT_PATTERN.findall(text):
                b = piece.encode("utf-8")
                if b:
                    chunks.append(list(b))

        # base vocab: 256 bytes
        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}

        for m in range(num_merges):
            # 1) poore data me pair frequencies gino
            counts = {}
            for ids in chunks:
                _get_pair_counts(ids, counts)
            if not counts:
                break
            # 2) sabse common pair chuno
            pair = max(counts, key=counts.get)
            if counts[pair] < 2:
                break  # ab koi pair repeat nahi ho raha -> rukh jao
            # 3) naya token id do aur har jagah merge kar do
            new_id = 256 + m
            chunks = [_merge(ids, pair, new_id) for ids in chunks]
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (m + 1) % 100 == 0:
                print(f"  merge {m + 1}/{num_merges}: {pair} -> {new_id} "
                      f"({vocab[new_id]!r}) x{counts[pair]}")

        self.merges = merges
        self.vocab = vocab
        return self

    # ------------------------------------------------------------------ #
    # SPECIAL TOKENS  (FIM / chat tokens — inhe kabhi tode nahi jaana chahiye)
    # ------------------------------------------------------------------ #
    def add_special_tokens(self, tokens):
        """`tokens` (strings ki list) ko vocab ke aage naye ids par register karo."""
        next_id = (max(self.vocab) + 1) if self.vocab else 256
        next_id = max(next_id, (max(self._inv_special) + 1) if self._inv_special else 0)
        for t in tokens:
            if t in self.special_tokens:
                continue
            self.special_tokens[t] = next_id
            self._inv_special[next_id] = t
            next_id += 1
        return self

    # ------------------------------------------------------------------ #
    # ENCODE  (text -> token ids)
    # ------------------------------------------------------------------ #
    def _encode_piece(self, ids):
        """Ek byte-list par seekhe hue merges apply karo (sabse pehle wale merge pehle)."""
        while len(ids) >= 2:
            counts = _get_pair_counts(ids, {})
            # wo pair chuno jo sabse pehle seekha gaya tha (sabse chhoti merge-id)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # aur koi merge possible nahi
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        """String ko token ids me badlo. Special tokens ko tode bina alag rakhta hai."""
        if self.special_tokens:
            # special tokens ke around split karo (unhe as-is rakho)
            pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                for piece in _SPLIT_PATTERN.findall(part):
                    ids.extend(self._encode_piece(list(piece.encode("utf-8"))))
        return ids

    # ------------------------------------------------------------------ #
    # DECODE  (token ids -> text)
    # ------------------------------------------------------------------ #
    def decode(self, ids):
        parts = []
        for i in ids:
            if i in self._inv_special:
                parts.append(self._inv_special[i].encode("utf-8"))
            elif i in self.vocab:
                parts.append(self.vocab[i])
            else:
                raise ValueError(f"unknown token id: {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # SAVE / LOAD
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    def save(self, path):
        """Tokenizer ko ek JSON file me save karo (merges + special tokens)."""
        data = {
            "type": "bpe-byte-level",
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        # base vocab
        tok.vocab = {i: bytes([i]) for i in range(256)}
        # merges ko order me rebuild karo (vocab bhi saath saath ban jaata hai)
        merges_sorted = sorted(data["merges"], key=lambda x: x[2])
        for a, b, idx in merges_sorted:
            tok.merges[(a, b)] = idx
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        for t, idx in data["special_tokens"].items():
            tok.special_tokens[t] = idx
            tok._inv_special[idx] = t
        return tok
