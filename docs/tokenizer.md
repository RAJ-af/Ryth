# Scratch Byte-Level BPE Tokenizer

A Byte-Pair Encoding (BPE) tokenizer written from scratch in pure Python — the
same family of algorithm used by GPT-2 / Code Llama / StarCoder, kept small and
readable. Located in `tokenizer/bpe.py` (`BPETokenizer`).

## Architecture

BPE turns text into a sequence of integer token ids in four conceptual steps:

1. **Byte-level start.** Text is encoded to UTF-8 **bytes**, so every input maps
   to tokens in the range `0..255` to begin with. This is the **byte fallback**:
   any character in any language is representable, so there is **no "unknown"
   token**.
2. **Count pairs.** Across the corpus, count how often each adjacent token pair
   occurs.
3. **Merge the most frequent pair** into a new single token, added to the vocab.
4. **Repeat** until the target vocabulary size is reached.

Frequent code patterns (e.g. `"de"+"f" → "def"`, `" re"+"turn"`) become single
tokens on their own.

### Pre-tokenization & Unicode Normalization

Before counting pairs, text is normalized to Unicode NFC form (applied in `train`
and `encode`). `decode()` does not normalize and joins raw byte tokens, ensuring that
`decode(encode(x)) == unicodedata.normalize("NFC", x)` for all inputs.

Pre-tokenization splits text with a GPT-2 style regular expression enhanced for
script awareness (e.g. Devanagari words including combining marks like matras, virama,
nukta, anusvara, and ZWJ/ZWNJ `\u200d`/`\u200c`, while excluding danda `।` and double
danda `॥` so punctuation splits separately).

## BPE training

```python
from tokenizer import BPETokenizer, DEFAULT_SPECIAL_TOKENS, build_special_tokens

texts = ["def add(a, b):\n    return a + b", "class Foo: pass", ...]

tok = BPETokenizer()
tok.train(texts, vocab_size=8000, verbose=True)
tok.add_special_tokens(build_special_tokens(reserved=256))
```

- `train(texts, vocab_size, verbose=False)` learns merges until `vocab_size` is
  reached (or until no pair repeats). `vocab_size` must be ≥ 256 (the base bytes).
- Training is deterministic for a given corpus.

## Vocabulary Extension (`extend`)

Existing tokenizers can be extended with new domain/language data without changing
any existing token IDs or merges:

```python
tok_v2 = BPETokenizer.load("tok/tokenizer.json")
tok_v2.extend(new_texts, add_vocab=1000, verbose=True)
tok_v2.save("tok_v2/tokenizer.json")
```

- All existing merges, special token IDs, and base vocabulary remain identical.
- New merges learned from `new_texts` are assigned contiguous IDs starting above the maximum existing ID.
- `version` increments automatically and `sha256` hash is recomputed.

## Encoding

```python
ids = tok.encode("def add(a, b): return a + b")
# -> [ ... token ids ... ]
```

- Splits on special tokens first (they are never broken apart), then applies the
  learned merges to each piece.

## Decoding

```python
text = tok.decode(ids)   # exact round-trip for in-vocab ids
```

- Concatenates the byte sequences behind each token and UTF-8 decodes them
  (`errors="replace"` guards against partial byte sequences).

## Save

```python
tok.save("tok/tokenizer.json")
```

Writes a JSON file containing `version` (v2), `sha256` checksum, `merges` (`[[a, b, new_id], ...]`),
and `special_tokens`. The base 256-byte vocab is implicit and rebuilt on load.

## Load

```python
tok = BPETokenizer.load("tok/tokenizer.json")
```

Rebuilds the vocab by replaying merges in order, restores special tokens, and verifies the `sha256` checksum
when present. Old v1 tokenizer JSON files (without `sha256` or `version`) load seamlessly for backward compatibility.

## Special tokens & Reserved slots

`add_special_tokens([...])` registers strings at ids **above** the learned vocab.
By default (`--reserved 256`), `build_special_tokens(reserved=256)` reserves 256 slot tokens
(`<|reserved_0|>` to `<|reserved_255|>`) followed by the chat and FIM special tokens (`DEFAULT_SPECIAL_TOKENS`):

```
<|reserved_0|> ... <|reserved_255|>
<|endoftext|> <|pad|>
<|fim_prefix|> <|fim_suffix|> <|fim_middle|>      # fill-in-the-middle
<|system|> <|user|> <|assistant|> <|end|>          # chat
```

The FIM sentinels match those produced by the RDE **FIM Builder**, so the two
components line up.

## Vocabulary

- `tok.vocab_size` → total tokens (base bytes + learned merges + special tokens).
- `tok.merges` → the learned `(pair) → id` mapping.
- `tok.vocab` → `id → bytes` for every non-special token.

For `vocab_size ≤ 65536`, RDE stores token ids as `uint16` (2 bytes each).

## Byte fallback

Because the base vocabulary is the full 256 bytes, **any** text encodes and
decodes losslessly, even scripts the tokenizer never saw during training (e.g.
Hindi, Tamil, Urdu). Such text simply uses more tokens (less compression) until
you include it in the training corpus. There is never an unknown token.

## CLI

```bash
ryth-tokenizer train  --files 'src/**/*.py' --vocab 8000 --reserved 256 --out tok
ryth-tokenizer extend --base tok/tokenizer.json --files 'hindi/*.txt' --add_vocab 1000 --out tok_v2
ryth-tokenizer encode --tokenizer tok/tokenizer.json --text "def add"
ryth-tokenizer decode --tokenizer tok/tokenizer.json --ids "100 101 102"
```

## Example

See [`examples/example_train_tokenizer.py`](../examples/example_train_tokenizer.py)
and [`examples/example_decode_tokens.py`](../examples/example_decode_tokens.py).
