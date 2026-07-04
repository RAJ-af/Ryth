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

### Pre-tokenization

Before counting pairs, text is split with a GPT-2-style regular expression so
merges never cross word/space boundaries (which produces cleaner tokens). The
default pattern is ASCII-oriented (ideal for code); for full-Unicode word
splitting you can install `regex` and swap the pattern (see the note in
`bpe.py`).

## BPE training

```python
from tokenizer import BPETokenizer, DEFAULT_SPECIAL_TOKENS

texts = ["def add(a, b):\n    return a + b", "class Foo: pass", ...]

tok = BPETokenizer()
tok.train(texts, vocab_size=8000, verbose=True)
tok.add_special_tokens(DEFAULT_SPECIAL_TOKENS)
```

- `train(texts, vocab_size, verbose=False)` learns merges until `vocab_size` is
  reached (or until no pair repeats). `vocab_size` must be ≥ 256 (the base bytes).
- Training is deterministic for a given corpus.

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

Writes a JSON file containing the merge list (`[[a, b, new_id], ...]`) and the
special-token table. The base 256-byte vocab is implicit and rebuilt on load.

## Load

```python
tok = BPETokenizer.load("tok/tokenizer.json")
```

Rebuilds the vocab by replaying merges in order, then restores special tokens.

## Special tokens

`add_special_tokens([...])` registers strings at ids **above** the learned vocab.
They are treated atomically during encoding/decoding and never split. The default
set (`DEFAULT_SPECIAL_TOKENS`) includes:

```
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
ryth-tokenizer train  --files 'src/**/*.py' --vocab 8000 --out tok
ryth-tokenizer train  --data  jsonl_dir     --vocab 8000 --out tok
ryth-tokenizer encode --tokenizer tok/tokenizer.json --text "def add"
ryth-tokenizer decode --tokenizer tok/tokenizer.json --ids "100 101 102"
```

## Example

See [`examples/example_train_tokenizer.py`](../examples/example_train_tokenizer.py)
and [`examples/example_decode_tokens.py`](../examples/example_decode_tokens.py).
