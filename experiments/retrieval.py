"""Small, exact kNN-LM experiment over explicit tokenized documents."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from model.config import RythConfig
from model.decoder import RythForCausalLM


def load_model(checkpoint, config_path=None):
    """Load a model/checkpoint.py checkpoint, or supply model config JSON explicitly."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = json.loads(Path(config_path).read_text()) if config_path else ckpt.get("metadata", {}).get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint lacks model metadata.config; pass --config model-config.json")
    model = RythForCausalLM(RythConfig(**config))
    model.load_state_dict(ckpt["model"])
    return model.eval()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(doc, vocab):
    t = torch.as_tensor(doc)
    if t.ndim != 1 or t.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("each document must be a one-dimensional integer token sequence")
    t = t.to(device="cpu", dtype=torch.long)
    if t.numel() and (t.min() < 0 or t.max() >= vocab):
        raise ValueError("token outside vocabulary")
    return t


def _fingerprint(tokens):
    return hashlib.sha256(tokens.numpy().tobytes()).hexdigest()


def _windows(tokens, length):
    # One-token overlap: every prediction is scored once, without crossing documents.
    for start in range(0, len(tokens) - 1, length):
        targets = tokens[start + 1:start + length + 1]
        yield tokens[start:start + len(targets)], targets


class KNNMemory:
    def __init__(self, keys, values, vocab_size, *, fingerprints=(), checkpoint_hash=None, datastore_hash=None):
        if keys.ndim != 2 or keys.shape[0] == 0 or keys.shape[1] == 0:
            raise ValueError("keys must be nonempty [entries, hidden]")
        if values.ndim != 1 or len(values) != len(keys) or vocab_size < 1:
            raise ValueError("values must match keys and vocabulary must be positive")
        if values.dtype not in (torch.int32, torch.int64) or values.min() < 0 or values.max() >= vocab_size:
            raise ValueError("invalid memory values")
        if not torch.isfinite(keys).all():
            raise ValueError("keys must be finite")
        self.keys = keys.detach().to(device="cpu", dtype=torch.float16).contiguous()
        self.values = values.detach().to(device="cpu", dtype=torch.int32).contiguous()
        self.vocab_size = int(vocab_size)
        self.fingerprints = tuple(fingerprints)
        self.checkpoint_hash = checkpoint_hash
        self.datastore_hash = datastore_hash
        if not torch.isfinite(self.keys).all():
            raise ValueError("keys overflow float16 storage")

    @classmethod
    @torch.no_grad()
    def build(cls, model, datastore_documents, *, max_entries=10000, seq_len=None, checkpoint_hash=None):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        length = model.config.max_seq_len if seq_len is None else seq_len
        if not 1 <= length <= model.config.max_seq_len:
            raise ValueError("seq_len must be within model context")
        keys, values, hashes = [], [], []
        count = 0
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            for doc in datastore_documents:
                t = _tokens(doc, model.config.vocab_size)
                hashes.append(_fingerprint(t))
                for x, y in _windows(t, length):
                    if count >= max_entries:
                        break
                    take = min(len(y), max_entries - count)
                    hidden, _ = model.decoder(x[:take].unsqueeze(0).to(device))
                    keys.append(hidden[0].cpu().to(torch.float16))
                    values.append(y[:take].to(torch.int32))
                    count += take
        finally:
            model.train(was_training)
        if not keys:
            raise ValueError("datastore needs at least one next-token pair")
        return cls(torch.cat(keys), torch.cat(values), model.config.vocab_size,
                   fingerprints=hashes, checkpoint_hash=checkpoint_hash)

    def save(self, path):
        torch.save(dict(keys=self.keys, values=self.values, vocab_size=self.vocab_size,
                        fingerprints=self.fingerprints, checkpoint_hash=self.checkpoint_hash,
                        datastore_hash=self.datastore_hash), path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, map_location="cpu", weights_only=True))

    def query(self, hidden, *, k=8, temperature=1.0, query_chunk=32, key_chunk=4096):
        """Squared-L2 exact search with bounded distance matrices; returns dense probabilities."""
        if hidden.ndim < 1 or hidden.shape[-1] != self.keys.shape[1]:
            raise ValueError("hidden dimension does not match memory")
        if k < 1 or temperature <= 0 or not math.isfinite(temperature) or query_chunk < 1 or key_chunk < 1:
            raise ValueError("k, temperature and chunk sizes must be positive")
        if not torch.isfinite(hidden).all():
            raise ValueError("hidden must be finite")
        shape = hidden.shape[:-1]
        flat = hidden.detach().reshape(-1, hidden.shape[-1]).to(device="cpu", dtype=torch.float32)
        result = torch.zeros((len(flat), self.vocab_size), dtype=torch.float32)
        k = min(k, len(self.keys))
        for start in range(0, len(flat), query_chunk):
            q = flat[start:start + query_chunk]
            best_d = torch.empty((len(q), 0))
            best_i = torch.empty((len(q), 0), dtype=torch.long)
            for offset in range(0, len(self.keys), key_chunk):
                keys = self.keys[offset:offset + key_chunk].float()
                distances = (q.square().sum(-1, keepdim=True) + keys.square().sum(-1)[None, :] - 2 * q @ keys.T).clamp_min_(0)
                indices = torch.arange(offset, offset + len(keys)).expand(len(q), -1)
                ds = torch.cat((best_d, distances), dim=1)
                ix = torch.cat((best_i, indices), dim=1)
                best_d, pos = ds.topk(min(k, ds.shape[1]), largest=False, dim=1)
                best_i = ix.gather(1, pos)
            weights = torch.softmax(-best_d / temperature, dim=-1)
            result[start:start + len(q)].scatter_add_(1, self.values[best_i].long(), weights)
        return result.reshape(*shape, self.vocab_size).to(hidden.device)

    def mix(self, logits, hidden, *, weight=0.25, k=8, temperature=1.0):
        if not 0 <= weight <= 1 or logits.shape[:-1] != hidden.shape[:-1] or logits.shape[-1] != self.vocab_size:
            raise ValueError("invalid weight or incompatible logits/hidden shape")
        model_probs = torch.softmax(logits.float(), dim=-1)
        if weight == 0:
            return model_probs
        return (1 - weight) * model_probs + weight * self.query(hidden, k=k, temperature=temperature)


@torch.no_grad()
def evaluate(model, memory, eval_documents, *, seq_len=None, weight=0.25, k=8, temperature=1.0):
    """Score separate documents; reject exact documents used to build the datastore."""
    length = model.config.max_seq_len if seq_len is None else seq_len
    if not 1 <= length <= model.config.max_seq_len:
        raise ValueError("seq_len must be within model context")
    if memory.vocab_size != model.config.vocab_size:
        raise ValueError("memory/model vocabulary mismatch")
    if not 0 <= weight <= 1 or k < 1 or temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("invalid interpolation parameters")
    model_nll = mixed_nll = 0.0
    count = 0
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for doc in eval_documents:
            t = _tokens(doc, model.config.vocab_size)
            if _fingerprint(t) in memory.fingerprints:
                raise ValueError("evaluation document matches a datastore document")
            for x, y in _windows(t, length):
                hidden, _ = model.decoder(x.unsqueeze(0).to(device))
                logits = model.lm_head(hidden)
                target = y.to(device).reshape(1, -1, 1)
                logp = torch.log_softmax(logits.float(), dim=-1)
                model_nll -= logp.gather(-1, target).sum().item()
                target_logp = logp.gather(-1, target)
                if weight == 0:
                    mixed_logp = target_logp
                else:
                    knn_logp = memory.query(hidden, k=k, temperature=temperature).gather(-1, target).log()
                    if weight == 1:
                        mixed_logp = knn_logp
                    else:
                        mixed_logp = torch.logaddexp(target_logp + math.log1p(-weight), knn_logp + math.log(weight))
                mixed_nll -= mixed_logp.sum().item()
                count += len(y)
    finally:
        model.train(was_training)
    if not count:
        raise ValueError("evaluation needs at least one next-token pair")
    def ppl(nll):
        mean = nll / count
        return math.exp(mean) if mean < 709 else float("inf")
    return {"tokens": count, "model_ppl": ppl(model_nll), "knn_ppl": ppl(mixed_nll)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", help="model config JSON if checkpoint has no model config")
    p.add_argument("--datastore", required=True, help="train documents JSON: list of token-ID lists")
    p.add_argument("--eval", required=True, help="held-out documents JSON")
    p.add_argument("--memory", required=True, help="memory file to write, or read with --load-memory")
    p.add_argument("--load-memory", action="store_true")
    p.add_argument("--max-entries", type=int, default=10000)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--weight", type=float, default=0.25)
    args = p.parse_args()
    if Path(args.datastore).resolve() == Path(args.eval).resolve() or os.path.samefile(args.datastore, args.eval):
        p.error("datastore and eval must be separate files")
    model = load_model(args.checkpoint, args.config)
    digest = file_hash(args.checkpoint)
    train = json.loads(Path(args.datastore).read_text())
    heldout = json.loads(Path(args.eval).read_text())
    if args.load_memory:
        memory = KNNMemory.load(args.memory)
        if memory.checkpoint_hash != digest or memory.datastore_hash != file_hash(args.datastore):
            p.error("memory was built with a different checkpoint or datastore file")
    else:
        memory = KNNMemory.build(model, train, max_entries=args.max_entries, seq_len=args.seq_len, checkpoint_hash=digest)
        memory.datastore_hash = file_hash(args.datastore)
        memory.save(args.memory)
    print(json.dumps(evaluate(model, memory, heldout, seq_len=args.seq_len, weight=args.weight, k=args.k, temperature=args.temperature), indent=2))


if __name__ == "__main__":
    main()
