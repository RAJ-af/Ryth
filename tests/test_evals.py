"""Unit tests for the evals package.

Chat-template + metrics + execution pure stdlib; generation/ppl need torch.
Run:  python -m pytest tests/test_evals.py -v
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from tokenizer.bpe import BPETokenizer
from evals.chat_template import (CHAT_TOKENS, extract_assistant,
                                 register_chat_tokens, render)


def _tok() -> BPETokenizer:
    """Tokenize karne layak fresh tokenizer (encode/decode roundtrip kaam kare)."""
    tok = BPETokenizer()
    tok.train(["hello world"], vocab_size=350, verbose=False)
    return tok


def test_register_and_render_roundtrip():
    tok = _tok()
    ids = register_chat_tokens(tok)
    assert set(ids) == {"<|system|>", "<|user|>", "<|assistant|>", "<|end|>"}
    # idempotent: dobara register karne par ids na badlein
    assert register_chat_tokens(tok) == ids
    msg = [{"role": "user", "content": "hello"}]
    s = render(msg, add_generation_prompt=True)
    assert s == "<|user|>hello<|end|><|assistant|>"
    # special tokens encode me intact rehne chahiye
    enc = tok.encode(s)
    assert "<|end|>" in tok.decode(enc)


def test_render_system_and_assistant_roles():
    msgs = [
        {"role": "system", "content": "You are Ryth."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert render(msgs) == ("<|system|>You are Ryth.<|end|>"
                            "<|user|>hi<|end|>"
                            "<|assistant|>hello<|end|>")


def test_extract_assistant_cuts_at_end():
    full = "<|user|>q<|end|><|assistant|>def f():\n    return 1<|end|>"
    assert extract_assistant(full) == "def f():\n    return 1"
    # bina <|end|> ke poora tail milta hai
    assert extract_assistant("x<|assistant|>abc") == "abc"
    assert extract_assistant("no markers here") == ""


# --------------------------------------------------------------------- #
# metrics — pass@k
# --------------------------------------------------------------------- #

from evals.metrics import aggregate, pass_at_k


def test_pass_at_k_known_values():
    assert pass_at_k(100, 100, 1) == 1.0          # sab correct
    assert pass_at_k(100, 0, 1) == 0.0            # kuch correct nahi
    assert abs(pass_at_k(4, 1, 1) - 0.25) < 1e-9   # c/n
    assert abs(pass_at_k(10, 5, 1) - 0.5) < 1e-9   # simple: 1-(n-c)/n
    # monotone: zyada k => zyada ya barabar chance
    assert pass_at_k(20, 5, 5) >= pass_at_k(20, 5, 1)


def test_aggregate_mean_over_tasks():
    res = [{"task_id": "a", "n": 10, "n_passed": 10},
           {"task_id": "b", "n": 10, "n_passed": 5}]
    out = aggregate(res, ks=(1, 2))
    assert abs(out["pass@1"] - 0.75) < 1e-9       # (1.0 + 0.5)/2
    assert out["pass@2"] >= out["pass@1"]


# --------------------------------------------------------------------- #
# execution — sandboxed subprocess runner
# --------------------------------------------------------------------- #

from evals.execution import run_program


def test_run_program_pass_fail_timeout_syntax():
    r = run_program("print('PASS')\n")
    assert r.ok and r.stdout.strip() == "PASS" and not r.timed_out
    r = run_program("raise AssertionError('boom')\n")
    assert not r.ok and "AssertionError" in r.stderr
    r = run_program("while True:\n    pass\n", timeout_s=1.0)
    assert not r.ok and r.timed_out
    r = run_program("def broken(:\n")
    assert not r.ok and not r.timed_out          # SyntaxError = non-zero exit


# --------------------------------------------------------------------- #
# datasets — problem loader + fixtures
# --------------------------------------------------------------------- #

import gzip

from evals.datasets import Problem, load_problems

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def test_load_problems_jsonl():
    ps = load_problems(os.path.join(_FIXTURES, "humaneval_tiny.jsonl"))
    assert [p.task_id for p in ps] == ["tiny/1", "tiny/2"]
    p = ps[0]
    assert isinstance(p, Problem)
    assert p.entry_point == "add_two"
    assert "candidate" in p.test and p.prompt.startswith("def ")


def test_load_problems_gz(tmpdir):
    src = os.path.join(_FIXTURES, "humaneval_tiny.jsonl")
    gz = os.path.join(str(tmpdir), "p.jsonl.gz")
    with open(src, "rb") as fin, gzip.open(gz, "wb") as fout:
        fout.write(fin.read())
    assert len(load_problems(gz)) == 2


def test_load_problems_missing_key_raises(tmpdir):
    bad = os.path.join(str(tmpdir), "bad.jsonl")
    with open(bad, "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_id": "x"}) + "\n")
    try:
        load_problems(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "prompt" in str(e)


# --------------------------------------------------------------------- #
# generation — ckpt loading + sampling (torch)
# --------------------------------------------------------------------- #

import torch

from model import RythConfig, RythForCausalLM
from model.checkpoint import save_checkpoint
from evals.generation import (extract_code, find_eos, load_model,
                              sample_completion, truncate_at_stops)


def test_truncate_at_stops():
    assert truncate_at_stops("abc```more", ("```",)) == "abc"
    assert truncate_at_stops("no stops", ("```",)) == "no stops"
    assert truncate_at_stops("a\nb\nclass X", ("\nclass", "\ndef")) == "a\nb"


def test_extract_code_strips_fences():
    assert extract_code("```python\nprint(1)\n```") == "print(1)"
    assert extract_code("plain code") == "plain code"


def test_extract_code_preserves_indent_without_fences():
    # function-body continuation ka leading indent KEEMTI hai (ruling)
    assert extract_code("    return a + b\n") == "    return a + b"


def test_sample_completion_with_real_tiny_model():
    # NOTE: preset classmethods apne defaults override nahi karne dete (config.py:104),
    # isliye chhota model seedha RythConfig(...) se banta hai.
    tok = _tok()
    cfg = RythConfig(vocab_size=tok.vocab_size, max_seq_len=64,
                     d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = RythForCausalLM(cfg)
    out = sample_completion(model, tok, "def f():\n",
                            max_new_tokens=8, temperature=1.0)
    assert isinstance(out, str)


def test_find_eos_none_by_default():
    assert find_eos(_tok()) is None
    tok = _tok()
    tok.add_special_tokens(["<|end|>"])
    assert find_eos(tok) == tok.special_tokens["<|end|>"]


def test_load_model_from_checkpoint(tmpdir):
    tok = _tok()
    cfg = RythConfig(vocab_size=tok.vocab_size, max_seq_len=32,
                     d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = RythForCausalLM(cfg)
    ck = os.path.join(str(tmpdir), "best.pt")
    save_checkpoint(ck, model, cfg)
    loaded = load_model(ck, tok.vocab_size, preset=None, seq_len=32)
    assert loaded.config.vocab_size == tok.vocab_size   # attr = .config
    x = torch.randint(0, tok.vocab_size, (1, 4))
    logits, _ = loaded(x)
    assert logits.shape[-1] == tok.vocab_size


# --------------------------------------------------------------------- #
# humaneval — pass@k runner
# --------------------------------------------------------------------- #

from evals.humaneval import evaluate, save_results

_HE = os.path.join(_FIXTURES, "humaneval_tiny.jsonl")


def test_humaneval_perfect_sampler_scores_100():
    problems = load_problems(_HE)
    canon = {p.task_id: p.canonical_solution for p in problems}
    res = evaluate(problems, sampler=lambda prompt: canon.get(
        _id_from_prompt(prompt, problems), ""), n_samples=3, ks=(1,))
    assert abs(res["pass_at_k"]["pass@1"] - 1.0) < 1e-9


def test_humaneval_empty_sampler_scores_0():
    problems = load_problems(_HE)
    res = evaluate(problems, sampler=lambda prompt: "", n_samples=2, ks=(1,))
    assert res["pass_at_k"]["pass@1"] == 0.0


def test_humaneval_results_file_written(tmpdir):
    problems = load_problems(_HE)[:1]
    canon = problems[0].canonical_solution
    res = evaluate(problems, sampler=lambda prompt: canon, n_samples=2, ks=(1,))
    out = os.path.join(str(tmpdir), "res.json")
    save_results(res, out)
    with open(out, encoding="utf-8") as f:
        back = json.load(f)
    assert back["tasks"][0]["n_passed"] == 2


def test_humaneval_meta_echoes_config():
    # global constraint: results me config echo hota hai
    problems = load_problems(_HE)
    res = evaluate(problems, sampler=lambda prompt: "", n_samples=1,
                   ks=(1,), temperature=0.5, top_k=10, max_new_tokens=32,
                   progress=lambda *a, **k: None)
    m = res["meta"]
    assert (m["task"], m["mode"], m["n_samples"], m["temperature"],
            m["top_k"], m["max_new_tokens"]) == ("humaneval", "base",
                                                 1, 0.5, 10, 32)


def _id_from_prompt(prompt, problems):           # test helper (sampler DI demo)
    for p in problems:
        if prompt.startswith(p.prompt.split("\n")[0]):
            return p.task_id
    return ""


# --------------------------------------------------------------------- #
# mbpp — assert-based runner + field adapter
# --------------------------------------------------------------------- #

from evals.mbpp import evaluate as mbpp_evaluate, load_mbpp

_MBPP = os.path.join(_FIXTURES, "mbpp_tiny.jsonl")


def test_load_mbpp_adapter_maps_fields():
    ps = load_mbpp(_MBPP)
    assert ps[0].task_id == "m1"
    assert ps[0].prompt.startswith("Write a function add2")
    assert ps[0].test == "assert add2(2, 3) == 5\nassert add2(0, 0) == 0"


def test_load_mbpp_tolerates_canonical_rows(tmpdir):
    # already-canonical rows (prompt/test) bhi chale — HumanEval-style file reuse
    src = os.path.join(str(tmpdir), "canon.jsonl")
    with open(src, "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_id": "c1", "prompt": "def g():\n",
                            "test": "assert g() is not None\n",
                            "entry_point": "g"}) + "\n")
    ps = load_mbpp(src)
    assert ps[0].task_id == "c1" and ps[0].entry_point == "g"


def test_mbpp_perfect_sampler_scores_100():
    problems = load_mbpp(_MBPP)
    canon = {p.task_id: p.canonical_solution for p in problems}
    res = mbpp_evaluate(problems, sampler=lambda prompt: canon.get(
        _mbpp_id(prompt, problems), ""), n_samples=2, ks=(1,))
    assert abs(res["pass_at_k"]["pass@1"] - 1.0) < 1e-9


def _mbpp_id(prompt, problems):
    for p in problems:
        if prompt.startswith(p.prompt.split(".")[0][:20]):
            return p.task_id
    return ""


# --------------------------------------------------------------------- #
# ppl — held-out perplexity
# --------------------------------------------------------------------- #

from evals.ppl import evaluate_files, perplexity


def test_perplexity_finite_deterministic_and_reasonable():
    tok = _tok()
    cfg = RythConfig(vocab_size=tok.vocab_size, max_seq_len=128,
                     d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    torch.manual_seed(0)
    model = RythForCausalLM(cfg)
    text = ("def add(a, b):\n    return a + b\n" * 20)
    p1 = perplexity(model, tok, text, seq_len=64)
    p2 = perplexity(model, tok, text, seq_len=64)
    assert p1 == p2 and p1 > 1.0                    # random init => high ppl


def test_perplexity_short_text_is_inf():
    tok = _tok()
    cfg = RythConfig(vocab_size=tok.vocab_size, max_seq_len=16,
                     d_model=32, n_layers=1, n_heads=2, n_kv_heads=1)
    model = RythForCausalLM(cfg)
    assert perplexity(model, tok, "", seq_len=8) == float("inf")


def test_evaluate_files_reads_labels(tmpdir):
    tok = _tok()
    cfg = RythConfig(vocab_size=tok.vocab_size, max_seq_len=16,
                     d_model=32, n_layers=1, n_heads=2, n_kv_heads=1)
    model = RythForCausalLM(cfg)
    f = os.path.join(str(tmpdir), "py.txt")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("return 1\n" * 5)
    out = evaluate_files(model, tok, {"py": f}, seq_len=8)
    assert set(out) == {"py"} and isinstance(out["py"], float)


# --------------------------------------------------------------------- #
# cli — ryth-eval entry point
# --------------------------------------------------------------------- #


def tk_path(tok):                                 # helper: tokenizer ko file me save karo
    p = os.path.join(tempfile.mkdtemp(prefix="ryth_tok_"), "tokenizer.json")
    tok.save(p)                                   # API verified: bpe.py:170
    return p


def test_cli_ppl_smoke(tmpdir):
    from evals.cli import main

    tok = _tok()
    cfg = RythConfig(vocab_size=tok.vocab_size, max_seq_len=64,
                     d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = RythForCausalLM(cfg)
    ck = os.path.join(str(tmpdir), "b.pt")
    save_checkpoint(ck, model, cfg)
    txt = os.path.join(str(tmpdir), "t.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("def f():\n    return 1\n" * 10)
    outj = os.path.join(str(tmpdir), "r.json")
    rc = main(["ppl", "--ckpt", ck, "--tokenizer", tk_path(tok),
               "--files", f"py={txt}", "--out", outj])
    assert rc == 0
    with open(outj, encoding="utf-8") as f:
        assert "py" in json.load(f)["perplexity"]

# ── W2: baseline tooling ─────────────────────────────────────────────────────

def test_apply_limit_slices_and_defaults():
    from evals.cli import apply_limit

    probs = list("abcde")
    assert apply_limit(probs, None) == probs          # default: no slice
    assert apply_limit(probs, 0) == probs             # 0/negative => no slice
    assert apply_limit(probs, 3) == list("abc")
    assert apply_limit([], 5) == []


def test_cli_has_limit_flag():
    import contextlib
    import io
    from evals.cli import main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        main(["humaneval", "--help"])
    assert "--limit" in buf.getvalue()
