"""Notebook exec-simulation — ryth_kaggle_train.ipynb ka notebook-level test.

W1-revision directive #7: sirf unit-tests nahi — NOTEBOOK ke cells ko ACTUAL
exec karo (paths tmp par substitute karke, GLOBAL subprocess.run patch karke —
cell ka apna `import subprocess` real module laata hai, isliye ns-injection
kaafi nahi) aur Run-All order / skip-guards / fail-paths VERIFY karo.

Kya cover hota hai:
  * sabhi code cells ka syntax gate (ast.parse)
  * A1 fresh-run: commands sahi ORDER me, artifacts markers ke saath
  * A1 idempotent re-run: train/pack dobara NAHI chalte
  * partial-attach resume: corpus attach -> restore, tokenizer banega
  * adhoora corpus -> RuntimeError, expensive stage command jari NAHI hota
  * B1 bina dataset / bina CUDA -> RuntimeError (SystemExit nahi — IPython)
  * B2 checkpoint ke bina -> RuntimeError
"""

import ast
import json
import os
import subprocess as _real_subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "notebooks", "ryth_kaggle_train.ipynb")


def _cells():
    nb = json.load(open(NB, encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        src = c["source"]
        if isinstance(src, list):
            src = "".join(src)
        out.append((c["cell_type"], src))
    return out


# ------------------------------------------------------------------ #
# Level 1: har code cell syntax-valid ho (magics strip karke)
# ------------------------------------------------------------------ #

def test_all_code_cells_parse():
    n_code = 0
    for ctype, src in _cells():
        if ctype != "code":
            continue
        n_code += 1
        clean = "\n".join(ln for ln in src.splitlines()
                          if not ln.lstrip().startswith(("%", "!")))
        ast.parse(clean)
    assert n_code >= 14


def test_no_systemexit_in_production_guards():
    # IPython SystemExit ask-exit weirdness — RuntimeError hi hona chahiye
    for i, (ctype, src) in enumerate(_cells()):
        if ctype != "code":
            continue
        assert "raise SystemExit" not in src, f"cell {i} SystemExit use karta hai"


# ------------------------------------------------------------------ #
# Level 2: production-flow simulation (cells 9 -> 2 -> 3 -> 5 -> 6)
# ------------------------------------------------------------------ #

class CmdRecorder:
    """GLOBAL subprocess.run replacement — record + artifacts materialize."""

    def __init__(self, work):
        self.cmds = []
        self.work = str(work)
        self.on_cmd = None

    def __call__(self, cmd, shell=False, **kw):
        self.cmds.append(cmd)
        self._materialize(cmd)
        if self.on_cmd:
            self.on_cmd(cmd)

        class R:
            returncode = 0
        return R()

    def _materialize(self, cmd):
        w = self.work

        def mk(p, obj=None):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(obj if isinstance(obj, str) else json.dumps(obj))

        if "w1_build_corpus" in cmd:
            for s in range(27):
                mk(os.path.join(w, "corpus_out", "stage", f"src{s:02d}", "_DONE"),
                   "{}")
            mk(os.path.join(w, "corpus_out", "_SUMMARY.json"),
               {"total_bytes": 2_500_000_000})
            mk(os.path.join(w, "val_src", "_VAL_DONE"), "{}")
        elif "w1_train_tokenizer" in cmd:
            mk(os.path.join(w, "tok", "tokenizer.json"),
               '{"type":"bpe-byte-level"}')
            mk(os.path.join(w, "tok", "_DONE"),
               '{"vocab_size":24576,"specials":9}')
        elif "w1_tokenizer_efficiency" in cmd:
            mk(os.path.join(w, "tokenizer_efficiency.json"),
               {"hf:wikipedia-hi": {"chars_per_token": 2.2,
                                    "tokens_per_char": 0.45},
                "_aggregate": {"chars_per_token": 3.1}})
        elif "w1_pack_rds" in cmd:
            mk(os.path.join(w, "rds_w1", "final", "manifest.json"),
               {"shards": [], "seq_len": 1024})
        elif cmd.startswith("git clone"):
            repo = cmd.split()[-1]
            os.makedirs(os.path.join(repo, ".git"), exist_ok=True)


@pytest.fixture()
def recorder(monkeypatch):
    """subprocess.run globally patch — cell-internal import bhi stub dekhega."""
    state = {"rec": None}

    def factory(work):
        rec = CmdRecorder(work)
        state["rec"] = rec
        monkeypatch.setattr(_real_subprocess, "run", rec)
        return rec

    yield factory


def _substitute(src, tmp):
    src = src.replace("/kaggle/working", str(tmp / "kworking"))
    src = src.replace("/kaggle/input/w1-prep", str(tmp / "kinput" / "w1-prep"))
    return src


def _run_cell(src, tmp_path, monkeypatch, extra_ns=None):
    """Cell exec; cwd restore kyunke cell os.chdir karta hai."""
    cwd = os.getcwd()
    try:
        ns = {"__name__": "__nb__"}
        if extra_ns:
            ns.update(extra_ns)
        exec(compile(ast.parse(_substitute(src, tmp_path)), "<nb-cell>",
                     "exec"), ns)
        return ns
    finally:
        os.chdir(cwd)


NB_CELLS = None                          # lazy global (json ek baar padho)


def _lazy_cells():
    global NB_CELLS
    if NB_CELLS is None:
        NB_CELLS = _cells()
    return NB_CELLS


def test_a1_fresh_run_command_order_and_rerun_skips(tmp_path, recorder,
                                                    monkeypatch):
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "kworking" / "prep"
    fake = recorder(work)
    cells = dict(enumerate(_lazy_cells()))

    _run_cell(cells[9][1], tmp_path, monkeypatch)      # WALKTHROUGH=False gate

    _run_cell(cells[2][1], tmp_path, monkeypatch)      # A1

    names = [next((t for t in ("probe_stack", "build_corpus",
                               "train_tokenizer", "tokenizer_efficiency",
                               "pack_rds") if t in c), c)
             for c in fake.cmds]
    assert any("clone" in c or "pull" in c for c in fake.cmds[:2])
    idx = {t: names.index(t) for t in ("build_corpus", "train_tokenizer",
                                       "tokenizer_efficiency", "pack_rds")}
    assert idx["build_corpus"] < idx["train_tokenizer"] < \
        idx["tokenizer_efficiency"] < idx["pack_rds"]
    assert names.count("probe_stack") == 2             # python + c probes
    all_cmds = " ".join(fake.cmds)
    assert "--val-count 40" in all_cmds and "--total-gb 2.4" in all_cmds
    assert "--val-out" in all_cmds                     # val split WORK me

    # RE-RUN: mehngi stages dobara NAHI (markers se skip)
    n_before = len(fake.cmds)
    _run_cell(cells[2][1], tmp_path, monkeypatch)
    rerun = fake.cmds[n_before:]
    assert not any("train_tokenizer" in c for c in rerun), rerun
    assert not any("pack_rds" in c for c in rerun)
    assert not any("efficiency" in c for c in rerun)
    assert any("build_corpus" in c for c in rerun)     # idempotent-skip, sasta


def test_a1_partial_attach_resume(tmp_path, recorder, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "kinput" / "w1-prep"
    work = tmp_path / "kworking" / "prep"
    # attached dataset me SIRF corpus (tok/rds kabhi bane hi nahi)
    for s in range(27):
        d = inp / "corpus_out" / "stage" / f"src{s:02d}"
        d.mkdir(parents=True)
        (d / "_DONE").write_text("{}", encoding="utf-8")
    (inp / "corpus_out" / "_SUMMARY.json").write_text(
        '{"total_bytes": 2500000000}', encoding="utf-8")

    fake = recorder(work)
    cells = dict(enumerate(_lazy_cells()))
    _run_cell(cells[2][1], tmp_path, monkeypatch)

    assert (work / "corpus_out" / "_SUMMARY.json").exists()   # restore hua
    assert any("train_tokenizer" in c for c in fake.cmds)     # phir bhi bana
    assert any("pack_rds" in c for c in fake.cmds)


def test_a1_incomplete_corpus_raises_and_stops(tmp_path, recorder,
                                               monkeypatch):
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "kworking" / "prep"
    fake = recorder(work)

    def short(cmd):
        if "w1_build_corpus" in cmd:
            sp = work / "corpus_out" / "_SUMMARY.json"
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text('{"total_bytes": 100}', encoding="utf-8")

    fake.on_cmd = short
    cells = dict(enumerate(_lazy_cells()))
    with pytest.raises(RuntimeError, match="ADHOORA"):
        _run_cell(cells[2][1], tmp_path, monkeypatch)
    assert not any("train_tokenizer" in c for c in fake.cmds)
    assert not any("pack_rds" in c for c in fake.cmds)


def test_b1_without_dataset_is_runtimeerror(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = dict(enumerate(_lazy_cells()))
    with pytest.raises(RuntimeError, match="w1-prep"):
        _run_cell(cells[5][1], tmp_path, monkeypatch)


def test_b1_without_cuda_is_runtimeerror(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "kinput" / "w1-prep"
    (inp / "tok").mkdir(parents=True)
    (inp / "tok" / "tokenizer.json").write_text("{}", encoding="utf-8")

    import types
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    cells = dict(enumerate(_lazy_cells()))
    with pytest.raises(RuntimeError, match="CUDA"):
        _run_cell(cells[5][1], tmp_path, monkeypatch,
                  extra_ns={"torch": fake_torch})


def test_b2_without_checkpoint_is_runtimeerror(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = dict(enumerate(_lazy_cells()))
    with pytest.raises(RuntimeError, match="checkpoint"):
        _run_cell(cells[6][1], tmp_path, monkeypatch)


def test_a2_runs_after_a1_artifacts(tmp_path, recorder, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "kworking" / "prep"
    recorder(work)
    cells = dict(enumerate(_lazy_cells()))
    ns = _run_cell(cells[2][1], tmp_path, monkeypatch)
    ns["WORK"] = str(work)
    cwd = os.getcwd()
    try:
        exec(compile(ast.parse(_substitute(cells[3][1], tmp_path)),
                     "<nb-cell>", "exec"), ns)
    finally:
        os.chdir(cwd)
    out = capsys.readouterr().out
    assert "(skip" not in out
    assert "chars/token" in out
