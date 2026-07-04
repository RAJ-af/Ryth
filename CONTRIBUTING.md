# Contributing to Ryth

Thanks for your interest in Ryth! This is an early-stage project building a
coding-first LLM entirely from scratch. Contributions of all sizes are welcome.

## Ground rules

- **Be respectful.** See the [Code of Conduct](CODE_OF_CONDUCT.md).
- **Keep it from scratch.** Ryth's philosophy is to build core components
  ourselves (tokenizer, data engine, and later the model/training). Please avoid
  adding heavy third-party ML dependencies to the core packages.
- **Pure standard library** for the tokenizer and RDE core. Optional speedups
  (e.g. `xxhash`) must degrade gracefully when not installed.

## Development setup

```bash
git clone https://github.com/RAJ-af/Ryth.git
cd Ryth
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -q                 # full suite
python tests/test_rde.py  # or run a file directly (no pytest needed)
```

All tests are **pure standard library** and must pass before a PR is merged.

## Making a change

1. Fork and create a branch: `git checkout -b feature/my-change`
2. Keep changes focused; match the surrounding code style.
3. Add or update tests for any behavior change.
4. Update docs (`docs/` and `README.md`) so they still match the code.
5. Run `pytest -q` and make sure everything is green.
6. Open a pull request describing **what** and **why**.

## Coding style

- Clear names, small functions, docstrings on public APIs.
- No debug prints in library code (CLI/verbose output is fine behind a flag).
- No dead code, no unused imports, no leftover `TODO`s in merged code.

## Reporting bugs / requesting features

Open a [GitHub issue](https://github.com/RAJ-af/Ryth/issues) with:
- what you expected vs. what happened,
- a minimal repro (commands + inputs),
- your Python version and OS.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for where the project is headed. Good first areas:
tokenizer improvements, new language detectors, quality-analyzer signals, and
RDE v2 features (import graph, AST cache, dataset diff).
