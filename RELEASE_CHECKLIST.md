# Release Checklist

A reusable pre-release checklist for Ryth. Current release: **v0.3.0** (scratch
tokenizer + Ryth Data Engine + model core + training engine). Copy the version
into the boxes below for each new release.

## 0. Security (do this first) 🔒

- [ ] **Revoke any GitHub token / password that was ever shared in chat or a log**
      at https://github.com/settings/tokens, and change the account password.
      A credential pasted into a chat/log must be treated as compromised.
- [ ] Confirm **no secrets** are in the repo:
      ```bash
      git grep -nE 'ghp_|github_pat_|password|secret|BEGIN .*PRIVATE KEY' || echo "clean"
      ```
- [ ] `.gitignore` excludes secrets, datasets, checkpoints (`*.pt`), and build
      artifacts (it does).
- [ ] Push over **SSH** (`git@github.com:RAJ-af/Ryth.git`) — no token in any file,
      command, or commit.

## 1. Scope

- [ ] Release contains **only implemented, tested** components.
- [ ] Current pillars: scratch **tokenizer**, **RDE** v1.1, **model core**
      (v0.2.0), **training engine** (v0.3.0).
- [ ] Documentation reflects the code exactly — no unimplemented features claimed.

## 2. Code quality

- [ ] `python -m py_compile` passes for all packages, tests, examples.
- [ ] No unused imports (AST scan clean).
- [ ] No `TODO`/`FIXME`/`XXX`/`HACK` left in code.
- [ ] No stray references to old module paths (`pycoder`, `rde`, `demo`).
- [ ] No duplicate modules; library code has no debug prints (CLI output is intentional).

## 3. Tests

- [ ] Tokenizer tests pass — `pytest tests/test_tokenizer.py` (8).
- [ ] RDE tests pass — `pytest tests/test_rde.py` (21).
- [ ] Model tests pass — `pytest tests/test_model.py` (44, needs torch).
- [ ] Training tests pass — `pytest tests/test_training.py` (28, needs torch).
- [ ] Full suite green — `pip install -e ".[dev]" && pytest -q` (**101 total**).
- [ ] All examples run — `python examples/example_*.py`.

## 4. Packaging

- [ ] `pip install -e .` succeeds (core is pure standard library, no required deps).
- [ ] `pip install -e ".[model]"` / `".[train]"` pull in torch and work.
- [ ] Console scripts resolve on PATH: `ryth-tokenizer`, `ryth-rde`, `ryth-train`.
- [ ] CLI workflows verified: `build → verify → inspect → stats → manifest`, and
      `ryth-train --data_dir … --max_steps …` + `--resume latest`.
- [ ] `pyproject.toml` metadata correct (name, **version**, license, URLs, packages).

## 5. Documentation

- [ ] `README.md` — intro, vision, features (all four pillars), architecture,
      install, quick start, structure, examples, dataset format, tokenizer, model,
      training, roadmap, license.
- [ ] `docs/` — architecture, tokenizer, dataset_engine, rds_format, model,
      training, quickstart, faq.
- [ ] `CHANGELOG.md`, `ROADMAP.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
      `SECURITY.md`, `LICENSE` present and current.
- [ ] Skim rendered Markdown on GitHub after the push (links/diagrams).

## 6. Final review

- [ ] Read the diff / file tree once more.
- [ ] Confirm author/URLs in `pyproject.toml`, `LICENSE`, and docs are correct.

## 7. Publish

> SSH is already configured. Do not place any token in a file, command, or commit.

```bash
cd Ryth
git add .
git commit -m "vX.Y.Z <title>"
git push origin main
```

## 8. Tag the release

```bash
git tag -a vX.Y.Z -m "<title>"
git push origin vX.Y.Z
```

Then on GitHub: **Releases → Draft a new release → choose tag `vX.Y.Z` →** set the
title → paste the matching `CHANGELOG.md` section → Publish.

## 9. Post-release

- [ ] Verify `pip install git+https://github.com/RAJ-af/Ryth.git` works in a
      clean environment.
- [ ] Open tracking issues for the next milestone (currently Phase 5 — 30M prototype).
