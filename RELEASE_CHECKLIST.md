# Release Checklist — v0.1.0 "Foundation Release"

Use this before publishing the first public release of Ryth.

## 0. Security (do this first) 🔒

- [ ] **Revoke the GitHub Personal Access Token that was shared in chat** at
      https://github.com/settings/tokens, and **change the account password.**
      A token pasted into a chat/log must be treated as compromised.
- [ ] Confirm **no secrets** are in the repo:
      ```bash
      git grep -nE 'ghp_|github_pat_|password|secret|BEGIN .*PRIVATE KEY' || echo "clean"
      ```
- [ ] `.gitignore` excludes secrets, datasets, and build artifacts (it does).

## 1. Scope

- [x] Release contains **only implemented, tested** components: the scratch
      tokenizer and RDE v1.1.
- [x] Model and training engine are **not** shipped (Phase 3+ on the roadmap).
- [x] Documentation reflects the code exactly — no unimplemented features.

## 2. Code quality

- [x] `python -m py_compile` passes for all packages, tests, examples.
- [x] No unused imports (AST scan clean).
- [x] No `TODO`/`FIXME`/`XXX`/`HACK` left in code.
- [x] No stray references to old module paths (`pycoder`, `rde`, `demo`).
- [x] No duplicate modules.
- [x] Library code has no debug prints (CLI/verbose output is intentional).

## 3. Tests

- [x] Tokenizer tests pass — `python tests/test_tokenizer.py` (8/8).
- [x] RDE tests pass — `python tests/test_rde.py` (15/15).
- [x] All examples run — `python examples/example_*.py`.
- [ ] (Optional) `pip install -e ".[dev]" && pytest -q` in a clean venv.

## 4. Packaging

- [x] `pip install -e .` succeeds (pure standard library, no required deps).
- [x] Console scripts resolve on PATH: `ryth-tokenizer`, `ryth-rde`.
- [x] CLI workflow verified: `build → verify → inspect → stats → manifest`.
- [x] `pyproject.toml` metadata correct (name, version `0.1.0`, license, URLs).

## 5. Documentation

- [x] `README.md` — intro, vision, features, architecture, install, quick start,
      structure, examples, dataset format, tokenizer, roadmap, license.
- [x] `docs/` — architecture, tokenizer, dataset_engine, rds_format, quickstart, faq.
- [x] `CHANGELOG.md`, `ROADMAP.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
      `SECURITY.md`, `LICENSE` present.
- [ ] Skim rendered Markdown on GitHub after the first push (links/diagrams).

## 6. Final review

- [ ] Read the diff / file tree once more.
- [ ] Confirm author/URLs in `pyproject.toml`, `LICENSE`, and docs are correct.

## 7. Publish (run locally — nothing is pushed for you)

> Authenticate with a **fresh** token via git's credential prompt when pushing.
> Do not place any token in a file, a command, or a commit.

```bash
cd Ryth
git init
git add .
git commit -m "v0.1.0 Foundation Release"
git branch -M main
git remote add origin https://github.com/RAJ-af/Ryth.git
git push -u origin main
```

## 8. Tag the release

```bash
git tag -a v0.1.0 -m "Foundation Release: scratch tokenizer + Ryth Data Engine"
git push origin v0.1.0
```

Then on GitHub: **Releases → Draft a new release → choose tag `v0.1.0` →**
title **"Foundation Release"** → paste the `CHANGELOG.md` v0.1.0 section → Publish.

## 9. Post-release

- [ ] Verify `pip install git+https://github.com/RAJ-af/Ryth.git` works in a
      clean environment.
- [ ] Open tracking issues for Phase 3 (Training Engine).
