# Security Policy

## Supported Versions

Ryth is in early development. Security fixes are applied to the latest release.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅         |

## Reporting a Vulnerability

If you discover a security issue, please **do not open a public issue**. Instead:

1. Use GitHub's **private vulnerability reporting** (Security tab → "Report a
   vulnerability") on https://github.com/RAJ-af/Ryth, or
2. Contact the maintainer privately.

Please include a description, reproduction steps, and potential impact. We aim to
acknowledge reports within a few days.

## Scope & safe usage notes

- **Untrusted code inputs:** The RDE `Validator` compiles Python files with
  `compile(..., "exec")` to check syntax. This does **not execute** the code, but
  parsing untrusted input still carries some risk — run dataset builds in an
  isolated environment when processing untrusted repositories.
- **Sandboxing:** RDE is a data-processing pipeline, not a security sandbox. It
  reads files and writes binary shards; it does not run downloaded code.
- **Never commit secrets.** API tokens, keys, and passwords must never be placed
  in the repository, issues, or pull requests. `.gitignore` excludes common
  secret files, but review your commits. If a credential is ever exposed, **revoke
  it immediately.**
