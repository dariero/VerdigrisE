---
name: upgrade-dependencies
description: "Update VerdigrisE dependencies while preserving its Python 3.14, hash-locked, public-clone installation contract. Use when the user asks to add, remove, upgrade, pin, or investigate dependencies, regenerate pylock.toml, or repair dependency reproducibility."
---

# Upgrade dependencies

1. Read the installation and development sections of `README.md`, plus `pyproject.toml` and `pylock.toml`.
2. Confirm the requested direct-dependency scope and target. Inspect the current constraint and locked version before editing. If the approved outcome is already satisfied, report a no-op; do not regenerate the lock merely to create a diff. A request to upgrade does not authorize adding or removing unrelated dependencies, changing Python support, or introducing new tooling.
3. Record `uv --version` and whether `uv.lock` already exists. If regenerating with a different uv version from the one documented in `README.md`, update that generator-version statement. Preserve any pre-existing `uv.lock`; stop for direction if the documented workflow would overwrite or delete it.
4. Edit only the approved abstract requirement in `pyproject.toml`, preserving `requires-python = ">=3.14,<3.15"` unless a Python-policy change was explicitly approved.
5. When `uv.lock` was absent before the task, regenerate the exact lock from the repository root with the documented sequence:

   ```bash
   uv export \
     --format pylock.toml \
     --all-groups \
     --no-emit-project \
     --python 3.14 \
     --prerelease disallow \
     --no-header \
     -o pylock.toml
   rm uv.lock
   ```

6. Verify that `pylock.toml` retains `requires-python = "==3.14.*"`, hash-bearing artifacts, all dependency groups, and public package sources. Reject local paths, editable installs, sibling checkouts, private indexes, and unintended prereleases.
7. If `.venv` does not exist, create it:

   ```bash
   uv venv --python 3.14
   ```

8. Synchronize and validate the locked environment:

   ```bash
   uv pip sync --preview-features pylock --require-hashes pylock.toml
   uv pip check
   ```

9. Run the free deterministic suite and branch-coverage gate:

   ```bash
   .venv/bin/python -m pytest eval/ -q
   .venv/bin/python -m pytest --cov --cov-report=term-missing eval/ -q
   ```

10. Inspect the full `pyproject.toml` and `pylock.toml` diff. Reject unrelated direct changes and unexplained transitive churn; do not use an all-package refresh to broaden the approved scope. Explain necessary transitive changes, compatibility implications, and documentation updates. Do not run paid provider tests without separate per-run approval.
