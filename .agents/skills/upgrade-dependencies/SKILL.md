---
name: upgrade-dependencies
description: "Update VerdigrisE dependencies while preserving its Python 3.14, hash-locked, public-clone installation contract. Use when the user asks to add, remove, upgrade, pin, or investigate dependencies, regenerate pylock.toml, or repair dependency reproducibility."
---

# Upgrade dependencies

1. Read the installation and development sections of `README.md`, plus `pyproject.toml`, `uv.lock`, and `pylock.toml`.
2. Inspect `git status --short --branch` and the current diffs for `pyproject.toml`, `uv.lock`, and `pylock.toml`; stop if unrelated work overlaps those files. Confirm the requested direct-dependency scope and target. Inspect the current constraint and locked version before editing. If the approved outcome is already satisfied, report a no-op; do not regenerate the lock merely to create a diff. A request to upgrade does not authorize adding or removing unrelated dependencies, changing Python support, or introducing new tooling.
3. Record `uv --version` and run lock-generation commands through the generator version documented in `README.md`. The commands below use uv 0.11.30 explicitly. A uv-version change is separate tooling scope: update the CI version and archive checksum, README generator-version statement, and skill commands together only when explicitly approved. Never delete the committed `uv.lock`.
4. Edit only the approved abstract requirement in `pyproject.toml`, preserving `requires-python = ">=3.14,<3.15"` unless a Python-policy change was explicitly approved.
5. Update the committed solver state from the repository root. For an approved upgrade within an unchanged constraint, target only that package:

   ```bash
   uvx --from uv==0.11.30 uv lock \
     --upgrade-package <package> \
     --python 3.14 \
     --prerelease disallow
   ```

   After an approved abstract requirement edit, retain existing lock preferences and resolve only required changes:

   ```bash
   uvx --from uv==0.11.30 uv lock \
     --python 3.14 \
     --prerelease disallow
   ```

   Do not use `--upgrade` or another all-package refresh unless that broader scope was explicitly approved.
6. Export the exact public-install contract from the committed solver state:

   ```bash
   uvx --from uv==0.11.30 uv export \
     --frozen \
     --format pylock.toml \
     --all-groups \
     --no-emit-project \
     --python 3.14 \
     --prerelease disallow \
     --no-header \
     -o pylock.toml
   ```

7. Check the solver state and reproduce the public lock independently:

   ```bash
   uvx --from uv==0.11.30 uv lock \
     --check \
     --python 3.14 \
     --prerelease disallow
   generated_dir="$(mktemp -d)"
   uvx --from uv==0.11.30 uv export \
     --frozen \
     --format pylock.toml \
     --all-groups \
     --no-emit-project \
     --python 3.14 \
     --prerelease disallow \
     --no-header \
     --quiet \
     -o "$generated_dir/pylock.generated.toml"
   cmp pylock.toml "$generated_dir/pylock.generated.toml"
   ```

   Verify that `pylock.toml` retains `requires-python = "==3.14.*"`, hash-bearing artifacts, every declared runtime and dependency-group root, and public package sources. The PEP 751 export does not retain transitive group provenance; do not claim that it does. Reject local paths, editable installs, sibling checkouts, private indexes, and unintended prereleases.
8. If `.venv` does not exist, create it:

   ```bash
   uv venv --python 3.14
   ```

9. Synchronize and validate the locked environment:

   ```bash
   uv pip sync --preview-features pylock --require-hashes pylock.toml
   uv pip check
   ```

10. Run the free deterministic suite and branch-coverage gate:

   ```bash
   .venv/bin/python -m pytest eval/ -q
   .venv/bin/python -m pytest --cov --cov-report=term-missing eval/ -q
   ```

11. Inspect the full `pyproject.toml`, `uv.lock`, and `pylock.toml` diff. Reject unrelated direct changes and unexplained transitive churn; do not use an all-package refresh to broaden the approved scope. Explain necessary transitive changes, compatibility implications, and documentation updates. Do not run paid provider tests without separate per-run approval.
