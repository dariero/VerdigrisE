---
name: ship
description: "Publish a completed VerdigrisE change by validating the atomic diff, committing it, pushing a task branch, opening a ready pull request, requesting Codex review, and enabling automatic squash merge. Use when the user asks to ship, publish, or open and merge a pull request."
---

# Ship a change

1. Read `AGENTS.md`, then inspect `git status --short --branch`, the unstaged diff, the staged diff, the branch diff against `origin/main`, and recent repository history.
2. Confirm the intended change is atomic and that unrelated user work will remain untouched. Stop if the shipping scope is ambiguous.
3. Fetch `origin/main`. Verify that the current branch is a `codex/<short-slug>` task branch whose pull-request base is `main`; never commit on `main`. If the branch predates current `origin/main`, assess whether it must be reconciled before shipping and preserve shared history and user work.
4. Stage only intended paths; never use `git add -A`. Exclude `.env*`, credentials, `.index/`, caches, `.DS_Store`, editor state, and other generated files. Staging first ensures pre-commit includes intended new files in its all-files run.
5. Run the free validation commands:

   ```bash
   .venv/bin/pre-commit run --all-files
   .venv/bin/python -m pytest -c pytest.ini eval/ -q
   ```

   If pre-commit modifies files, inspect the changes, restage the explicit intended paths, and rerun the command until every hook passes.

   Do not run a paid tier as part of shipping. Use `run-paid-evaluation` separately if the user explicitly approves one invocation.
6. Inspect `git diff --cached`, `git diff --cached --check`, and `git diff --cached --name-only`. Verify that the staged result is complete, portable, and contains no secrets.
7. Create one concise conventional commit consistent with repository history, without an AI attribution trailer, then push the task branch.
8. Open a ready pull request, not a draft. Include a short outcome summary and the exact free validation command and result.
9. Post this review request on the pull request:

   ```text
   @codex review for deterministic/RagaliQ ownership, dependency reproducibility, Python 3.14 compatibility, public-clone portability, paid-call safety, golden-fixture integrity, marker correctness, public API compatibility, and unintended behaviour changes
   ```

10. Wait for Codex review and any repository checks to complete. Resolve every actionable finding; after a pushed fix, rerun the free suite and obtain updated review coverage. Do not enable automatic merge while an actionable finding or failing check remains.
11. Verify that the ready pull request targets `main`, is mergeable, and contains only the intended atomic change. Enable automatic merge using the squash method:

    ```bash
    gh pr merge --auto --squash <PR>
    ```

    Report any branch-protection, review, or check requirement that prevents auto-merge.
12. Do not update issues, boards, labels, milestones, releases, or delete local or remote branches.
