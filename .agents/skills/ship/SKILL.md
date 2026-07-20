---
name: ship
description: "Publish a completed VerdigrisE change by validating the atomic diff, committing it, pushing a task branch, opening a ready pull request, requesting Codex review, enabling automatic squash merge, and proving the exact merge revision's post-merge checks. Use when the user asks to ship, publish, or open and merge a pull request."
---

# Ship a change

1. Read `AGENTS.md`, then inspect `git status --short --branch`, the unstaged diff, the staged diff, the branch diff against `origin/main`, and recent repository history.
2. Confirm the intended change is atomic and that unrelated user work will remain untouched. Stop if the shipping scope is ambiguous.
3. Fetch `origin/main`. Verify that the current branch is a `codex/<short-slug>` task branch whose pull-request base is `main`; never commit on `main`. If the branch predates current `origin/main`, assess whether it must be reconciled before shipping and preserve shared history and user work.
4. Stage only intended paths; never use `git add -A`. Exclude `.env*`, credentials, `.index/`, caches, `.DS_Store`, editor state, and other generated files. Staging first ensures pre-commit includes intended new files in its all-files run.
5. Run the free validation commands:

   ```bash
   .venv/bin/pre-commit run --all-files
   .venv/bin/python -m pytest eval/ -q
   .venv/bin/python -m pytest --cov --cov-report=term-missing eval/ -q
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

10. Wait for Codex review and every repository check to complete. Resolve every actionable finding. Any push changes the pull-request head and invalidates all earlier Codex review coverage: rerun the free suite, post the canonical review request again, and wait for a new completed review before proceeding. Do not enable automatic merge while an actionable finding, unresolved review thread, or failing check remains.
11. Prove that Codex reviewed the current pull-request head. Read the latest completed Codex review response, record the SHA shown in its `Reviewed commit` field, resolve that reference to a full commit SHA, and require exact equality with the current `headRefOid`:

    ```bash
    PR_NUMBER='replace-with-PR-number'
    CODEX_REVIEWED_REF='replace-with-latest-Codex-reviewed-SHA'
    pr_head=$(gh pr view "$PR_NUMBER" --json headRefOid --jq .headRefOid)
    codex_reviewed_head=$(gh api "repos/{owner}/{repo}/commits/$CODEX_REVIEWED_REF" --jq .sha)
    test "$codex_reviewed_head" = "$pr_head"
    ```

    Do not infer exact-head coverage from a review timestamp, reaction, or review of an earlier commit. If the latest completed Codex response does not identify a reviewed commit, request a manual exact-head review and wait. Repeat this equality gate immediately before enabling automatic merge so a concurrent push cannot make the review stale.
12. Verify that the ready pull request targets `main`, is mergeable, and contains only the intended atomic change. Enable automatic merge using the squash method:

    ```bash
    set -euo pipefail
    : "${PR_NUMBER:?set PR_NUMBER to the pull-request number}"
    : "${CODEX_REVIEWED_REF:?set CODEX_REVIEWED_REF to the latest reviewed SHA}"
    pr_head=$(gh pr view "$PR_NUMBER" --json headRefOid --jq .headRefOid)
    codex_reviewed_head=$(gh api "repos/{owner}/{repo}/commits/$CODEX_REVIEWED_REF" --jq .sha)
    test "$codex_reviewed_head" = "$pr_head"
    gh pr merge --auto --squash --match-head-commit "$pr_head" "$PR_NUMBER"
    ```

    Report any branch-protection, review, or check requirement that prevents auto-merge.
13. Wait until GitHub reports the pull request as `MERGED`, resolve its exact squash-merge revision, and require both the published GitHub `main` ref and `origin/main` to equal that revision:

    ```bash
    set -euo pipefail
    : "${PR_NUMBER:?set PR_NUMBER to the merged pull-request number}"
    test "$(gh pr view "$PR_NUMBER" --json state --jq .state)" = "MERGED"
    test "$(gh pr view "$PR_NUMBER" --json baseRefName --jq .baseRefName)" = "main"
    merge_sha=$(gh pr view "$PR_NUMBER" --json mergeCommit --jq '.mergeCommit.oid // empty')
    test -n "$merge_sha"
    test "$(gh api repos/{owner}/{repo}/git/ref/heads/main --jq .object.sha)" = "$merge_sha"
    git fetch origin main
    test "$(git rev-parse refs/remotes/origin/main)" = "$merge_sha"
    ```

    This equality is deliberately strict. If `main` advances before the delivery evidence is complete, stop and report the mismatch instead of substituting an ancestry check or silently attributing later evidence to this delivery.
14. Wait for the exact merge SHA's applicable post-merge workflows. Require the `CI` push run and the default-setup `CodeQL` dynamic run. Also require the native `Dependency Graph` dynamic run when the pull request changed `uv.lock`. Workflow registration is asynchronous, so poll briefly for each expected run before treating it as missing:

    ```bash
    set -euo pipefail
    : "${PR_NUMBER:?set PR_NUMBER to the merged pull-request number}"
    : "${merge_sha:?resolve merge_sha from the merged pull request}"

    needs_dependency_graph=$(
      gh pr view "$PR_NUMBER" --json files --jq '
        any(.files[]; .path == "uv.lock")
      '
    )

    wait_for_exact_run() {
      local workflow=$1
      local event=$2
      local attempt=0
      local run_id=

      while [ "$attempt" -lt 60 ]; do
        run_id=$(
          gh run list \
            --commit "$merge_sha" \
            --branch main \
            --workflow "$workflow" \
            --event "$event" \
            --limit 20 \
            --json databaseId,headSha \
            --jq 'map(select(.headSha == "'"$merge_sha"'")) | first | .databaseId // empty'
        )
        if [ -n "$run_id" ]; then
          break
        fi
        attempt=$((attempt + 1))
        sleep 5
      done

      if [ -z "$run_id" ]; then
        printf 'No exact-SHA %s run appeared for %s\n' "$workflow" "$merge_sha" >&2
        return 1
      fi

      if ! gh run watch "$run_id" --exit-status; then
        gh run view "$run_id" --json workflowName,event,headSha,status,conclusion,url
        return 1
      fi
      test "$(gh run view "$run_id" --json workflowName --jq .workflowName)" = "$workflow"
      test "$(gh run view "$run_id" --json event --jq .event)" = "$event"
      test "$(gh run view "$run_id" --json headSha --jq .headSha)" = "$merge_sha"
      test "$(gh run view "$run_id" --json status --jq .status)" = "completed"
      test "$(gh run view "$run_id" --json conclusion --jq .conclusion)" = "success"
      gh run view "$run_id" --json workflowName,event,headSha,status,conclusion,url
    }

    wait_for_exact_run "CI" "push"
    wait_for_exact_run "CodeQL" "dynamic"
    if [ "$needs_dependency_graph" = "true" ]; then
      wait_for_exact_run "Dependency Graph" "dynamic"
    fi

    free_check_count=$(
      gh api "repos/{owner}/{repo}/commits/$merge_sha/check-runs?per_page=100" --jq '
        [
          .check_runs[]
          | select(
              .name == "Free deterministic validation"
              and .status == "completed"
              and .conclusion == "success"
            )
        ] | length
      '
    )
    test "$free_check_count" -ge 1
    test "$(gh api repos/{owner}/{repo}/git/ref/heads/main --jq .object.sha)" = "$merge_sha"
    ```

    A missing, skipped, cancelled, timed-out, or failed expected run is a delivery failure. Report its run URL when available and stop. Do not dispatch, rerun, or automatically fix a post-merge workflow without new maintainer approval.
15. Only after the post-merge evidence passes, return the workspace to a clean, synchronized `main`. Delete only the obsolete local task branch after proving that it is the merged pull request's `codex/` head. Squash merging means the guarded local deletion requires `-D` because the task commit is not an ancestor of `main`:

    ```bash
    set -euo pipefail
    : "${PR_NUMBER:?set PR_NUMBER to the merged pull-request number}"
    test "$(gh pr view "$PR_NUMBER" --json state --jq .state)" = "MERGED"
    merge_sha=$(gh pr view "$PR_NUMBER" --json mergeCommit --jq '.mergeCommit.oid // empty')
    test -n "$merge_sha"
    test "$(gh api repos/{owner}/{repo}/git/ref/heads/main --jq .object.sha)" = "$merge_sha"
    test -z "$(git status --porcelain)"
    task_branch=$(git branch --show-current)
    test "${task_branch#codex/}" != "$task_branch"
    pr_head=$(gh pr view "$PR_NUMBER" --json headRefOid --jq .headRefOid)
    test "$(git rev-parse "$task_branch")" = "$pr_head"
    git switch main
    git fetch --prune origin
    git pull --ff-only origin main
    test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
    test "$(git rev-parse HEAD)" = "$merge_sha"
    test -z "$(git status --porcelain)"
    git branch -D "$task_branch"
    test -z "$(git branch --list "$task_branch")"
    test "$(git branch --show-current)" = "main"
    test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
    test -z "$(git status --porcelain)"
    ```

    Never manually delete the remote task branch; allow the repository's automatic branch deletion to own that action, and let `git fetch --prune` remove the obsolete tracking reference. Never delete or modify an unrelated local or remote branch. Stop and report the mismatch instead of weakening any guard.
16. Do not update issues, boards, labels, milestones, or releases.
