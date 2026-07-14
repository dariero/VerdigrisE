# VerdigrisE repository guidance

## Project boundaries

VerdigrisE is a public, local-first sandbox for inspectable RAG mechanics. Keep the corpus, vectors, retrieval order, prompt payload, answer, and evaluation record directly auditable.

- Preserve the flat architecture: `corpus.py` owns the executable fixture, `models.py` owns capture contracts, `pipeline.py` owns the synchronous embed/index/retrieve/generate flow, and `eval/` owns verification.
- Preserve explicit NumPy cosine retrieval, stable-id tie-breaking, top-2 context, citation metadata, verbatim prompt capture, validated index persistence, and exact abstention unless the task intentionally changes those contracts.
- Deterministic pytest owns ids, rank, collision materialization, values, units, qualifiers, citations, prompt bytes, distance math, provider mappings, and exact abstention. RagaliQ owns only faithfulness and answer relevance after those exact checks pass.
- Do not introduce document ingestion, a vector database, reranking, services, packaging, or other production layers as incidental changes.
- Treat executable source, tests, and configuration as more authoritative than comments or prose. Resolve conflicts and update stale documentation instead of copying it here.

## Tooling and dependencies

- Support Python 3.14 only, as defined by `.python-version`, `pyproject.toml`, and `pylock.toml`.
- Use uv for environment management. `pyproject.toml` is the abstract dependency authority; the hash-bearing `pylock.toml` is the exact public-install contract. Change them together.
- Public clones must install the published `ragaliq` artifact. Never commit an editable sibling checkout, local path, machine-specific URL, or private package source.
- Do not add or remove direct dependencies, change constraints, broaden the Python range, or introduce development tooling without explicit approval. Use `.agents/skills/upgrade-dependencies/SKILL.md` for approved dependency work.
- Ruff is the configured formatter and linter. Mypy strictly checks the application modules and RagaliQ adapter. Pytest-cov enforces the measured branch-coverage floor over that application/adapter scope. Pre-commit runs the static tools and repository-hygiene hooks. This repository has no build backend or task runner; use only the commands documented in `README.md`.

## Testing and paid-call safety

After code, fixture, dependency, or test changes, run the free deterministic suite and branch-coverage gate:

```bash
.venv/bin/python -m pytest eval/ -q
.venv/bin/python -m pytest --cov --cov-report=term-missing eval/ -q
```

For documentation-only changes, verify every documented command and source reference against the repository. The default pytest marker policy must continue to exclude `openai` and `rag_test`; any CI must use that free boundary and must not expose provider keys or select paid markers.

Any command that can contact OpenAI or Anthropic requires explicit approval for that exact invocation. This includes paid pytest tiers, `pipeline.py ingest`, `pipeline.py ask`, and live use of `ask()`. Approval does not carry to a manually initiated retry, another command, or another tier. Before requesting approval, state the exact command, providers and models, selected cases, nominal request fan-out, provider-retry uncertainty, and cost uncertainty. The RagaliQ cost limit is an approximate post-test guard, not a hard pre-spend cap. Never print, persist, or commit provider keys. Use `.agents/skills/run-paid-evaluation/SKILL.md` for paid work.

## Git and pull requests

- `main` is protected. Start task branches from current `origin/main`; use `codex/<short-slug>` for Codex work and never push directly to `main`.
- Preserve unrelated user work. Keep commits and pull requests atomic, stage explicit paths, and exclude secrets, `.env` files, `.index/`, caches, editor files, and other generated artifacts.
- Use a concise conventional commit subject consistent with repository history. Do not add AI attribution trailers unless requested.
- Before shipping, run every pre-commit hook against all files, run the supported free suite, and inspect the complete staged diff.
- When asked to ship, open a ready pull request, request the repository-specific Codex review below, resolve actionable findings, and enable automatic squash merge. After merge, return to synchronized `main` and delete only the obsolete merged local `codex/` task branch. Do not update issues, boards, labels, milestones, releases, remote branches, or unrelated local branches unless separately requested.
- Use `.agents/skills/ship/SKILL.md` for the publishing workflow.

## Authoritative references

- `README.md`: purpose, installation, execution tiers, architecture, and development commands
- `corpus.py`: executable corpus and golden contract
- `pipeline.py` and `models.py`: runtime and public capture contracts
- `eval/test_verdigrise.py`: deterministic and paid ownership boundaries
- `eval/ragaliq_adapter.py`: RagaliQ integration boundary
- `pyproject.toml` and `pylock.toml`: dependency, interpreter, pytest, marker, and coverage policy

## Review guidelines

Report only actionable issues introduced by the pull request, with the affected path and concrete consequence. Prioritize correctness, safety, compatibility, reproducibility, and regression risk; omit style-only findings already clear from local conventions.

- Check retrieval rank, stable-id tie-breaking, distance calculation, citations, prompt bytes, index schema and fingerprints, per-file atomic replacement, fail-closed validation of a stable persisted pair, generation parsing, and `RagRecord` alignment for unintended behavior changes. Do not claim crash transactionality or concurrent reader/writer safety; the current implementation has no durability sync, locking, rollback, or versioned-generation switch.
- Reject transfers of exact facts, units, qualifiers, ordering, citations, or abstention from deterministic pytest to probabilistic RagaliQ judging.
- Flag any path that could execute OpenAI or Anthropic calls without an explicit paid marker and per-run approval.
- Require `pyproject.toml` and `pylock.toml` consistency, hashes, Python `==3.14.*` resolution, and published-package provenance; flag unexplained lock churn.
- Check Python 3.14 compatibility and avoid assumptions derived from another interpreter version.
- Reject absolute local paths, sibling-checkout requirements, private indexes, credentials, secret values, or developer-machine state that breaks a public clone.
- Treat corpus order, stable ids, verbatim text, golden literals, collision siblings, exact absence, fixed vectors, expected ranks, and expected answers as coupled fixture data.
- Flag backward-incompatible changes to `ask()`, the CLI, capture models, persisted-index schema, or documented RagaliQ integration surfaces unless intentional and documented.
- Ensure pytest markers and any CI selection preserve the free default and cannot silently include paid tests.
- Flag stale README commands, versions, model names, file layouts, ownership claims, and missing documentation for behavior changes.
- Exclude generated files and unrelated edits from atomic pull requests.

Use this repository-specific review request:

```text
@codex review for deterministic/RagaliQ ownership, dependency reproducibility, Python 3.14 compatibility, public-clone portability, paid-call safety, golden-fixture integrity, marker correctness, public API compatibility, and unintended behaviour changes
```
