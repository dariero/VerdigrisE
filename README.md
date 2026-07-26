# VerdigrisE: An Inspectable RAG Mechanics Sandbox

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![RagaliQ 0.2.0](https://img.shields.io/badge/RagaliQ-0.2.0-7c3aed.svg)](https://pypi.org/project/ragaliq/0.2.0/)
[![CI](https://github.com/dariero/VerdigrisE/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/dariero/VerdigrisE/actions/workflows/ci.yml)
![Status: Public Sandbox](https://img.shields.io/badge/status-public%20sandbox-6b7280.svg)
![Evaluation: Deterministic First](https://img.shields.io/badge/evaluation-deterministic%20first-0f766e.svg)

**VerdigrisE** is a public, local-first RAG sandbox built around an executable alchemical-blossom grimoire fixture. It keeps corpus strings, embedding rows, indexed metadata, top-2 retrieval, prompt bytes, generated answers, and evaluation records inspectable — **numeric source conflicts**, **near-synonym collisions**, **mandatory qualifiers**, **original-unit preservation**, and **exact abstention** all become testable failures — while preserving a direct ladder toward citation-critical technical-document systems.

---

## Why VerdigrisE?

When a RAG pipeline answers fluently, how do you know the number came from the right source? How do you catch an answer that swaps a near-synonym's dosage for the one you asked about? How do you prove the system abstains when the corpus genuinely lacks the fact?

Production retrieval systems can hide source attribution errors behind fluent, grounded prose. VerdigrisE makes those errors observable with numeric source conflicts, near-synonym collisions, mandatory qualifiers, original-unit preservation, and an exact out-of-corpus abstention case — each materialized as a golden case with exact expected evidence and forbidden literals.

> **Scope:** This rung deliberately stops before PDF parsing, table extraction, hybrid retrieval, reranking, access controls, or service orchestration. It is for tracing the core mechanics and fixing evaluation ownership before adding production infrastructure — see [Production Boundary](#production-boundary).

---

## Assertions That Cannot Fail

A passing test proves an assertion ran. It does not prove the assertion constrains anything.

The distinction has a name in QA practice. A *test case* exercises the system; an *assertion* is the claim inside it that must hold; a *quality gate* is the rule that stops a change when an assertion breaks. None of those three guarantees the fourth thing, which is the only one that matters: that the assertion would have failed had the behaviour been wrong. An assertion that would pass against a broken implementation is decorative. It costs the same to run as a real one and it reports the same green.

The standard way to measure this is *mutation testing*: deliberately break the code, run the suite, and see whether it notices. A break the suite catches is a killed mutant, and it proves the assertion covering it is load-bearing. A break the suite ignores is a surviving mutant, and it names a defect class the suite cannot see. Line coverage cannot substitute for this, because a line executed by a tautological assertion is a covered line.

VerdigrisE inverts the usual arrangement. In an ordinary project the mutants are hypothetical and a tool injects them. Here they are already planted, by hand, in `corpus.py`, and they are the fixture's entire reason for existing:

- Three grimoires prescribe three conflicting doses for the same elixir, so an answer that cites the wrong source is a wrong number rather than a stylistic slip.
- `moonpetal-silver-vapor` and `moonflower-golden-vapor` (`corpus.py:47`, `corpus.py:59`) are near-synonyms with different quantities, so swapping them is undetectable by fluency.
- `shadeglass-orchid-harvest` (`corpus.py:110`) carries a qualifier that is only valid when the plant was grown entirely in shade, so dropping the condition turns a correct value into a wrong instruction.
- One golden case asks for a fact the corpus genuinely does not contain, so confident invention is a failure rather than a judgement call.

The pipeline is the code under test. The evaluation suite is what is being validated. That inversion sets the standard: a green suite against a softened corpus has not passed, it has failed, and it has failed silently. Weakening a trap to make a test go green destroys the only evidence the test was ever producing.

This is not hypothetical here. A mutation sweep of the free suite found that every assertion it failed to kill shared one shape: a value read from the artifact under test and compared against itself. Deleting the shade qualifier from the evidence text left the suite green, because the qualifier was asserted against `expected_answer`, which the fixture also authored. `FixedAnswerGenerator` (`eval/test_verdigrise.py:297`) returns `case.expected_answer` verbatim, so any assertion comparing the answer to the fixture's own literal is a statement about the fixture agreeing with itself. Each of those assertions looked correct when read. Each passed forever.

The repairs are visible in the suite. `test_required_qualifiers_appear_in_the_expected_evidence_text` (`eval/test_verdigrise.py:710`) now checks the qualifier against the corpus text rather than against the answer. `test_every_collision_sibling_conflict_literal_is_forbidden` (`:730`) checks completeness of every trap family rather than one. `test_corpus_order_is_pinned` (`:412`) and `test_condition_metadata_is_pinned` (`:431`) pin fixture data that was previously free to drift.

Because reading an assertion is not enough to tell the two apart, `.agents/skills/maintain-golden-contract/SKILL.md:31-52` makes the proof mechanical. Every new or changed assertion must name the single edit that should break it, and that edit must actually be executed: snapshot the file's bytes, apply one minimal mutation, run the free suite with markers selected explicitly, confirm the named node FAILS, restore the exact bytes, confirm green, and record the FAILED node id in the pull-request body.

Reasoning that an assertion would fail does not satisfy the step. If the loop comes back green, it has not proved the assertion is fine. It has found a defect.

---

## Key Features

| Capability | What It Does | How It Helps |
|---|---|---|
| **Executable Grimoire Fixture** | Keeps eight short corpus entries and seven golden cases together in Python | Source facts and expected failures remain directly inspectable |
| **Adversarial Retrieval Traps** | Exercises source-scoped numbers, near-synonyms, qualifiers, and genuine absence | Plausible wrong-source answers become detectable failures |
| **Explicit NumPy Retrieval** | Shows normalization, cosine similarity, distance, top-2 ranking, and tie-breaking | Vector mechanics remain visible without a database abstraction |
| **Transactional Persistence** | Publishes each normalized `float32` matrix and fingerprinted manifest as an immutable generation, then atomically switches one active pointer | Readers resolve either the complete old or complete new generation during a save; validation still rejects malformed, mismatched, stale, or numerically invalid state |
| **Exact Prompt Capture** | Preserves verbatim retrieved text, citation labels, request messages, and answer text | Every generation input is available for deterministic assertions and audit review |
| **Deterministic-First Tests** | Uses fixed vectors, fixed answers, fake provider endpoints, and exact assertions by default | Local regression behavior does not depend on API availability or judge variance |
| **All-Golden Provider Acceptance** | Runs every golden case through real OpenAI embeddings and generation when explicitly selected | The configured providers are checked without weakening the free default suite |
| **RagaliQ Semantic Boundary** | Uses a canned transport for free structural coverage and native Claude judging for paid semantics | Exact facts stay with pytest; faithfulness and relevance stay with RagaliQ |

---

## Installation

VerdigrisE supports Python 3.14 (`>=3.14,<3.15`). RagaliQ 0.2.0 sets the same minimum, and Python 3.14 is the only version resolved and tested by this sandbox.

Install [uv](https://docs.astral.sh/uv/), then create and synchronize the environment from the repository root:

```bash
uv venv --python 3.14
uv pip sync --preview-features pylock --require-hashes pylock.toml
uv pip check
```

`pyproject.toml` is the abstract dependency authority: NumPy, OpenAI, and Pydantic are runtime dependencies; pytest, pytest-cov, and RagaliQ are test/evaluation dependencies; mypy, pre-commit, pre-commit-hooks, and Ruff are development tooling. The committed `uv.lock` preserves uv's solver decisions and dependency graph. The universal, hash-bearing `pylock.toml` is exported from that solver state and remains the exact cross-platform public-install contract. The current locks are validated and reproduced with uv 0.11.30 for Python 3.14, and install the public `ragaliq==0.2.0` release rather than relying on an adjacent checkout.

> **Platform support:** The command examples use POSIX shell syntax and paths. The free validation path is exercised locally on macOS and continuously on Ubuntu through GitHub Actions; native Windows command syntax and execution are not currently tested or claimed.

Set provider keys only for explicitly selected paid paths:

```bash
export OPENAI_API_KEY="replace-with-your-openai-api-key"
export ANTHROPIC_API_KEY="replace-with-your-anthropic-api-key"
```

VerdigrisE does not load `.env` files. `OPENAI_API_KEY` owns live embedding and generation calls. `ANTHROPIC_API_KEY` owns RagaliQ's native Claude judge calls. The free default suite requires neither key.

> **Paid-call boundary:** Do not run a paid command without explicit approval. Every VerdigrisE-owned OpenAI client disables SDK retries with `max_retries=0` and applies a 120-second timeout to each network operation. That timeout is not a whole-command deadline, and a timed-out request may still be processed or billed. RagaliQ's `--ragaliq-cost-limit` is an approximate post-test guard based on recorded token counts, not a strict pre-spend cap.

Before the corpus embedding request, ingestion rejects an existing index or generations target that is a symbolic link or not a directory. Save-time validation repeats these checks; the preflight cannot prevent later permission or capacity failures, or a target change between preflight and save.

---

## Quick Start

### Python API

The public entry point is synchronous and returns the complete capture model:

```python
from pipeline import ask

record = ask("What quantity of pearl salt and which vapor are specified for ground moonpetal?")

print(record.retrieved_ids)
print(record.context_payload)
print(record.answer)
```

Run it from the repository root after paid ingestion:

```bash
.venv/bin/python -c \
  'from pipeline import ask; print(ask("What quantity of pearl salt and which vapor are specified for ground moonpetal?").model_dump_json(indent=2))'
```

> **Lifecycle:** `ask()` resolves one active generation under `.index/`, rejects an index stale relative to the fixture or configured embedding model before provider initialization, embeds the question separately, retrieves top-2, generates once, and returns a `RagRecord`. It makes paid OpenAI calls.

### Pytest Integration

Run the free deterministic suite:

```bash
.venv/bin/python -m pytest eval/ -q
```

The native pytest configuration in `pyproject.toml` strictly registers markers and excludes both paid provider acceptance and paid native judge tests by default.

### CLI

```bash
# Paid: batch-embed the corpus, persist a generation, and atomically activate it
.venv/bin/python pipeline.py ingest --debug

# Paid: embed one query, retrieve top-2, and generate one answer
.venv/bin/python pipeline.py ask \
  "Under GRIM-UMBRAL-BOTANY, when may a shadeglass orchid be harvested?" \
  --debug
```

With `--debug`, the additional `verdigrise_embedding_debug` line includes only stage, model, input count, vector dimensions, matrix shape, and stable ids; that diagnostic line does not print corpus text, query text, vector values, or secrets. The normal `ask` command output still prints the complete `RagRecord`, including the question, retrieved text, context payload, generation messages, and answer, so treat aggregate stdout as content-bearing.

---

## Evaluation Ownership

VerdigrisE splits evaluation between two owners — exact concerns belong to deterministic pytest, semantic concerns belong to RagaliQ:

| Concern | Owner | Assertion Surface |
|---|---|---|
| Expected ids and rank order | Deterministic pytest | Fixed-vector retrieval records |
| Collision sibling present in top-2 context | Deterministic pytest | Stable ids, verbatim sibling text, and forbidden literals |
| Exact numbers, units, and qualifiers | Deterministic pytest | Golden `must_contain` and `forbidden` fields |
| Exact abstention phrase | Deterministic pytest | Byte-for-byte `INSUFFICIENT_CONTEXT` equality |
| Verbatim context membership and order | Deterministic pytest | Captured `context_payload` |
| Citation metadata and answer citations | Deterministic pytest | Raw entry, `RetrievedChunk`, context label, and answer |
| Distance and similarity consistency | Deterministic pytest | Similarity in `[-1, 1]`, distance in `[0, 2]`, and exact `distance = 1 - cosine_similarity` |
| Provider request and response mappings | Deterministic pytest | Controlled endpoint fakes |
| Prose grounding or faithfulness | RagaliQ | `faithfulness` evaluator |
| Answer relevance | RagaliQ | `relevance` evaluator |

The controlled suite is the authoritative exact regression layer. The full opt-in paid OpenAI tier runs every golden case through the configured embedding and generation providers, then checks a narrower live-acceptance subset: top-2 size and intended-source presence, materialized collisions and context literals, the near-synonym ordering constraint, exact abstention, required and forbidden answer literals and qualifiers, and the expected grimoire identifier and answer citation. It does not replace deterministic proof of exact fixed-vector ranks, distances, prompt bytes, canned answers, or provider request and response mappings. The paid RagaliQ tier receives only answerable cases after those live assertions and owns no ids, numeric literals, units, qualifiers, citations, ordering, or abstention behavior.

---

## Fixture Format

The corpus is a Python fixture, not a document-loader output:

```python
{
    "id": "verdigris-dose-verdant",
    "text": "For tarnish fever, the Verdant Crucible prescribes ...",
    "grimoire_id": "GRIM-VERDANT",
    "folio": 21,
    "subject": "verdigris blossom elixir",
    "fact_type": "tarnish-fever dosage",
    "condition": "distilled in copper and administered after dusk",
}
```

The corpus must contain at least one entry. Every entry has a stable id, verbatim text, `grimoire_id`, `folio`, subject, fact type, and condition. The id, text, subject, fact type, and condition must each contain a non-whitespace character; validation checks presence without trimming or normalizing the stored values. At least one citation field must be populated. Either field may be `None`, but not both; every supplied string citation value must contain a non-whitespace character, and `folio` may also be an integer. IDs and evidence text must also be UTF-8 encodable so every accepted entry can be fingerprinted and persisted.

The complete retrieval-metadata tree, including additional fields, must be acyclic and JSON-persistable: dictionaries with UTF-8 string keys, list or tuple sequences, and JSON scalar leaves with finite floats and UTF-8 strings. This contract is enforced before corpus embedding and at direct-index and persisted-load boundaries. Validation does not trim or rewrite accepted scalar values; JSON persistence represents both list and tuple sequences as arrays.

### Golden Cases

| Case | Expected Evidence | Materialized Conflict or Gap |
|---|---|---|
| `numeric-source-verdigris-dose` | `3 drams`, copper, after dusk, `GRIM-VERDANT` | `9 drams`, amber glass, dawn, `GRIM-AMBER` |
| `numeric-source-amber-dose` | `9 drams`, amber glass, dawn, `GRIM-AMBER` | `15 drams`, basalt, lunar eclipse, `GRIM-OBSIDIAN-PETAL` |
| `numeric-source-obsidian-dose` | `15 drams`, basalt, lunar eclipse, `GRIM-OBSIDIAN-PETAL` | `3 drams`, copper, after dusk, `GRIM-VERDANT` |
| `near-synonym-moonpetal-vapor` | Moonpetal, `11 grains`, silver sleep vapor | Moonflower, `17 grains`, golden waking vapor |
| `conditional-shadeglass-harvest` | `7 moon-phases` only when grown entirely in shade | `10 moon-phases` when grown in full sun |
| `conditional-shadeglass-direct-sun` | Direct sun exposure invalidates the shadeglass harvest window | Sunspire permits `10 moon-phases` when grown in full sun |
| `absent-moonpetal-dew-shelf-life` | Exact `INSUFFICIENT_CONTEXT` | No bottled-dew storage or spoilage duration exists |

Each dosage case materializes one declared collision sibling in top-2 context and separately forbids answer leakage from both non-target dosage sources. The separate `asterquartz-powdering` entry supplies `Mohs hardness 8`. Corpus text preserves `drams`, `grains`, `moon-phases`, and `Mohs hardness` verbatim rather than normalizing units.

---

## RagRecord Capture

`RetrievedChunk` preserves one ranked result. Its metadata is detached from caller-owned
containers and recursively frozen after validation:

```python
class RetrievedChunk(BaseModel):
    id: str
    text: str
    metadata: Mapping[str, object]
    distance: float
    similarity: float
```

`RagRecord` captures the complete query lifetime:

```python
class RagRecord(BaseModel):
    question: str
    retrieved_ids: tuple[str, ...]
    retrieved_chunks: tuple[RetrievedChunk, ...]
    distances: tuple[float, ...]
    context_payload: str
    generation_messages: tuple[PromptMessage, ...]
    answer: str
```

The model validators reject rank misalignment between `retrieved_ids`, `retrieved_chunks`, and `distances`. They also require each retrieval metric to be finite, constrain similarity to `[-1, 1]` and distance to `[0, 2]`, and enforce `distance = 1 - similarity` within an absolute `1e-12` tolerance without clamping or rewriting caller values. Rank-bearing sequences are immutable tuples, and nested metadata mappings and sequences are recursively read-only, so a valid capture cannot silently drift after construction. Metadata accepts only string-keyed mappings, ordered sequences, and JSON scalar leaves; unsupported mutable objects are rejected. Capture object schemas are closed: unknown model fields, including misspelled `model_copy(update=...)` keys and unknown nested chunk or prompt-message fields, are rejected instead of discarded, while retrieval metadata remains the explicitly extensible mapping. Constructors continue to accept ordinary lists and dictionaries, updated model copies are revalidated, and `model_dump()` and `model_dump_json()` preserve the public array and object shapes.

```text
raw corpus dictionary
  -> text-embedding-3-small corpus batch
  -> float64 norm calculation, then normalized float32 rows aligned with stable ids and metadata
  -> validated NumpyVectorIndex
  -> separate text-embedding-3-small query vector
  -> deterministic cosine top-2
  -> rank-ordered RetrievedChunk values
  -> verbatim citation-labelled context payload
  -> gpt-5.6-luna generation at temperature 0
  -> complete RagRecord
  -> deterministic assertions
  -> RagaliQ faithfulness and relevance residue
```

---

## Pytest Reference

### Execution Tiers

| Tier | Command | Calls and Ownership |
|---|---|---|
| Free deterministic contracts | `.venv/bin/python -m pytest eval/ -q` | Fixed embeddings and answers, provider fakes, exact contracts, plus canned-transport RagaliQ wiring |
| Free deterministic branch coverage | `.venv/bin/python -m pytest --cov --cov-report=term-missing eval/ -q` | The same free suite measures application/adapter branches and enforces the 95% floor |
| Paid all-golden OpenAI acceptance | `.venv/bin/python -m pytest -m "openai and not rag_test" eval/ -q` | One corpus embedding batch plus seven query embeddings and seven generations: at most 15 OpenAI SDK attempts |
| Paid cross-family semantic evaluation | `.venv/bin/python -m pytest -m "openai and rag_test" --ragaliq-cost-limit 5.00 eval/ -q` | One corpus embedding batch plus six query embeddings and six generations: at most 13 OpenAI SDK attempts; then six answerable cases judged by native RagaliQ Claude faithfulness and relevance |

The module-scoped paid pipeline embeds the corpus once, while each selected parametrized node asks only its own case. For `n` selected nodes from one paid tier, the maximum OpenAI SDK fan-out is `1 + 2n` attempts: one corpus embedding, `n` query embeddings, and `n` generations. A single selected node therefore makes at most 3 attempts. The complete all-golden tier makes at most 15 attempts for seven cases; the complete semantic tier makes at most 13 OpenAI attempts for six answerable cases before RagaliQ judging. Early failures can reduce these counts, and `max_retries=0` prevents OpenAI SDK retries from multiplying them. A timeout does not prove provider-side cancellation, so final cost remains uncertain. RagaliQ's Claude transport retains its own retry behavior and model-output-dependent fan-out.

### Markers

```python
@pytest.mark.openai   # Paid OpenAI embedding or generation calls
@pytest.mark.rag_test # Paid native RagaliQ Claude judge calls
```

The semantic test carries both markers because it judges live OpenAI answers. Before any provider fixture executes, selected `openai` nodes fail if `OPENAI_API_KEY` is unset or empty, and selected `rag_test` nodes fail if `ANTHROPIC_API_KEY` is unset or empty. Credential-free `--collect-only` remains available for inspecting node selection.

---

## RagaliQ Integration Reference

VerdigrisE installs the published `ragaliq==0.2.0` artifact from PyPI. The release's source tag resolves to the same commit whose public API was inspected for this integration.

| Provenance Field | Value |
|---|---|
| Package source | [`ragaliq==0.2.0` on PyPI](https://pypi.org/project/ragaliq/0.2.0/) |
| Package version | `0.2.0` |
| Source tag | [`v0.2.0`](https://github.com/dariero/RagaliQ/releases/tag/v0.2.0) |
| Inspected commit | `ac1c7ce9e38c9308736afd4400c5a9471c25255c` |
| Free integration path | `CannedJudgeTransport -> BaseJudge -> RagaliQ` |
| Paid integration path | Native `rag_tester` fixture with `ClaudeJudge` |
| Evaluators | `faithfulness`, `relevance` |
| Semantic case handoff | `id`, derived `name`, `query`, `context=[record.context_payload]`, `response`, and semantic tags |

Public objects used are `RagaliQ`, `RAGTestCase`, `RAGTestResult`, `BaseJudge`, `JudgeConfig`, `JudgeTransport`, `TransportResponse`, and `DEFAULT_JUDGE_MODEL`.

RagaliQ 0.2.0 strips surrounding whitespace from the mapped query, response, and each context entry; the source `RagRecord` retains provider text verbatim. The mapping passes `context=[record.context_payload]`, so both retrieved chunks are concatenated into one logical RagaliQ document. This integration therefore cannot expose per-chunk source attribution or context precision. VerdigrisE deliberately leaves RagaliQ's `expected_answer` and `expected_facts` unset. Exact answers, facts, units, qualifiers, citations, ordering, and abstention remain deterministic pytest responsibilities.

Exact signatures relied upon:

```text
RagaliQ.__init__(
    self,
    judge: Literal["claude", "openai"] | LLMJudge = "claude",
    evaluators: list[str] | None = None,
    default_threshold: float = 0.7,
    judge_config: JudgeConfig | None = None,
    api_key: str | None = None,
    max_concurrency: int = 5,
    max_judge_concurrency: int = 20,
    fail_fast: bool = False,
) -> None

RagaliQ.evaluate(self, test_case: RAGTestCase) -> RAGTestResult

BaseJudge.__init__(
    self,
    transport: JudgeTransport,
    config: JudgeConfig | None = None,
    *,
    trace_collector: TraceCollector | None = None,
    max_concurrency: int = 20,
) -> None

async JudgeTransport.send(
    self,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_JUDGE_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> TransportResponse
```

RagaliQ's native pytest plugin supplies `rag_tester`, `ClaudeJudge`, retrying transport, `TraceCollector`, estimated cost summaries, and `--ragaliq-cost-limit`. Its `openai` judge selector is declared but raises `NotImplementedError`, so VerdigrisE does not use it.

---

## Architecture

```
./VerdigrisE/
├── .agents/
│   └── skills/                  # Golden-contract, paid-evaluation, ship, and upgrade procedures
├── .github/
│   ├── CODEOWNERS               # Review routing for policy and contract surfaces
│   └── workflows/
│       └── ci.yml               # Locked, secret-free pull-request validation
├── .pre-commit-config.yaml      # Ruff and repository-hygiene commit hooks
├── .python-version              # Canonical Python 3.14 interpreter line
├── .env.example                 # Provider-key placeholders only
├── .gitignore                   # Secret, environment, index, and cache exclusions
├── AGENTS.md                    # Repository guidance, ownership rules, and review checklist
├── LICENSE                      # MIT license text
├── README.md                    # Setup, contracts, commands, and boundaries
├── SECURITY.md                  # Private vulnerability reporting policy
├── config.py                    # Model, abstention, retrieval, and index constants
├── corpus.py                    # Corpus dictionaries and golden cases
├── models.py                    # RetrievedChunk, PromptMessage, and RagRecord
├── pipeline.py                  # Embed, index, retrieve, prompt, generate, and CLI
├── pylock.toml                  # Universal, hash-bearing exact environment
├── pyproject.toml               # Metadata, dependencies, pytest, and coverage policy
├── uv.lock                      # Solver decisions and dependency graph
└── eval/
    ├── __init__.py
    ├── check_agent_policy_symbols.py  # Resolves agent-policy symbol references against tracked source
    ├── conftest.py              # Flat-module imports and fail-closed paid-credential preflight
    ├── ragaliq_adapter.py       # Canned structural transport and case mapping
    └── test_verdigrise.py       # Exact, provider-acceptance, and semantic tiers
```

The four pipeline stages are:

1. Embed the ordered corpus batch or one separate query with the same model.
2. Index aligned corpus rows or search the normalized matrix for deterministic top-2.
3. Build the exact citation-labelled context and generation messages.
4. Generate without rewriting the response and capture every stage in `RagRecord`.

Deterministic query tests construct `NumpyVectorIndex` in memory. After `index()` returns, the index owns a recursive snapshot of caller-supplied metadata; later caller mutations cannot change its fingerprint, retrieval captures, or saved manifest. Free integrity tests also run ingestion, persist and reload the index, and prove that wrong-dtype, non-finite, zero, non-unit, fingerprint-mismatched, fixture-stale, and configured-model-stale state is rejected before provider initialization. Persisted rows must already use the `float32` dtype and have unit length within an absolute `1e-5` tolerance; loading validates the fingerprinted representation directly instead of casting or renormalizing it. CLI `ingest` publishes under `.index/`; CLI `ask` resolves and validates the active generation.

```text
.index/
├── active.json
└── generations/
    └── <32-character-generation-id>/
        ├── manifest.json
        └── vectors.npy
```

`save()` writes and synchronizes both files inside a writer-unique hidden staging directory, synchronizes the directory, atomically publishes it as an immutable generation, and synchronizes `generations/`. It then writes and synchronizes a writer-unique temporary pointer, replaces `active.json` once with `os.replace`, and synchronizes the index directory. `load()` reads that pointer once and validates only the selected immutable generation. On supported local macOS and Ubuntu filesystems, an interruption before the pointer switch leaves the previous generation active; a concurrent reader therefore sees a complete old or new generation rather than a mixed pair. Concurrent writers use disjoint paths, and the last completed pointer switch determines the active generation.

For compatibility, `load()` also accepts the previous root-level `manifest.json` plus `vectors.npy` layout only when `active.json` is absent. New saves never mutate that legacy pair, and any present active pointer is authoritative and fail-closed. Completed inactive generations are retained so readers that already selected them remain safe. This sandbox does not serialize writer order, compare and swap revisions, prune retained or crash-orphaned generations, roll back after activation, guarantee network-filesystem behavior, or bind a generation id into durable `RagRecord` replay provenance.

---

## Development

Ruff is the repository's formatter and linter. Mypy strictly checks the application modules and RagaliQ adapter. Pytest-cov measures branch coverage over the same application/adapter scope and enforces a clean integer `fail_under = 95`; paid provider paths remain excluded. Pre-commit runs the static tools, the repository-hygiene hooks, and one repository-specific gate before commits: `agent-policy-symbols` runs `eval/check_agent_policy_symbols.py`, which resolves every backticked symbol reference in `AGENTS.md` and `.agents/skills/*/SKILL.md` against the names tracked modules actually bind, so a procedure file cannot keep naming a constant that was renamed. It runs over the whole set rather than the changed files, because renaming a constant in a module invalidates a reference in a policy file the same commit never touches. The same check runs as a free-suite node, so a clone that never installs the hook still catches the drift. Every hook uses `uv run --no-sync` and the hash-locked `.venv`; hook execution does not create separate environments or resolve additional packages. The sandbox still has no build backend, package-publication layer, or task runner. uv owns environment creation and exact dependency synchronization.

GitHub Actions validates every ready pull request against its prospective merge result and validates `main` after each merge. CI installs the hash-locked Python 3.14 environment from a clean checkout, validates the pre-commit configuration, runs every repository hook, and runs the deterministic branch-coverage gate with OpenAI and RagaliQ paid markers explicitly excluded. Provider key variables are explicitly empty throughout the job, and the workflow does not reference provider secrets.

GitHub natively ingests the committed `uv.lock` into the dependency graph, preserving exact versions and dependency relationships for Dependabot monitoring. CI independently requires `uv.lock` to satisfy `pyproject.toml` and requires its frozen export to match the committed `pylock.toml` byte for byte. The hash-bearing `pylock.toml` remains the public-install contract; it is not submitted as a second dependency inventory, so the graph has one full-fidelity lock authority and no repository workflow needs write permission for dependency submission.

Install the Git hook once per clone:

```bash
.venv/bin/pre-commit install
```

Run every configured hook against all tracked files:

```bash
.venv/bin/pre-commit run --all-files
```

Some hooks apply safe fixes. Review any resulting edits and rerun the command until every hook passes.

After changing `.pre-commit-config.yaml`, validate its schema:

```bash
.venv/bin/pre-commit validate-config
```

Check formatting and linting without changing files:

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
```

Run strict static checking over the application and adapter scope:

```bash
.venv/bin/python -m mypy \
  config.py corpus.py models.py pipeline.py eval/ragaliq_adapter.py
```

Apply Ruff's safe lint fixes, then format the tree:

```bash
.venv/bin/ruff check --fix .
.venv/bin/ruff format .
```

Run the free deterministic suite:

```bash
.venv/bin/python -m pytest eval/ -q
```

Run the same free suite with deterministic branch coverage over `config.py`, `corpus.py`, `models.py`, `pipeline.py`, and `eval/ragaliq_adapter.py`:

```bash
.venv/bin/python -m pytest --cov --cov-report=term-missing eval/ -q
```

When dependency metadata changes, update the committed solver state and regenerate its standardized public-install export with the same Python policy:

```bash
uvx --from uv==0.11.30 uv lock \
  --python 3.14 \
  --prerelease disallow
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

The project export path preserves the selected Python minor in `pylock.toml` as `==3.14.*`. Change `pyproject.toml` when the abstract requirements change, and update `uv.lock` and `pylock.toml` together whenever solver state changes. CI requires `uv.lock` to satisfy `pyproject.toml` and its frozen export to match the committed `pylock.toml` byte for byte.

Model and provider integration remains behind explicit paid markers. A failing all-golden paid test is an acceptance result for the configured model pair, not permission to weaken a fixture expectation.

---

## Documentation

- [corpus.py](corpus.py) — Executable fixture and golden contract
- [pipeline.py](pipeline.py) — Runtime data flow and object lifetimes
- [eval/test_verdigrise.py](eval/test_verdigrise.py) — Deterministic and paid ownership boundaries
- [eval/ragaliq_adapter.py](eval/ragaliq_adapter.py) — Exact RagaliQ context handoff and free structural runner

---

## Comparison with Alternatives

| Capability | ChromaDB | FAISS | Pure NumPy Cosine |
|---|---|---|---|
| Setup overhead | Medium | Medium, including a native binary | Low |
| Dependency weight | High for this fixture | Higher than NumPy alone | Lowest |
| Metadata support | Native | Separate sidecar required | Explicit dictionaries and JSON |
| Persistence | Native | Vector index serialization | Immutable `.npy`-plus-manifest generations with an active pointer |
| Distance transparency | Backend configuration mediates semantics | Normalization and metric choice remain external | Normalization, dot product, distance, and tie-break are visible |
| Teaching value | Database lifecycle | ANN index mechanics | Complete small-corpus data flow |
| Later replacement | Collection adapter | Index plus metadata adapter | Existing provider/index/result seams |

VerdigrisE keeps pure NumPy. Normalization uses a `float64` norm accumulator before storing `float32` rows, avoiding overflow or underflow for finite `float32` inputs. The persisted implementation validates those exact rows without silently transforming them, and bounds floating-point cosine results to `[-1, 1]` before ranking and capture. It adds fixture and vector fingerprints without changing the visible cosine calculation:

```text
raw_similarity = dot(L2_normalized_document, L2_normalized_query)
cosine_similarity = clip(raw_similarity, -1, 1)
distance = 1 - cosine_similarity
rank = descending bounded similarity, then ascending stable chunk id
```

---

## Production Boundary

The next rungs are intentionally visible but not implemented:

1. Replace hand-authored strings with table-aware and clause-aware ingestion while retaining stable source spans and version lineage.
2. Bind answers and evaluations to an exact index generation and retain durable provenance for replay across source editions.
3. Add lexical retrieval, metadata filters, hybrid fusion, and reranking behind the current retrieval-result boundary.
4. Enforce project, document, clause, table, row, or cell access policy before evidence reaches the prompt.
5. Add claim-to-evidence validation, source-span citations, and durable answer audit events.
6. Move the synchronous local boundary into an observable backend with adaptive retry and backoff, rate-limit, cache, cost, and failure policies.
7. Expand RagaliQ semantic evaluation over versioned production fixtures without transferring exact compliance assertions to probabilistic judges.

VerdigrisE remains a sandbox: it does not claim PDF support, table extraction, production storage, web serving, generated reports, or deployment readiness.

### What Each Untaken Rung Costs

A list of next steps is indistinguishable from a list written without evaluating them. The table below states, for each rung this sandbox does not take, why it was not taken, what not taking it costs, and the scale at which that cost begins to bite. Measured rows come from `benchmark_retrieval.py`; unmeasured rows say so rather than estimating.

Reproduce every measured number with:

```bash
.venv/bin/python benchmark_retrieval.py                          # default grid, peaks near 0.8 GiB
.venv/bin/python benchmark_retrieval.py --rows 1000 10000 100000 # adds the largest point, peaks near 5 GiB
```

The default grid stops at `n` = 10,000 so the one-command run is safe on an ordinary machine. The `n` = 100,000 row is opt-in because building an index that size peaks around 5 GiB, which is a cost of the float64 normalization accumulator rather than of the benchmark. Both invocations make no provider calls, need no API key, use synthetic vectors from a pinned seed, and print the machine and library versions they ran on. Timings are machine-specific; the shape of the curve is the finding, not the absolute milliseconds.

| Rung not taken | Why | Cost of not taking it | Scale at which it bites |
|---|---|---|---|
| Vector database instead of an in-memory NumPy scan | Normalization, dot product, distance, and tie-breaking stay visible; a backend would mediate all four | The whole matrix is resident: `pipeline.py:206` stores `float32` rows, so `4 * n * d` bytes. No metadata filter, no incremental upsert, no query language | Measured 585.94 MiB resident at `n` = 100,000, `d` = 1536. Peak is several times that: building the index there took 5,524 MiB, because `_normalize` (`pipeline.py:378-390`) holds the float32 input, a float64 copy, a float64 quotient, and the float32 result at once, then revalidates through another float64 pass |
| Vectorized top-`k` selection instead of a total sort | `pipeline.py:417-420` sorts by `(-similarity, id)`, and expressing that exact documented tie-break as a readable key is worth more here than selection speed | The key is interpreted Python called once per row, and all `n` rows are ordered to return `TOP_K`, which is 2. This is the operation that ends the design, not the vector math | The sort overtakes the matrix product somewhere in the low tens of rows, and the exact point is **not robustly measurable** on ordinary hardware: at that size both operations cost a few microseconds and trial-to-trial variation exceeds the gap between them. Five trials at `d` = 1536 located it at 32, 8, 32, 32, and 32 rows. That instability is itself the finding, because it says the crossover is not a size at which anything fails. Where it matters is at scale, and there it is stable: at `n` = 100,000, `d` = 1536 the sort dominates the matrix product by roughly a factor of three, consistently across nine runs, though neither figure is stable enough to publish |
| Approximate nearest neighbour instead of exact search | Exact search has recall 1.0 and no index-build step or tuning parameters to explain | `pipeline.py:416` touches every row on every query. Latency is linear in `n` before the sort above dominates it | Measured: at `n` = 100,000 and `d` = 128 the matrix product is around a millisecond inside a search of tens of milliseconds, consistently across nine runs, so exact search is not the binding constraint at any size this sandbox reaches |
| Document ingestion and chunking | The corpus is an executable fixture, not loader output, so retrieval granularity is authoring granularity | There is no chunking code at all, so one entry is one indivisible retrieval unit. A long document can be represented, since `corpus.py:17` types `text` as an unrestricted `str` and `validate_corpus` requires only that it be non-blank and UTF-8 encodable, but it would be retrieved whole or not at all, with no sub-document span to cite. Growing the corpus means editing `corpus.py` by hand and hand-authoring a matching fixture vector | Binds immediately: the corpus is 8 entries and every one is hand-written |
| Incremental index updates | An index that cannot change after construction cannot drift from its fingerprint | `pipeline.py:395-396` raises on a second `index()` call, so any corpus change re-embeds the entire corpus in one paid request | Binds at the first corpus edit |
| Provider retry and backoff | Retries multiply spend, and an unbounded fan-out is worse here than a failed run | `OPENAI_MAX_RETRIES = 0` (`config.py:8`) means one transient network failure fails the whole invocation with no recovery. The 120-second timeout (`config.py:9`) is per operation, not a command deadline, and a timed-out request may still be billed | Binds on the first flaky connection |
| Embedding cache | A cache would hide which calls are actually made, which is the thing this sandbox exists to show | Every `ask()` re-embeds its question. Asking the same question twice costs twice | Binds on the second identical query |
| Request batching and concurrency | The synchronous path is readable top to bottom | `pipeline.py:130-134` sends the entire corpus as one `embeddings.create` call with no chunking and no size check, so the ceiling is the provider's per-request input limit rather than anything this repository sets. Not verifiable from repository source | Not measured; the limit is provider-side |
| Compare-and-swap on the active pointer | Writer-unique staging, `fsync` chains, and atomic publication are taken; only revision checking is not | Two concurrent writers both succeed and the last `os.replace` wins, so a slower writer can overwrite a newer generation's activation | Binds with two concurrent writers |
| Managed secrets | Keys are read from the process environment and nowhere else, which removes a whole class of accidental commit | `pipeline.py:91-96` reads `OPENAI_API_KEY` directly. No rotation, no scoping, no short-lived credentials, and `ask` output is content-bearing | Not measured; a policy limit rather than a scale limit |
| Logging, tracing, and metrics | `--debug` prints shapes and stable ids and deliberately withholds text, vectors, and secrets | A paid run that fails before persistence leaves no artifact except the pytest output. One that fails after `pipeline.py:473` publishes the generation leaves a complete, content-bearing generation under `.index/generations/` that no pointer references, because the `finally` block cleans only staging paths. There is no timing, no event stream, and no way to reconstruct what a past run did, so distinguishing an orphan from the active generation means reading `active.json` by hand | Not measured; binds the first time a paid run needs a post-mortem |
| Packaging and deployment | `pyproject.toml:33` sets `package = false`, so the only interface is the documented commands | Not installable and has no entry point, which is also why every command carries a `.venv/bin/python` prefix and why Windows is not claimed | Not measured; binds on first use outside this checkout |

Three regimes, not one crossover. A NumPy call costs something fixed before any arithmetic happens: argument parsing, dtype and shape resolution, and BLAS dispatch. Measured at one row, that floor is about 0.0027 ms and is the same at `d` = 12, `d` = 128, and `d` = 1536, because dispatch does not care how wide the vectors are. Below roughly a thousand rows the matrix product is measuring that floor rather than the arithmetic, which is why its cost barely moves across a 128-fold increase in width: about 0.0023 ms against about 0.0035 ms at `n` = 8. By `n` = 10,000 the same comparison is about 0.026 ms against about 2.0 ms, a factor of roughly 75, and the arithmetic is plainly in charge.

The regime is the robust result. The crossover between the two operations is not, and this section deliberately does not publish a number for it. It falls in the low tens of rows, where both operations cost single-digit microseconds, and there the difference between them is smaller than the run-to-run variation, so repeated determination on a step-2 grid returns different answers: five trials at `d` = 1536 gave 32, 8, 32, 32, and 32 rows. Publishing a point estimate, or even a tight bracket, would claim a precision the hardware does not deliver. It also would not matter if it did, because a reader who took "the sort wins at fourteen documents" as "this design fails at fourteen documents" would be wrong by four orders of magnitude.

The decision-relevant number is absolute latency for a single query, and it is stated here without a verdict because this repository sets no latency target. What counts as acceptable is a deployment decision that VerdigrisE does not make.

| Corpus size | Single-query search latency |
|---|---|
| 8, the shipped corpus | tens of microseconds |
| 1,000 | a few hundred microseconds |
| 10,000 | single-digit milliseconds |
| 100,000 | tens of milliseconds, occasionally past 100 |

No numeric interval is published, and that is a result rather than caution. Two earlier revisions tried: first single values, then the lowest and highest of nine runs per cell with endpoints rounded outward. Both were falsified by the next run. The nine-run ranges were checked cell by cell to contain every observation behind them, and a single further run from a clean clone landed outside four of nine cells immediately.

Run-to-run spread reaches 41% at 100,000 entries, which is larger than the differences a reader would want an interval to resolve, so any interval drawn from a finite sample is a statement about that sample rather than about the system. Order of magnitude is what survives, and it is stated with its edge rather than as a clean decade. The first three labels held across every one of thirty runs using the documented command. At 100,000 entries the observations cross out of the tens, so that row's label says so; treating it as a clean decade would repeat in one word the error the intervals made in two numbers.

One comparison is robust enough to state. Wider vectors are consistently slower at every size measured: the `d` = 12 and `d` = 1536 observations never overlapped at any corpus size, though both stay within the same order of magnitude. The slowest observations anywhere were at 100,000 entries and `d` = 1536.

This is the same standard already applied to the crossover above, retracted rather than re-bracketed once trial noise exceeded the gap being measured.

Read against a threshold a reader supplies: single-query search is tens of microseconds at the shipped corpus size, a few hundred microseconds at 1,000 entries, single-digit milliseconds at 10,000, and tens of milliseconds at 100,000, occasionally past a hundred. The resident matrix at `n` = 100,000 and `d` = 1536 is 585.94 MiB, which is exact arithmetic on row count and width rather than a timing, and is the one figure here that carries full precision.

One measured fact inverts the expected answer and is the one that applies today. At the size this sandbox actually runs, 8 entries at width 12, the matrix product and the sort together are about 0.0037 ms of an approximately 0.030 ms search, so the two operations that scale account for **at most about an eighth** of it.

The remaining seven eighths is deliberately left **unassigned** rather than attributed. Two things prevent attributing it. The isolated matrix product writes through a preallocated buffer and clips in place, while `search()` at `pipeline.py:416` allocates both results, so the isolated figure understates the real one and the difference lands in the remainder. And the three timings are medians aggregated independently, so subtracting them does not yield the median of the difference. The remainder therefore contains query normalization and the construction of two `RetrievedChunk` capture models, but also allocation, method-call overhead, and whatever else `search()` does between its first and last line. Naming only the first two would be an ownership claim the measurement does not support.

What survives is the conclusion that matters: at the real corpus size the scaling operations are a minority of an already negligible cost, so optimizing either would change nothing a user could perceive.

---

## Why "VerdigrisE"?

**Verdigris** + **E**val = **VerdigrisE**

Verdigris is the green patina that surfaces on weathered copper — corrosion made visible. It is also the fixture's central subject: the **verdigris blossom elixir**, a compound whose dosage is prescribed three conflicting ways by three rival grimoires, and whose canonical preparation is distilled in copper. The **E** stands for the evaluation of RAG (Retrieval-Augmented Generation) mechanics that the sandbox exists to make inspectable.

Because a RAG system's most dangerous failures are the ones that never surface. VerdigrisE exists to make wrong-source numbers, swapped near-synonyms, dropped qualifiers, and confident answers-from-nothing visible as exact, repeatable test failures.

---

## License

VerdigrisE is licensed under the [MIT License](LICENSE). The RagaliQ dependency is separately MIT licensed — see [RagaliQ's LICENSE](https://github.com/dariero/RagaliQ/blob/v0.2.0/LICENSE).
