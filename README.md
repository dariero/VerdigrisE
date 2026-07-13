# VerdigrisE: An Inspectable RAG Mechanics Sandbox

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![RagaliQ 0.2.0](https://img.shields.io/badge/RagaliQ-0.2.0-7c3aed.svg)](https://pypi.org/project/ragaliq/0.2.0/)
![Status: Public Sandbox](https://img.shields.io/badge/status-public%20sandbox-6b7280.svg)
![Evaluation: Deterministic First](https://img.shields.io/badge/evaluation-deterministic%20first-0f766e.svg)

**VerdigrisE** is a public, local-first RAG sandbox built around an executable alchemical-blossom grimoire fixture. It keeps corpus strings, embedding rows, indexed metadata, top-2 retrieval, prompt bytes, generated answers, and evaluation records inspectable — **numeric source conflicts**, **near-synonym collisions**, **mandatory qualifiers**, **original-unit preservation**, and **exact abstention** all become testable failures — while preserving a direct ladder toward citation-critical technical-document systems.

---

## Why VerdigrisE?

When a RAG pipeline answers fluently, how do you know the number came from the right source? How do you catch an answer that swaps a near-synonym's dosage for the one you asked about? How do you prove the system abstains when the corpus genuinely lacks the fact?

Production retrieval systems can hide source attribution errors behind fluent, grounded prose. VerdigrisE makes those errors observable with numeric source conflicts, near-synonym collisions, mandatory qualifiers, original-unit preservation, and an exact out-of-corpus abstention case — each materialized as a golden case with exact expected evidence and forbidden literals.

> **Scope:** This rung deliberately stops before PDF parsing, table extraction, hybrid retrieval, reranking, access controls, or service orchestration. It is for tracing the core mechanics and fixing evaluation ownership before adding production infrastructure — see [Production Boundary](#production-boundary).

---

## Key Features

| Capability | What It Does | How It Helps |
|---|---|---|
| **Executable Grimoire Fixture** | Keeps eight short corpus entries and six golden cases together in Python | Source facts and expected failures remain directly inspectable |
| **Adversarial Retrieval Traps** | Exercises source-scoped numbers, near-synonyms, qualifiers, and genuine absence | Plausible wrong-source answers become detectable failures |
| **Explicit NumPy Retrieval** | Shows normalization, cosine similarity, distance, top-2 ranking, and tie-breaking | Vector mechanics remain visible without a database abstraction |
| **Validated Persistence** | Stores normalized rows and an inspectable manifest with corpus and vector fingerprints | Loading rejects incomplete files, fingerprint mismatches, and indexes stale relative to `corpus.py` |
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

`pyproject.toml` is the abstract dependency authority: NumPy, OpenAI, and Pydantic are runtime dependencies; pytest and RagaliQ are test/evaluation dependencies; mypy, pre-commit, pre-commit-hooks, and Ruff are development tooling. The universal, hash-bearing `pylock.toml` records the exact cross-platform environment. It was generated with uv 0.11.28 for Python 3.14, and installs the public `ragaliq==0.2.0` release rather than relying on an adjacent checkout.

An editable `../RagaliQ` install is an optional maintainer-only co-development override, not the public installation contract. Re-running the locked sync command restores the published RagaliQ artifact.

Set provider keys only for explicitly selected paid paths:

```bash
export OPENAI_API_KEY="replace-with-your-openai-api-key"
export ANTHROPIC_API_KEY="replace-with-your-anthropic-api-key"
```

VerdigrisE does not load `.env` files. `OPENAI_API_KEY` owns live embedding and generation calls. `ANTHROPIC_API_KEY` owns RagaliQ's native Claude judge calls. The free default suite requires neither key.

> **Paid-call boundary:** Do not run a paid command without explicit approval. RagaliQ's `--ragaliq-cost-limit` is an approximate post-test guard based on recorded token counts, not a strict pre-spend cap.

---

## Quick Start

### Python API

The public entry point is synchronous and returns the complete capture model:

```python
from pipeline import ask

record = ask(
    "What quantity of pearl salt and which vapor are specified for ground moonpetal?"
)

print(record.retrieved_ids)
print(record.context_payload)
print(record.answer)
```

Run it from the repository root after paid ingestion:

```bash
.venv/bin/python -c \
  'from pipeline import ask; print(ask("What quantity of pearl salt and which vapor are specified for ground moonpetal?").model_dump_json(indent=2))'
```

> **Lifecycle:** `ask()` loads `.index/`, rejects a fixture-stale index, embeds the question separately, retrieves top-2, generates once, and returns a `RagRecord`. It makes paid OpenAI calls.

### Pytest Integration

Run the free deterministic suite:

```bash
.venv/bin/python -m pytest -c pytest.ini eval/ -q
```

`pytest.ini` excludes both paid provider acceptance and paid native judge tests by default.

### CLI

```bash
# Paid: batch-embed the corpus and persist the validated index
.venv/bin/python pipeline.py ingest --debug

# Paid: embed one query, retrieve top-2, and generate one answer
.venv/bin/python pipeline.py ask \
  "Under GRIM-UMBRAL-BOTANY, when may a shadeglass orchid be harvested?" \
  --debug
```

Safe debug output includes only stage, model, input count, vector dimensions, matrix shape, and stable ids. It does not print corpus text, query text, vector values, or secrets.

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
| Distance and similarity consistency | Deterministic pytest | `distance = 1 - cosine_similarity` |
| Provider request and response mappings | Deterministic pytest | Controlled endpoint fakes |
| Prose grounding or faithfulness | RagaliQ | `faithfulness` evaluator |
| Answer relevance | RagaliQ | `relevance` evaluator |

The controlled suite is the authoritative exact regression layer. The paid OpenAI tier re-applies the golden contract to every case but remains opt-in. The paid RagaliQ tier receives only answerable cases after exact checks and owns no ids, numeric literals, units, qualifiers, citations, ordering, or abstention behavior.

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

Every entry has a stable id, verbatim text, `grimoire_id`, `folio`, subject, fact type, and condition. At least one citation field must be populated, and `folio` may be an integer or string.

### Golden Cases

| Case | Expected Evidence | Materialized Conflict or Gap |
|---|---|---|
| `numeric-source-verdigris-dose` | `3 drams`, copper, after dusk, `GRIM-VERDANT` | `9 drams`, amber glass, dawn, `GRIM-AMBER` |
| `numeric-source-amber-dose` | `9 drams`, amber glass, dawn, `GRIM-AMBER` | `15 drams`, basalt, lunar eclipse, `GRIM-OBSIDIAN-PETAL` |
| `numeric-source-obsidian-dose` | `15 drams`, basalt, lunar eclipse, `GRIM-OBSIDIAN-PETAL` | `3 drams`, copper, after dusk, `GRIM-VERDANT` |
| `near-synonym-moonpetal-vapor` | Moonpetal, `11 grains`, silver sleep vapor | Moonflower, `17 grains`, golden waking vapor |
| `conditional-shadeglass-harvest` | `7 moon-phases` only when grown entirely in shade | `10 moon-phases` when grown in full sun |
| `absent-moonpetal-dew-shelf-life` | Exact `INSUFFICIENT_CONTEXT` | No bottled-dew storage or spoilage duration exists |

The separate `asterquartz-powdering` entry supplies `Mohs hardness 8`. Corpus text preserves `drams`, `grains`, `moon-phases`, and `Mohs hardness` verbatim rather than normalizing units.

---

## RagRecord Capture

`RetrievedChunk` preserves one ranked result:

```python
class RetrievedChunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, object]
    distance: float
    similarity: float
```

`RagRecord` captures the complete query lifetime:

```python
class RagRecord(BaseModel):
    question: str
    retrieved_ids: list[str]
    retrieved_chunks: list[RetrievedChunk]
    distances: list[float]
    context_payload: str
    generation_messages: list[PromptMessage]
    answer: str
```

The model validator rejects rank misalignment between `retrieved_ids`, `retrieved_chunks`, and `distances`.

```text
raw corpus dictionary
  -> text-embedding-3-small corpus batch
  -> float32 matrix aligned with stable ids and metadata
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
| Free deterministic contracts | `.venv/bin/python -m pytest -c pytest.ini eval/ -q` | Fixed embeddings and answers, provider fakes, exact contracts, plus canned-transport RagaliQ wiring |
| Paid all-golden OpenAI acceptance | `.venv/bin/python -m pytest -c pytest.ini -o addopts='' -m "openai and not rag_test" eval/ -q` | One corpus embedding batch, then live retrieval and generation for every golden case |
| Paid cross-family semantic evaluation | `.venv/bin/python -m pytest -c pytest.ini -o addopts='' -m "openai and rag_test" --ragaliq-cost-limit 5.00 eval/ -q` | Live OpenAI answers judged by native RagaliQ Claude faithfulness and relevance |

### Markers

```python
@pytest.mark.openai   # Paid OpenAI embedding or generation calls
@pytest.mark.rag_test # Paid native RagaliQ Claude judge calls
```

The semantic test carries both markers because it judges live OpenAI answers. `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are independently checked before the corresponding fixtures execute.

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
| Exact context handoff | `context=[record.context_payload]` |

Public objects used are `RagaliQ`, `RAGTestCase`, `RAGTestResult`, `BaseJudge`, `JudgeConfig`, `JudgeTransport`, `TransportResponse`, and `DEFAULT_JUDGE_MODEL`.

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
├── .pre-commit-config.yaml      # Ruff and repository-hygiene commit hooks
├── .python-version              # Canonical Python 3.14 interpreter line
├── .env.example                 # Provider-key placeholders only
├── .gitignore                   # Secret, environment, index, and cache exclusions
├── README.md                    # Setup, contracts, commands, and boundaries
├── config.py                    # Model, abstention, retrieval, and index constants
├── corpus.py                    # Corpus dictionaries and golden cases
├── models.py                    # RetrievedChunk, PromptMessage, and RagRecord
├── pipeline.py                  # Embed, index, retrieve, prompt, generate, and CLI
├── pylock.toml                  # Universal, hash-bearing exact environment
├── pyproject.toml               # Non-package metadata and dependency authority
├── pytest.ini                   # Free-default marker policy
└── eval/
    ├── __init__.py
    ├── conftest.py              # Flat-module import boundary
    ├── ragaliq_adapter.py       # Canned structural transport and case mapping
    └── test_verdigrise.py       # Exact, provider-acceptance, and semantic tiers
```

The four pipeline stages are:

1. Embed the ordered corpus batch or one separate query with the same model.
2. Index aligned corpus rows or search the normalized matrix for deterministic top-2.
3. Build the exact citation-labelled context and generation messages.
4. Generate without rewriting the response and capture every stage in `RagRecord`.

Deterministic tests construct `NumpyVectorIndex` in memory. CLI `ingest` persists it under `.index/`; CLI `ask` reloads and validates it.

---

## Development

Ruff is the repository's formatter and linter. Mypy strictly checks the application modules and RagaliQ adapter. Pre-commit runs both tools and repository-hygiene hooks before commits. Every hook uses `uv run --no-sync` and the hash-locked `.venv`; hook execution does not create separate environments or resolve additional packages. The sandbox still has no coverage tooling, build backend, package-publication layer, or task runner. uv owns environment creation and exact dependency synchronization.

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
.venv/bin/python -m pytest -c pytest.ini eval/ -q
```

When dependency metadata changes, regenerate the standardized lock with the same Python policy:

```bash
uv export \
  --format pylock.toml \
  --all-groups \
  --no-emit-project \
  --python 3.14 \
  --prerelease disallow \
  --no-header \
  -o pylock.toml
rm uv.lock  # transient uv project lock; pylock.toml is the committed contract
```

The project export path preserves the selected Python minor in `pylock.toml` as `==3.14.*`. Dependency changes must update `pyproject.toml` and `pylock.toml` together.

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
| Persistence | Native | Vector index serialization | Explicit `.npy` plus manifest |
| Distance transparency | Backend configuration mediates semantics | Normalization and metric choice remain external | Normalization, dot product, distance, and tie-break are visible |
| Teaching value | Database lifecycle | ANN index mechanics | Complete small-corpus data flow |
| Later replacement | Collection adapter | Index plus metadata adapter | Existing provider/index/result seams |

VerdigrisE keeps pure NumPy. The persisted implementation adds fixture and vector fingerprints without changing the visible cosine calculation:

```text
cosine_similarity = dot(L2_normalized_document, L2_normalized_query)
distance = 1 - cosine_similarity
rank = descending similarity, then ascending stable chunk id
```

---

## Production Boundary

The next rungs are intentionally visible but not implemented:

1. Replace hand-authored strings with table-aware and clause-aware ingestion while retaining stable source spans and version lineage.
2. Version corpus snapshots and indexes so answers and evaluations can be replayed against an exact source edition.
3. Add lexical retrieval, metadata filters, hybrid fusion, and reranking behind the current retrieval-result boundary.
4. Enforce project, document, clause, table, row, or cell access policy before evidence reaches the prompt.
5. Add claim-to-evidence validation, source-span citations, and durable answer audit events.
6. Move the synchronous local boundary into an observable backend with retry, rate-limit, cache, cost, and failure policies.
7. Expand RagaliQ semantic evaluation over versioned production fixtures without transferring exact compliance assertions to probabilistic judges.

VerdigrisE remains a sandbox: it does not claim PDF support, table extraction, production storage, web serving, CI integration, generated reports, or deployment readiness.

---

## Why "VerdigrisE"?

**Verdigris** + **E**val = **VerdigrisE**

Verdigris is the green patina that surfaces on weathered copper — corrosion made visible. It is also the fixture's central subject: the **verdigris blossom elixir**, a compound whose dosage is prescribed three conflicting ways by three rival grimoires, and whose canonical preparation is distilled in copper. The **E** stands for the evaluation of RAG (Retrieval-Augmented Generation) mechanics that the sandbox exists to make inspectable.

Because a RAG system's most dangerous failures are the ones that never surface. VerdigrisE exists to make wrong-source numbers, swapped near-synonyms, dropped qualifiers, and confident answers-from-nothing visible as exact, repeatable test failures.

---

## License

VerdigrisE is licensed under the [MIT License](LICENSE). The RagaliQ dependency is separately MIT licensed — see [RagaliQ's LICENSE](https://github.com/dariero/RagaliQ/blob/v0.2.0/LICENSE).
