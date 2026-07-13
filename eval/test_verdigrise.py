"""VerdigrisE deterministic contracts and explicitly gated semantic integrations.

Ownership boundary:
    deterministic pytest: ids, rank, collision materialization, values, units,
        qualifiers, exact abstention, prompt bytes, citations, and distance math;
    RagaliQ: faithfulness and answer relevance after deterministic checks pass.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from ragaliq import RAGTestResult

from config import (
    ABSTENTION_PHRASE,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    TOP_K,
)
from corpus import CORPUS, GOLDEN_CASES, CorpusEntry, validate_corpus
from eval.ragaliq_adapter import (
    CannedJudgeTransport,
    build_ragaliq_runner,
    to_ragaliq_case,
)
from models import PromptMessage, RagRecord
from pipeline import (
    GenerationResponseError,
    NumpyVectorIndex,
    OpenAIAnswerGenerator,
    OpenAIEmbeddingProvider,
    RagPipeline,
    build_generation_messages,
    ingest_corpus,
    parse_generation_response,
)


class GoldenCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    question: str
    expected_retrieved_id: str | None
    expected_ranked_ids: list[str]
    collision_sibling_ids: list[str]
    expected_value: str | None
    forbidden_values: list[str]
    expected_grimoire_id: str | None
    required_qualifiers: list[str]
    must_contain: list[str]
    forbidden: list[str]
    expected_answer: str
    expect_abstention: bool


GOLDEN = [GoldenCase.model_validate(case) for case in GOLDEN_CASES]
GOLDEN_BY_ID = {case.case_id: case for case in GOLDEN}
ANSWERABLE = [case for case in GOLDEN if not case.expect_abstention]
ABSTAINING = [case for case in GOLDEN if case.expect_abstention]
COLLISION_CASES = [case for case in ANSWERABLE if case.collision_sibling_ids]
QUALIFIED_CASES = [case for case in ANSWERABLE if case.required_qualifiers]


# Dimensions isolate the dosage, vapor, harvest, hardness, and absent-query
# families while forcing each declared sibling to rank second without a tie.
_CORPUS_VECTORS: dict[str, list[float]] = {
    "verdigris-dose-verdant": [4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "verdigris-dose-amber": [4, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "verdigris-dose-obsidian": [4, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    "moonpetal-silver-vapor": [0, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0],
    "moonflower-golden-vapor": [0, 0, 0, 0, 4, 0, 3, 0, 0, 0, 0, 0],
    "shadeglass-orchid-harvest": [0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 0],
    "sunspire-orchid-harvest": [0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 0],
    "asterquartz-powdering": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
}
_QUESTION_VECTORS: dict[str, list[float]] = {
    GOLDEN_BY_ID["numeric-source-verdigris-dose"].question: [4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    GOLDEN_BY_ID["numeric-source-amber-dose"].question: [4, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    GOLDEN_BY_ID["numeric-source-obsidian-dose"].question: [4, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    GOLDEN_BY_ID["near-synonym-moonpetal-vapor"].question: [0, 0, 0, 0, 4, 4, 1, 0, 0, 0, 0, 0],
    GOLDEN_BY_ID["conditional-shadeglass-harvest"].question: [0, 0, 0, 0, 0, 0, 0, 4, 4, 1, 0, 0],
    GOLDEN_BY_ID["absent-moonpetal-dew-shelf-life"].question: [0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 4],
}


class FixedEmbeddingProvider:
    """Controlled vectors make exact retrieval assertions local and reproducible."""

    model = EMBEDDING_MODEL

    def __init__(self) -> None:
        self._vectors_by_text = {
            entry["text"]: _CORPUS_VECTORS[entry["id"]] for entry in CORPUS
        } | _QUESTION_VECTORS

    def embed(
        self,
        inputs: list[str],
        *,
        input_ids: list[str],
        stage: str,
        debug: bool = False,
    ) -> np.ndarray:
        del input_ids, stage, debug
        try:
            return np.asarray([self._vectors_by_text[text] for text in inputs], dtype=np.float32)
        except KeyError as exc:
            raise AssertionError(f"No fixed vector for input: {exc.args[0]!r}") from exc


class FixedAnswerGenerator:
    """Controlled responses isolate deterministic contracts from generation variance."""

    def __init__(self) -> None:
        self.answers = {case.question: case.expected_answer for case in GOLDEN}
        self.calls: list[tuple[str, list[PromptMessage]]] = []

    def generate(self, question: str, messages: list[PromptMessage]) -> str:
        self.calls.append((question, messages))
        return self.answers[question]


@pytest.fixture(scope="session")
def deterministic_pipeline() -> RagPipeline:
    embedder = FixedEmbeddingProvider()
    vectors = embedder.embed(
        [entry["text"] for entry in CORPUS],
        input_ids=[entry["id"] for entry in CORPUS],
        stage="corpus",
    )
    index = NumpyVectorIndex(dimension=vectors.shape[1], embedding_model=embedder.model)
    index.index(CORPUS, vectors)
    return RagPipeline(index=index, embedder=embedder, generator=FixedAnswerGenerator())


@pytest.fixture(scope="session")
def records(deterministic_pipeline: RagPipeline) -> dict[str, RagRecord]:
    return {case.case_id: deterministic_pipeline.ask(case.question) for case in GOLDEN}


def test_corpus_has_stable_identity_and_grimoire_citations() -> None:
    validate_corpus()
    assert len({entry["id"] for entry in CORPUS}) == len(CORPUS)
    for entry in CORPUS:
        assert set(entry) >= {"id", "text", "grimoire_id", "folio"}
        assert entry["grimoire_id"] is not None or entry["folio"] is not None
        assert entry["folio"] is None or isinstance(entry["folio"], (int, str))


def test_fixture_units_and_genuine_absence_are_explicit() -> None:
    entries = {entry["id"]: entry for entry in CORPUS}
    expected_unit_literals = {
        "verdigris-dose-verdant": "3 drams",
        "moonpetal-silver-vapor": "11 grains",
        "shadeglass-orchid-harvest": "7 moon-phases",
        "asterquartz-powdering": "Mohs hardness 8",
    }
    for entry_id, literal in expected_unit_literals.items():
        assert literal in entries[entry_id]["text"]
    corpus_text = "\n".join(entry["text"] for entry in CORPUS)
    lowered = corpus_text.lower()
    for absent_term in (
        "moonpetal dew",
        "bottled",
        "shelf life",
        "storage",
        "stored",
        "spoil",
    ):
        assert absent_term not in lowered


def test_three_source_dosage_trap_has_distinct_sources_values_and_conditions() -> None:
    ids = {
        "verdigris-dose-verdant",
        "verdigris-dose-amber",
        "verdigris-dose-obsidian",
    }
    entries = [entry for entry in CORPUS if entry["id"] in ids]
    assert len(entries) == 3
    assert len({entry["subject"] for entry in entries}) == 1
    assert len({entry["fact_type"] for entry in entries}) == 1
    assert len({entry["grimoire_id"] for entry in entries}) == 3
    assert len({entry["condition"] for entry in entries}) == 3
    joined = "\n".join(entry["text"] for entry in entries)
    for value in ("3 drams", "9 drams", "15 drams"):
        assert value in joined


def test_near_synonym_pair_is_distinct_and_factually_conflicting() -> None:
    entries = {entry["id"]: entry for entry in CORPUS}
    moonpetal = entries["moonpetal-silver-vapor"]
    moonflower = entries["moonflower-golden-vapor"]
    assert moonpetal["subject"] != moonflower["subject"]
    assert moonpetal["fact_type"] == moonflower["fact_type"]
    assert "11 grains" in moonpetal["text"]
    assert "silver sleep vapor" in moonpetal["text"]
    assert "17 grains" in moonflower["text"]
    assert "golden waking vapor" in moonflower["text"]


@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: case.case_id)
def test_expected_source_and_rank_order(case: GoldenCase, records: dict[str, RagRecord]) -> None:
    record = records[case.case_id]
    assert len(record.retrieved_ids) == TOP_K
    assert record.retrieved_ids == case.expected_ranked_ids
    if case.expected_retrieved_id is not None:
        assert case.expected_retrieved_id in record.retrieved_ids


@pytest.mark.parametrize("case", COLLISION_CASES, ids=lambda case: case.case_id)
def test_collision_sibling_materializes_in_generation_context(
    case: GoldenCase, records: dict[str, RagRecord]
) -> None:
    record = records[case.case_id]
    materialized_ids = set(case.collision_sibling_ids)
    assert materialized_ids.issubset(record.retrieved_ids), (
        f"Not every collision sibling {case.collision_sibling_ids} reached top-{TOP_K}: "
        f"{record.retrieved_ids}"
    )
    chunks_by_id = {chunk.id: chunk for chunk in record.retrieved_chunks}
    for sibling_id in materialized_ids:
        assert chunks_by_id[sibling_id].text in record.context_payload
    for value in case.forbidden_values:
        assert value in record.context_payload


def test_near_synonym_target_ranks_above_collision_sibling(
    records: dict[str, RagRecord],
) -> None:
    case = GOLDEN_BY_ID["near-synonym-moonpetal-vapor"]
    ranked = records[case.case_id].retrieved_ids
    assert ranked.index(case.expected_retrieved_id) < ranked.index(case.collision_sibling_ids[0])


@pytest.mark.parametrize("case", ANSWERABLE, ids=lambda case: case.case_id)
def test_required_and_forbidden_facts(case: GoldenCase, records: dict[str, RagRecord]) -> None:
    record = records[case.case_id]
    assert record.answer == case.expected_answer
    for literal in case.must_contain:
        assert literal in record.answer
    for literal in case.forbidden:
        assert literal not in record.answer


@pytest.mark.parametrize("case", ANSWERABLE, ids=lambda case: case.case_id)
def test_expected_value_and_conflicting_values_are_owned_deterministically(
    case: GoldenCase, records: dict[str, RagRecord]
) -> None:
    answer = records[case.case_id].answer
    assert case.expected_value is not None
    assert case.expected_value in answer
    for value in case.forbidden_values:
        assert value not in answer


@pytest.mark.parametrize("case", QUALIFIED_CASES, ids=lambda case: case.case_id)
def test_value_only_answer_cannot_pass_required_qualifiers(
    case: GoldenCase, records: dict[str, RagRecord]
) -> None:
    answer = records[case.case_id].answer
    assert case.expected_value in answer
    for qualifier in case.required_qualifiers:
        assert qualifier in answer


@pytest.mark.parametrize("case", ABSTAINING, ids=lambda case: case.case_id)
def test_abstention_is_exact(case: GoldenCase, records: dict[str, RagRecord]) -> None:
    assert case.expected_answer == ABSTENTION_PHRASE
    assert records[case.case_id].answer == ABSTENTION_PHRASE


@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: case.case_id)
def test_prompt_contains_only_ranked_chunks_verbatim(
    case: GoldenCase, records: dict[str, RagRecord]
) -> None:
    record = records[case.case_id]
    retrieved = set(record.retrieved_ids)
    for entry in CORPUS:
        if entry["id"] in retrieved:
            assert record.context_payload.count(entry["text"]) == 1
        else:
            assert entry["text"] not in record.context_payload
    expected_blocks = []
    for chunk in record.retrieved_chunks:
        expected_blocks.append(
            f"[CONTEXT id={chunk.id} grimoire_id={chunk.metadata['grimoire_id']!r} "
            f"folio={chunk.metadata['folio']!r}]\n"
            f"{chunk.text}\n"
            f"[END CONTEXT id={chunk.id}]"
        )
    assert record.context_payload == "\n\n".join(expected_blocks)
    assert record.generation_messages[1].content == (
        f"QUESTION:\n{record.question}\n\nCONTEXT:\n{record.context_payload}"
    )


@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: case.case_id)
def test_distance_similarity_and_rank_fields_align(
    case: GoldenCase, records: dict[str, RagRecord]
) -> None:
    record = records[case.case_id]
    assert record.retrieved_ids == [chunk.id for chunk in record.retrieved_chunks]
    assert record.distances == [chunk.distance for chunk in record.retrieved_chunks]
    for chunk in record.retrieved_chunks:
        assert chunk.distance == pytest.approx(1.0 - chunk.similarity)
        assert set(chunk.model_dump()) == {
            "id",
            "text",
            "metadata",
            "distance",
            "similarity",
        }


@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: case.case_id)
def test_grimoire_metadata_survives_retrieval_and_prompt(
    case: GoldenCase, records: dict[str, RagRecord]
) -> None:
    record = records[case.case_id]
    corpus_by_id = {entry["id"]: entry for entry in CORPUS}
    for chunk in record.retrieved_chunks:
        raw = corpus_by_id[chunk.id]
        assert chunk.metadata == {
            key: value for key, value in raw.items() if key not in {"id", "text"}
        }
        label = (
            f"[CONTEXT id={chunk.id} grimoire_id={chunk.metadata['grimoire_id']!r} "
            f"folio={chunk.metadata['folio']!r}]"
        )
        assert label in record.context_payload


@pytest.mark.parametrize("case", ANSWERABLE, ids=lambda case: case.case_id)
def test_expected_chunk_and_grimoire_references_survive_answer_construction(
    case: GoldenCase, records: dict[str, RagRecord]
) -> None:
    record = records[case.case_id]
    expected = next(
        chunk for chunk in record.retrieved_chunks if chunk.id == case.expected_retrieved_id
    )
    assert expected.metadata["grimoire_id"] == case.expected_grimoire_id
    assert case.expected_grimoire_id in record.answer
    assert f"[{case.expected_retrieved_id}]" in record.answer


def test_prompt_builder_is_directly_callable_without_a_client(
    records: dict[str, RagRecord],
) -> None:
    case = GOLDEN_BY_ID["numeric-source-verdigris-dose"]
    record = records[case.case_id]
    context, messages = build_generation_messages(record.question, record.retrieved_chunks)
    assert context == record.context_payload
    assert messages == record.generation_messages
    assert ABSTENTION_PHRASE in messages[0].content
    assert "Repeat the supporting" in messages[0].content
    assert "`grimoire_id` verbatim" in messages[0].content


def test_cosine_ties_use_ascending_stable_id() -> None:
    entries: list[CorpusEntry] = [CORPUS[0], CORPUS[7]]
    index = NumpyVectorIndex(dimension=2, embedding_model=EMBEDDING_MODEL)
    index.index(entries, np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    hits = index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2)
    assert [hit.id for hit in hits] == sorted(entry["id"] for entry in entries)


def test_index_persistence_preserves_rank_and_grimoire_metadata(tmp_path: Path) -> None:
    embedder = FixedEmbeddingProvider()
    vectors = embedder.embed(
        [entry["text"] for entry in CORPUS],
        input_ids=[entry["id"] for entry in CORPUS],
        stage="corpus",
    )
    index = NumpyVectorIndex(dimension=vectors.shape[1], embedding_model=embedder.model)
    index.index(CORPUS, vectors)
    index.save(tmp_path)
    loaded = NumpyVectorIndex.load(tmp_path)
    case = GOLDEN_BY_ID["numeric-source-verdigris-dose"]
    hits = loaded.search(np.asarray(_QUESTION_VECTORS[case.question], dtype=np.float32))
    assert [hit.id for hit in hits] == case.expected_ranked_ids
    assert hits[0].metadata["grimoire_id"] == "GRIM-VERDANT"
    assert hits[0].metadata["folio"] == 21
    assert hits[0].metadata["condition"] == ("distilled in copper and administered after dusk")


def test_zero_vector_is_rejected() -> None:
    index = NumpyVectorIndex(dimension=2, embedding_model=EMBEDDING_MODEL)
    with pytest.raises(ValueError, match="zero vectors"):
        index.index([CORPUS[0]], np.asarray([[0.0, 0.0]], dtype=np.float32))


class _FakeEmbeddingsEndpoint:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
            ]
        )


def test_real_embedding_adapter_batches_and_debugs_safely(capsys: Any) -> None:
    endpoint = _FakeEmbeddingsEndpoint()
    client = SimpleNamespace(embeddings=endpoint)
    matrix = OpenAIEmbeddingProvider(client).embed(
        ["first", "second"],
        input_ids=["id-1", "id-2"],
        stage="corpus",
        debug=True,
    )
    assert endpoint.kwargs == {
        "model": EMBEDDING_MODEL,
        "input": ["first", "second"],
        "encoding_format": "float",
    }
    assert matrix.shape == (2, 3)
    assert matrix.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    output = capsys.readouterr().out
    assert f"model={EMBEDDING_MODEL}" in output
    assert "input_count=2" in output
    assert "dimensions=3" in output
    assert "shape=(2, 3)" in output
    assert "ids=['id-1', 'id-2']" in output
    assert "verdigrise_embedding_debug" in output
    assert "first" not in output


class _FakeCompletionsEndpoint:
    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self.kwargs: dict[str, object] = {}
        self._content = content
        self._finish_reason = finish_reason

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=self._finish_reason,
                    message=SimpleNamespace(content=self._content, refusal=None),
                )
            ]
        )


def _fake_generation_client(endpoint: _FakeCompletionsEndpoint) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=endpoint))


def test_real_generation_adapter_preserves_structured_text_and_citations() -> None:
    raw = '{"answer":"3 drams","sources":["verdigris-dose-verdant"]}'
    endpoint = _FakeCompletionsEndpoint(raw)
    generator = OpenAIAnswerGenerator(_fake_generation_client(endpoint))
    messages = [
        PromptMessage(role="system", content="system"),
        PromptMessage(role="user", content="user"),
    ]
    assert generator.generate("question", messages) == raw
    assert endpoint.kwargs["model"] == GENERATION_MODEL
    assert endpoint.kwargs["temperature"] == GENERATION_TEMPERATURE
    assert endpoint.kwargs["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_generation_parser_rejects_missing_text_instead_of_discarding_it() -> None:
    response = _FakeCompletionsEndpoint(None).create()
    with pytest.raises(GenerationResponseError, match="no text content"):
        parse_generation_response(response)


def test_ragaliq_canned_runner_executes_structural_wiring_locally(
    records: dict[str, RagRecord],
) -> None:
    case = GOLDEN_BY_ID["numeric-source-verdigris-dose"]
    record = records[case.case_id]
    transport = CannedJudgeTransport()
    runner = build_ragaliq_runner(transport)
    test_case = to_ragaliq_case(record, case_id=case.case_id)
    result = runner.evaluate(test_case)
    assert isinstance(result, RAGTestResult)
    assert result.passed
    assert result.scores == {"faithfulness": 1.0, "relevance": 0.9}
    assert result.test_case.context == [record.context_payload]
    assert len(transport.calls) == 3


requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for explicitly selected paid tests",
)
requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY is required for explicitly selected RagaliQ judge tests",
)


@pytest.fixture(scope="module")
def live_openai_records(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, RagRecord]:
    """Paid fixture: embed once, then ask every golden question once."""

    client = OpenAI()
    embedder = OpenAIEmbeddingProvider(client)
    index = ingest_corpus(
        embedder=embedder,
        output_directory=tmp_path_factory.mktemp("live-openai-index"),
        debug=True,
    )
    pipeline = RagPipeline(
        index=index,
        embedder=embedder,
        generator=OpenAIAnswerGenerator(client),
        debug=True,
    )
    return {case.case_id: pipeline.ask(case.question) for case in GOLDEN}


def _assert_live_golden_contract(case: GoldenCase, record: RagRecord) -> None:
    """Apply exact fixture ownership before any probabilistic semantic evaluation."""

    assert len(record.retrieved_ids) == TOP_K

    if case.expected_retrieved_id is not None:
        assert case.expected_retrieved_id in record.retrieved_ids

    if case.collision_sibling_ids:
        assert set(case.collision_sibling_ids).issubset(record.retrieved_ids)
        context_by_id = {chunk.id: chunk.text for chunk in record.retrieved_chunks}
        for sibling_id in case.collision_sibling_ids:
            assert context_by_id[sibling_id] in record.context_payload
        for value in case.forbidden_values:
            assert value in record.context_payload

    if case.case_id == "near-synonym-moonpetal-vapor":
        assert record.retrieved_ids.index(case.expected_retrieved_id) < record.retrieved_ids.index(
            case.collision_sibling_ids[0]
        )

    if case.expect_abstention:
        assert record.answer == ABSTENTION_PHRASE
        return

    assert case.expected_value is not None
    assert case.expected_value in record.answer
    for qualifier in case.required_qualifiers:
        assert qualifier in record.answer
    for literal in case.must_contain:
        assert literal in record.answer
    for literal in case.forbidden:
        assert literal not in record.answer

    expected = next(
        chunk for chunk in record.retrieved_chunks if chunk.id == case.expected_retrieved_id
    )
    assert expected.metadata["grimoire_id"] == case.expected_grimoire_id
    assert case.expected_grimoire_id in record.answer
    assert f"[{case.expected_retrieved_id}]" in record.answer


@pytest.mark.openai
@requires_openai
@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: case.case_id)
def test_real_openai_all_golden_acceptance(
    case: GoldenCase,
    live_openai_records: dict[str, RagRecord],
) -> None:
    """Paid: every golden case constrains the configured retrieval and generation models."""

    _assert_live_golden_contract(case, live_openai_records[case.case_id])


@pytest.mark.openai
@pytest.mark.rag_test
@requires_openai
@requires_anthropic
@pytest.mark.parametrize("case", ANSWERABLE, ids=lambda case: case.case_id)
def test_ragaliq_claude_semantic_residue_on_live_answers(
    case: GoldenCase,
    live_openai_records: dict[str, RagRecord],
    rag_tester: Any,
) -> None:
    """Paid: native cross-family judge owns faithfulness and answer relevance only."""

    record = live_openai_records[case.case_id]
    _assert_live_golden_contract(case, record)
    test_case = to_ragaliq_case(record, case_id=case.case_id)
    result = rag_tester.evaluate(test_case)
    assert result.passed
    assert set(result.scores) == {"faithfulness", "relevance"}
