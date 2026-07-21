"""VerdigrisE deterministic contracts and explicitly gated semantic integrations.

Ownership boundary:
    deterministic pytest: ids, rank, collision materialization, values, units,
        qualifiers, exact abstention, prompt bytes, citations, and distance math;
    RagaliQ: faithfulness and answer relevance after deterministic checks pass.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from ragaliq import RAGTestResult
from ragaliq.judges import DEFAULT_JUDGE_MODEL

import pipeline as pipeline_module
from config import (
    ABSTENTION_PHRASE,
    DISTANCE_DEFINITION,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    INDEX_ACTIVE_FILENAME,
    INDEX_GENERATIONS_DIRECTORY,
    INDEX_MANIFEST_FILENAME,
    INDEX_POINTER_SCHEMA_VERSION,
    INDEX_SCHEMA_VERSION,
    INDEX_VECTOR_FILENAME,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_SECONDS,
    TIE_BREAK_RULE,
    TOP_K,
)
from corpus import CORPUS, GOLDEN_CASES, CorpusEntry, validate_corpus
from eval import conftest as eval_conftest
from eval.ragaliq_adapter import (
    CannedJudgeTransport,
    build_ragaliq_runner,
    to_ragaliq_case,
)
from models import PromptMessage, RagRecord, RetrievedChunk
from pipeline import (
    AnswerGenerator,
    EmbeddingProvider,
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
DOSAGE_CASES = [case for case in GOLDEN if case.case_id.startswith("numeric-source-")]


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
    GOLDEN_BY_ID["conditional-shadeglass-direct-sun"].question: [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4,
        4,
        3,
        0,
        0,
    ],
    GOLDEN_BY_ID["absent-moonpetal-dew-shelf-life"].question: [0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 4],
}


class FixedEmbeddingProvider:
    """Controlled vectors make exact retrieval assertions local and reproducible."""

    model = EMBEDDING_MODEL

    def __init__(self) -> None:
        self._vectors_by_text = {
            entry["text"]: _CORPUS_VECTORS[entry["id"]] for entry in CORPUS
        } | _QUESTION_VECTORS
        self.calls: list[tuple[list[str], list[str], str, bool]] = []

    def embed(
        self,
        inputs: list[str],
        *,
        input_ids: list[str],
        stage: str,
        debug: bool = False,
    ) -> np.ndarray:
        self.calls.append((list(inputs), list(input_ids), stage, debug))
        try:
            return np.asarray([self._vectors_by_text[text] for text in inputs], dtype=np.float32)
        except KeyError as exc:
            raise AssertionError(f"No fixed vector for input: {exc.args[0]!r}") from exc


def _persist_fixed_index(
    directory: Path, *, debug: bool = False
) -> tuple[FixedEmbeddingProvider, NumpyVectorIndex]:
    embedder = FixedEmbeddingProvider()
    index = ingest_corpus(embedder=embedder, output_directory=directory, debug=debug)
    return embedder, index


def _read_active_pointer(directory: Path) -> dict[str, Any]:
    pointer = json.loads((directory / INDEX_ACTIVE_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(pointer, dict)
    return pointer


def _active_generation_directory(directory: Path) -> Path:
    generation_id = _read_active_pointer(directory)["generation_id"]
    assert isinstance(generation_id, str)
    return directory / INDEX_GENERATIONS_DIRECTORY / generation_id


def _index_file_path(directory: Path, filename: str) -> Path:
    return _active_generation_directory(directory) / filename


def _read_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads(
        _index_file_path(directory, INDEX_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert isinstance(manifest, dict)
    return manifest


def _write_manifest(directory: Path, manifest: object) -> None:
    _index_file_path(directory, INDEX_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_vectors(directory: Path, vectors: np.ndarray, *, update_digest: bool) -> None:
    vector_path = _index_file_path(directory, INDEX_VECTOR_FILENAME)
    np.save(vector_path, vectors, allow_pickle=False)
    if update_digest:
        manifest = _read_manifest(directory)
        manifest["vectors_sha256"] = hashlib.sha256(vector_path.read_bytes()).hexdigest()
        _write_manifest(directory, manifest)


def _write_active_pointer(directory: Path, pointer: object) -> None:
    (directory / INDEX_ACTIVE_FILENAME).write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _generation_directories(directory: Path) -> list[Path]:
    generations_directory = directory / INDEX_GENERATIONS_DIRECTORY
    return sorted(
        path
        for path in generations_directory.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def _two_entry_index(vectors: list[list[float]]) -> NumpyVectorIndex:
    index = NumpyVectorIndex(dimension=2, embedding_model=EMBEDDING_MODEL)
    index.index(CORPUS[:2], np.asarray(vectors, dtype=np.float32))
    return index


def _first_hit_id(directory: Path) -> str:
    loaded = NumpyVectorIndex.load(directory)
    return loaded.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)[0].id


class _MatrixEmbeddingProvider:
    model = EMBEDDING_MODEL

    def __init__(self, matrix: object) -> None:
        self._matrix = matrix

    def embed(
        self,
        inputs: list[str],
        *,
        input_ids: list[str],
        stage: str,
        debug: bool = False,
    ) -> Any:
        del inputs, input_ids, stage, debug
        return self._matrix


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
    assert (
        "Direct sun exposure invalidates that harvest window."
        in entries["shadeglass-orchid-harvest"]["text"]
    )
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


def test_each_dosage_case_forbids_every_non_target_source() -> None:
    source_literals = {
        "verdigris-dose-verdant": {
            "3 drams",
            "GRIM-VERDANT",
            "distilled in copper",
            "administered after dusk",
            "[verdigris-dose-verdant]",
        },
        "verdigris-dose-amber": {
            "9 drams",
            "GRIM-AMBER",
            "distilled in amber glass",
            "administered at dawn",
            "[verdigris-dose-amber]",
        },
        "verdigris-dose-obsidian": {
            "15 drams",
            "GRIM-OBSIDIAN-PETAL",
            "distilled in basalt",
            "lunar eclipse",
            "[verdigris-dose-obsidian]",
        },
    }
    assert len(DOSAGE_CASES) == len(source_literals)
    for case in DOSAGE_CASES:
        assert case.expected_retrieved_id in source_literals
        expected_forbidden = set().union(
            *(
                literals
                for source_id, literals in source_literals.items()
                if source_id != case.expected_retrieved_id
            )
        )
        assert expected_forbidden.issubset(case.forbidden)


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
    assert list(record.retrieved_ids) == case.expected_ranked_ids
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
    assert record.retrieved_ids == tuple(chunk.id for chunk in record.retrieved_chunks)
    assert record.distances == tuple(chunk.distance for chunk in record.retrieved_chunks)
    for chunk in record.retrieved_chunks:
        assert -1.0 <= chunk.similarity <= 1.0
        assert 0.0 <= chunk.distance <= 2.0
        assert chunk.distance == 1.0 - chunk.similarity
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
    assert messages == list(record.generation_messages)
    assert ABSTENTION_PHRASE in messages[0].content
    assert "Repeat the supporting" in messages[0].content
    assert "`grimoire_id` verbatim" in messages[0].content


def _capture_contract_record(metadata: dict[str, object] | None = None) -> RagRecord:
    chunk = RetrievedChunk(
        id="verdigris-dose-verdant",
        text="Verbatim evidence.",
        metadata=metadata
        or {
            "grimoire_id": "GRIM-VERDANT",
            "folio": 21,
            "audit": {"qualifiers": ["after dusk"]},
        },
        distance=0.25,
        similarity=0.75,
    )
    return RagRecord(
        question="What is the scoped fact?",
        retrieved_ids=[chunk.id],
        retrieved_chunks=[chunk],
        distances=[chunk.distance],
        context_payload="[CONTEXT] Verbatim evidence.",
        generation_messages=[
            PromptMessage(role="system", content="Use only context."),
            PromptMessage(role="user", content="What is the scoped fact?"),
        ],
        answer="The scoped fact [verdigris-dose-verdant].",
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        pytest.param(
            {"retrieved_ids": ["wrong-id"]},
            "retrieved_ids must match",
            id="ids",
        ),
        pytest.param(
            {"distances": [0.5]},
            "distances must match",
            id="distances",
        ),
    ],
)
def test_rag_record_rejects_rank_misalignment(override: dict[str, object], message: str) -> None:
    chunk = RetrievedChunk(
        id="verdigris-dose-verdant",
        text="Verbatim evidence.",
        metadata={"grimoire_id": "GRIM-VERDANT", "folio": 21},
        distance=0.25,
        similarity=0.75,
    )
    payload: dict[str, object] = {
        "question": "What is the scoped fact?",
        "retrieved_ids": [chunk.id],
        "retrieved_chunks": [chunk],
        "distances": [chunk.distance],
        "context_payload": "[CONTEXT] Verbatim evidence.",
        "generation_messages": [
            {"role": "system", "content": "Use only context."},
            {"role": "user", "content": "What is the scoped fact?"},
        ],
        "answer": "The scoped fact [verdigris-dose-verdant].",
    }
    payload.update(override)

    with pytest.raises(ValidationError, match=message):
        RagRecord.model_validate(payload)


def test_capture_rank_collections_are_immutable() -> None:
    record = _capture_contract_record()

    assert isinstance(record.retrieved_ids, tuple)
    assert isinstance(record.retrieved_chunks, tuple)
    assert isinstance(record.distances, tuple)
    assert isinstance(record.generation_messages, tuple)
    with pytest.raises(TypeError):
        record.retrieved_ids[0] = "wrong-id"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.retrieved_chunks[0] = record.retrieved_chunks[0]  # type: ignore[index]
    with pytest.raises(TypeError):
        record.distances[0] = 2.0  # type: ignore[index]
    with pytest.raises(TypeError):
        record.generation_messages[0] = record.generation_messages[0]  # type: ignore[index]


def test_chunk_metadata_is_detached_and_recursively_immutable() -> None:
    qualifiers = ["after dusk"]
    audit = {"qualifiers": qualifiers}
    metadata: dict[str, object] = {
        "grimoire_id": "GRIM-VERDANT",
        "folio": 21,
        "audit": audit,
    }
    record = _capture_contract_record(metadata)

    metadata["folio"] = 999
    qualifiers.append("at dawn")
    audit["unexpected"] = True

    chunk_metadata = record.retrieved_chunks[0].metadata
    assert len(chunk_metadata) == 3
    assert chunk_metadata["folio"] == 21
    nested_audit = chunk_metadata["audit"]
    assert isinstance(nested_audit, Mapping)
    assert nested_audit["qualifiers"] == ("after dusk",)
    assert "unexpected" not in nested_audit
    with pytest.raises(TypeError):
        chunk_metadata["folio"] = 999  # type: ignore[index]
    with pytest.raises(TypeError):
        nested_audit["qualifiers"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        nested_audit["qualifiers"][0] = "at dawn"  # type: ignore[index]


def test_chunk_metadata_rejects_unsupported_mutable_leaves() -> None:
    mutable_leaf = SimpleNamespace(value=1)

    with pytest.raises(ValidationError, match="JSON scalar leaves"):
        RetrievedChunk(
            id="verdigris-dose-verdant",
            text="Verbatim evidence.",
            metadata={"grimoire_id": "GRIM-VERDANT", "mutable": mutable_leaf},
            distance=0.25,
            similarity=0.75,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_chunk_metadata_rejects_non_finite_float_leaves(value: float) -> None:
    with pytest.raises(ValidationError, match="float leaves must be finite"):
        RetrievedChunk(
            id="verdigris-dose-verdant",
            text="Verbatim evidence.",
            metadata={"grimoire_id": "GRIM-VERDANT", "value": value},
            distance=0.25,
            similarity=0.75,
        )


def test_chunk_metadata_rejects_non_string_nested_mapping_keys() -> None:
    with pytest.raises(ValidationError, match="mapping keys must be strings"):
        RetrievedChunk(
            id="verdigris-dose-verdant",
            text="Verbatim evidence.",
            metadata={"grimoire_id": "GRIM-VERDANT", "nested": {1: "invalid"}},
            distance=0.25,
            similarity=0.75,
        )


def test_immutable_captures_preserve_dump_and_json_container_shapes() -> None:
    record = _capture_contract_record()

    dumped = record.model_dump()
    assert isinstance(dumped["retrieved_ids"], list)
    assert isinstance(dumped["retrieved_chunks"], list)
    assert isinstance(dumped["distances"], list)
    assert isinstance(dumped["generation_messages"], list)
    metadata = dumped["retrieved_chunks"][0]["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["audit"]["qualifiers"] == ["after dusk"]
    assert json.loads(record.model_dump_json()) == dumped
    assert RagRecord.model_validate_json(record.model_dump_json()) == record


def test_immutable_capture_supports_deep_model_copy() -> None:
    record = _capture_contract_record()

    copied = record.model_copy(deep=True)

    assert copied == record
    assert copied is not record
    assert copied.retrieved_chunks[0] is not record.retrieved_chunks[0]
    assert copied.model_dump() == record.model_dump()


def test_capture_model_copy_revalidates_updates() -> None:
    record = _capture_contract_record()

    updated = record.model_copy(update={"retrieved_ids": [record.retrieved_ids[0]]})

    assert updated.retrieved_ids == record.retrieved_ids
    assert isinstance(updated.retrieved_ids, tuple)
    deep_updated = record.model_copy(update={"answer": "Updated answer."}, deep=True)
    assert deep_updated.answer == "Updated answer."
    assert deep_updated.retrieved_chunks[0] is not record.retrieved_chunks[0]
    with pytest.raises(ValidationError, match="retrieved_ids must match"):
        record.model_copy(update={"retrieved_ids": ["wrong-id"]})
    with pytest.raises(ValidationError, match="JSON scalar leaves"):
        record.retrieved_chunks[0].model_copy(
            update={"metadata": {"mutable": SimpleNamespace(value=1)}}
        )


def test_immutable_capture_supports_pickle_round_trip() -> None:
    record = _capture_contract_record()

    restored = pickle.loads(pickle.dumps(record))

    assert restored == record
    assert restored is not record
    assert restored.model_dump() == record.model_dump()
    with pytest.raises(TypeError):
        restored.retrieved_chunks[0].metadata["folio"] = 999


def test_cosine_ties_use_ascending_stable_id() -> None:
    positive_entries: list[CorpusEntry] = [CORPUS[0], CORPUS[7]]
    negative_entry = CORPUS[1]
    entries = [*positive_entries, negative_entry]
    rounding_edge = [0.0862179771065712, 0.996276319026947]
    index = NumpyVectorIndex(dimension=2, embedding_model=EMBEDDING_MODEL)
    index.index(
        entries,
        np.asarray(
            [rounding_edge, rounding_edge, [-value for value in rounding_edge]],
            dtype=np.float32,
        ),
    )
    hits = index.search(np.asarray(rounding_edge, dtype=np.float32), top_k=3)

    assert [hit.id for hit in hits[:2]] == sorted(entry["id"] for entry in positive_entries)
    assert hits[2].id == negative_entry["id"]
    assert [hit.similarity for hit in hits] == [1.0, 1.0, -1.0]
    assert [hit.distance for hit in hits] == [0.0, 0.0, 2.0]


@pytest.mark.parametrize(
    "vector",
    [
        pytest.param(
            [np.finfo(np.float32).max, np.finfo(np.float32).max],
            id="largest-finite",
        ),
        pytest.param(
            [np.nextafter(np.float32(0.0), np.float32(1.0)), np.float32(0.0)],
            id="smallest-positive-subnormal",
        ),
    ],
)
def test_finite_float32_extremes_normalize_without_collapsing(
    tmp_path: Path, vector: list[np.float32]
) -> None:
    index = NumpyVectorIndex(dimension=2, embedding_model=EMBEDDING_MODEL)
    matrix = np.asarray([vector], dtype=np.float32)
    index.index([CORPUS[0]], matrix)
    index.save(tmp_path)

    hits = NumpyVectorIndex.load(tmp_path).search(matrix[0], top_k=1)

    assert len(hits) == 1
    assert 0.0 < hits[0].similarity <= 1.0
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= hits[0].distance < 1e-5
    assert hits[0].distance == 1.0 - hits[0].similarity


def test_unpaid_ingest_persists_a_reloadable_current_index(tmp_path: Path) -> None:
    embedder, index = _persist_fixed_index(tmp_path, debug=True)
    assert embedder.calls == [
        (
            [entry["text"] for entry in CORPUS],
            [entry["id"] for entry in CORPUS],
            "corpus",
            True,
        )
    ]
    active_pointer = _read_active_pointer(tmp_path)
    assert active_pointer["schema_version"] == INDEX_POINTER_SCHEMA_VERSION
    generation_id = active_pointer["generation_id"]
    assert isinstance(generation_id, str)
    assert len(generation_id) == 32
    assert set(generation_id) <= set("0123456789abcdef")
    assert _index_file_path(tmp_path, INDEX_MANIFEST_FILENAME).is_file()
    assert _index_file_path(tmp_path, INDEX_VECTOR_FILENAME).is_file()
    assert not (tmp_path / INDEX_MANIFEST_FILENAME).exists()
    assert not (tmp_path / INDEX_VECTOR_FILENAME).exists()
    assert index.embedding_model == EMBEDDING_MODEL
    assert index.dimension == len(next(iter(_CORPUS_VECTORS.values())))
    assert index.indexed_corpus_sha256 == NumpyVectorIndex.corpus_sha256(CORPUS)

    manifest = _read_manifest(tmp_path)
    assert manifest["schema_version"] == INDEX_SCHEMA_VERSION
    assert manifest["embedding_model"] == EMBEDDING_MODEL
    assert manifest["dimension"] == index.dimension
    assert manifest["distance_definition"] == DISTANCE_DEFINITION
    assert manifest["tie_break_rule"] == TIE_BREAK_RULE
    assert manifest["corpus_sha256"] == index.indexed_corpus_sha256

    vector_path = _index_file_path(tmp_path, INDEX_VECTOR_FILENAME)
    persisted_vectors = np.load(vector_path, allow_pickle=False)
    assert persisted_vectors.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        np.linalg.norm(persisted_vectors.astype(np.float64), axis=1),
        1.0,
        rtol=0.0,
        atol=1e-5,
    )

    loaded = NumpyVectorIndex.load(tmp_path)
    roundtrip_directory = tmp_path / "roundtrip"
    loaded.save(roundtrip_directory)
    assert (
        _index_file_path(roundtrip_directory, INDEX_VECTOR_FILENAME).read_bytes()
        == vector_path.read_bytes()
    )
    case = GOLDEN_BY_ID["numeric-source-verdigris-dose"]
    hits = loaded.search(np.asarray(_QUESTION_VECTORS[case.question], dtype=np.float32))
    assert [hit.id for hit in hits] == case.expected_ranked_ids
    assert hits[0].metadata["grimoire_id"] == "GRIM-VERDANT"
    assert hits[0].metadata["folio"] == 21
    assert hits[0].metadata["condition"] == ("distilled in copper and administered after dusk")


def test_repeated_save_retains_old_generation_and_activates_new_generation(
    tmp_path: Path,
) -> None:
    old_index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])
    new_index = _two_entry_index([[0.0, 1.0], [1.0, 0.0]])
    old_index.save(tmp_path)
    old_generation = _active_generation_directory(tmp_path)

    new_index.save(tmp_path)
    new_generation = _active_generation_directory(tmp_path)

    assert new_generation != old_generation
    assert _generation_directories(tmp_path) == sorted([old_generation, new_generation])
    assert _first_hit_id(old_generation) == CORPUS[0]["id"]
    assert _first_hit_id(tmp_path) == CORPUS[1]["id"]


def test_legacy_schema_v1_pair_loads_until_a_generation_is_activated(tmp_path: Path) -> None:
    source_directory = tmp_path / "source"
    legacy_directory = tmp_path / "legacy"
    legacy_directory.mkdir()
    old_index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])
    new_index = _two_entry_index([[0.0, 1.0], [1.0, 0.0]])
    old_index.save(source_directory)
    source_generation = _active_generation_directory(source_directory)
    legacy_manifest_path = legacy_directory / INDEX_MANIFEST_FILENAME
    legacy_vector_path = legacy_directory / INDEX_VECTOR_FILENAME
    legacy_manifest_path.write_bytes((source_generation / INDEX_MANIFEST_FILENAME).read_bytes())
    legacy_vector_path.write_bytes((source_generation / INDEX_VECTOR_FILENAME).read_bytes())
    legacy_manifest_bytes = legacy_manifest_path.read_bytes()
    legacy_vector_bytes = legacy_vector_path.read_bytes()

    assert _first_hit_id(legacy_directory) == CORPUS[0]["id"]

    new_index.save(legacy_directory)

    assert legacy_manifest_path.read_bytes() == legacy_manifest_bytes
    assert legacy_vector_path.read_bytes() == legacy_vector_bytes
    assert _first_hit_id(legacy_directory) == CORPUS[1]["id"]


@pytest.mark.parametrize(
    ("pointer", "message"),
    [
        pytest.param([], "JSON object", id="non-object"),
        pytest.param(
            {"schema_version": True, "generation_id": "0" * 32},
            "pointer schema version",
            id="boolean-schema",
        ),
        pytest.param(
            {"schema_version": 1.0, "generation_id": "0" * 32},
            "pointer schema version",
            id="float-schema",
        ),
        pytest.param(
            {
                "schema_version": INDEX_POINTER_SCHEMA_VERSION + 1,
                "generation_id": "0" * 32,
            },
            "pointer schema version",
            id="unsupported-schema",
        ),
        pytest.param(
            {"schema_version": INDEX_POINTER_SCHEMA_VERSION},
            "generation id",
            id="missing-generation",
        ),
        pytest.param(
            {
                "schema_version": INDEX_POINTER_SCHEMA_VERSION,
                "generation_id": "../outside",
            },
            "generation id",
            id="traversal-generation",
        ),
        pytest.param(
            {
                "schema_version": INDEX_POINTER_SCHEMA_VERSION,
                "generation_id": "A" * 32,
            },
            "generation id",
            id="uppercase-generation",
        ),
    ],
)
def test_index_load_rejects_an_invalid_active_pointer(
    tmp_path: Path, pointer: object, message: str
) -> None:
    _write_active_pointer(tmp_path, pointer)

    with pytest.raises(ValueError, match=message):
        NumpyVectorIndex.load(tmp_path)


def test_index_load_rejects_a_missing_active_generation(tmp_path: Path) -> None:
    _write_active_pointer(
        tmp_path,
        {
            "schema_version": INDEX_POINTER_SCHEMA_VERSION,
            "generation_id": "0" * 32,
        },
    )

    with pytest.raises(FileNotFoundError, match="Active index generation"):
        NumpyVectorIndex.load(tmp_path)


def test_invalid_active_pointer_never_falls_back_to_a_legacy_pair(tmp_path: Path) -> None:
    source_directory = tmp_path / "source"
    index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])
    index.save(source_directory)
    source_generation = _active_generation_directory(source_directory)
    (tmp_path / INDEX_MANIFEST_FILENAME).write_bytes(
        (source_generation / INDEX_MANIFEST_FILENAME).read_bytes()
    )
    (tmp_path / INDEX_VECTOR_FILENAME).write_bytes(
        (source_generation / INDEX_VECTOR_FILENAME).read_bytes()
    )
    _write_active_pointer(tmp_path, [])

    with pytest.raises(ValueError, match="JSON object"):
        NumpyVectorIndex.load(tmp_path)


def test_dangling_active_symlink_never_falls_back_to_a_legacy_pair(tmp_path: Path) -> None:
    source_directory = tmp_path / "source"
    index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])
    index.save(source_directory)
    source_generation = _active_generation_directory(source_directory)
    (tmp_path / INDEX_MANIFEST_FILENAME).write_bytes(
        (source_generation / INDEX_MANIFEST_FILENAME).read_bytes()
    )
    (tmp_path / INDEX_VECTOR_FILENAME).write_bytes(
        (source_generation / INDEX_VECTOR_FILENAME).read_bytes()
    )
    (tmp_path / INDEX_ACTIVE_FILENAME).symlink_to(tmp_path / "missing-active.json")

    with pytest.raises(ValueError, match="symbolic link"):
        NumpyVectorIndex.load(tmp_path)


def test_index_load_rejects_a_symlinked_storage_directory(tmp_path: Path) -> None:
    source_directory = tmp_path / "source"
    symlinked_directory = tmp_path / "symlinked"
    index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])
    index.save(source_directory)
    symlinked_directory.symlink_to(source_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="storage directory.*symbolic link"):
        NumpyVectorIndex.load(symlinked_directory)


def test_active_generation_directory_symlink_is_rejected(tmp_path: Path) -> None:
    source_directory = tmp_path / "source"
    target_directory = tmp_path / "target"
    generation_id = "0" * 32
    index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])
    index.save(source_directory)
    source_generation = _active_generation_directory(source_directory)
    (target_directory / INDEX_GENERATIONS_DIRECTORY).mkdir(parents=True)
    (target_directory / INDEX_GENERATIONS_DIRECTORY / generation_id).symlink_to(
        source_generation,
        target_is_directory=True,
    )
    _write_active_pointer(
        target_directory,
        {
            "schema_version": INDEX_POINTER_SCHEMA_VERSION,
            "generation_id": generation_id,
        },
    )

    with pytest.raises(ValueError, match="generation path.*symbolic link"):
        NumpyVectorIndex.load(target_directory)


@pytest.mark.parametrize(
    "symlink_filename",
    [INDEX_MANIFEST_FILENAME, INDEX_VECTOR_FILENAME],
)
def test_active_generation_file_symlink_is_rejected(tmp_path: Path, symlink_filename: str) -> None:
    source_directory = tmp_path / "source"
    target_directory = tmp_path / "target"
    generation_id = "0" * 32
    index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])
    index.save(source_directory)
    source_generation = _active_generation_directory(source_directory)
    target_generation = target_directory / INDEX_GENERATIONS_DIRECTORY / generation_id
    target_generation.mkdir(parents=True)
    other_filename = (
        INDEX_VECTOR_FILENAME
        if symlink_filename == INDEX_MANIFEST_FILENAME
        else INDEX_MANIFEST_FILENAME
    )
    (target_generation / symlink_filename).symlink_to(source_generation / symlink_filename)
    (target_generation / other_filename).write_bytes(
        (source_generation / other_filename).read_bytes()
    )
    _write_active_pointer(
        target_directory,
        {
            "schema_version": INDEX_POINTER_SCHEMA_VERSION,
            "generation_id": generation_id,
        },
    )

    with pytest.raises(ValueError, match="generation files.*symbolic links"):
        NumpyVectorIndex.load(target_directory)


def test_save_rejects_a_symlinked_generations_directory(tmp_path: Path) -> None:
    index_directory = tmp_path / "index"
    outside_directory = tmp_path / "outside"
    index_directory.mkdir()
    outside_directory.mkdir()
    (index_directory / INDEX_GENERATIONS_DIRECTORY).symlink_to(
        outside_directory,
        target_is_directory=True,
    )
    index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="storage directory.*symbolic link"):
        index.save(index_directory)


def test_save_synchronizes_new_parent_directories_bottom_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_directory = tmp_path / "parent" / "index"
    synchronized_directories: list[Path] = []
    real_fsync_directory = NumpyVectorIndex._fsync_directory

    def record_directory_sync(directory: Path) -> None:
        synchronized_directories.append(directory)
        real_fsync_directory(directory)

    monkeypatch.setattr(
        NumpyVectorIndex,
        "_fsync_directory",
        staticmethod(record_directory_sync),
    )

    _two_entry_index([[1.0, 0.0], [0.0, 1.0]]).save(index_directory)

    assert synchronized_directories[:3] == [
        index_directory.parent,
        tmp_path,
        index_directory,
    ]


def test_save_enforces_the_durability_operation_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generations_directory = tmp_path / INDEX_GENERATIONS_DIRECTORY
    generations_directory.mkdir()
    active_path = tmp_path / INDEX_ACTIVE_FILENAME
    events: list[str] = []
    real_fsync_file = NumpyVectorIndex._fsync_file
    real_fsync_directory = NumpyVectorIndex._fsync_directory
    real_replace = pipeline_module.os.replace

    def record_file_sync(path: Path) -> None:
        if path.name == INDEX_VECTOR_FILENAME:
            events.append("fsync vectors")
        elif path.name == INDEX_MANIFEST_FILENAME:
            events.append("fsync manifest")
        else:
            events.append("fsync active pointer")
        real_fsync_file(path)

    def record_directory_sync(directory: Path) -> None:
        if directory == generations_directory:
            events.append("fsync generations directory")
        elif directory == tmp_path:
            events.append("fsync index directory")
        else:
            events.append("fsync staging directory")
        real_fsync_directory(directory)

    def record_replace(source: Path, destination: Path) -> None:
        destination_path = Path(destination)
        if destination_path == active_path:
            events.append("replace active pointer")
        else:
            events.append("publish generation")
        real_replace(source, destination)

    monkeypatch.setattr(
        NumpyVectorIndex,
        "_fsync_file",
        staticmethod(record_file_sync),
    )
    monkeypatch.setattr(
        NumpyVectorIndex,
        "_fsync_directory",
        staticmethod(record_directory_sync),
    )
    monkeypatch.setattr(pipeline_module.os, "replace", record_replace)

    _two_entry_index([[1.0, 0.0], [0.0, 1.0]]).save(tmp_path)

    assert events == [
        "fsync vectors",
        "fsync manifest",
        "fsync staging directory",
        "publish generation",
        "fsync generations directory",
        "fsync active pointer",
        "replace active pointer",
        "fsync index directory",
    ]


def test_cleanup_failure_does_not_mask_the_publication_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generations_directory = tmp_path / INDEX_GENERATIONS_DIRECTORY
    real_replace = pipeline_module.os.replace

    def fail_generation_publication(source: Path, destination: Path) -> None:
        if Path(destination).parent == generations_directory:
            raise OSError("simulated publication failure")
        real_replace(source, destination)

    def fail_cleanup(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        del path, missing_ok
        raise PermissionError("simulated cleanup failure")

    monkeypatch.setattr(pipeline_module.os, "replace", fail_generation_publication)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)

    with pytest.raises(OSError, match="simulated publication failure"):
        _two_entry_index([[1.0, 0.0], [0.0, 1.0]]).save(tmp_path)


@pytest.mark.parametrize(
    ("failure_phase", "expected_generation_count"),
    [
        pytest.param("generation", 1, id="before-generation-publication"),
        pytest.param("active", 2, id="before-active-switch"),
    ],
)
def test_interrupted_save_preserves_the_previous_active_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
    expected_generation_count: int,
) -> None:
    old_index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])
    new_index = _two_entry_index([[0.0, 1.0], [1.0, 0.0]])
    old_index.save(tmp_path)
    active_path = tmp_path / INDEX_ACTIVE_FILENAME
    generations_directory = tmp_path / INDEX_GENERATIONS_DIRECTORY
    old_pointer_bytes = active_path.read_bytes()
    real_replace = pipeline_module.os.replace

    def fail_at_selected_phase(source: Path, destination: Path) -> None:
        destination_path = Path(destination)
        fail_generation = (
            failure_phase == "generation" and destination_path.parent == generations_directory
        )
        fail_active = failure_phase == "active" and destination_path == active_path
        if fail_generation or fail_active:
            raise OSError("simulated interrupted save")
        real_replace(source, destination)

    monkeypatch.setattr(pipeline_module.os, "replace", fail_at_selected_phase)

    with pytest.raises(OSError, match="simulated interrupted save"):
        new_index.save(tmp_path)

    assert active_path.read_bytes() == old_pointer_bytes
    assert _first_hit_id(tmp_path) == CORPUS[0]["id"]
    assert len(_generation_directories(tmp_path)) == expected_generation_count
    assert not list(generations_directory.glob(".*.tmp"))
    assert not list(tmp_path.glob(f".{INDEX_ACTIVE_FILENAME}.*.tmp"))


def test_reader_uses_the_generation_captured_before_an_active_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])
    new_index = _two_entry_index([[0.0, 1.0], [1.0, 0.0]])
    old_index.save(tmp_path)
    old_vector_path = _index_file_path(tmp_path, INDEX_VECTOR_FILENAME)
    digest_captured = Event()
    resume_reader = Event()
    real_file_sha256 = NumpyVectorIndex._file_sha256

    def pause_after_old_digest(path: Path) -> str:
        digest = real_file_sha256(path)
        if path == old_vector_path and not digest_captured.is_set():
            digest_captured.set()
            if not resume_reader.wait(timeout=5):
                raise AssertionError("reader was not resumed")
        return digest

    monkeypatch.setattr(
        NumpyVectorIndex,
        "_file_sha256",
        staticmethod(pause_after_old_digest),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        reader = executor.submit(_first_hit_id, tmp_path)
        assert digest_captured.wait(timeout=5)
        try:
            new_index.save(tmp_path)
        finally:
            resume_reader.set()
        assert reader.result(timeout=5) == CORPUS[0]["id"]

    assert _first_hit_id(tmp_path) == CORPUS[1]["id"]


def test_concurrent_writers_publish_disjoint_complete_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])
    second_index = _two_entry_index([[0.0, 1.0], [1.0, 0.0]])
    generations_directory = tmp_path / INDEX_GENERATIONS_DIRECTORY
    publish_barrier = Barrier(2)
    real_replace = pipeline_module.os.replace

    def synchronize_generation_publication(source: Path, destination: Path) -> None:
        destination_path = Path(destination)
        if destination_path.parent == generations_directory:
            publish_barrier.wait(timeout=5)
        real_replace(source, destination)

    monkeypatch.setattr(
        pipeline_module.os,
        "replace",
        synchronize_generation_publication,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(first_index.save, tmp_path),
            executor.submit(second_index.save, tmp_path),
        ]
        for future in futures:
            future.result(timeout=10)

    generation_directories = _generation_directories(tmp_path)
    assert len(generation_directories) == 2
    assert _active_generation_directory(tmp_path) in generation_directories
    assert {_first_hit_id(path) for path in generation_directories} == {
        CORPUS[0]["id"],
        CORPUS[1]["id"],
    }
    assert _first_hit_id(tmp_path) in {CORPUS[0]["id"], CORPUS[1]["id"]}


def test_post_switch_sync_failure_leaves_the_new_generation_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_index = _two_entry_index([[1.0, 0.0], [0.0, 1.0]])
    new_index = _two_entry_index([[0.0, 1.0], [1.0, 0.0]])
    old_index.save(tmp_path)
    real_fsync_directory = NumpyVectorIndex._fsync_directory

    def fail_final_index_sync(directory: Path) -> None:
        if directory == tmp_path:
            raise OSError("simulated final directory sync failure")
        real_fsync_directory(directory)

    monkeypatch.setattr(
        NumpyVectorIndex,
        "_fsync_directory",
        staticmethod(fail_final_index_sync),
    )

    with pytest.raises(OSError, match="final directory sync failure"):
        new_index.save(tmp_path)

    assert _first_hit_id(tmp_path) == CORPUS[1]["id"]
    assert len(_generation_directories(tmp_path)) == 2


def test_schema_v1_load_accepts_a_custom_model_and_unknown_manifest_fields(
    tmp_path: Path,
) -> None:
    index = NumpyVectorIndex(dimension=1, embedding_model="custom-embedding-model")
    index.index([CORPUS[0]], np.asarray([[2.0]], dtype=np.float32))
    index.save(tmp_path)
    vector_path = _index_file_path(tmp_path, INDEX_VECTOR_FILENAME)
    vectors = np.load(vector_path, allow_pickle=False)
    vectors[0, 0] = np.float32(1.0 + 5e-6)
    _rewrite_vectors(tmp_path, vectors, update_digest=True)
    manifest = _read_manifest(tmp_path)
    manifest["future_extension"] = {"ignored": True}
    _write_manifest(tmp_path, manifest)

    loaded = NumpyVectorIndex.load(tmp_path)
    assert loaded.dimension == 1
    assert loaded.embedding_model == "custom-embedding-model"
    roundtrip_directory = tmp_path / "roundtrip"
    loaded.save(roundtrip_directory)
    assert (
        _index_file_path(roundtrip_directory, INDEX_VECTOR_FILENAME).read_bytes()
        == vector_path.read_bytes()
    )


@pytest.mark.parametrize(
    ("dimension", "embedding_model", "message"),
    [
        pytest.param(True, EMBEDDING_MODEL, "dimension", id="boolean-dimension"),
        pytest.param(1.0, EMBEDDING_MODEL, "dimension", id="float-dimension"),
        pytest.param(0, EMBEDDING_MODEL, "dimension", id="zero-dimension"),
        pytest.param(1, "", "model", id="empty-model"),
        pytest.param(1, "   ", "model", id="whitespace-model"),
    ],
)
def test_index_constructor_rejects_invalid_identity(
    dimension: object, embedding_model: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        NumpyVectorIndex(dimension=dimension, embedding_model=embedding_model)


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param([], id="not-an-array"),
        pytest.param(np.ones(12, dtype=np.float32), id="not-a-matrix"),
        pytest.param(np.ones((1, 12), dtype=np.float32), id="wrong-row-count"),
        pytest.param(np.empty((len(CORPUS), 0), dtype=np.float32), id="zero-columns"),
        pytest.param(
            np.full((len(CORPUS), 12), np.nan, dtype=np.float32),
            id="non-finite",
        ),
    ],
)
def test_unpaid_ingest_rejects_an_invalid_embedding_matrix(tmp_path: Path, matrix: object) -> None:
    with pytest.raises(ValueError, match="Corpus embedder returned an invalid matrix"):
        ingest_corpus(
            embedder=_MatrixEmbeddingProvider(matrix),
            output_directory=tmp_path,
        )
    assert not (tmp_path / INDEX_ACTIVE_FILENAME).exists()
    assert not (tmp_path / INDEX_GENERATIONS_DIRECTORY).exists()
    assert not (tmp_path / INDEX_MANIFEST_FILENAME).exists()
    assert not (tmp_path / INDEX_VECTOR_FILENAME).exists()


@pytest.mark.parametrize(
    "missing_filename",
    [INDEX_MANIFEST_FILENAME, INDEX_VECTOR_FILENAME],
)
def test_index_load_rejects_an_incomplete_file_pair(tmp_path: Path, missing_filename: str) -> None:
    _persist_fixed_index(tmp_path)
    _index_file_path(tmp_path, missing_filename).unlink()
    with pytest.raises(FileNotFoundError, match="Active index generation"):
        NumpyVectorIndex.load(tmp_path)


@pytest.mark.parametrize(
    "present_filename",
    [INDEX_MANIFEST_FILENAME, INDEX_VECTOR_FILENAME],
)
def test_index_load_rejects_an_incomplete_legacy_pair(
    tmp_path: Path, present_filename: str
) -> None:
    (tmp_path / present_filename).write_bytes(b"incomplete")

    with pytest.raises(FileNotFoundError, match="No complete index"):
        NumpyVectorIndex.load(tmp_path)


def test_index_load_rejects_a_non_object_manifest(tmp_path: Path) -> None:
    _persist_fixed_index(tmp_path)
    _write_manifest(tmp_path, [])
    with pytest.raises(ValueError, match="JSON object"):
        NumpyVectorIndex.load(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param("schema_version", True, "schema version", id="boolean-schema"),
        pytest.param("schema_version", 1.0, "schema version", id="float-schema"),
        pytest.param(
            "schema_version",
            INDEX_SCHEMA_VERSION + 1,
            "schema version",
            id="unsupported-schema",
        ),
        pytest.param("dimension", True, "dimension", id="boolean-dimension"),
        pytest.param("dimension", 1.0, "dimension", id="float-dimension"),
        pytest.param("dimension", 0, "dimension", id="zero-dimension"),
        pytest.param("embedding_model", None, "model", id="missing-model"),
        pytest.param("embedding_model", "", "model", id="empty-model"),
        pytest.param("embedding_model", "   ", "model", id="whitespace-model"),
    ],
)
def test_index_load_rejects_invalid_manifest_identity(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    _persist_fixed_index(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest[field] = value
    _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match=message):
        NumpyVectorIndex.load(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param(
            "distance_definition",
            "raw dot product",
            "distance definition",
            id="distance",
        ),
        pytest.param(
            "tie_break_rule",
            "insertion order",
            "tie-breaking rule",
            id="tie-break",
        ),
    ],
)
def test_index_load_rejects_changed_retrieval_policy(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    _persist_fixed_index(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest[field] = value
    _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match=message):
        NumpyVectorIndex.load(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param("not-list", "non-empty list", id="entries-not-list"),
        pytest.param("empty-list", "non-empty list", id="entries-empty"),
        pytest.param("entry-not-object", "contain id, text, and metadata", id="entry-not-object"),
        pytest.param("missing-field", "contain id, text, and metadata", id="missing-field"),
        pytest.param("empty-id", "non-empty string", id="empty-id"),
        pytest.param("duplicate-id", "Duplicate indexed entry id", id="duplicate-id"),
        pytest.param("empty-text", "text is invalid", id="empty-text"),
        pytest.param("metadata-not-object", "metadata is invalid", id="metadata-not-object"),
        pytest.param("missing-citation", "missing citation fields", id="missing-citation"),
        pytest.param("invalid-grimoire", "grimoire_id is invalid", id="invalid-grimoire"),
        pytest.param("invalid-folio", "folio is invalid", id="invalid-folio"),
        pytest.param("no-citation", "no citation metadata", id="no-citation"),
    ],
)
def test_index_load_rejects_invalid_entries(tmp_path: Path, mutation: str, message: str) -> None:
    _persist_fixed_index(tmp_path)
    manifest = _read_manifest(tmp_path)
    entries = manifest["entries"]
    assert isinstance(entries, list)

    if mutation == "not-list":
        manifest["entries"] = {}
    elif mutation == "empty-list":
        manifest["entries"] = []
    elif mutation == "entry-not-object":
        entries[0] = []
    else:
        entry = entries[0]
        assert isinstance(entry, dict)
        if mutation == "missing-field":
            entry.pop("metadata")
        elif mutation == "empty-id":
            entry["id"] = ""
        elif mutation == "duplicate-id":
            second = entries[1]
            assert isinstance(second, dict)
            second["id"] = entry["id"]
        elif mutation == "empty-text":
            entry["text"] = ""
        elif mutation == "metadata-not-object":
            entry["metadata"] = []
        else:
            metadata = entry["metadata"]
            assert isinstance(metadata, dict)
            if mutation == "missing-citation":
                metadata.pop("folio")
            elif mutation == "invalid-grimoire":
                metadata["grimoire_id"] = 1
            elif mutation == "invalid-folio":
                metadata["folio"] = True
            elif mutation == "no-citation":
                metadata["grimoire_id"] = None
                metadata["folio"] = None
            else:
                raise AssertionError(f"Unhandled mutation: {mutation}")

    _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match=message):
        NumpyVectorIndex.load(tmp_path)


def test_index_load_rejects_a_corpus_fingerprint_mismatch(tmp_path: Path) -> None:
    _persist_fixed_index(tmp_path)
    manifest = _read_manifest(tmp_path)
    entries = manifest["entries"]
    assert isinstance(entries, list)
    first = entries[0]
    assert isinstance(first, dict)
    first["text"] = f"{first['text']} Drifted."
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="corpus fingerprint"):
        NumpyVectorIndex.load(tmp_path)


def test_index_load_rejects_a_vector_fingerprint_mismatch(tmp_path: Path) -> None:
    _persist_fixed_index(tmp_path)
    vector_path = _index_file_path(tmp_path, INDEX_VECTOR_FILENAME)
    vectors = np.load(vector_path, allow_pickle=False)
    vectors[0, 0] += np.float32(0.25)
    _rewrite_vectors(tmp_path, vectors, update_digest=False)

    with pytest.raises(ValueError, match="vector fingerprint"):
        NumpyVectorIndex.load(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param("row-misalignment", "not rank-aligned", id="row-misalignment"),
        pytest.param("wrong-dtype", "float32 dtype", id="wrong-dtype"),
        pytest.param("non-finite", "non-finite", id="non-finite"),
        pytest.param("zero-vector", "zero vectors", id="zero-vector"),
        pytest.param("non-unit", "unit-norm tolerance", id="non-unit"),
    ],
)
def test_index_load_rejects_invalid_persisted_vectors(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _persist_fixed_index(tmp_path)
    vector_path = _index_file_path(tmp_path, INDEX_VECTOR_FILENAME)
    vectors = np.load(vector_path, allow_pickle=False)
    if mutation == "row-misalignment":
        vectors = vectors[:-1]
    elif mutation == "wrong-dtype":
        vectors = vectors.astype(np.float64)
    elif mutation == "non-finite":
        vectors[0, 0] = np.nan
    elif mutation == "zero-vector":
        vectors[0] = 0.0
    elif mutation == "non-unit":
        vectors[0] *= np.float32(2.0)
    else:
        raise AssertionError(f"Unhandled mutation: {mutation}")
    _rewrite_vectors(tmp_path, vectors, update_digest=True)

    with pytest.raises(ValueError, match=message):
        NumpyVectorIndex.load(tmp_path)


def test_stale_current_corpus_is_rejected_before_provider_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _persist_fixed_index(tmp_path)
    stale_corpus = [entry.copy() for entry in CORPUS]
    stale_corpus[0]["text"] += " Drifted."
    monkeypatch.setattr(pipeline_module, "CORPUS", stale_corpus)
    monkeypatch.setattr(
        pipeline_module,
        "_real_client",
        lambda: pytest.fail("provider client initialized before index validation"),
    )

    with pytest.raises(ValueError, match="stale relative to corpus.py"):
        pipeline_module._real_pipeline(index_directory=tmp_path)


def test_stale_embedding_model_is_rejected_before_provider_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _persist_fixed_index(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest["embedding_model"] = "stale-embedding-model"
    _write_manifest(tmp_path, manifest)
    monkeypatch.setattr(
        pipeline_module,
        "_real_client",
        lambda: pytest.fail("provider client initialized before index validation"),
    )

    with pytest.raises(ValueError, match="differs from configured model"):
        pipeline_module._real_pipeline(index_directory=tmp_path)


def test_real_client_uses_exact_bounded_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    client = object()
    constructor_kwargs: dict[str, object] = {}

    def build_client(**kwargs: object) -> object:
        constructor_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(pipeline_module, "OpenAI", build_client)

    assert OPENAI_MAX_RETRIES == 0
    assert OPENAI_TIMEOUT_SECONDS == 120.0
    assert pipeline_module._real_client() is client
    assert constructor_kwargs == {
        "max_retries": 0,
        "timeout": 120.0,
    }


def test_real_client_rejects_missing_key_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        pipeline_module,
        "OpenAI",
        lambda **_kwargs: pytest.fail("client constructed without a credential"),
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        pipeline_module._real_client()


def test_zero_vector_is_rejected() -> None:
    index = NumpyVectorIndex(dimension=2, embedding_model=EMBEDDING_MODEL)
    with pytest.raises(ValueError, match="zero vectors"):
        index.index([CORPUS[0]], np.asarray([[0.0, 0.0]], dtype=np.float32))


@pytest.mark.parametrize(
    ("query", "message"),
    [
        pytest.param([0.0, 0.0], "zero vectors", id="zero"),
        pytest.param([np.nan, 0.0], "non-finite", id="non-finite"),
    ],
)
def test_invalid_query_vector_is_rejected(query: list[float], message: str) -> None:
    index = NumpyVectorIndex(dimension=2, embedding_model=EMBEDDING_MODEL)
    index.index([CORPUS[0]], np.asarray([[1.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match=message):
        index.search(np.asarray(query, dtype=np.float32), top_k=1)


class _FakeEmbeddingsEndpoint:
    def __init__(self, data: list[SimpleNamespace] | None = None) -> None:
        self.kwargs: dict[str, object] = {}
        self._data = (
            data
            if data is not None
            else [
                SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
            ]
        )

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(data=self._data)


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
    assert matrix.dtype == np.float32
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


@pytest.mark.parametrize(
    ("inputs", "input_ids"),
    [
        pytest.param([], [], id="empty"),
        pytest.param(["first"], [], id="missing-id"),
        pytest.param(["first"], ["id-1", "id-2"], id="extra-id"),
    ],
)
def test_embedding_adapter_rejects_empty_or_misaligned_requests(
    inputs: list[str], input_ids: list[str]
) -> None:
    endpoint = _FakeEmbeddingsEndpoint()
    with pytest.raises(ValueError, match="non-empty and rank-aligned"):
        OpenAIEmbeddingProvider(SimpleNamespace(embeddings=endpoint)).embed(
            inputs,
            input_ids=input_ids,
            stage="corpus",
        )
    assert endpoint.kwargs == {}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        pytest.param([], "indices are not contiguous", id="missing-all-indices"),
        pytest.param(
            [
                SimpleNamespace(index=0, embedding=[1.0]),
                SimpleNamespace(index=0, embedding=[0.0]),
            ],
            "indices are not contiguous",
            id="duplicate-index",
        ),
        pytest.param(
            [
                SimpleNamespace(index=0, embedding=[1.0]),
                SimpleNamespace(index=2, embedding=[0.0]),
            ],
            "indices are not contiguous",
            id="gapped-index",
        ),
        pytest.param(
            [
                SimpleNamespace(index=0, embedding=1.0),
                SimpleNamespace(index=1, embedding=0.0),
            ],
            "Unexpected embedding matrix shape",
            id="not-a-matrix",
        ),
        pytest.param(
            [
                SimpleNamespace(index=0, embedding=[]),
                SimpleNamespace(index=1, embedding=[]),
            ],
            "Unexpected embedding matrix shape",
            id="zero-dimension",
        ),
        pytest.param(
            [
                SimpleNamespace(index=0, embedding=[float("nan")]),
                SimpleNamespace(index=1, embedding=[1.0]),
            ],
            "non-finite value",
            id="non-finite",
        ),
    ],
)
def test_embedding_adapter_rejects_malformed_provider_responses(
    data: list[SimpleNamespace], message: str
) -> None:
    endpoint = _FakeEmbeddingsEndpoint(data)
    with pytest.raises(ValueError, match=message):
        OpenAIEmbeddingProvider(SimpleNamespace(embeddings=endpoint)).embed(
            ["first", "second"],
            input_ids=["id-1", "id-2"],
            stage="corpus",
        )


class _FakeCompletionsEndpoint:
    def __init__(
        self,
        content: str | None,
        finish_reason: str = "stop",
        *,
        refusal: str | None = None,
        choice_count: int = 1,
    ) -> None:
        self.kwargs: dict[str, object] = {}
        self._content = content
        self._finish_reason = finish_reason
        self._refusal = refusal
        self._choice_count = choice_count

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=self._finish_reason,
                    message=SimpleNamespace(content=self._content, refusal=self._refusal),
                )
                for _ in range(self._choice_count)
            ]
        )


def _fake_generation_client(endpoint: _FakeCompletionsEndpoint) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=endpoint))


def test_real_generation_adapter_preserves_structured_text_and_citations() -> None:
    raw = '\n{"answer":"3 drams","sources":["verdigris-dose-verdant"]}\n'
    endpoint = _FakeCompletionsEndpoint(raw)
    generator = OpenAIAnswerGenerator(_fake_generation_client(endpoint))
    messages = [
        PromptMessage(role="system", content="system"),
        PromptMessage(role="user", content="user"),
    ]
    assert generator.generate("question", messages) == raw
    assert endpoint.kwargs == {
        "model": GENERATION_MODEL,
        "temperature": GENERATION_TEMPERATURE,
        "max_completion_tokens": 300,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
    }


@pytest.mark.parametrize(
    ("content", "finish_reason", "refusal", "choice_count", "message"),
    [
        pytest.param("answer", "stop", None, 0, "Expected one generation choice, got 0", id="none"),
        pytest.param("answer", "stop", None, 2, "Expected one generation choice, got 2", id="many"),
        pytest.param("partial", "length", None, 1, "did not finish normally", id="truncated"),
        pytest.param(None, "stop", None, 1, "no text content", id="missing-text"),
        pytest.param(None, "stop", "policy refusal", 1, "Refusal: policy refusal", id="refusal"),
        pytest.param("", "stop", None, 1, "empty text content", id="empty"),
        pytest.param(" \n", "stop", None, 1, "empty text content", id="whitespace"),
    ],
)
def test_generation_parser_rejects_incomplete_provider_responses(
    content: str | None,
    finish_reason: str,
    refusal: str | None,
    choice_count: int,
    message: str,
) -> None:
    response = _FakeCompletionsEndpoint(
        content,
        finish_reason,
        refusal=refusal,
        choice_count=choice_count,
    ).create()
    with pytest.raises(GenerationResponseError, match=message):
        parse_generation_response(response)


@pytest.mark.parametrize("case", ANSWERABLE, ids=lambda case: case.case_id)
def test_rag_record_maps_exactly_to_ragaliq_case(
    case: GoldenCase, records: dict[str, RagRecord]
) -> None:
    record = records[case.case_id]
    test_case = to_ragaliq_case(record, case_id=case.case_id)
    assert test_case.model_dump() == {
        "id": case.case_id,
        "name": f"VerdigrisE semantic residue for {case.case_id}",
        "query": record.question,
        "context": [record.context_payload],
        "response": record.answer,
        "expected_answer": None,
        "expected_facts": None,
        "tags": ["verdigrise", "semantic-residue"],
    }


def test_ragaliq_case_applies_published_whitespace_normalization(
    records: dict[str, RagRecord],
) -> None:
    case = GOLDEN_BY_ID["numeric-source-verdigris-dose"]
    record = records[case.case_id]
    spaced_record = record.model_copy(
        update={
            "question": f" \n{record.question}\n ",
            "context_payload": f" \n{record.context_payload}\n ",
            "answer": f" \n{record.answer}\n ",
        }
    )
    test_case = to_ragaliq_case(spaced_record, case_id=case.case_id)
    assert test_case.query == record.question
    assert test_case.context == [record.context_payload]
    assert test_case.response == record.answer


def test_ragaliq_canned_runner_executes_structural_wiring_locally(
    records: dict[str, RagRecord],
) -> None:
    case = GOLDEN_BY_ID["numeric-source-verdigris-dose"]
    record = records[case.case_id]
    transport = CannedJudgeTransport()
    runner = build_ragaliq_runner(transport)
    assert runner.evaluator_names == ["faithfulness", "relevance"]
    assert runner.default_threshold == 0.7
    test_case = to_ragaliq_case(record, case_id=case.case_id)
    result = runner.evaluate(test_case)
    assert isinstance(result, RAGTestResult)
    assert result.passed
    assert result.scores == {"faithfulness": 1.0, "relevance": 0.9}
    assert result.test_case == test_case
    assert result.judge_tokens_used == 90
    assert len(transport.calls) == 3
    for call in transport.calls:
        assert set(call) == {
            "system_prompt",
            "user_prompt",
            "model",
            "temperature",
            "max_tokens",
        }
        assert call["model"] == DEFAULT_JUDGE_MODEL
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == 1024

    user_prompts: list[str] = []
    for call in transport.calls:
        user_prompt = call["user_prompt"]
        assert isinstance(user_prompt, str)
        user_prompts.append(user_prompt)
    routing = sorted(
        (
            record.question in prompt,
            record.answer in prompt,
            record.context_payload in prompt,
        )
        for prompt in user_prompts
    )
    assert routing == [
        (False, False, True),
        (False, True, False),
        (True, True, False),
    ]


def _marked_item(*markers: str) -> Any:
    selected = set(markers)
    return SimpleNamespace(get_closest_marker=lambda name: object() if name in selected else None)


@pytest.mark.parametrize(
    ("markers", "environment", "expected"),
    [
        pytest.param((), {}, [], id="free-node"),
        pytest.param(("openai",), {}, ["OPENAI_API_KEY"], id="openai-missing"),
        pytest.param(
            ("openai",),
            {"OPENAI_API_KEY": ""},
            ["OPENAI_API_KEY"],
            id="openai-empty",
        ),
        pytest.param(
            ("openai",),
            {"OPENAI_API_KEY": "present"},
            [],
            id="openai-present-without-anthropic",
        ),
        pytest.param(("rag_test",), {}, ["ANTHROPIC_API_KEY"], id="anthropic-missing"),
        pytest.param(
            ("openai", "rag_test"),
            {"OPENAI_API_KEY": "present"},
            ["ANTHROPIC_API_KEY"],
            id="semantic-anthropic-missing",
        ),
        pytest.param(
            ("openai", "rag_test"),
            {"OPENAI_API_KEY": "present", "ANTHROPIC_API_KEY": "present"},
            [],
            id="semantic-credentials-present",
        ),
    ],
)
def test_paid_credential_preflight_requires_only_selected_marker_keys(
    markers: tuple[str, ...], environment: dict[str, str], expected: list[str]
) -> None:
    item = _marked_item(*markers)

    assert eval_conftest._missing_paid_credentials([item], environment) == expected


def test_paid_credential_preflight_preserves_credential_free_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    session = SimpleNamespace(
        config=SimpleNamespace(option=SimpleNamespace(collectonly=True)),
        items=[_marked_item("openai", "rag_test")],
    )

    eval_conftest.pytest_collection_finish(session)


def test_paid_credential_preflight_fails_before_selected_paid_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    session = SimpleNamespace(
        config=SimpleNamespace(option=SimpleNamespace(collectonly=False)),
        items=[_marked_item("openai", "rag_test")],
    )

    with pytest.raises(
        pytest.UsageError,
        match=(
            "Explicitly selected paid tests require non-empty environment variables: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY"
        ),
    ):
        eval_conftest.pytest_collection_finish(session)


def _build_ingested_pipeline(
    *,
    embedder: EmbeddingProvider,
    generator: AnswerGenerator,
    output_directory: Path,
    debug: bool = False,
) -> RagPipeline:
    """Embed the corpus once without asking any golden question eagerly."""

    index = ingest_corpus(
        embedder=embedder,
        output_directory=output_directory,
        debug=debug,
    )
    return RagPipeline(
        index=index,
        embedder=embedder,
        generator=generator,
        debug=debug,
    )


def _build_live_openai_pipeline(*, output_directory: Path) -> RagPipeline:
    """Build the paid fixture through the same policy-bound client as runtime paths."""

    client = pipeline_module._real_client()
    return _build_ingested_pipeline(
        embedder=OpenAIEmbeddingProvider(client),
        generator=OpenAIAnswerGenerator(client),
        output_directory=output_directory,
        debug=True,
    )


def test_live_openai_pipeline_uses_runtime_client_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = object()
    expected_pipeline = object()
    build_kwargs: dict[str, object] = {}
    monkeypatch.setattr(pipeline_module, "_real_client", lambda: client)

    def capture_build(**kwargs: object) -> object:
        build_kwargs.update(kwargs)
        return expected_pipeline

    monkeypatch.setitem(globals(), "_build_ingested_pipeline", capture_build)

    assert _build_live_openai_pipeline(output_directory=tmp_path) is expected_pipeline
    embedder = build_kwargs["embedder"]
    generator = build_kwargs["generator"]
    assert isinstance(embedder, OpenAIEmbeddingProvider)
    assert isinstance(generator, OpenAIAnswerGenerator)
    assert embedder._client is client
    assert generator._client is client
    assert build_kwargs["output_directory"] == tmp_path
    assert build_kwargs["debug"] is True


@pytest.fixture(scope="module")
def live_openai_pipeline(
    tmp_path_factory: pytest.TempPathFactory,
) -> RagPipeline:
    """Paid fixture: embed the corpus once; selected tests ask their own case."""

    return _build_live_openai_pipeline(
        output_directory=tmp_path_factory.mktemp("live-openai-index"),
    )


def test_live_openai_fixture_delegates_to_policy_bound_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_pipeline = object()
    captured_directories: list[Path] = []

    def capture_build(*, output_directory: Path) -> object:
        captured_directories.append(output_directory)
        return expected_pipeline

    monkeypatch.setitem(globals(), "_build_live_openai_pipeline", capture_build)
    tmp_path_factory = SimpleNamespace(mktemp=lambda _prefix: tmp_path)

    assert live_openai_pipeline.__wrapped__(tmp_path_factory) is expected_pipeline
    assert captured_directories == [tmp_path]


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


def _ask_and_assert_live_case(pipeline: RagPipeline, case: GoldenCase) -> RagRecord:
    """Ask one selected case and enforce exact ownership before semantic judging."""

    record = pipeline.ask(case.question)
    _assert_live_golden_contract(case, record)
    return record


@pytest.mark.parametrize(
    "selected_cases",
    [
        pytest.param(GOLDEN[:1], id="one-selected-case"),
        pytest.param(ANSWERABLE, id="six-answerable-cases"),
        pytest.param(GOLDEN, id="seven-golden-cases"),
    ],
)
def test_paid_pipeline_fan_out_matches_selected_cases(
    tmp_path: Path,
    selected_cases: list[GoldenCase],
) -> None:
    embedder = FixedEmbeddingProvider()
    generator = FixedAnswerGenerator()
    pipeline = _build_ingested_pipeline(
        embedder=embedder,
        generator=generator,
        output_directory=tmp_path,
    )

    assert [call[2] for call in embedder.calls] == ["corpus"]
    assert generator.calls == []

    records = [_ask_and_assert_live_case(pipeline, case) for case in selected_cases]
    selected_questions = [case.question for case in selected_cases]

    assert [record.question for record in records] == selected_questions
    assert [call[0] for call in embedder.calls[1:]] == [
        [question] for question in selected_questions
    ]
    assert [call[1] for call in embedder.calls[1:]] == [["<query>"] for _ in selected_questions]
    assert [call[2] for call in embedder.calls[1:]] == ["query" for _ in selected_questions]
    assert [question for question, _ in generator.calls] == selected_questions
    assert len(embedder.calls) + len(generator.calls) == 1 + 2 * len(selected_cases)


@pytest.mark.openai
@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: case.case_id)
def test_real_openai_all_golden_acceptance(
    case: GoldenCase,
    live_openai_pipeline: RagPipeline,
) -> None:
    """Paid: every golden case constrains the configured retrieval and generation models."""

    _ask_and_assert_live_case(live_openai_pipeline, case)


@pytest.mark.openai
@pytest.mark.rag_test
@pytest.mark.parametrize("case", ANSWERABLE, ids=lambda case: case.case_id)
def test_ragaliq_claude_semantic_residue_on_live_answers(
    case: GoldenCase,
    live_openai_pipeline: RagPipeline,
    rag_tester: Any,
) -> None:
    """Paid: native cross-family judge owns faithfulness and answer relevance only."""

    record = _ask_and_assert_live_case(live_openai_pipeline, case)
    test_case = to_ragaliq_case(record, case_id=case.case_id)
    result = rag_tester.evaluate(test_case)
    assert result.passed
    assert set(result.scores) == {"faithfulness", "relevance"}
