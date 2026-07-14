"""VerdigrisE deterministic contracts and explicitly gated semantic integrations.

Ownership boundary:
    deterministic pytest: ids, rank, collision materialization, values, units,
        qualifiers, exact abstention, prompt bytes, citations, and distance math;
    RagaliQ: faithfulness and answer relevance after deterministic checks pass.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from ragaliq import RAGTestResult
from ragaliq.judges import DEFAULT_JUDGE_MODEL

import pipeline as pipeline_module
from config import (
    ABSTENTION_PHRASE,
    DISTANCE_DEFINITION,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    INDEX_MANIFEST_FILENAME,
    INDEX_SCHEMA_VERSION,
    INDEX_VECTOR_FILENAME,
    TIE_BREAK_RULE,
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


def _read_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / INDEX_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    return manifest


def _write_manifest(directory: Path, manifest: object) -> None:
    (directory / INDEX_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_vectors(directory: Path, vectors: np.ndarray, *, update_digest: bool) -> None:
    vector_path = directory / INDEX_VECTOR_FILENAME
    np.save(vector_path, vectors, allow_pickle=False)
    if update_digest:
        manifest = _read_manifest(directory)
        manifest["vectors_sha256"] = hashlib.sha256(vector_path.read_bytes()).hexdigest()
        _write_manifest(directory, manifest)


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
    assert (tmp_path / INDEX_MANIFEST_FILENAME).is_file()
    assert (tmp_path / INDEX_VECTOR_FILENAME).is_file()
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

    loaded = NumpyVectorIndex.load(tmp_path)
    case = GOLDEN_BY_ID["numeric-source-verdigris-dose"]
    hits = loaded.search(np.asarray(_QUESTION_VECTORS[case.question], dtype=np.float32))
    assert [hit.id for hit in hits] == case.expected_ranked_ids
    assert hits[0].metadata["grimoire_id"] == "GRIM-VERDANT"
    assert hits[0].metadata["folio"] == 21
    assert hits[0].metadata["condition"] == ("distilled in copper and administered after dusk")


def test_schema_v1_load_accepts_a_custom_model_and_unknown_manifest_fields(
    tmp_path: Path,
) -> None:
    index = NumpyVectorIndex(dimension=1, embedding_model="custom-embedding-model")
    index.index([CORPUS[0]], np.asarray([[2.0]], dtype=np.float32))
    index.save(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest["future_extension"] = {"ignored": True}
    _write_manifest(tmp_path, manifest)

    loaded = NumpyVectorIndex.load(tmp_path)
    assert loaded.dimension == 1
    assert loaded.embedding_model == "custom-embedding-model"


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
    assert not (tmp_path / INDEX_MANIFEST_FILENAME).exists()
    assert not (tmp_path / INDEX_VECTOR_FILENAME).exists()


@pytest.mark.parametrize(
    "missing_filename",
    [INDEX_MANIFEST_FILENAME, INDEX_VECTOR_FILENAME],
)
def test_index_load_rejects_an_incomplete_file_pair(tmp_path: Path, missing_filename: str) -> None:
    _persist_fixed_index(tmp_path)
    (tmp_path / missing_filename).unlink()
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
    vector_path = tmp_path / INDEX_VECTOR_FILENAME
    vectors = np.load(vector_path, allow_pickle=False)
    vectors[0, 0] += np.float32(0.25)
    _rewrite_vectors(tmp_path, vectors, update_digest=False)

    with pytest.raises(ValueError, match="vector fingerprint"):
        NumpyVectorIndex.load(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param("row-misalignment", "not rank-aligned", id="row-misalignment"),
        pytest.param("non-finite", "non-finite", id="non-finite"),
        pytest.param("zero-vector", "zero vectors", id="zero-vector"),
    ],
)
def test_index_load_rejects_invalid_persisted_vectors(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _persist_fixed_index(tmp_path)
    vector_path = tmp_path / INDEX_VECTOR_FILENAME
    vectors = np.load(vector_path, allow_pickle=False)
    if mutation == "row-misalignment":
        vectors = vectors[:-1]
    elif mutation == "non-finite":
        vectors[0, 0] = np.nan
    elif mutation == "zero-vector":
        vectors[0] = 0.0
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


def test_zero_vector_is_rejected() -> None:
    index = NumpyVectorIndex(dimension=2, embedding_model=EMBEDDING_MODEL)
    with pytest.raises(ValueError, match="zero vectors"):
        index.index([CORPUS[0]], np.asarray([[0.0, 0.0]], dtype=np.float32))


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


requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for explicitly selected paid tests",
)
requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY is required for explicitly selected RagaliQ judge tests",
)


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


@pytest.fixture(scope="module")
def live_openai_pipeline(
    tmp_path_factory: pytest.TempPathFactory,
) -> RagPipeline:
    """Paid fixture: embed the corpus once; selected tests ask their own case."""

    client = OpenAI()
    return _build_ingested_pipeline(
        embedder=OpenAIEmbeddingProvider(client),
        generator=OpenAIAnswerGenerator(client),
        output_directory=tmp_path_factory.mktemp("live-openai-index"),
        debug=True,
    )


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
@requires_openai
@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: case.case_id)
def test_real_openai_all_golden_acceptance(
    case: GoldenCase,
    live_openai_pipeline: RagPipeline,
) -> None:
    """Paid: every golden case constrains the configured retrieval and generation models."""

    _ask_and_assert_live_case(live_openai_pipeline, case)


@pytest.mark.openai
@pytest.mark.rag_test
@requires_openai
@requires_anthropic
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
