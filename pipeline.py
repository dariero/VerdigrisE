"""VerdigrisE pipeline with four explicit stages and a validated cosine index.

Reader map:
    stage 1  EmbeddingProvider.embed            -> corpus or query vector matrix
    stage 2  NumpyVectorIndex.index/search      -> vectors, metadata, ranked top-2
    stage 3  build_generation_messages          -> exact context and request
    stage 4  AnswerGenerator.generate/RagRecord -> verbatim answer and full capture

The stage labels are the compact teaching path. Provider injection, persisted
fingerprints, and capture models retain the base auditability around that path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from config import (
    ABSTENTION_PHRASE,
    DISTANCE_DEFINITION,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    INDEX_ACTIVE_FILENAME,
    INDEX_DIRECTORY,
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
from corpus import CORPUS, CorpusEntry, validate_corpus
from models import PromptMessage, RagRecord, RetrievedChunk

type _Float32Array = NDArray[np.float32]

_FLOAT32_UNIT_NORM_TOLERANCE = 1e-5


class _IndexEntry(TypedDict):
    id: str
    text: str
    metadata: dict[str, object]


class EmbeddingProvider(Protocol):
    model: str

    def embed(
        self,
        inputs: list[str],
        *,
        input_ids: list[str],
        stage: str,
        debug: bool = False,
    ) -> _Float32Array:
        """Return one vector row per input string."""


class AnswerGenerator(Protocol):
    def generate(self, question: str, messages: list[PromptMessage]) -> str:
        """Return the model response without rewriting its content."""


class GenerationResponseError(RuntimeError):
    """Raised when generation does not contain one complete text response."""


def _require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in the process environment; "
            "VerdigrisE does not read dotenv files."
        )


def _safe_embedding_debug(
    *, model: str, stage: str, input_ids: list[str], matrix: _Float32Array
) -> None:
    """Print shapes and stable ids, never input text, secrets, or vector values."""

    print(
        "verdigrise_embedding_debug "
        f"stage={stage} model={model} input_count={matrix.shape[0]} "
        f"dimensions={matrix.shape[1]} shape={matrix.shape} ids={input_ids!r}"
    )


class OpenAIEmbeddingProvider:
    """Thin adapter over the real OpenAI embeddings call."""

    model = EMBEDDING_MODEL

    def __init__(self, client: Any) -> None:
        self._client = client

    def embed(
        self,
        inputs: list[str],
        *,
        input_ids: list[str],
        stage: str,
        debug: bool = False,
    ) -> _Float32Array:
        if not inputs or len(inputs) != len(input_ids):
            raise ValueError("inputs and input_ids must be non-empty and rank-aligned")

        response = self._client.embeddings.create(
            model=self.model,
            input=inputs,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        indices = [item.index for item in ordered]
        if indices != list(range(len(inputs))):
            raise ValueError(f"Embedding response indices are not contiguous: {indices}")

        matrix = np.asarray([item.embedding for item in ordered], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(inputs) or matrix.shape[1] == 0:
            raise ValueError(f"Unexpected embedding matrix shape: {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("Embedding matrix contains a non-finite value")
        if debug:
            _safe_embedding_debug(
                model=self.model,
                stage=stage,
                input_ids=input_ids,
                matrix=matrix,
            )
        return matrix


def parse_generation_response(response: Any) -> str:
    """Return exactly one complete text choice, preserving citations and JSON."""

    choices = response.choices
    if len(choices) != 1:
        raise GenerationResponseError(f"Expected one generation choice, got {len(choices)}")
    choice = choices[0]
    if choice.finish_reason != "stop":
        raise GenerationResponseError(
            f"Generation did not finish normally: {choice.finish_reason!r}"
        )
    content = choice.message.content
    if not isinstance(content, str):
        refusal = getattr(choice.message, "refusal", None)
        detail = f" Refusal: {refusal}" if refusal else ""
        raise GenerationResponseError(f"Generation returned no text content.{detail}")
    if not content.strip():
        raise GenerationResponseError("Generation returned empty text content")
    return content


class OpenAIAnswerGenerator:
    """Thin adapter over the real OpenAI Chat Completions call."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def generate(self, question: str, messages: list[PromptMessage]) -> str:
        del question  # The exact question is already present in messages.
        response = self._client.chat.completions.create(
            model=GENERATION_MODEL,
            temperature=GENERATION_TEMPERATURE,
            max_completion_tokens=300,
            messages=[message.model_dump() for message in messages],
        )
        return parse_generation_response(response)


class NumpyVectorIndex:
    """Exact cosine index with explicit metadata and deterministic ranking.

    Stored rows are L2-normalized. Query rows are normalized at search time.
    Similarity is their dot product bounded to [-1, 1]. Distance is exactly one
    minus similarity.
    Equal similarities are ordered by ascending stable chunk id.
    """

    def __init__(self, *, dimension: int, embedding_model: str) -> None:
        dimension, embedding_model = self._validate_identity(dimension, embedding_model)
        self.dimension = dimension
        self.embedding_model = embedding_model
        self._vectors: _Float32Array = np.empty((0, dimension), dtype=np.float32)
        self._entries: list[_IndexEntry] = []

    @staticmethod
    def _validate_identity(dimension: object, embedding_model: object) -> tuple[int, str]:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Index dimension must be a positive integer")
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            raise ValueError("Index embedding model must be a non-empty string")
        return dimension, embedding_model

    @staticmethod
    def _make_index_entries(entries: Sequence[CorpusEntry]) -> list[_IndexEntry]:
        return [
            {
                "id": entry["id"],
                "text": entry["text"],
                "metadata": {
                    key: value for key, value in entry.items() if key not in {"id", "text"}
                },
            }
            for entry in entries
        ]

    @staticmethod
    def _validate_index_entries(entries: object) -> list[_IndexEntry]:
        if not isinstance(entries, list) or not entries:
            raise ValueError("Index manifest entries must be a non-empty list")
        seen_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"id", "text", "metadata"}:
                raise ValueError("Every indexed entry must contain id, text, and metadata")
            chunk_id = entry["id"]
            text = entry["text"]
            metadata = entry["metadata"]
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Indexed entry id must be a non-empty string")
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate indexed entry id: {chunk_id}")
            seen_ids.add(chunk_id)
            if not isinstance(text, str) or not text:
                raise ValueError(f"Indexed entry text is invalid for {chunk_id}")
            if not isinstance(metadata, dict):
                raise ValueError(f"Indexed entry metadata is invalid for {chunk_id}")
            if not {"grimoire_id", "folio"}.issubset(metadata):
                raise ValueError(
                    f"Indexed entry metadata is missing citation fields for {chunk_id}"
                )
            grimoire_id = metadata.get("grimoire_id")
            folio = metadata.get("folio")
            if grimoire_id is not None and not isinstance(grimoire_id, str):
                raise ValueError(f"Indexed grimoire_id is invalid for {chunk_id}")
            if folio is not None and (isinstance(folio, bool) or not isinstance(folio, (int, str))):
                raise ValueError(f"Indexed folio is invalid for {chunk_id}")
            if grimoire_id is None and folio is None:
                raise ValueError(f"Indexed entry has no citation metadata: {chunk_id}")
        return cast(list[_IndexEntry], entries)

    @staticmethod
    def _entries_sha256(entries: list[_IndexEntry]) -> str:
        canonical = json.dumps(
            entries,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb+") as handle:
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _ensure_directory(cls, directory: Path) -> None:
        missing_directories: list[Path] = []
        current = directory
        while not current.exists():
            if current.is_symlink():
                raise ValueError(f"Index storage directory must not be a symbolic link: {current}")
            missing_directories.append(current)
            current = current.parent
        if current.is_symlink():
            raise ValueError(f"Index storage directory must not be a symbolic link: {current}")

        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Index storage path must be a real directory: {directory}")
        for created_directory in missing_directories:
            cls._fsync_directory(created_directory.parent)

    @staticmethod
    def _validate_generation_id(value: object) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 32
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                "Active index generation id must be 32 lowercase hexadecimal characters"
            )
        return value

    @classmethod
    def corpus_sha256(cls, entries: Sequence[CorpusEntry]) -> str:
        """Fingerprint ordered ids, verbatim text, and all retrieval metadata."""

        indexed_entries = cls._make_index_entries(entries)
        cls._validate_index_entries(indexed_entries)
        return cls._entries_sha256(indexed_entries)

    @property
    def indexed_corpus_sha256(self) -> str:
        return self._entries_sha256(self._entries)

    @staticmethod
    def _unit_norms(matrix: _Float32Array) -> NDArray[np.float64]:
        return np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)

    @classmethod
    def _validate_unit_rows(cls, matrix: _Float32Array) -> None:
        if matrix.dtype.kind != "f" or matrix.dtype.itemsize != np.dtype(np.float32).itemsize:
            raise ValueError("Vector matrix must use the float32 dtype")
        if not np.isfinite(matrix).all():
            raise ValueError("Vector matrix contains a non-finite value")
        norms = cls._unit_norms(matrix)
        if np.any(norms == 0.0):
            raise ValueError("Cosine similarity is undefined for zero vectors")
        if np.any(np.abs(norms - 1.0) > _FLOAT32_UNIT_NORM_TOLERANCE):
            raise ValueError("Vector matrix contains a row outside the float32 unit-norm tolerance")

    @classmethod
    def _normalize(cls, matrix: _Float32Array) -> _Float32Array:
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError(f"Expected a two-dimensional vector matrix, got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("Vector matrix contains a non-finite value")
        float64_matrix = matrix.astype(np.float64)
        norms = np.linalg.norm(float64_matrix, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("Cosine similarity is undefined for zero vectors")
        normalized = (float64_matrix / norms).astype(np.float32)
        cls._validate_unit_rows(normalized)
        return normalized

    def index(self, entries: Sequence[CorpusEntry], vectors: _Float32Array) -> None:
        """Insert one raw corpus entry beside each embedding vector row."""

        if self._entries:
            raise RuntimeError("This VerdigrisE index is immutable after indexing")
        if vectors.shape != (len(entries), self.dimension):
            raise ValueError(
                f"Expected vector shape {(len(entries), self.dimension)}, got {vectors.shape}"
            )

        self._vectors = self._normalize(vectors.astype(np.float32))
        self._entries = self._validate_index_entries(self._make_index_entries(entries))

    def search(self, query_vector: _Float32Array, *, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """Return best-first results using the documented tie-breaking rule."""

        if not self._entries:
            raise RuntimeError("Index is empty; run ingestion before querying")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if query_vector.shape != (self.dimension,):
            raise ValueError(f"Expected query shape {(self.dimension,)}, got {query_vector.shape}")

        normalized_query = self._normalize(query_vector[None, :].astype(np.float32))[0]
        similarities = np.clip(self._vectors @ normalized_query, -1.0, 1.0)
        ranked_indices = sorted(
            range(len(self._entries)),
            key=lambda index: (-float(similarities[index]), self._entries[index]["id"]),
        )[: min(top_k, len(self._entries))]

        results: list[RetrievedChunk] = []
        for index in ranked_indices:
            similarity = float(similarities[index])
            entry = self._entries[index]
            results.append(
                RetrievedChunk(
                    id=entry["id"],
                    text=entry["text"],
                    metadata=dict(entry["metadata"]),
                    distance=1.0 - similarity,
                    similarity=similarity,
                )
            )
        return results

    def save(self, directory: Path) -> None:
        """Publish one immutable generation, then atomically make it active."""

        if not self._entries:
            raise RuntimeError("Cannot save an empty index")
        self._validate_unit_rows(self._vectors)
        self._ensure_directory(directory)
        generations_directory = directory / INDEX_GENERATIONS_DIRECTORY
        self._ensure_directory(generations_directory)
        generation_id = uuid.uuid4().hex
        staging_directory = generations_directory / f".{generation_id}.tmp"
        generation_directory = generations_directory / generation_id
        staging_directory.mkdir()
        vector_path = staging_directory / INDEX_VECTOR_FILENAME
        manifest_path = staging_directory / INDEX_MANIFEST_FILENAME
        active_path = directory / INDEX_ACTIVE_FILENAME
        temporary_active_path = directory / f".{INDEX_ACTIVE_FILENAME}.{generation_id}.tmp"
        try:
            np.save(vector_path, self._vectors, allow_pickle=False)
            manifest = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "embedding_model": self.embedding_model,
                "dimension": self.dimension,
                "distance_definition": DISTANCE_DEFINITION,
                "tie_break_rule": TIE_BREAK_RULE,
                "corpus_sha256": self.indexed_corpus_sha256,
                "vectors_sha256": self._file_sha256(vector_path),
                "entries": self._entries,
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._fsync_file(vector_path)
            self._fsync_file(manifest_path)
            self._fsync_directory(staging_directory)
            os.replace(staging_directory, generation_directory)
            self._fsync_directory(generations_directory)

            active_pointer = {
                "generation_id": generation_id,
                "schema_version": INDEX_POINTER_SCHEMA_VERSION,
            }
            temporary_active_path.write_text(
                json.dumps(active_pointer, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._fsync_file(temporary_active_path)
            os.replace(temporary_active_path, active_path)
            self._fsync_directory(directory)
        finally:
            with suppress(OSError):
                temporary_active_path.unlink(missing_ok=True)
            with suppress(OSError):
                vector_path.unlink(missing_ok=True)
            with suppress(OSError):
                manifest_path.unlink(missing_ok=True)
            with suppress(OSError):
                staging_directory.rmdir()

    @classmethod
    def _resolve_index_files(cls, directory: Path) -> tuple[Path, Path]:
        if directory.is_symlink():
            raise ValueError(f"Index storage directory must not be a symbolic link: {directory}")
        active_path = directory / INDEX_ACTIVE_FILENAME
        if active_path.is_symlink():
            raise ValueError("Active index pointer must not be a symbolic link")
        if active_path.exists():
            if not active_path.is_file():
                raise ValueError("Active index pointer must be a regular file")
            raw_pointer = json.loads(active_path.read_text(encoding="utf-8"))
            if not isinstance(raw_pointer, dict):
                raise ValueError("Active index pointer must be a JSON object")
            pointer = cast(dict[str, object], raw_pointer)
            pointer_schema_version = pointer.get("schema_version")
            if (
                isinstance(pointer_schema_version, bool)
                or not isinstance(pointer_schema_version, int)
                or pointer_schema_version != INDEX_POINTER_SCHEMA_VERSION
            ):
                raise ValueError("Unsupported active index pointer schema version")
            generation_id = cls._validate_generation_id(pointer.get("generation_id"))
            generations_directory = directory / INDEX_GENERATIONS_DIRECTORY
            generation_directory = generations_directory / generation_id
            manifest_path = generation_directory / INDEX_MANIFEST_FILENAME
            vector_path = generation_directory / INDEX_VECTOR_FILENAME
            if generations_directory.is_symlink() or generation_directory.is_symlink():
                raise ValueError("Active index generation path must not be a symbolic link")
            if manifest_path.is_symlink() or vector_path.is_symlink():
                raise ValueError("Active index generation files must not be symbolic links")
            if (
                not generation_directory.is_dir()
                or not manifest_path.is_file()
                or not vector_path.is_file()
            ):
                raise FileNotFoundError(
                    f"Active index generation {generation_id} is incomplete at {directory}"
                )
            return manifest_path, vector_path

        legacy_manifest_path = directory / INDEX_MANIFEST_FILENAME
        legacy_vector_path = directory / INDEX_VECTOR_FILENAME
        if legacy_manifest_path.is_symlink() or legacy_vector_path.is_symlink():
            raise ValueError("Legacy index files must not be symbolic links")
        if legacy_manifest_path.is_file() and legacy_vector_path.is_file():
            return legacy_manifest_path, legacy_vector_path
        raise FileNotFoundError(f"No complete index at {directory}. Run: python pipeline.py ingest")

    @classmethod
    def load(cls, directory: Path) -> NumpyVectorIndex:
        """Resolve one active immutable generation and validate row alignment."""

        manifest_path, vector_path = cls._resolve_index_files(directory)

        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw_manifest, dict):
            raise ValueError("Index manifest must be a JSON object")
        manifest = cast(dict[str, object], raw_manifest)
        schema_version = manifest.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != INDEX_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported index schema version")
        if manifest.get("distance_definition") != DISTANCE_DEFINITION:
            raise ValueError("Persisted index uses a different distance definition")
        if manifest.get("tie_break_rule") != TIE_BREAK_RULE:
            raise ValueError("Persisted index uses a different tie-breaking rule")
        dimension, embedding_model = cls._validate_identity(
            manifest.get("dimension"), manifest.get("embedding_model")
        )
        corpus_digest = manifest.get("corpus_sha256")
        vector_digest = manifest.get("vectors_sha256")
        entries = cls._validate_index_entries(manifest.get("entries"))
        if corpus_digest != cls._entries_sha256(entries):
            raise ValueError("Index manifest corpus fingerprint does not match its entries")
        if not isinstance(vector_digest, str) or vector_digest != cls._file_sha256(vector_path):
            raise ValueError("Index vector fingerprint does not match its manifest")

        vectors = np.load(vector_path, allow_pickle=False)
        if vectors.shape != (len(entries), dimension):
            raise ValueError("Persisted vectors and manifest entries are not rank-aligned")
        persisted_vectors = cast(_Float32Array, vectors)
        cls._validate_unit_rows(persisted_vectors)
        index = cls(dimension=dimension, embedding_model=embedding_model)
        index._vectors = persisted_vectors
        index._entries = entries
        return index


SYSTEM_INSTRUCTIONS = f"""Answer only from the supplied CONTEXT blocks.
Do not use outside knowledge or infer an unstated value.
For every supported statement, cite the supporting stable chunk id in square
brackets, for example [verdigris-dose-verdant].
Context labels provide `grimoire_id` and `folio`. Repeat the supporting
`grimoire_id` verbatim in every supported answer.
If the answer is unsupported, return exactly {ABSTENTION_PHRASE} and nothing else."""


def build_context_payload(chunks: Sequence[RetrievedChunk]) -> str:
    """Build the exact context string without changing any retrieved text."""

    blocks: list[str] = []
    for chunk in chunks:
        grimoire_id = chunk.metadata.get("grimoire_id")
        folio = chunk.metadata.get("folio")
        label = f"[CONTEXT id={chunk.id} grimoire_id={grimoire_id!r} folio={folio!r}]"
        blocks.append(f"{label}\n{chunk.text}\n[END CONTEXT id={chunk.id}]")
    return "\n\n".join(blocks)


def build_generation_messages(
    question: str, chunks: Sequence[RetrievedChunk]
) -> tuple[str, list[PromptMessage]]:
    """Return the context payload and exact testable generation request."""

    context_payload = build_context_payload(chunks)
    messages = [
        PromptMessage(role="system", content=SYSTEM_INSTRUCTIONS),
        PromptMessage(
            role="user",
            content=f"QUESTION:\n{question}\n\nCONTEXT:\n{context_payload}",
        ),
    ]
    return context_payload, messages


class RagPipeline:
    """Application service with injected embedding, index, and generation boundaries."""

    def __init__(
        self,
        *,
        index: NumpyVectorIndex,
        embedder: EmbeddingProvider,
        generator: AnswerGenerator,
        debug: bool = False,
    ) -> None:
        if index.embedding_model != embedder.model:
            raise ValueError(
                "Corpus and query embedding models differ: "
                f"{index.embedding_model!r} != {embedder.model!r}"
            )
        self._index = index
        self._embedder = embedder
        self._generator = generator
        self._debug = debug

    def ask(self, question: str) -> RagRecord:
        if not question.strip():
            raise ValueError("question must not be empty")

        # Stage 1 – embed the query separately in the corpus embedding space.
        query_matrix = self._embedder.embed(
            [question],
            input_ids=["<query>"],
            stage="query",
            debug=self._debug,
        )
        if query_matrix.shape != (1, self._index.dimension):
            raise ValueError(
                f"Query embedding shape {query_matrix.shape} does not match index dimension "
                f"{self._index.dimension}"
            )
        # Stage 2 – retrieve the deterministic cosine top-2 with metadata intact.
        chunks = self._index.search(query_matrix[0], top_k=TOP_K)

        # Stage 3 – assemble the exact, verbatim, citation-labelled request.
        context_payload, messages = build_generation_messages(question, chunks)

        # Stage 4 – preserve the answer and every preceding stage in RagRecord.
        answer = self._generator.generate(question, messages)
        return RagRecord(
            question=question,
            retrieved_ids=tuple(chunk.id for chunk in chunks),
            retrieved_chunks=tuple(chunks),
            distances=tuple(chunk.distance for chunk in chunks),
            context_payload=context_payload,
            generation_messages=tuple(messages),
            answer=answer,
        )


def ingest_corpus(
    *,
    embedder: EmbeddingProvider,
    output_directory: Path = INDEX_DIRECTORY,
    debug: bool = False,
) -> NumpyVectorIndex:
    """Run stages 1 and 2 for the corpus, then persist the validated index."""

    validate_corpus()
    texts = [entry["text"] for entry in CORPUS]
    ids = [entry["id"] for entry in CORPUS]
    vectors = embedder.embed(
        texts,
        input_ids=ids,
        stage="corpus",
        debug=debug,
    )
    if (
        not isinstance(vectors, np.ndarray)
        or vectors.ndim != 2
        or vectors.shape[0] != len(CORPUS)
        or vectors.shape[1] == 0
        or not np.isfinite(vectors).all()
    ):
        shape = getattr(vectors, "shape", None)
        raise ValueError(f"Corpus embedder returned an invalid matrix: {shape}")
    index = NumpyVectorIndex(dimension=vectors.shape[1], embedding_model=embedder.model)
    index.index(CORPUS, vectors)
    index.save(output_directory)
    return index


def _real_client() -> OpenAI:
    _require_api_key()
    return OpenAI(
        max_retries=OPENAI_MAX_RETRIES,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )


def _real_pipeline(*, index_directory: Path = INDEX_DIRECTORY, debug: bool = False) -> RagPipeline:
    index = NumpyVectorIndex.load(index_directory)
    if index.embedding_model != EMBEDDING_MODEL:
        raise ValueError(
            "Persisted index embedding model differs from configured model: "
            f"{index.embedding_model!r} != {EMBEDDING_MODEL!r}; run ingest again"
        )
    expected_corpus_digest = NumpyVectorIndex.corpus_sha256(CORPUS)
    if index.indexed_corpus_sha256 != expected_corpus_digest:
        raise ValueError("Persisted index is stale relative to corpus.py; run ingest again")
    client = _real_client()
    embedder = OpenAIEmbeddingProvider(client)
    return RagPipeline(
        index=index,
        embedder=embedder,
        generator=OpenAIAnswerGenerator(client),
        debug=debug,
    )


def ask(question: str) -> RagRecord:
    """Public synchronous entry point using the persisted default index."""

    return _real_pipeline().ask(question)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verdigrise",
        description="Inspect VerdigrisE grimoire retrieval mechanics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Embed and persist the corpus")
    ingest_parser.add_argument("--debug", action="store_true", help="Print safe shape metadata")
    ingest_parser.add_argument("--index-dir", type=Path, default=INDEX_DIRECTORY)

    ask_parser = subparsers.add_parser("ask", help="Query the persisted index")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--debug", action="store_true", help="Print safe shape metadata")
    ask_parser.add_argument("--index-dir", type=Path, default=INDEX_DIRECTORY)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "ingest":
        client = _real_client()
        index = ingest_corpus(
            embedder=OpenAIEmbeddingProvider(client),
            output_directory=args.index_dir,
            debug=args.debug,
        )
        print(
            json.dumps(
                {
                    "index_directory": str(args.index_dir),
                    "corpus_entries": len(CORPUS),
                    "dimensions": index.dimension,
                    "embedding_model": index.embedding_model,
                },
                indent=2,
            )
        )
        return 0

    pipeline = _real_pipeline(index_directory=args.index_dir, debug=args.debug)
    print(pipeline.ask(args.question).model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
