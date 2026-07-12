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
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from openai import OpenAI

from config import (
    ABSTENTION_PHRASE,
    DISTANCE_DEFINITION,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    INDEX_DIRECTORY,
    INDEX_MANIFEST_FILENAME,
    INDEX_SCHEMA_VERSION,
    INDEX_VECTOR_FILENAME,
    TIE_BREAK_RULE,
    TOP_K,
)
from corpus import CORPUS, CorpusEntry, validate_corpus
from models import PromptMessage, RagRecord, RetrievedChunk


class EmbeddingProvider(Protocol):
    model: str

    def embed(
        self,
        inputs: list[str],
        *,
        input_ids: list[str],
        stage: str,
        debug: bool = False,
    ) -> np.ndarray:
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
    *, model: str, stage: str, input_ids: list[str], matrix: np.ndarray
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
    ) -> np.ndarray:
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
    Similarity is their dot product. Distance is exactly one minus similarity.
    Equal similarities are ordered by ascending stable chunk id.
    """

    def __init__(self, *, dimension: int, embedding_model: str) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.embedding_model = embedding_model
        self._vectors = np.empty((0, dimension), dtype=np.float32)
        self._entries: list[dict[str, object]] = []

    @staticmethod
    def _make_index_entries(entries: Sequence[CorpusEntry]) -> list[dict[str, object]]:
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
    def _validate_index_entries(entries: object) -> list[dict[str, object]]:
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
            if folio is not None and (
                isinstance(folio, bool) or not isinstance(folio, (int, str))
            ):
                raise ValueError(f"Indexed folio is invalid for {chunk_id}")
            if grimoire_id is None and folio is None:
                raise ValueError(f"Indexed entry has no citation metadata: {chunk_id}")
        return entries

    @staticmethod
    def _entries_sha256(entries: list[dict[str, object]]) -> str:
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
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError(f"Expected a two-dimensional vector matrix, got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("Vector matrix contains a non-finite value")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("Cosine similarity is undefined for zero vectors")
        return (matrix / norms).astype(np.float32)

    def index(self, entries: Sequence[CorpusEntry], vectors: np.ndarray) -> None:
        """Insert one raw corpus entry beside each embedding vector row."""

        if self._entries:
            raise RuntimeError("This VerdigrisE index is immutable after indexing")
        if vectors.shape != (len(entries), self.dimension):
            raise ValueError(
                f"Expected vector shape {(len(entries), self.dimension)}, got {vectors.shape}"
            )

        self._vectors = self._normalize(vectors.astype(np.float32))
        self._entries = self._validate_index_entries(self._make_index_entries(entries))

    def search(self, query_vector: np.ndarray, *, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """Return best-first results using the documented tie-breaking rule."""

        if not self._entries:
            raise RuntimeError("Index is empty; run ingestion before querying")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if query_vector.shape != (self.dimension,):
            raise ValueError(
                f"Expected query shape {(self.dimension,)}, got {query_vector.shape}"
            )

        normalized_query = self._normalize(query_vector[None, :].astype(np.float32))[0]
        similarities = self._vectors @ normalized_query
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
                    metadata=dict(entry["metadata"]),  # type: ignore[arg-type]
                    distance=1.0 - similarity,
                    similarity=similarity,
                )
            )
        return results

    def save(self, directory: Path) -> None:
        """Persist normalized rows and an inspectable metadata manifest."""

        if not self._entries:
            raise RuntimeError("Cannot save an empty index")
        directory.mkdir(parents=True, exist_ok=True)
        vector_path = directory / INDEX_VECTOR_FILENAME
        manifest_path = directory / INDEX_MANIFEST_FILENAME
        temporary_vector_path = directory / f".{INDEX_VECTOR_FILENAME}.tmp.npy"
        temporary_manifest_path = directory / f".{INDEX_MANIFEST_FILENAME}.tmp"
        try:
            np.save(temporary_vector_path, self._vectors, allow_pickle=False)
            manifest = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "embedding_model": self.embedding_model,
                "dimension": self.dimension,
                "distance_definition": DISTANCE_DEFINITION,
                "tie_break_rule": TIE_BREAK_RULE,
                "corpus_sha256": self.indexed_corpus_sha256,
                "vectors_sha256": self._file_sha256(temporary_vector_path),
                "entries": self._entries,
            }
            temporary_manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_vector_path, vector_path)
            os.replace(temporary_manifest_path, manifest_path)
        finally:
            temporary_vector_path.unlink(missing_ok=True)
            temporary_manifest_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: Path) -> "NumpyVectorIndex":
        """Load a previously ingested index and validate row alignment."""

        manifest_path = directory / INDEX_MANIFEST_FILENAME
        vector_path = directory / INDEX_VECTOR_FILENAME
        if not manifest_path.exists() or not vector_path.exists():
            raise FileNotFoundError(
                f"No complete index at {directory}. Run: python pipeline.py ingest"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("Unsupported index schema version")
        if manifest.get("distance_definition") != DISTANCE_DEFINITION:
            raise ValueError("Persisted index uses a different distance definition")
        if manifest.get("tie_break_rule") != TIE_BREAK_RULE:
            raise ValueError("Persisted index uses a different tie-breaking rule")
        dimension = manifest.get("dimension")
        embedding_model = manifest.get("embedding_model")
        corpus_digest = manifest.get("corpus_sha256")
        vector_digest = manifest.get("vectors_sha256")
        entries = cls._validate_index_entries(manifest.get("entries"))
        if not isinstance(dimension, int) or not isinstance(embedding_model, str):
            raise ValueError("Index manifest has invalid model or dimension fields")
        if corpus_digest != cls._entries_sha256(entries):
            raise ValueError("Index manifest corpus fingerprint does not match its entries")
        if not isinstance(vector_digest, str) or vector_digest != cls._file_sha256(vector_path):
            raise ValueError("Index vector fingerprint does not match its manifest")

        vectors = np.load(vector_path, allow_pickle=False)
        if vectors.shape != (len(entries), dimension):
            raise ValueError("Persisted vectors and manifest entries are not rank-aligned")
        index = cls(dimension=dimension, embedding_model=embedding_model)
        index._vectors = index._normalize(vectors.astype(np.float32))
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
        label = (
            f"[CONTEXT id={chunk.id} grimoire_id={grimoire_id!r} folio={folio!r}]"
        )
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
            retrieved_ids=[chunk.id for chunk in chunks],
            retrieved_chunks=chunks,
            distances=[chunk.distance for chunk in chunks],
            context_payload=context_payload,
            generation_messages=messages,
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
    return OpenAI()


def _real_pipeline(*, index_directory: Path = INDEX_DIRECTORY, debug: bool = False) -> RagPipeline:
    client = _real_client()
    embedder = OpenAIEmbeddingProvider(client)
    index = NumpyVectorIndex.load(index_directory)
    expected_corpus_digest = NumpyVectorIndex.corpus_sha256(CORPUS)
    if index.indexed_corpus_sha256 != expected_corpus_digest:
        raise ValueError("Persisted index is stale relative to corpus.py; run ingest again")
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
