"""Stable data contracts shared by VerdigrisE and its evaluation suite."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievedChunk(BaseModel):
    """One ranked retrieval result with verbatim text and citation metadata."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    metadata: dict[str, object]
    distance: float
    similarity: float


class PromptMessage(BaseModel):
    """One exact message passed to the generation API."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user"]
    content: str


class RagRecord(BaseModel):
    """Evaluation capture for one complete VerdigrisE retrieval and generation run."""

    model_config = ConfigDict(frozen=True)

    question: str
    retrieved_ids: list[str] = Field(description="Stable ids in retrieval rank order")
    retrieved_chunks: list[RetrievedChunk]
    distances: list[float] = Field(description="One minus cosine similarity in rank order")
    context_payload: str
    generation_messages: list[PromptMessage]
    answer: str

    @model_validator(mode="after")
    def rank_aligned_fields_must_match(self) -> RagRecord:
        chunk_ids = [chunk.id for chunk in self.retrieved_chunks]
        chunk_distances = [chunk.distance for chunk in self.retrieved_chunks]
        if self.retrieved_ids != chunk_ids:
            raise ValueError("retrieved_ids must match retrieved_chunks rank order")
        if self.distances != chunk_distances:
            raise ValueError("distances must match retrieved_chunks rank order")
        return self
