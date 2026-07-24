"""Stable data contracts shared by VerdigrisE and its evaluation suite."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from math import isclose, isfinite
from types import MappingProxyType
from typing import Any, Literal, Self, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class _ImmutableMapping[Key, Value](Mapping[Key, Value]):
    """Read-only mapping that remains safe to share during a deep model copy."""

    __slots__ = ("__values",)

    def __init__(self, values: Mapping[Key, Value]) -> None:
        self.__values = MappingProxyType(dict(values))

    def __getitem__(self, key: Key) -> Value:
        return self.__values[key]

    def __iter__(self) -> Iterator[Key]:
        return iter(self.__values)

    def __len__(self) -> int:
        return len(self.__values)

    def __deepcopy__(self, memo: dict[int, object]) -> _ImmutableMapping[Key, Value]:
        del memo
        return self

    @override
    def __reduce__(self) -> tuple[object, tuple[dict[Key, Value]]]:
        return type(self), (dict(self.__values),)


class _CaptureModel(BaseModel):
    """Frozen, closed base that revalidates any explicitly updated copy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @override
    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        values = self.model_dump(round_trip=True)
        update_values = dict(update)
        if deep:
            values = deepcopy(values)
            update_values = deepcopy(update_values)
        values.update(update_values)
        return type(self).model_validate(values)


def _freeze_metadata_value(value: object) -> object:
    """Recursively detach and freeze the JSON-like containers metadata can own."""

    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("metadata mapping keys must be strings")
        return _ImmutableMapping({key: _freeze_metadata_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if type(value) is float and not isfinite(value):
        raise ValueError("metadata float leaves must be finite")
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise ValueError(
        "metadata values must use string-keyed mappings, ordered sequences, and JSON scalar leaves"
    )


def _serialize_metadata_value(value: object) -> object:
    """Restore ordinary JSON-shaped containers at the serialization boundary."""

    if isinstance(value, Mapping):
        return {key: _serialize_metadata_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize_metadata_value(item) for item in value]
    return value


class RetrievedChunk(_CaptureModel):
    """One ranked retrieval result with verbatim text and citation metadata."""

    id: str
    text: str
    metadata: Mapping[str, object]
    distance: float = Field(ge=0.0, le=2.0, allow_inf_nan=False)
    similarity: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)

    @field_validator("metadata", mode="after")
    @classmethod
    def metadata_must_be_immutable(cls, metadata: Mapping[str, object]) -> Mapping[str, object]:
        return _ImmutableMapping(
            {key: _freeze_metadata_value(value) for key, value in metadata.items()}
        )

    @model_validator(mode="after")
    def retrieval_metrics_must_match(self) -> RetrievedChunk:
        if not isclose(
            self.distance,
            1.0 - self.similarity,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("distance must equal one minus similarity within 1e-12")
        return self

    @field_serializer("metadata")
    def serialize_metadata(self, metadata: Mapping[str, object]) -> dict[str, object]:
        return {key: _serialize_metadata_value(value) for key, value in metadata.items()}


class PromptMessage(_CaptureModel):
    """One exact message passed to the generation API."""

    role: Literal["system", "user"]
    content: str


class RagRecord(_CaptureModel):
    """Evaluation capture for one complete VerdigrisE retrieval and generation run."""

    question: str
    retrieved_ids: tuple[str, ...] = Field(description="Stable ids in retrieval rank order")
    retrieved_chunks: tuple[RetrievedChunk, ...]
    distances: tuple[float, ...] = Field(description="One minus cosine similarity in rank order")
    context_payload: str
    generation_messages: tuple[PromptMessage, ...]
    answer: str

    @field_serializer("retrieved_ids")
    def serialize_retrieved_ids(self, values: tuple[str, ...]) -> list[str]:
        return list(values)

    @field_serializer("retrieved_chunks")
    def serialize_retrieved_chunks(
        self, values: tuple[RetrievedChunk, ...]
    ) -> list[RetrievedChunk]:
        return list(values)

    @field_serializer("distances")
    def serialize_distances(self, values: tuple[float, ...]) -> list[float]:
        return list(values)

    @field_serializer("generation_messages")
    def serialize_generation_messages(
        self, values: tuple[PromptMessage, ...]
    ) -> list[PromptMessage]:
        return list(values)

    @model_validator(mode="after")
    def rank_aligned_fields_must_match(self) -> RagRecord:
        chunk_ids = tuple(chunk.id for chunk in self.retrieved_chunks)
        chunk_distances = tuple(chunk.distance for chunk in self.retrieved_chunks)
        if self.retrieved_ids != chunk_ids:
            raise ValueError("retrieved_ids must match retrieved_chunks rank order")
        if self.distances != chunk_distances:
            raise ValueError("distances must match retrieved_chunks rank order")
        return self
