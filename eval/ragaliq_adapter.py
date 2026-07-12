"""VerdigrisE RagaliQ adapters discovered from the local 0.2.0 source tree.

RagaliQ owns semantic residue only. The deterministic suite owns exact ids,
numbers, citations, prompt bytes, ranking, distances, and abstention.
"""

from __future__ import annotations

import json

from ragaliq import RAGTestCase, RagaliQ
from ragaliq.judges import (
    DEFAULT_JUDGE_MODEL,
    BaseJudge,
    JudgeConfig,
    JudgeTransport,
    TransportResponse,
)

from models import RagRecord


class CannedJudgeTransport:
    """Deterministic transport that executes RagaliQ parsing and evaluators locally."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def send(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> TransportResponse:
        del temperature, max_tokens
        self.calls.append(
            {"system_prompt": system_prompt, "user_prompt": user_prompt, "model": model}
        )
        combined = (system_prompt + user_prompt).lower()
        if "extract" in combined and "claim" in combined:
            text = json.dumps({"claims": ["The response states the scoped corpus fact."]})
        elif "verify" in combined and "claim" in combined:
            text = json.dumps(
                {"verdict": "SUPPORTED", "evidence": "The supplied context supports it."}
            )
        elif "relevance" in combined or "relevant" in combined:
            text = json.dumps({"score": 0.9, "reasoning": "The response answers the query."})
        else:
            raise AssertionError("Canned transport received an unrecognized RagaliQ prompt")
        return TransportResponse(
            text=text,
            input_tokens=20,
            output_tokens=10,
            model=model,
        )


def build_ragaliq_runner(transport: JudgeTransport) -> RagaliQ:
    """Wire a canned transport through the real two-evaluator RagaliQ path."""

    judge = BaseJudge(
        transport=transport,
        config=JudgeConfig(
            model=DEFAULT_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=1024,
        ),
    )
    return RagaliQ(
        judge=judge,
        evaluators=["faithfulness", "relevance"],
        default_threshold=0.7,
    )


def to_ragaliq_case(record: RagRecord, *, case_id: str) -> RAGTestCase:
    """Pass exactly what generation saw to RagaliQ, without reloading by id."""

    return RAGTestCase(
        id=case_id,
        name=f"VerdigrisE semantic residue for {case_id}",
        query=record.question,
        context=[record.context_payload],
        response=record.answer,
        tags=["verdigrise", "semantic-residue"],
    )
