---
name: maintain-golden-contract
description: "Safely evolve VerdigrisE's executable corpus, adversarial golden cases, fixed retrieval vectors, prompts, and exact evaluation contracts. Use when changing corpus.py, retrieval behavior, prompts, citations, abstention, fixtures, golden expectations, or the deterministic-versus-RagaliQ ownership boundary."
---

# Maintain the golden contract

1. Read the evaluation ownership and fixture sections of `README.md`, then inspect `corpus.py`, `config.py`, `models.py`, `pipeline.py`, and the relevant tests in `eval/test_verdigrise.py`.
2. Classify each requested assertion before editing:
   - Keep ids, rank, collisions, literals, units, qualifiers, citations, prompt bytes, distance math, provider mappings, and exact abstention in deterministic pytest.
   - Keep only prose faithfulness and answer relevance in RagaliQ, after exact checks pass.
3. Treat corpus order, stable ids, verbatim text, citation metadata, conditions, and genuine absence as contract data. Do not casually rename, reorder, normalize, or paraphrase them.
4. Update only the coupled data affected by the requested contract:
   - For a corpus-entry change, update `CORPUS` with its `_CORPUS_VECTORS` row and every dependent golden expectation.
   - For a case-only change, update `GOLDEN_CASES` with its `_QUESTION_VECTORS` row; do not rewrite unchanged corpus data or vectors.
   - Update `GoldenCase` validation or deterministic assertions only when the schema or ownership contract changes.
   - Keep expected ranked ids, collision siblings, required and forbidden literals, qualifiers, citations, answers, and abstention state internally consistent.
5. Ensure every declared collision sibling reaches top-2 context and that the intended source ranks correctly. Preserve `distance = 1 - cosine_similarity` and stable-id tie-breaking unless deliberately changing the retrieval contract.
6. For an abstention case, prove the requested fact remains genuinely absent from the entire corpus and preserve exact `INSUFFICIENT_CONTEXT` equality.
7. Preserve verbatim retrieved text, citation-labelled context, request-message capture, and rank alignment in `RagRecord`.
8. Never weaken a deterministic expectation to accommodate a paid model result. A paid failure is an acceptance result for the configured providers, not evidence that the fixture should move to probabilistic ownership.
9. Update `README.md` when the fixture table, public behavior, model configuration, ownership boundary, or documented command changes.
10. Run only the free deterministic suite:

    ```bash
    .venv/bin/python -m pytest -c pytest.ini eval/ -q
    ```

    Use `run-paid-evaluation` only after separate approval for a specific paid invocation.
