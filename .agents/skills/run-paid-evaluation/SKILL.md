---
name: run-paid-evaluation
description: "Run VerdigrisE's opt-in live OpenAI and RagaliQ Claude paths under a per-invocation approval gate. Use when the user requests live provider acceptance, semantic judge evaluation, paid ingestion or querying, live use of ask(), or a retry of any paid run."
---

# Run a paid evaluation

1. Run the free deterministic suite first:

   ```bash
   .venv/bin/python -m pytest -c pytest.ini eval/ -q
   ```

   Stop on deterministic failure; do not spend provider calls to diagnose a broken exact layer.
2. Select exactly one paid command. Before requesting approval, report the exact command, selected cases, provider key names, models, nominal request fan-out, provider-retry uncertainty, and cost uncertainty without displaying credential values:
   - OpenAI all-golden acceptance uses `text-embedding-3-small` for one corpus batch and six query embeddings, then `gpt-5.6-luna` for six generations: nominally 13 OpenAI requests.
   - Cross-family semantic evaluation makes the same nominal 13 OpenAI requests, then evaluates five answerable cases with RagaliQ's installed `DEFAULT_JUDGE_MODEL`. Nominal Claude fan-out is 10 base requests plus one verification request per extracted claim, so the total is model-output-dependent.
   - `pipeline.py ingest` makes one OpenAI corpus-embedding request. `pipeline.py ask` and live `ask()` each make one OpenAI query-embedding request and one generation request after an index exists.
   Inspect the installed provider SDK and RagaliQ transport retry settings before quoting a maximum. Provider-managed retries can multiply these nominal counts within one invocation.
3. Obtain explicit approval for the exact command immediately before executing it. A general investigation request, an available API key, or approval for another tier is not approval. Treat every manually initiated retry as a new paid invocation.
4. Check that the required environment variables are present without printing their values. Do not load, display, persist, or log secrets.
5. Run the approved command once.

   Paid all-golden OpenAI acceptance; requires `OPENAI_API_KEY`:

   ```bash
   .venv/bin/python -m pytest -c pytest.ini -o addopts='' -m "openai and not rag_test" eval/ -q
   ```

   Paid cross-family semantic evaluation; requires both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`:

   ```bash
   .venv/bin/python -m pytest -c pytest.ini -o addopts='' -m "openai and rag_test" --ragaliq-cost-limit 5.00 eval/ -q
   ```

6. Do not manually rerun a failed command, broaden marker selection, or run both tiers. Provider-managed retries may already have occurred inside the approved invocation. Report passes, failures, skips, and any recorded estimated cost. State that `--ragaliq-cost-limit` is an approximate post-test guard, not a strict pre-spend cap.
7. Apply deterministic golden checks before interpreting semantic scores. Do not weaken fixtures or move exact ownership to RagaliQ because a live provider fails acceptance.
8. Treat these live CLI paths as separately paid commands requiring their own exact approval:

   ```bash
   .venv/bin/python pipeline.py ingest --debug
   .venv/bin/python pipeline.py ask "<question>" --debug
   ```
