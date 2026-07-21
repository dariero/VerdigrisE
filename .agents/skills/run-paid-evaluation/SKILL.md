---
name: run-paid-evaluation
description: "Run VerdigrisE's opt-in live OpenAI and RagaliQ Claude paths under a per-invocation approval gate. Use when the user requests live provider acceptance, semantic judge evaluation, paid ingestion or querying, live use of ask(), or a retry of any paid run."
---

# Run a paid evaluation

1. Run the free deterministic suite and branch-coverage gate first:

   ```bash
   .venv/bin/python -m pytest eval/ -q
   .venv/bin/python -m pytest --cov --cov-report=term-missing eval/ -q
   ```

   Stop on deterministic failure; do not spend provider calls to diagnose a broken exact layer.
2. Select exactly one paid command. Before requesting approval, report the exact command, selected cases, provider key names, models, the maximum OpenAI SDK attempt fan-out under `max_retries=0`, the 120-second per-network-operation OpenAI timeout, separate RagaliQ/Anthropic retry uncertainty, and timeout/cost uncertainty without displaying credential values:
   - Full OpenAI all-golden acceptance uses `text-embedding-3-small` for one corpus batch and seven query embeddings, then `gpt-5.6-luna` for seven generations: at most 15 OpenAI SDK attempts.
   - Full cross-family semantic evaluation uses one corpus batch, six query embeddings, and six generations: at most 13 OpenAI SDK attempts. It then evaluates six answerable cases with RagaliQ's installed `DEFAULT_JUDGE_MODEL`.
   - Selecting `n` parametrized nodes from one paid tier makes at most `1 + 2n` OpenAI SDK attempts: one shared corpus embedding and `n` query-and-generation pairs. A single selected node therefore makes at most 3 OpenAI SDK attempts. For `n` selected semantic nodes, nominal Claude fan-out is `2n` base requests plus one verification request per extracted claim, so the total is model-output-dependent.
   - `pipeline.py ingest` makes at most one OpenAI SDK attempt. `pipeline.py ask` and live `ask()` each make at most one OpenAI query-embedding attempt and one generation attempt after an index exists.
   Confirm that every selected OpenAI path uses the policy-bound client. `max_retries=0` prevents OpenAI SDK retry multiplication. The 120-second timeout applies to each network operation, not the whole invocation, and a timed-out request may still be processed or billed. Inspect RagaliQ and Anthropic transport settings before quoting a Claude maximum because that tier retains its own retries.
3. Obtain explicit approval for the exact command immediately before executing it. A general investigation request, an available API key, or approval for another tier is not approval. Treat every manually initiated retry as a new paid invocation.
4. Check that the required environment variables are present without printing their values. Do not load, display, persist, or log secrets.
5. Run the approved command once.

   Paid all-golden OpenAI acceptance; requires `OPENAI_API_KEY`:

   ```bash
   .venv/bin/python -m pytest -o addopts='' -m "openai and not rag_test" eval/ -q
   ```

   Paid cross-family semantic evaluation; requires both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`:

   ```bash
   .venv/bin/python -m pytest -o addopts='' -m "openai and rag_test" --ragaliq-cost-limit 5.00 eval/ -q
   ```

6. Do not manually rerun a failed command, broaden marker selection, or run both tiers. No automatic OpenAI SDK retry occurs, but RagaliQ or Anthropic may retry inside the semantic invocation. Report passes, failures, skips, timeouts, and any recorded estimated cost. State that a timed-out request may still be processed or billed and that `--ragaliq-cost-limit` is an approximate post-test guard, not a strict pre-spend cap.
7. Apply deterministic golden checks before interpreting semantic scores. Do not weaken fixtures or move exact ownership to RagaliQ because a live provider fails acceptance.
8. Treat these live CLI paths as separately paid commands requiring their own exact approval:

   ```bash
   .venv/bin/python pipeline.py ingest --debug
   .venv/bin/python pipeline.py ask "<question>" --debug
   ```
