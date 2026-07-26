"""Measure where VerdigrisE retrieval time goes as the corpus grows.

Run it:

    .venv/bin/python benchmark_retrieval.py

Makes zero provider calls and needs no API key. Every vector is synthetic and
drawn from a pinned seed, so two runs on one machine agree and a reader can
reproduce the tables in the README's Production Boundary section.

What it measures, and why those two operations.

`NumpyVectorIndex.search` (`pipeline.py:405-431`) does its work in two steps
that scale differently:

- `pipeline.py:416` computes `self._vectors @ normalized_query`, one BLAS-backed
  matrix-vector product over a contiguous float32 array of shape (n, d). The
  arithmetic is proportional to n * d and runs entirely in compiled code.
- `pipeline.py:417-420` calls `sorted(range(n), key=...)`. The key is
  interpreted Python, invoked once per row, and each invocation extracts a NumPy
  scalar, indexes a list, reads a dict, negates, and allocates a tuple. Then the
  sort performs O(n log n) tuple comparisons. It orders all n rows to return
  TOP_K, which is 2 (`config.py:11`).

Three regimes, not one crossover.

A NumPy call has a fixed cost before any arithmetic happens: argument parsing,
dtype and shape resolution, and BLAS dispatch. That cost does not depend on d.
So at small n the matrix product measures dispatch, not arithmetic, and its time
is flat across widths. Only once n * d is large enough does the arithmetic
dominate and the time start scaling with d. Reporting a single "crossover"
without saying which regime it sits in invites the reader to conclude the design
fails at that corpus size, which is false.

The script therefore reports the dispatch floor separately, sweeps n across four
orders of magnitude at three widths, and prints absolute single-query latency so
a reader can apply their own acceptability threshold. This repository does not
set one; that is a deployment decision.

Timing method. All arrays are allocated once per grid point, outside the timing
loop, and the matrix product writes through a preallocated output buffer, so
allocation is not charged to either operation. Medians are reported because a
single scheduler preemption skews a mean and these runs are short.

Vector width. The embedding dimension is not a constant in this repository: it
is taken from the provider's response at runtime (`pipeline.py:708`,
`dimension=vectors.shape[1]`). The widths swept below are illustrative choices
for measurement, not repository-derived facts, with one exception: 12 is the
width of the hand-authored fixture vectors in `eval/test_verdigrise.py`.

Memory. The default grid stops at n = 10,000, which peaks around 0.8 GiB. The
n = 100,000 row published in the README is opt-in because it peaks around 5 GiB:

    .venv/bin/python benchmark_retrieval.py --rows 1000 10000 100000

That peak is not this script's to avoid. It belongs to `NumpyVectorIndex.index`,
which calls `_normalize` (`pipeline.py:378-390`), and that holds the float32
input, a float64 copy, a float64 quotient, and the float32 result at once, then
revalidates through another float64 pass in `_unit_rows`. Peak ingest memory is
therefore several times the resident matrix, which is a real and deliberate cost
of the float64 accumulator that keeps float32 extremes from collapsing, not an
artefact of measurement. This script generates its own rows as float32 and
normalizes them in blocks so it adds nothing avoidable on top.

Timings are machine-specific and vary between runs on one machine. Measured over
five trials at 21 repetitions, spread stayed under 13 percent at every grid point
except n = 100,000 at d = 12, where it reached 38 percent. Report figures to the
precision those trials support and no further; the shape of the curve is the
finding, and the absolute milliseconds are not portable, which is why the script
prints the machine and library versions it ran on. Do not draw a conclusion that
depends on a figure sitting on one side of a round threshold.
"""

from __future__ import annotations

import argparse
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable, Sequence

import numpy as np

from config import TOP_K
from pipeline import NumpyVectorIndex

SEED = 20260726
DEFAULT_ROWS = (8, 16, 32, 64, 128, 512, 1_000, 10_000)
LARGEST_MEASURED_ROWS = 100_000
DEFAULT_WIDTHS = (12, 128, 1_536)
OPERATING_POINT_ROWS = 8
OPERATING_POINT_WIDTH = 12
CROSSOVER_ROWS = tuple(range(6, 49, 2))
CROSSOVER_TRIALS = 5
_NORMALIZE_BLOCK_ROWS = 4_096
# getrusage reports bytes on macOS and kilobytes on Linux.
_MAXRSS_DIVISOR = 2**20 if sys.platform == "darwin" else 2**10


def _synthetic_entries(count: int) -> list[dict[str, object]]:
    """Build corpus entries that satisfy index validation without any real text.

    Ids are zero-padded so they sort lexicographically, which keeps the
    tie-break comparison at `pipeline.py:419` doing representative string work.
    """

    return [
        {
            "id": f"synthetic-entry-{position:09d}",
            "text": f"Synthetic evidence sentence number {position}.",
            "grimoire_id": "GRIM-SYNTHETIC",
            "folio": position,
            "subject": "synthetic subject",
            "fact_type": "synthetic fact",
            "condition": "synthetic condition",
        }
        for position in range(count)
    ]


def _unit_rows(rows: int, width: int, generator: np.random.Generator) -> np.ndarray:
    """Return float32 rows the index will accept, normalized as ingestion would.

    Generated directly as float32 and normalized in row blocks. The obvious
    implementation, drawing float64 and converting, holds three full-size arrays
    at once; at the largest default grid point that is over 2 GiB of avoidable
    temporaries on top of the matrix itself. The float64 accumulator is still
    used for the norm, matching `pipeline.py:378-390`, but only one block wide.
    """

    matrix = generator.standard_normal((rows, width), dtype=np.float32)
    for start in range(0, rows, _NORMALIZE_BLOCK_ROWS):
        block = matrix[start : start + _NORMALIZE_BLOCK_ROWS]
        norms = np.linalg.norm(block.astype(np.float64), axis=1, keepdims=True)
        block[:] = (block.astype(np.float64) / norms).astype(np.float32)
    return matrix


def _median_ms(operation: Callable[[], object], repeats: int) -> float:
    """Return median wall-clock milliseconds over `repeats` calls, after a warm-up.

    The warm-up call is discarded. The first touch of a freshly allocated array
    pays page faults that belong to allocation, not to the operation, and
    charging them to the matrix product is exactly the error this script exists
    to avoid repeating.
    """

    operation()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def dispatch_floor(width: int, repeats: int) -> float:
    """Return the fixed per-call cost of the matrix product, with n = 1.

    One row makes the arithmetic negligible, so what remains is argument
    parsing, dtype and shape resolution, and BLAS dispatch. If the matrix
    product at larger n costs about this much, it is measuring overhead.
    """

    generator = np.random.default_rng(SEED)
    stored = _unit_rows(1, width, generator)
    query = _unit_rows(1, width, generator)[0]
    buffer = np.empty(1, dtype=np.float32)

    def probe() -> None:
        np.matmul(stored, query, out=buffer)
        np.clip(buffer, -1.0, 1.0, out=buffer)

    return _median_ms(probe, repeats)


def measure(rows: int, width: int, repeats: int) -> dict[str, float]:
    """Time one search, and the two operations inside it, at this corpus size."""

    generator = np.random.default_rng(SEED)
    vectors = _unit_rows(rows, width, generator)
    index = NumpyVectorIndex(dimension=width, embedding_model="benchmark-embedding-model")
    index.index(_synthetic_entries(rows), vectors)
    query = _unit_rows(1, width, generator)[0]

    # Reach past the public surface on purpose: the point is to attribute cost
    # to the two lines inside search, which the public call fuses.
    stored = index._vectors
    entries = index._entries
    normalized = index._normalize(query[None, :])[0]

    # Preallocated so neither timed operation is charged for allocation.
    buffer = np.empty(rows, dtype=np.float32)
    similarities = np.clip(stored @ normalized, -1.0, 1.0)

    def matmul() -> None:
        np.matmul(stored, normalized, out=buffer)
        np.clip(buffer, -1.0, 1.0, out=buffer)

    def python_sort() -> None:
        sorted(
            range(len(entries)),
            key=lambda position: (-float(similarities[position]), entries[position]["id"]),
        )[:TOP_K]

    def full_search() -> None:
        index.search(query)

    matmul_ms = _median_ms(matmul, repeats)
    sort_ms = _median_ms(python_sort, repeats)
    search_ms = _median_ms(full_search, repeats)
    return {
        "rows": float(rows),
        "width": float(width),
        "resident_mib": stored.nbytes / 2**20,
        "matmul_ms": matmul_ms,
        "sort_ms": sort_ms,
        "search_ms": search_ms,
        "sort_share": sort_ms / (matmul_ms + sort_ms) * 100.0 if matmul_ms + sort_ms else 0.0,
    }


def resolve_crossover(
    width: int, repeats: int, trials: int = CROSSOVER_TRIALS
) -> dict[str, object]:
    """Locate the corpus size where the sort first costs at least the matrix product.

    Two distinct error sources bound how precisely this can be stated, and both
    are reported rather than one being hidden behind the other.

    The first is grid resolution: a transition reported at some sampled n only
    means it happened above the previous sample and at or below this one, so the
    honest claim is a half-open interval whose width is the grid step.

    The second is measurement noise. Near the transition both operations cost a
    few microseconds, and run-to-run variation on one machine is around ten
    percent, which is enough to move the located point by a grid step or more.
    Refining the grid past the noise floor would buy apparent precision the
    measurement does not have, so the determination is repeated instead and the
    spread across trials is reported alongside the grid step.
    """

    located: list[int] = []
    for _ in range(trials):
        previous = None
        for count in CROSSOVER_ROWS:
            record = measure(count, width, repeats)
            if record["sort_ms"] >= record["matmul_ms"]:
                located.append(count)
                break
            previous = count
        else:
            located.append(0)
        _ = previous
    step = CROSSOVER_ROWS[1] - CROSSOVER_ROWS[0]
    found = [value for value in located if value]
    return {
        "width": width,
        "trials": located,
        "low": min(found) - step if found else 0,
        "high": max(found) if found else 0,
        "step": step,
    }


def _print_environment() -> None:
    print("VerdigrisE retrieval benchmark")
    print(f"  seed              {SEED}")
    print(f"  python            {platform.python_version()}")
    print(f"  numpy             {np.__version__}")
    print(f"  platform          {platform.platform()}")
    print(f"  machine           {platform.machine()}")
    print(f"  TOP_K             {TOP_K}")
    print()


def _report_operating_point(repeats: int) -> None:
    """Print the breakdown at the size this repository actually runs."""

    record = measure(OPERATING_POINT_ROWS, OPERATING_POINT_WIDTH, repeats)
    scaling = record["matmul_ms"] + record["sort_ms"]
    remainder = record["search_ms"] - scaling
    print(
        f"operating point: n = {OPERATING_POINT_ROWS} entries at d = {OPERATING_POINT_WIDTH}"
        " (the shipped corpus)"
    )
    print(f"  full search             {record['search_ms']:.4f} ms")
    print(
        f"  matrix product + sort   {scaling:.4f} ms "
        f"({scaling / record['search_ms'] * 100:.1f}% of search)"
    )
    print(
        f"  everything else         {remainder:.4f} ms "
        f"({remainder / record['search_ms'] * 100:.1f}% of search)"
    )
    print("  the remainder is query normalization plus construction of TOP_K capture models\n")


def run(rows: Sequence[int], widths: Sequence[int], repeats: int) -> list[dict[str, float]]:
    _print_environment()

    print("fixed per-call cost of the matrix product, measured at n = 1")
    for width in widths:
        print(f"  d = {width:>5}   {dispatch_floor(width, repeats):.5f} ms")
    print("  flat across d confirms this is dispatch, not arithmetic\n")

    _report_operating_point(repeats)

    print("crossover, resolved on a step-2 grid and repeated to expose timing noise")
    print(f"  {'width':>7} {'bracket':>18} {'grid step':>11}   trials")
    for width in widths:
        found = resolve_crossover(width, repeats)
        bracket = f"{found['low']} < n <= {found['high']}"
        print(f"  {width:>7} {bracket:>18} {found['step']:>11}   {found['trials']}")
    print("  a single value here would state the result more precisely than it resolves\n")

    results: list[dict[str, float]] = []
    for width in widths:
        print(f"vector width d = {width}")
        header = (
            f"  {'rows n':>9} {'resident MiB':>13} {'matmul ms':>11} "
            f"{'py sort ms':>11} {'search ms':>11} {'sort share':>11}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for count in rows:
            record = measure(count, width, repeats)
            results.append(record)
            print(
                f"  {count:>9,} {record['resident_mib']:>13.2f} {record['matmul_ms']:>11.4f} "
                f"{record['sort_ms']:>11.4f} {record['search_ms']:>11.4f} "
                f"{record['sort_share']:>10.1f}%"
            )
        crossover = next(
            (int(r["rows"]) for r in results if r["width"] == width and r["sort_share"] > 50.0),
            None,
        )
        if crossover is None:
            print("  the Python sort does not reach 50% of the two operations at any measured n")
        else:
            print(f"  the Python sort passes 50% of the two operations at n = {crossover:,}")
        print()

    print("matrix-product cost at fixed n across widths, the regime test")
    print(f"  {'rows n':>9} " + " ".join(f"{'d=' + str(w):>13}" for w in widths))
    for count in rows:
        row = [next(r for r in results if r["rows"] == count and r["width"] == w) for w in widths]
        print(f"  {count:>9,} " + " ".join(f"{r['matmul_ms']:>13.4f}" for r in row))
    print("  flat across a row means dispatch-bound; rising across a row means arithmetic-bound")
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _MAXRSS_DIVISOR
    largest = max(rows) if rows else 0
    print(f"\npeak process memory for this run: {peak_mib:.0f} MiB at n = {largest:,}")
    if largest < LARGEST_MEASURED_ROWS:
        print(
            f"  the README also publishes n = {LARGEST_MEASURED_ROWS:,}, which is opt-in because it\n"
            f"  peaks around 5 GiB: --rows 1000 10000 {LARGEST_MEASURED_ROWS}"
        )
    print()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)
    run(args.rows, args.widths, args.repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
