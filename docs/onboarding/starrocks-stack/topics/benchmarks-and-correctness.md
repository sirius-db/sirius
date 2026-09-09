# Benchmarks and checking answers

[Back to the guide](../README.md)

**Topic:** validation. **Features:** query runs, reference answers and comparison.
**Modules:** benchmark shell scripts and Python tools.
This document describes the reviewed source at `7610840c`.

## Why there are several tools

A fast query is useful only if its answer is correct. This branch separates
running the experiment, producing a reference answer, and comparing the results.

| Tool | Responsibility |
| --- | --- |
| `bench.sh` | Run selected queries through StarRocks, record time and save output. |
| `oracle.py` | Run equivalent queries in DuckDB to produce reference results. An oracle here means a reference answer. |
| `compare.py` | Check saved results against the reference, including every available run. |
| `cn-distribution.py` | Inspect per-CN records to see whether the intended nodes did work. |

The [benchmark README](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/benchmarks/tpch/README.md)
describes the workflow. Correct output alone does not prove that all intended
GPUs participated; that is why the distribution tool is separate.

## What the query kit contains

The branch carries all 22 TPC-H queries, a standard decision-support query set.
The kit reads Parquet files through StarRocks `FILES()` expressions and substitutes
the dataset location and scale factor into the SQL.

Some query text differs from the stock set. Query 11 scales its fraction by the
dataset scale and adds a tie-breaker to its ordering. Queries 8 and 9 reorder
tables in the `FROM` clause to avoid a documented poor plan when file statistics
are missing. Both engines must run the same text for a useful comparison.
The exact changes and their historical reasons are in
[QUERY-DEVIATIONS.md](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/benchmarks/tpch/QUERY-DEVIATIONS.md).

## Cold and warm runs must both count

A **cold run** includes startup or first-use effects. A **warm run** repeats work
after some initialization has happened. Comparing only the last warm answer can
hide a failure in the cold run.

For example, suppose the reference is `1`. The cold run returns `2`, and the next
run returns `1`. The comparison must fail overall. The inspected comparator does
fail this example; the [local reproduction record](../research/benchmark-validation.md)
documents the check.

The comparator examines each available run and keeps the worst result per query.
It checks row counts, then values. Numeric differences use a tolerance; text
values are compared after trimming surrounding whitespace. This is visible in
[compare.py](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/tools/compare.py).

## A verified gap in the correctness check

**NaN** means "not a number." It can appear after invalid floating-point
calculations. The comparator can report NaN as matching the finite value `1.0`.
This was reproduced with the real script. The cause is that a comparison such
as `NaN > tolerance` is false, so the code does not count the cell as different.

The benchmark extraction in
[draft #1738](https://github.com/sirius-db/sirius/pull/1738) leaves this defect
unfixed. It needs an explicit policy for non-finite values before it can serve
as a reliable correctness check. See
[verification and open issues](verification-and-open-issues.md).

There is also a scope rule: the comparator checks the result files it receives.
If the reference directory contains all 22 queries but the run contains only
queries 1 and 6, ignoring the other references is intentional. Proving that a
full 22-query run is complete needs a list of expected queries.

## What was checked

The extraction passed shell syntax and Python compilation checks. Local fixtures
confirmed ordinary mismatch rejection and exposed the NaN false pass. The review
did not rerun a GPU benchmark campaign. Earlier SF500 observations in the
[MIG notes](memory-and-mig.md) are historical records, not new successful tests.

The benchmark and placement commits are in
[#1738](https://github.com/sirius-db/sirius/pull/1738). The experimental memory and
reproduction runbook is in [#1737](https://github.com/sirius-db/sirius/pull/1737).
Both were still drafts when their status was checked for this guide.
