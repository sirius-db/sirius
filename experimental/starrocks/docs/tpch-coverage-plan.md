# Plan: TPC-H coverage for the Sirius StarRocks CN

The goal is to take TPC-H at SF1000 from 8/22 to 22/22 on 4× GB200. This doc covers only the plan shapes the CN refuses or gets wrong. Runtime, failure handling and the StarRocks feedback are in the [other docs](README.md).

## Where we are

All 22 queries at SF1000, 4× GB200, one CN per GPU, every result checked against DuckDB:

- **8/22 pass:** q05 q06 q07 q08 q09 q12 q14 q19.
- **14 fail.** Each failure is the CN refusing or mistranslating a plan shape; none is a GPU execution failure.

| gap | queries | error today | cause |
|---|---|---|---|
| G1. Two-phase aggregation beyond SUM | q01 q02 q13 q15 q17 q22 | "two-phase aggregation supports SUM only" | The stack splits only SUM into a partial step and a merge step. The POC branch also does COUNT, MIN, MAX and AVG. |
| G2. RIGHT SEMI join (`TJoinOp` 7) | q04 q20 q21 | "hash join type is unsupported" | This is how the FE plans `EXISTS` / `IN`. Neither the stack nor the POC maps it. |
| G3. `LIMIT` becomes `LIMIT 0` ([#1963](https://github.com/sirius-db/sirius/issues/1963)) | q03 q10 | "input stream N was declared but the plan does not read it" | The translator writes `FetchRel.count_expr`, and the DuckDB consumer reads only `count`, so it sees 0. DuckDB then drops the subtree, including its exchange read. The POC branch avoided this by writing `count`. |
| G4. Column-layout bugs | q16 q18 | "descriptor error: slot N (tuple T) is not part of row_tuples" | The translator looks a column up in a table the operator's input row doesn't carry. |
| G5. Wrong result | q11 | result mismatch | A grouped sum compared with a scalar subquery returns no rows. |

q02, q18 and q21 also end in `LIMIT`, so each needs the G3 fix as well as its own.

## Steps

The order is set in [tpch-coverage-and-filters-plan.md](tpch-coverage-and-filters-plan.md), which interleaves these steps with the runtime-filter work so that each filter step lands on queries that already run. Milestone numbers (M0–M5) refer to that doc. Each step ends with a run of all 22 queries. The count in brackets is the most it can add, counting what each query's recorded plan needs, not only its first error.

### Step 0: measurement (M0)

- **Stable datasets on local disk.** Restore `/scratch`, or use `/raid`. The `/opt` copy is 3–6× slower.
- **`harness/bench.sh --queries all`:**
  - commit the 13 new FILES() query files;
  - add a coverage table with refusal reasons, `--warmup N`, and a geomean over a fixed query set.
- **A CPU baseline on the same machine:** StarRocks BEs, or DuckDB on the same parquet.
- **A filters-off rerun** of every passing query (`SIRIUS_CN_RUNTIME_FILTERS=0`). Both runs must match DuckDB.

### Step 1: `LIMIT` (G3, M0) [up to +2: q03, q10]

[#1963](https://github.com/sirius-db/sirius/issues/1963) is still open; `TransformFetchOp` on `main` reads only `count`.
- **Stopgap in the translator:** also write the old `count` / `offset` fields (`node_translator.rs:241`), as the POC branch does.
- **Fix in the consumer (#1963):** `TransformFetchOp` (`substrait/src/from_substrait.cpp:584`) reads `count_expr` / `offset_expr` when set, falling back to `count` / `offset`.
- **Test:** `ORDER BY … LIMIT` round-trips through the consumer with the right limit.
- If q03 or q10 still fails after the fix, the "declared but not read" error is a separate bug.

### Step 2: two-phase COUNT, MIN, MAX (G1, part 1; M1) [up to +3: q13, q15, q02]

Port the single-column rules of `partial_state.rs` from the POC branch ([design/multi-phase-aggregation.md](design/multi-phase-aggregation.md) Step 1).
- Derive the exchange type from the function and phase, never from the slot type.
- q02 also needs Step 1.

### Step 3: RIGHT SEMI join (G2, M2) [up to +3: q20, q04, q21]

- In `translate_hash_join` (`node_translator.rs`), map `RIGHT_SEMI_JOIN` to Substrait `RIGHT_SEMI`, with the right input's columns as output.
- Re-check q21's RIGHT ANTI with extra join conditions.
- q04 and q21 also need Step 2's COUNT, and q21 also needs Step 1.
- **Tests:** the failing queries' recorded FE fragments become translator fixtures.

### Step 4: two-phase AVG (G1, part 2; M4) [up to +3: q01, q17, q22]

AVG travels as an FP64 sum and a BIGINT count ([design/multi-phase-aggregation.md](design/multi-phase-aggregation.md) Step 2). The partial step's slot says `VARBINARY` in all three queries; both sides must agree on the two columns without reading it.

### Step 5: column-layout bugs (G4, M5) [up to +2: q16, q18]

Track slot → column position through every translation step, the way the StarRocks BE reads the plan. Start each query from a failing translator test built from its recorded fragments:
- **q16:** a null-aware anti join plus `COUNT(DISTINCT)`.
  - The recorded plan isn't three-phase: two dedupe aggregations with no functions, then a one-phase `count`. The classifier already accepts that.
  - Only the descriptor error is left.
- **q18:** an `IN (… GROUP BY … HAVING)` semi join. It also needs Step 1.

### Step 6: q11 (G5, M5) [+1]

Compare its recorded plan with DuckDB's answer. It returns 0 rows with no error. Suspects:
- the cross join against the one-row subquery;
- the double-versus-decimal comparison in its `HAVING`. If the cause is decimal precision, the fix belongs to [design/exact-decimal.md](design/exact-decimal.md).

## Expected coverage (SF1000)

| after step | passing |
|---|---|
| today | 8/22 |
| 1: `LIMIT` | ≤ 10 |
| 2: two-phase COUNT/MIN/MAX | ≤ 13 |
| 3: RIGHT SEMI | ≤ 16 |
| 4: two-phase AVG | ≤ 19 |
| 5: layout bugs | ≤ 21 |
| 6: q11 | ≤ 22 |

These are upper bounds; each step's 22-query run gives the real count.

## Lessons from Starburst's GPU work

Source: [Starburst + NVIDIA](https://www.starburst.io/blog/gpu-accelerated-sql-analytics-how-starburst-and-nvidia-deliver-industry-benchmark-speedups-on-gpu-infrastructure/). 18 TPC-H queries, 4.6× geomean on one Blackwell GPU.

- **No query fails for coverage:** they fall back to the CPU. We refuse, so coverage comes before speed, and fallback has its own plan ([design/cpu-fallback-and-deployment.md](design/cpu-fallback-and-deployment.md)).
- **Fix gaps by category,** not query by query: one aggregation operator, one join operator.
- **Q17 regressed 3× without dynamic filtering.** See [design/distributed-runtime-filters.md](design/distributed-runtime-filters.md); q17 gets its partitioned-join filter at M3/M4 of the [combined order](tpch-coverage-and-filters-plan.md).
- **Method:** warm-up runs, geomean, a CPU baseline on the same machine (Step 0).
