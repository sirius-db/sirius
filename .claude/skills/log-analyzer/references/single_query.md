# Single-query analysis playbook

The four analyses below are worth running on every query. The first two are about *time*, the last two are about *memory*.

You don't have to run them in order — pick what answers the user's question first. But running all four gives you a complete picture and often surfaces issues the user didn't ask about.

## 1. Operator time attribution

**Question answered:** Which operators or pipelines dominate query time?

**Inputs:** `_pipeline_aggregates.csv` (filtered to the target `query_begin_ts`), `pipeline_plan.json`.

**Method:**

1. Sum `sum_execution_time_ms` across all pipelines of the query — that's your denominator (call it `total_exec_ms`).
2. For each pipeline row, compute `% = sum_execution_time_ms / total_exec_ms * 100`.
3. Cross-reference `pipeline_id` to `pipeline_plan.json -> operator_index` (or look up the pipeline in `pipelines[].operators`) to name the operators.
4. Also surface `max_execution_time_ms` — a single slow task can be more diagnostic than the sum (e.g. one straggler task vs many balanced tasks).

**Interpreting the result:**
- A single pipeline at >40% is almost certainly your hotspot.
- If `max_execution_time_ms` is much larger than `sum_execution_time_ms / num_tasks`, you have skew across tasks.
- Scan pipelines (`GPU_PARQUET_SCAN`, `DUCKDB_SCAN`) often dominate small-scale TPC-H — that's expected.

## 2. Pipeline gap analysis

**Question answered:** Where is the query *not* doing work, and why?

**Inputs:** `_pipeline_aggregates.csv` (`pipeline_begin`, `pipeline_end`), `pipeline_plan.json` (for `barrier` info).

**Method:**

1. Sort pipelines by `pipeline_begin`.
2. For each consecutive pair, compute `gap_ms = pipeline_begin[i+1] - pipeline_end[i]` (in milliseconds).
3. Flag gaps > 10 ms (tune to the query's scale — for short queries even 1 ms gaps matter).
4. For each flagged gap, look up the upstream pipeline's `output.barrier` in `pipeline_plan.json`:
   - `barrier: FULL` — pipeline breaker (e.g. before a hash-join build, before MERGE_AGGREGATE). Expected.
   - `barrier: PIPELINE` or `barrier: PARTIAL` — gap is *not* expected from the plan. Often points at scheduler stalls or downgrade.

**Interpreting the result:**
- Big gaps before known pipeline breakers (FULL barriers) are normal but worth quantifying.
- Big gaps with non-FULL barriers are suspicious; cross-check `memory_history.csv` for downgrade activity in the same time window.
- Gaps at the very beginning of the query (between `query_begin_ts` and the first pipeline's `pipeline_begin`) reflect planning/setup time — surface separately.

## 3. Memory pressure timeline

**Question answered:** Did we get close to OOM? When?

**Inputs:** `memory_reservations.csv`.

**Method:**

1. The `max_pool` column is the GPU pool ceiling — should be constant within a query.
2. Plot or tabulate `total_reserved` and `memory_available` over `timestamp`.
3. Identify the timestamp where `memory_available` is at its minimum.
4. Cross-reference that timestamp against `_pipeline_aggregates.csv` (`pipeline_begin` / `pipeline_end`) to find which pipeline was active at the low-water mark.

**Interpreting the result:**
- `memory_available / max_pool < 0.10` means we got within 10% of OOM. This usually correlates with downgrade events (next section).
- A spike in `total_reserved` right before a `barrier: FULL` pipeline is the classic hash-table-build memory peak.

## 4. Downgrade detection

**Question answered:** Was data evicted from GPU memory and re-materialized? Excessive downgrade kills performance.

**Inputs:** `memory_history.csv`, `pipeline_plan.json` (to identify scan vs non-scan pipelines).

**Method:**

1. Filter `memory_history.csv` to rows where `peak_bytes_to_materialize_input > 0`.
2. For each such row, look up the pipeline's first operator type via `pipeline_plan.json -> pipelines[].operators[0].type`.
3. A non-zero value on a **scan** pipeline (`GPU_PARQUET_SCAN`, `DUCKDB_SCAN`, `TABLE_SCAN` on parquet) is expected — that's just the scan materializing input. **Ignore.**
4. A non-zero value on a **non-scan** pipeline means upstream data had been downgraded (spilled to host/disk) and had to be re-read. This is the diagnostic case.

**Interpreting the result:**
- A handful of small downgrade events: normal under memory pressure.
- Repeated downgrade on the same non-scan pipeline: the pipeline's input is being evicted and re-fetched in a loop — likely the slow-down culprit.
- Cross-reference downgrade timestamps with the memory pressure timeline (#3); they usually line up.

## Putting it together

The four analyses are most useful in combination. A typical narrative looks like:

> "Pipeline 8 (HASH_JOIN id=13 → FILTER id=14 → PROJECTION id=15 → UNGROUPED_AGGREGATE id=17) is 73% of query time. It has a 240 ms gap before it (matches the FULL barrier in the plan — that's the hash-table build wait, expected). Memory available drops to 8% of pool during pipeline 8 — borderline. Two downgrade events fire on pipeline 8's input from pipeline 3, suggesting the build side was evicted and refetched. Recommendation: hand off to optimization-advisor for HASH_JOIN tuning."

Lead the conversational summary with the top finding. Save the supporting tables to the REPORT.md file.
