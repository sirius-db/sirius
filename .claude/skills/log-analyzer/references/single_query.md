# Single-query analysis playbook

The five analyses below are worth running on every query. The first two are about *time*, the next two are about *memory*, and the fifth is about *GPU balance* (only meaningful on multi-GPU runs).

You don't have to run them in order — pick what answers the user's question first. But running all five gives you a complete picture and often surfaces issues the user didn't ask about.

## 1. Operator time attribution

**Question answered:** Which operators or pipelines dominate query time?

**Inputs:** `_operator_aggregates.csv` and `_pipeline_aggregates.csv` (both filtered to the target `query_begin_ts`), `pipeline_plan.json`.

**Method:**

1. For pipeline-level attribution, sum `sum_execution_time_ms` across all pipeline rows of the query — that's your denominator (call it `total_exec_ms`), then compute each pipeline's `% = sum_execution_time_ms / total_exec_ms * 100`.
2. For operator-level attribution (usually the more actionable view), use `_operator_aggregates.csv`: each row already carries `operator_type`, `operator_id`, and `sum_execution_time_ms` per operator — rank them directly, no plan join needed. Rows are ordered by run order within each pipeline.
3. If you need to name operators from a pipeline row, cross-reference `pipeline_id` to `pipeline_plan.json -> operator_index` (or `pipelines[].operators`).
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
- Big gaps with non-FULL barriers are suspicious; cross-check `downgrades.csv` for downgrade activity in the same time window (rows whose `timestamp` falls inside the gap).
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

**Question answered:** How much memory was evicted, from where, to where? Excessive downgrade kills performance.

**Primary input:** `downgrades.csv` — one row per satisfied downgrade request emitted by `downgrade_executor`. The columns describe the request directly:

- `source_tier` + `source_device_id` — which tier the data was evicted *from* (`GPU:N`, `HOST:-1`, `DISK:N`).
- `is_monitor` — `True` for background-pressure requests from `monitor_loop`, `False` for caller-initiated requests (e.g. a task that OOM'd and asked for free memory).
- `total_bytes`, `total_batches`, `duration_ms`, `throughput_mbs` — the size and cost of the request.
- `to_host_bytes` / `to_disk_bytes` (and matching batch counts) — where the evicted data went. The two add up to `total_bytes`.
- `repos_bytes` / `pipeline_queue_bytes` — attribution of the freed bytes to data sitting in repositories vs data queued for an upstream pipeline that hadn't run yet.

**Secondary input:** `memory_history.csv` — per-task view of whether a task had to re-materialize a downgraded input. A row with `peak_bytes_to_materialize_input > 0` on a **non-scan** pipeline means upstream data had previously been downgraded and this task had to re-read it. This is the *consumer-side* signal complementing the *producer-side* signal in `downgrades.csv`.

**Method:**

1. **How much was downgraded, and from where?** Group `downgrades.csv` by `source_tier`, sum `total_bytes`. This answers "how much memory was downgraded from GPUs vs from host." On multi-GPU runs, also group by `source_device_id` to see if one GPU is bearing the eviction load.
2. **Where did the data go?** Sum `to_host_bytes` and `to_disk_bytes` across the query. Eviction to host is the cheap path; eviction to disk is the expensive path — many `to_disk` rows means GPU+HOST were both under pressure.
3. **Was it pressure-driven or pipeline-driven?** Compare counts with `is_monitor=True` vs `False`. A query dominated by `is_monitor=False` requests means tasks themselves are asking for free memory (they expected to OOM) — usually a reservation-estimate or operator-placement issue. Dominated by `is_monitor=True` means background pressure was triggering eviction — usually a residency/policy issue.
4. **Which pipelines were active during the downgrades?** Match `downgrades.timestamp` against `_pipeline_aggregates.csv` (`pipeline_begin` ≤ ts ≤ `pipeline_end`) to attribute downgrade activity to specific pipelines. Then use `pipeline_plan.json` to name the operators.
5. **Did consumers pay for it?** Filter `memory_history.csv` to non-scan pipelines with `peak_bytes_to_materialize_input > 0`. Use `pipeline_plan.json -> pipelines[].operators[0].type` to identify scans (`GPU_PARQUET_SCAN`, `DUCKDB_SCAN`, `TABLE_SCAN`) and ignore those. Non-zero values on non-scan pipelines are tasks that paid the re-read cost — these are the diagnostic rows.

**Interpreting the result:**

- A handful of small downgrade events: normal under memory pressure.
- Large `total_bytes` evicted from a single GPU: that GPU was the residency hotspot. Check the GPU balance section (#5) — usually the same GPU is also doing more compute.
- Repeated downgrade with the same pipeline active each time: that pipeline's input is being evicted and re-fetched in a loop — likely the slow-down culprit. Cross-check via `memory_history.csv` for the matching non-scan re-materialization rows on that pipeline.
- Significant `to_disk_bytes`: the engine couldn't keep working set in HOST either. Severe pressure — surface prominently.
- Cross-reference downgrade timestamps with the memory pressure timeline (#3); the spikes usually line up.

## 5. GPU balance (multi-GPU runs only)

**Question answered:** Did the work spread evenly across the available GPUs, or did one GPU carry most of it?

**Inputs:** `task_outputs.csv` (primary), `task_inputs.csv`, `memory_history.csv`, `memory_reservations.csv` — all four carry a `gpu_id` column on multi-GPU logs. `pipeline_plan.json` (to name operators by pipeline).

**Precondition:** Skip this analysis on single-GPU runs (every `gpu_id` is the same value, so balance is trivially 1). You can detect a multi-GPU run by checking that `task_outputs.csv` has more than one distinct `gpu_id`, or that the Sirius config in the log mentions more than one GPU memory space.

**Method:**

1. **Per-pipeline balance.** Group `task_outputs.csv` by `(pipeline_id, gpu_id)` and aggregate `num_tasks`, `sum(num_rows)`, `sum(size_bytes)`, `sum(execution_time_ms)`. For each pipeline, compute the imbalance ratio:
   ```
   imbalance(pipeline) = max(sum_execution_time_ms[gpu]) / mean(sum_execution_time_ms[gpu])
   ```
   A perfectly balanced pipeline has ratio ≈ 1.0; ratio > 1.5 means at least one GPU did substantially more work than average.
2. **Per-operator-type balance.** Same idea but grouped by `(operator_type, gpu_id)` — useful when one operator type (e.g. `HASH_JOIN`) consistently lands on a single GPU across multiple pipelines.
3. **Memory-pressure asymmetry.** Group `memory_reservations.csv` by `gpu_id` and compare the minimum `memory_available` per GPU. If one GPU consistently runs hotter, it's bearing more of the residency.
4. **Downgrade asymmetry.** Group `downgrades.csv` rows where `source_tier == "GPU"` by `source_device_id`, sum `total_bytes`. Downgrade volume concentrated on one GPU is a strong signal that pipeline placement isn't matching where the data lives. (For the consumer side — which task had to re-read downgraded data — also group `memory_history.csv` rows with `peak_bytes_to_materialize_input > 0` by `gpu_id`.)

**Interpreting the result:**

- Even balance (ratio close to 1) across all pipelines: the scheduler is doing its job.
- One pipeline severely skewed but others balanced: that pipeline's input data is GPU-pinned to one device (e.g. a pinned table colocated on one GPU, or a join build side that only landed on one GPU). Check `memory_history.csv` rows on the skewed pipeline for cross-GPU transfer hints.
- Same GPU is heavy across *multiple* pipelines: likely a SCHED-RR locality bias that's snowballing — once data ends up on GPU N, every downstream task that consumes it prefers GPU N too.
- Severe skew + downgrade concentrated on the heavy GPU: that GPU is bottlenecking the entire query because it's also the one running out of memory. Surface this prominently — the fix is usually a placement change, not an operator-level optimization.

**Cross-query GPU balance (for benchmarking workloads):**

When the user runs many queries in the same log (e.g. a TPC-H benchmark), the same analysis can be applied at the *query* level: for each query, compute its per-GPU `sum_execution_time_ms` and check whether some queries are consistently lopsided. Queries that are well-balanced vs lopsided form two natural buckets, and the lopsided bucket is a useful starting point for operator/placement investigations.

## Putting it together

The four analyses are most useful in combination. A typical narrative looks like:

> "Pipeline 8 (HASH_JOIN id=13 → FILTER id=14 → PROJECTION id=15 → UNGROUPED_AGGREGATE id=17) is 73% of query time. It has a 240 ms gap before it (matches the FULL barrier in the plan — that's the hash-table build wait, expected). Memory available drops to 8% of pool during pipeline 8 — borderline. `downgrades.csv` shows two `GPU:0` evictions totaling 6.2 GB to host (timestamps inside pipeline 8's window), and `memory_history.csv` has a matching non-zero `peak_bytes_to_materialize_input` on a pipeline 8 task — the build side was evicted and refetched. Recommendation: hand off to optimization-advisor for HASH_JOIN tuning."

Lead the conversational summary with the top finding. Save the supporting tables to the REPORT.md file.
