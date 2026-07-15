---
name: log-analyzer
description: >
  Use this skill to analyze Sirius query execution logs — to find why a query is slow,
  diagnose validation errors (wrong results), spot excessive downgrading, identify pipeline
  gaps, or compare two runs of the same query. Trigger whenever the user mentions a Sirius
  log file, says "look at this log", asks about per-operator time, pipeline gaps, memory
  pressure, downgrade behavior, validation/correctness mismatches, or wants to compare two
  query runs. Also use as a hand-off target from race-check (for two-run comparisons),
  runtime-errors (for "where did the query stall before hang"), and validate (for diffing
  the bad run against a good run). This is the log-only counterpart to profile-analyzer
  (nsys) and optimization-advisor (source-level guidance).
---

# Sirius log analyzer

You are diagnosing Sirius query execution by parsing its logs into structured artifacts and reasoning about them. Logs are always available (no profiler required), so this is the cheapest first-look tool when something goes wrong.

## What this skill covers

Two modes, same underlying parser:

- **Single-query analysis** — one log, one query: what dominates time, where the gaps are, when memory got tight, whether downgrade was excessive.
- **Two-run comparison** — same SQL run twice (often in the same log): perf delta, memory delta, or — most importantly — *which operator first produces different row counts / bytes* when a query returns wrong results.

If the user is bringing a log file to you, **default to offering to parse it first**. Don't try to grep the raw log by hand for things the parser already structures.

## The underlying tool

`tools/log_analyzer/parse_logs.py` is the parser. Run it once per log file.

```bash
python3 tools/log_analyzer/parse_logs.py <log_file> [--out <dir>]
```

- Hard precondition: the log must contain `[trace]` and `[debug]` lines (set `SIRIUS_LOG_LEVEL=trace` when running Sirius). The script errors out otherwise.
- If `--out` is omitted, output goes to `./log_analysis/<log_basename>/`.

### What the parser produces

```
<out>/
├── _index.csv                  # one row per analytical query kept (select/with only)
├── _summary.json               # counts + complete_queries / incomplete_queries lists + format warnings
├── _pipeline_aggregates.csv    # one row per (query, pipeline_id) — pipeline window, operator_types, output/timing sums, all memory metrics
├── _operator_aggregates.csv    # one row per (query, pipeline_id, operator_id) — per-operator input/output/timing sums, ordered by run order within each pipeline
└── <ts>/                       # one folder per query, e.g. 2026-05-20_14-41-37.879
    ├── query.sql               # Full SQL text from the log
    ├── query_meta.json         # begin/end ts, duration, status (complete | incomplete)
    ├── pipeline_plan.json      # the structured DAG (pipelines, operators, dependencies, leaves, root)
    ├── pipeline_plan.txt       # raw "=== Pipeline Overview ===" block
    ├── memory_reservations.csv
    ├── task_inputs.csv
    ├── task_outputs.csv
    ├── memory_history.csv
    └── downgrades.csv         # one row per downgrade request (data evicted from a tier)
```

The full schema is in `tools/log_analyzer/README.md`. Skim it before doing anything non-obvious.

### Joining across the per-query CSVs

`task_inputs.csv`, `task_outputs.csv`, `memory_history.csv`, and `memory_reservations.csv` can be joined to one another inside a single query folder. Useful keys:

- `pipeline_id` — present on every row. Coarsest join; gives you "all reservation/input/output/history rows that belong to pipeline N".
- `task_id` — present on every row (in logs emitted after the multi-GPU executor split). One task produces exactly one `memory_reservation`, one `task_input`, one `task_output`, and one `memory_history` row, so `(pipeline_id, task_id)` is effectively a primary key across the four tables.

Typical joins:

- *"Which operator's task consumed the most memory?"* — join `memory_reservations` ↔ `task_outputs` on `(pipeline_id, task_id)`, then sort by `reserved_bytes` or by `peak_allocated_bytes`. You now have both the operator and the reservation that fed it.
- *"Did the task that downgraded its input also run slowly?"* — join `memory_history` (filter `peak_bytes_to_materialize_input > 0` on a non-scan pipeline) ↔ `task_outputs` on `(pipeline_id, task_id)` to get the offending task's `execution_time_ms`.
- *"How much did this task's actual output undershoot its reservation?"* — join `memory_reservations` ↔ `task_outputs` on `(pipeline_id, task_id)`; compare `reserved_bytes` to `peak_allocated_bytes`. Systematic over-reservation is a planner-estimate bug.

On older logs (pre multi-GPU split) `task_id` is `None` in the input/output/history CSVs and only the `pipeline_id` join is available. The parser still loads those logs; just lose the per-task granularity.

`downgrades.csv` does **not** participate in this join. Each downgrade is a process-wide eviction request emitted by `downgrade_executor`, not tied to a specific pipeline or task — it carries its own `(timestamp, source_tier, source_device_id)` identity instead. To correlate downgrades with what was running at the time, match `downgrades.timestamp` against the active pipelines window in `_pipeline_aggregates.csv` (`pipeline_begin` ≤ `downgrades.timestamp` ≤ `pipeline_end`).

### Picking a query to analyze

Read `_index.csv` (or `_summary.json -> queries.complete_queries / incomplete_queries`). Each row has `query_begin_ts`, `folder`, `status`, `duration_ms`, and `sql_preview` (a 120-char convenience truncation the *parser* applies for display — the full SQL lives at `<folder>/query.sql`). The folder name embeds the timestamp, e.g. `2026-05-20_14-41-37.879`.

Ask the user which query if it's not obvious, or surface the slowest / failed candidates from `_index.csv`.

## Mode A: Single-query analysis

See `references/single_query.md` for the full playbook. Quick summary of the analyses worth running on every query:

1. **Operator time attribution** — which operators dominate? Use `_operator_aggregates.csv` (`sum_execution_time_ms`, `max_execution_time_ms` per `operator_id`/`operator_type`) — it gives per-operator timing directly, no join to the plan needed. `_pipeline_aggregates.csv` has the pipeline-wide roll-up.
2. **Pipeline gap analysis** — for each pipeline, the gap from the previous pipeline's `pipeline_end` to this one's `pipeline_begin`. Large gaps point at pipeline breakers (`barrier: FULL` in the plan) or scheduler stalls.
3. **Memory pressure** — read `memory_reservations.csv` over time; `memory_available` dropping toward zero (and `total_reserved` approaching `max_pool`) means we got close to OOM and are likely thrashing.
4. **Downgrade detection** — read `downgrades.csv` directly. Each row is one satisfied downgrade request and carries the source tier (`GPU:N`, `HOST:-1`, etc.), `total_bytes`, `duration_ms`, throughput, and the `to_host` / `to_disk` split. Sum `total_bytes` by `source_tier` to answer "how much was evicted from GPUs vs from host." A handful is normal under pressure; a lot — especially `HOST→DISK` rows — is a red flag, downgrade is expensive.
5. **GPU balance** (multi-GPU runs only) — group `task_outputs.csv` by `gpu_id` to see whether operators distributed work evenly across GPUs, or whether one GPU did most of the work for a given pipeline / operator type. Severe imbalance points at scheduling or locality issues.

Cross-reference the pipeline_id in any of these CSVs against `pipeline_plan.json -> operator_index` to name the operator. Don't memorize pipeline numbers — they're query-local.

## Mode B: Two-run comparison

See `references/comparison.md` for the full playbook, but the matching rule is so important it lives here too.

### Matching rule (read this before comparing anything)

Two queries are comparable only if **BOTH** of these match:

1. **The full SQL string** — diff `<folder>/query.sql` between the two runs, *not* the parser's 120-char `sql_preview`.
2. **The list of operators** — same operator types and counts. Compare `pipeline_plan.json -> counts.operator_types` between the two queries.

If either differs, you are not looking at the same query, and any perf or correctness diff you draw is meaningless. Tell the user that the runs don't match and stop.

### Two common false-match traps

- **`sql_preview` collision.** The Sirius log captures the full SQL on `QueryBegin`, but the parser still derives a 120-char `sql_preview` field for display in `_index.csv`. Two different queries that share the same first 120 characters will have the same `sql_preview` even though their full SQL differs. Always diff `<folder>/query.sql` (the full text) and cross-check the operator list — they reflect the *actual* plan.
- **Different scan source.** `GPU_PARQUET_SCAN` vs `DUCKDB_SCAN` (or vs `TABLE_SCAN` on top of a Parquet read) is the same logical query but a different execution path. The two are not directly comparable — perf and memory characteristics differ for legitimate reasons. Call this out to the user as a finding, not a fault.

### When no comparable run exists in the current log

If the bad run's operator inventory doesn't match any other run of the same SQL in the current log, **don't** quietly compare across mismatched plans — that's exactly what the traps above forbid. Instead, ask the user: "I couldn't find a matching run in this log. Do you have another log (a successful test run, or a different failure log) where the same query ran with the same operators?" In Sirius's test setup the same workload is usually re-run across many logs, so this is a common and useful question. See `references/comparison.md` for full guidance.

### Comparing against an incomplete run

If one of the two runs has `status: incomplete` in `query_meta.json`, it likely didn't reach the `RESULT_COLLECTOR` operator. Compare only the operators that *did* run in the incomplete run against the same operators in the completed run. That tells you whether the incomplete run was diverging *before* it stopped — which often points at the failing step.

### The validation-error workflow (your highest-value use case)

When two runs have the same SQL + same operators but produce different result rows, the goal is to find the **earliest operator whose output diverges**:

1. For each pipeline in both runs, compare `sum_output_num_rows` and `sum_output_size_bytes` from `_pipeline_aggregates.csv`.
2. Walk the pipeline DAG from leaves (scans) toward the root (RESULT_COLLECTOR) — `pipeline_plan.json -> leaves` and `dependencies` give the order.
3. The first pipeline along that walk whose output diverges between runs is the suspect operator. Everything downstream will look different too, but the *first* divergence is the source.

If outputs match for every pipeline but the final result still differs, the divergence is below the pipeline level — likely inside an operator's internal shuffle/ordering. Surface that to the user.

## How to deliver findings

Produce **both**:

1. **An inline conversational summary** — 4–8 bullets at most, lead with the most actionable finding (e.g. "Pipeline 8 dominates 73% of execution time; HASH_JOIN id=13 builds 1.2GB hash table — likely the bottleneck"). Cite specific pipeline IDs and operator IDs so the user can navigate the artifacts.
2. **A saved markdown report** at `<parser_output>/REPORT.md` (or `<parser_output>/REPORT_<query_ts>.md` for a single-query report; `<parser_output>/COMPARE_<good_ts>_vs_<bad_ts>.md` for a comparison). This is what race-check / validate / runtime-errors will pick up when handing off. Use this template:

   ```markdown
   # <Single-query analysis | Comparison: <good_ts> vs <bad_ts>>
   - Query begin ts: ...
   - Status: complete | incomplete
   - Duration: ... ms

   ## Top findings
   1. ...

   ## Operator time attribution
   | Pipeline | Operators | sum_execution_time_ms | % of query |
   | ... |

   ## Pipeline gaps
   ...

   ## Memory pressure
   ...

   ## Downgrade events
   ...

   ## (Comparison mode) Earliest output divergence
   ...

   ## Reproduce
   - parser output dir: ...
   - relevant per-query folders: ...
   ```

Keep the inline summary terse; the report is where the supporting tables live.

## When to hand off to a sibling skill

After you've localized the issue, recommend the right next step:

- **race-check** — if the divergence between two runs looks scheduling-dependent (e.g. row count varies run-to-run on the same operator).
- **runtime-errors** — if the bad run is `status: incomplete` and ended in a crash/hang.
- **optimization-advisor** — if the user wants to *fix* a localized bottleneck (e.g. "HASH_JOIN is the hotspot — how do I make it faster?"). You found the where; advisor handles the how.
- **profile-analyzer** — if the user wants kernel-level detail (occupancy, memory bandwidth) on the hotspot you found.
- **validate** — for the actual validation harness once you've identified the suspect operator.

## Common pitfalls (re-emphasized)

- **Don't compare runs without checking operators match.** Matching on the parser-derived `sql_preview` is unreliable (it's a 120-char convenience field); matching on full SQL via `query.sql` still doesn't catch scan-source variation. Always also compare operator inventories.
- **`operator_types` in `_pipeline_aggregates.csv` is sourced from `task_outputs`, not `task_inputs`.** Scan pipelines emit no task_inputs but they do emit task_outputs — so the field is populated correctly for them.
- **When pairing pipelines across two runs, require BOTH `pipeline_num` AND the operator-chain string to match.** For two runs of the same SQL on the same plan, pipeline numbers are stable across runs — but chain strings repeat (multiple PARTITIONs, multiple CONCATs), so matching on chain alone silently mis-pairs. Requiring both protects against the case where a run actually had a different plan slip through.
- **When the validation walk finds no divergence, report exactly that.** Don't speculate that the bug "must be" in operator content / values. The likeliest explanation is that you didn't compare the right pair — ask the user whether other good runs are worth trying.
- **Incomplete queries are still parsed.** Don't ignore them — they often contain the most diagnostic information about a crash or hang. Just be careful to only compare operators that actually ran.

## Quick-reference: where to find things

| Question | Where to look |
| --- | --- |
| Which queries are in this log? | `_index.csv`, `_summary.json` |
| What operators does query X have? | `<folder>/pipeline_plan.json -> counts.operator_types` |
| Which pipeline does operator id=N live in? | `<folder>/pipeline_plan.json -> operator_index` |
| Per-operator execution time | `_operator_aggregates.csv` (per-operator, aggregated) or `<folder>/task_outputs.csv` (per-task) |
| Memory low-water mark | `<folder>/memory_reservations.csv -> memory_available` |
| Downgrade events (bytes evicted, source tier, where it went) | `<folder>/downgrades.csv` |
| Re-materialized inputs after a downgrade (per-task view) | `<folder>/memory_history.csv -> peak_bytes_to_materialize_input > 0` |
| Per-pipeline summary across one query | filter `_pipeline_aggregates.csv` by `query_begin_ts` |
| Per-operator summary across one query | filter `_operator_aggregates.csv` by `query_begin_ts` |
| How well work was balanced across GPUs (multi-GPU runs) | group `task_outputs.csv` by `gpu_id` (per pipeline or per operator type); see `references/single_query.md` § "GPU balance" |
| Host (pinned) pool memory at query start/end | `<folder>/query_meta.json -> pool_stats.host` |
| GPU device memory pool at query start/end (per GPU) | `<folder>/query_meta.json -> pool_stats.gpu[]` |
| Why my parser run looks weird | `_summary.json -> format_warnings`, then `tools/log_analyzer/README.md` |

## Pool stats: detecting memory leaks across queries

`query_meta.json` contains a `pool_stats` section that captures memory state at the query boundary:

```json
{
  "pool_stats": {
    "host": {
      "begin": {"allocated_bytes": 5242880, "peak_bytes": 307232768, "free_blocks": 5115},
      "end":   {"allocated_bytes": 5242880, "peak_bytes": 307232768, "free_blocks": 5115}
    },
    "gpu": [
      {
        "device_id": 0,
        "begin": {"allocated_bytes": 0, "peak_bytes": 104857600, "reserved_bytes": 107374182400},
        "end":   {"allocated_bytes": 0, "peak_bytes": 104857600, "reserved_bytes": 107374182400}
      }
    ]
  }
}
```

- **Host pool** (`[query_pool]` log tag): uses a `fixed_size_host_memory_resource` slab allocator. `free_blocks` drops during the query and should return to the QueryBegin value at QueryEnd. A persistent `end.allocated_bytes > begin.allocated_bytes` is a pinned-host memory leak.
- **GPU pool** (`[gpu_pool]` log tag, one line per GPU): uses a `reservation_aware_resource_adaptor`. `allocated_bytes` reflects live GPU allocations across all streams; `reserved_bytes` reflects the capacity claimed by active reservations. A persistent `end.allocated_bytes > begin.allocated_bytes` is a GPU device memory leak.

The two pool tags are distinguishable in the raw log: host stats use `[query_pool]`, GPU stats use `[gpu_pool]`.
