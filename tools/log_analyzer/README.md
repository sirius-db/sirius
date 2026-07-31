# Sirius log analyzer

Parses a Sirius log file into structured, per-query artifacts that a Claude Skill
(or any downstream tool) can query interactively to answer questions about query
plans, pipeline behavior, memory reservations, and per-task metrics.

## Usage

```bash
# From the repo root
python tools/log_analyzer/parse_logs.py log/sirius_2026-05-20.log

# Custom output directory
python tools/log_analyzer/parse_logs.py log/sirius_2026-05-20.log --out /tmp/log_run
```

If `--out` is omitted, output goes to `./log_analysis/<log_basename>/`.

## Hard preconditions

The script first checks that the log contains at least one `[trace]` and one
`[debug]` line. If not, it exits with:

```
ERROR: the logs were not capture with the trace log level
```

Configure your Sirius run with `SIRIUS_LOG_LEVEL=trace` to capture the full
set of metrics this parser depends on.

## Output layout

```
<out>/
├── _index.csv                  # one row per kept analytical query
├── _summary.json               # totals, parser version, format-drift warnings
├── _pipeline_aggregates.csv    # one row per (query, pipeline_id) — cross-query
├── _operator_aggregates.csv    # one row per (query, pipeline_id, operator_id) — cross-query
└── 2026-05-20_14-25-02.368/    # one folder per kept query, name = QueryBegin ts
    ├── query.sql               # SQL text as it appeared in the log (often truncated)
    ├── query_meta.json         # begin/end ts, duration, status, per-metric counts
    ├── pipeline_plan.json      # structured DAG (see schema below)
    ├── pipeline_plan.txt       # raw "=== Pipeline Overview ===" block
    ├── memory_reservations.csv
    ├── task_inputs.csv
    ├── task_outputs.csv
    └── memory_history.csv
```

### `_index.csv`

| Column | Description |
|---|---|
| `query_begin_ts` | Timestamp from the QueryBegin line |
| `query_end_ts` | Timestamp from the QueryEnd line (empty if `status=incomplete`) |
| `status` | `complete` if a matching QueryEnd was found, else `incomplete` |
| `duration_ms` | `end_ts - begin_ts` in milliseconds (empty if incomplete) |
| `folder` | Subfolder name containing the per-query artifacts |
| `sql_preview` | First 120 characters of the SQL as captured in the log |

### `_pipeline_aggregates.csv`

One row per (query, pipeline_id) holding the metrics that are only meaningful
at the pipeline level: the pipeline begin/end window, its set of operator
types, pipeline-wide output/timing sums, and all of the memory-reservation and
memory-history figures. The memory metrics live here (not in the per-operator
table) because the log emits them once per task for the whole pipeline chain —
they carry no `operator_id` and cannot be attributed to a single operator.

Per-operator input/output/timing sums live in `_operator_aggregates.csv` (see
below). The per-task **input** sums are intentionally *not* in this table:
summing `input_num_rows` across a pipeline double-counts data that flows from
one operator into the next, which produced misleading numbers.

| Column | Source | Notes |
|---|---|---|
| `query_begin_ts` | QueryBegin line | grouping key |
| `sql_preview` | first 120 chars of SQL | for human readability |
| `pipeline_id` | grouping key |  |
| `pipeline_begin` | `min(timestamp)` from `task_inputs.csv` | earliest task input |
| `pipeline_end` | `max(timestamp)` from `task_outputs.csv` | latest task output |
| `num_tasks` | distinct `task_id` count from `task_outputs.csv` | falls back to row count for pre-multi-GPU logs (no `task=` field) |
| `operator_types` | comma-joined distinct op types from `task_outputs.csv` | first-seen order; sourced from outputs so scan pipelines (`GPU_PARQUET_SCAN`, `DUCKDB_SCAN`) — which emit no task_inputs — are still populated |
| `sum_output_num_rows` | sum from `task_outputs.csv` |  |
| `sum_output_size_bytes` | sum from `task_outputs.csv` |  |
| `sum_output_peak_allocated_bytes` | sum from `task_outputs.csv` |  |
| `sum_execution_time_ms` | sum from `task_outputs.csv` | "total time across tasks" |
| `max_execution_time_ms` | max from `task_outputs.csv` | "slowest single task" |
| `sum_reserved_bytes` | sum from `memory_reservations.csv` |  |
| `min_memory_available` | min from `memory_reservations.csv` | snapshot — see note below |
| `max_total_reserved` | max from `memory_reservations.csv` | snapshot |
| `max_max_pool` | max from `memory_reservations.csv` | snapshot |
| `sum_history_input_basis` | sum from `memory_history.csv` |  |
| `sum_history_output_bytes` | sum from `memory_history.csv` |  |
| `sum_history_reservation_bytes` | sum from `memory_history.csv` |  |
| `sum_history_peak_bytes` | sum from `memory_history.csv` |  |
| `sum_history_peak_bytes_to_materialize_input` | sum from `memory_history.csv` |  |

> **Snapshot columns**: `memory_available`, `total_reserved`, `max_pool` from
> `memory_reservations.csv` are GPU-pool snapshots at reservation time, not
> per-task work. Summing them across tasks produces a misleading number; we
> use min/max instead. `history_reservation_bytes` (from `memory_history.csv`)
> is a separate per-record figure and IS summed.

### `_operator_aggregates.csv`

One row per (query, pipeline_id, operator_id) summarizing the per-task metrics
attributed to an individual operator. Grouped from `task_inputs.csv` /
`task_outputs.csv` (the only metrics that carry an `operator_id`), so — unlike
the pipeline table — the input sums here are correct: each operator's inputs
are its own, not the whole pipeline's.

Rows are sorted by `query_begin_ts`, then `pipeline_id`, then `operator_begin`
(so the operators within each pipeline appear in the order they run — `min(task_id)`
cannot order them because every task flows through every operator and so shares
the same min), then `operator_id` as a final tiebreak.

| Column | Source | Notes |
|---|---|---|
| `query_begin_ts` | QueryBegin line | grouping / sort key |
| `sql_preview` | first 120 chars of SQL | for human readability |
| `pipeline_id` | grouping / sort key |  |
| `operator_id` | grouping key | from the `(id=N)` field |
| `operator_type` | from `task_outputs.csv` | single type for this operator |
| `operator_begin` | `min(timestamp)` from `task_outputs.csv` | earliest output for this operator |
| `operator_end` | `max(timestamp + execution_time_ms)` from `task_outputs.csv` | latest completion for this operator |
| `num_tasks` | distinct `task_id` count from `task_outputs.csv` | falls back to row count for pre-multi-GPU logs |
| `sum_input_num_rows` | sum from `task_inputs.csv` |  |
| `sum_input_size_bytes` | sum from `task_inputs.csv` |  |
| `sum_output_num_rows` | sum from `task_outputs.csv` |  |
| `sum_output_size_bytes` | sum from `task_outputs.csv` |  |
| `sum_output_peak_allocated_bytes` | sum from `task_outputs.csv` |  |
| `sum_execution_time_ms` | sum from `task_outputs.csv` |  |
| `max_execution_time_ms` | max from `task_outputs.csv` |  |

### `pipeline_plan.json`

```jsonc
{
  "pipelines": [
    {
      "pipeline_num": 8,
      "operators": [
        {"type": "HASH_JOIN",          "id": 13},
        {"type": "FILTER",             "id": 14},
        {"type": "PROJECTION",         "id": 15},
        {"type": "PROJECTION",         "id": 16},
        {"type": "UNGROUPED_AGGREGATE","id": 17}
      ],
      "inputs": [
        {"from_pipeline": 7, "on_operator": {"type": "HASH_JOIN", "id": 13}, "port": "default", "barrier": "FULL"},
        {"from_pipeline": 3, "on_operator": {"type": "HASH_JOIN", "id": 13}, "port": "build",   "barrier": "FULL"}
      ],
      "dependencies": [3, 7],
      "output": {"to_operator": "MERGE_AGGREGATE", "port": "default", "barrier": "FULL"}
    }
  ],
  "operator_index": { "13": 8, "14": 8 },   // operator_id (str) -> pipeline_num
  "counts": {
    "pipelines": 11,
    "operators": 16,
    "scans": 2,
    "operator_types": {"HASH_JOIN": 1, "PROJECTION": 3}
  },
  "leaves": [0, 4],            // pipelines with no Input: line
  "root_pipeline": 10          // pipeline containing RESULT_COLLECTOR
}
```

## Per-metric CSV schemas

### `memory_reservations.csv`
`timestamp, pipeline_id, task_id, reserved_bytes, memory_available, total_reserved, max_pool`

### `task_inputs.csv`
`timestamp, pipeline_id, operator_type, operator_id, num_batches, num_rows, size_bytes`

> Multi-batch lines (e.g. `num rows: 0  0  1  0`) have their per-batch counts
> summed into a single `num_rows` total. `size_bytes` is already a total in
> the log.

### `task_outputs.csv`
`timestamp, pipeline_id, operator_type, operator_id, num_batches, num_rows, size_bytes, execution_time_ms, peak_allocated_bytes`

### `memory_history.csv`
`timestamp, pipeline_id, input_basis, output_bytes, reservation_bytes, peak_bytes, peak_bytes_to_materialize_input`

## Filtering rules

A query is kept only if the `QueryBegin: <sql>` line's SQL (stripped of
leading whitespace, case-insensitive) starts with `select ` or `with `.

Skipped: ATTACH, USE, SET, CREATE VIEW, CREATE TABLE, BEGIN, COMMIT, LOAD,
INSTALL, DROP, COPY, CALL, RESET, INSERT, …

## Keyed execution windows (SHAPE 1.7)

Current logs carry a correlation key on every execution-window line:
`[window] begin/end instance=<ptr> connection=<N> window=<N> query=<N>
outcome=<ok|unwind|cleanup_failed|begin_failed|->`, and the pool/SQL lines
inside a window repeat the same key. When `[window]` lines are present the
segmenter pairs segments by `(instance, connection, window)` — the
authoritative boundary — and joins the SQL by `(instance, connection, query)`.
`QuerySegment` then exposes `instance`, `connection_id`, `window_id`,
`query_id` and `outcome`. `verify_query_lifecycle_segments.py` runs a
concurrent two-connection scenario against the real engine and cross-checks
the segmenter's output line-by-line against a raw parse of the same log.

## Incomplete queries

A `QueryBegin` with no matching `QueryEnd` (log truncated, crash) is still
parsed with `status="incomplete"` in `query_meta.json` and `_index.csv`. On
keyed logs an incomplete window is capped at the next window-begin; on legacy
keyless logs the positional matching for QueryEnd stops at the next QueryBegin
to avoid swallowing the following query.

## Format drift detection

Each parser uses a cheap "loose prefix" substring check to find candidate
lines, then a strict regex to capture fields. When the loose prefix matches
but the strict regex doesn't, it's recorded under
`_summary.json -> format_warnings -> drift_counts` along with up to three
sample lines. The CLI exits 0 but writes a `WARNING:` to stderr so callers
notice.

If you see drift warnings, update the corresponding pattern in `patterns.py`
and bump `SHAPE_VERSION`.

## Static line anchors (the things to update when the log format changes)

These literal substrings are checked verbatim. Centralize all edits in
`patterns.py`.

| Anchor | Constant | Used by |
|---|---|---|
| `[info] [:] [query_pool] QueryBegin allocated=` | `QUERY_BEGIN_ANCHOR` | `segmenter.py` |
| `[info] [:] [query_pool] QueryEnd allocated=` | `QUERY_END_ANCHOR` | `segmenter.py` |
| `[info] [:] QueryBegin: ` | `QUERY_SQL_ANCHOR` | `segmenter.py` |
| `=== Pipeline Overview ===` | `PIPELINE_OVERVIEW_HEADER` | `plan_parser.py` |
| `=== Query Plan DAG ===` | `PIPELINE_OVERVIEW_END_MARKERS` | `plan_parser.py` (end of block) |
| `Acquiring memory reservation for pipeline` | `MEM_RESERVATION_ANCHOR` | `metrics/memory_reservation.py` |
| `executing on` | `TASK_INPUT_ANCHOR` | `metrics/task_input.py` |
| `produced` | `TASK_OUTPUT_ANCHOR` | `metrics/task_output.py` |
| `memory history record` | `MEM_HISTORY_ANCHOR` | `metrics/memory_history.py` |

## Module layout

```
tools/log_analyzer/
├── parse_logs.py        # CLI orchestrator
├── segmenter.py         # Splits log into per-query segments
├── plan_parser.py       # Parses === Pipeline Overview === block
├── aggregator.py        # Builds _pipeline_aggregates.csv + _operator_aggregates.csv
├── patterns.py          # All anchors + regexes (single source of truth)
├── validators.py        # Format-drift tracking
├── metrics/
│   ├── memory_reservation.py
│   ├── task_input.py
│   ├── task_output.py
│   └── memory_history.py
└── README.md            # this file
```

## Parser shape version

`patterns.SHAPE_VERSION = "1.0"` — recorded in `_summary.json`. Bump when
any pattern changes so a downstream skill can refuse to query data parsed
by an incompatible version.
