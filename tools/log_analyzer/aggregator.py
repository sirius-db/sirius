"""Build the per-pipeline and per-operator aggregate tables.

Two tables are produced from a single query's parsed metrics:

* ``_pipeline_aggregates.csv`` — one row per (query, pipeline_id). Holds the
  columns that are only meaningful at the pipeline level: the pipeline
  begin/end window, the set of operator types in the pipeline, pipeline-wide
  output/timing sums, and all of the memory-reservation and memory-history
  figures (which the log emits per task, not per operator, so they cannot be
  attributed to a single operator).

* ``_operator_aggregates.csv`` — one row per (query, pipeline_id, operator_id).
  Holds the input/output/timing sums attributed to an individual operator.
  Note the per-task input sums (``sum_input_num_rows`` etc.) only make sense at
  the operator level — summing them across a pipeline double-counts data that
  flows from one operator into the next — which is why they live here and not
  in the pipeline table.

See README.md for the rationale on which columns are sum'd vs min/max.
"""

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional


_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


PIPELINE_AGG_COLUMNS = [
    "query_begin_ts",
    "sql_preview",
    "pipeline_id",
    "pipeline_begin",
    "pipeline_end",
    "num_tasks",
    "operator_types",
    # task_outputs (pipeline-wide)
    "sum_output_num_rows",
    "sum_output_size_bytes",
    "sum_output_peak_allocated_bytes",
    "sum_execution_time_ms",
    "max_execution_time_ms",
    # memory_reservations
    "sum_reserved_bytes",
    "min_memory_available",
    "max_total_reserved",
    "max_max_pool",
    # memory_history (prefixed to disambiguate from reservation cols)
    "sum_history_input_basis",
    "sum_history_output_bytes",
    "sum_history_reservation_bytes",
    "sum_history_peak_bytes",
    "sum_history_peak_bytes_to_materialize_input",
]


OPERATOR_AGG_COLUMNS = [
    "query_begin_ts",
    "sql_preview",
    "pipeline_id",
    "operator_id",
    "operator_type",
    "operator_begin",
    "operator_end",
    "num_tasks",
    # task_inputs
    "sum_input_num_rows",
    "sum_input_size_bytes",
    # task_outputs
    "sum_output_num_rows",
    "sum_output_size_bytes",
    "sum_output_peak_allocated_bytes",
    "sum_execution_time_ms",
    "max_execution_time_ms",
]


def _group_by(rows: Iterable[dict], key) -> Dict:
    grouped: Dict = {}
    for r in rows:
        grouped.setdefault(key(r), []).append(r)
    return grouped


def _min_or_none(values):
    vs = [v for v in values if v is not None]
    return min(vs) if vs else None


def _max_or_none(values):
    vs = [v for v in values if v is not None]
    return max(vs) if vs else None


def _count_tasks(rows: List[dict]) -> int:
    """Distinct task_id count, falling back to row count for old logs where
    task_id is None (pre multi-GPU-executor logs carry no task= field)."""
    task_ids = {r["task_id"] for r in rows if r.get("task_id") is not None}
    return len(task_ids) if task_ids else len(rows)


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, _TS_FMT)
    except ValueError:
        return None


def _fmt_ts(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    # Match the log's millisecond precision (3 fractional digits).
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def build_pipeline_rows(
    query_begin_ts: str,
    sql_preview: str,
    task_inputs: List[dict],
    task_outputs: List[dict],
    memory_reservations: List[dict],
    memory_history: List[dict],
) -> List[dict]:
    """Build per-pipeline aggregate rows for a single query."""
    get_pid = lambda r: r["pipeline_id"]
    inputs_by_pid = _group_by(task_inputs, get_pid)
    outputs_by_pid = _group_by(task_outputs, get_pid)
    reservations_by_pid = _group_by(memory_reservations, get_pid)
    history_by_pid = _group_by(memory_history, get_pid)

    pipeline_ids = (
        set(inputs_by_pid)
        | set(outputs_by_pid)
        | set(reservations_by_pid)
        | set(history_by_pid)
    )

    out: List[dict] = []
    for pid in sorted(pipeline_ids):
        inps = inputs_by_pid.get(pid, [])
        outs = outputs_by_pid.get(pid, [])
        res = reservations_by_pid.get(pid, [])
        hist = history_by_pid.get(pid, [])

        pipeline_begin = _min_or_none(r["timestamp"] for r in inps)
        pipeline_end = _max_or_none(r["timestamp"] for r in outs)

        # Comma-joined distinct op types in first-seen order. We source from
        # task_outputs (not task_inputs) because scan pipelines like
        # GPU_PARQUET_SCAN and DUCKDB_SCAN produce data from a source and
        # therefore emit task_outputs but no task_inputs — sourcing from
        # task_inputs would leave operator_types empty for them.
        seen = []
        for r in outs:
            if r["operator_type"] not in seen:
                seen.append(r["operator_type"])
        operator_types = ",".join(seen)

        out.append(
            {
                "query_begin_ts": query_begin_ts,
                "sql_preview": sql_preview,
                "pipeline_id": pid,
                "pipeline_begin": pipeline_begin,
                "pipeline_end": pipeline_end,
                "num_tasks": _count_tasks(outs),
                "operator_types": operator_types,
                "sum_output_num_rows": sum(r["num_rows"] for r in outs),
                "sum_output_size_bytes": sum(r["size_bytes"] for r in outs),
                "sum_output_peak_allocated_bytes": sum(
                    r["peak_allocated_bytes"] for r in outs
                ),
                "sum_execution_time_ms": sum(r["execution_time_ms"] for r in outs),
                "max_execution_time_ms": (
                    max(r["execution_time_ms"] for r in outs) if outs else 0.0
                ),
                "sum_reserved_bytes": sum(r["reserved_bytes"] for r in res),
                "min_memory_available": _min_or_none(
                    r["memory_available"] for r in res
                ),
                "max_total_reserved": _max_or_none(r["total_reserved"] for r in res),
                "max_max_pool": _max_or_none(r["max_pool"] for r in res),
                "sum_history_input_basis": sum(r["input_basis"] for r in hist),
                "sum_history_output_bytes": sum(r["output_bytes"] for r in hist),
                "sum_history_reservation_bytes": sum(
                    r["reservation_bytes"] for r in hist
                ),
                "sum_history_peak_bytes": sum(r["peak_bytes"] for r in hist),
                "sum_history_peak_bytes_to_materialize_input": sum(
                    r["peak_bytes_to_materialize_input"] for r in hist
                ),
            }
        )
    return out


def build_operator_rows(
    query_begin_ts: str,
    sql_preview: str,
    task_inputs: List[dict],
    task_outputs: List[dict],
) -> List[dict]:
    """Build per-operator aggregate rows for a single query.

    Grouped by (pipeline_id, operator_id). Only task_inputs/task_outputs carry
    an operator_id, so the memory-reservation/history figures — which the log
    emits once per task for the whole pipeline chain — are intentionally absent
    here; they live in the pipeline table.

    Rows are returned sorted by (pipeline_id, operator_begin, operator_id) so
    that, once concatenated across queries and sorted by query_begin_ts, the
    operators within each pipeline appear in the order they run.
    """
    get_key = lambda r: (r["pipeline_id"], r["operator_id"])
    inputs_by_op = _group_by(task_inputs, get_key)
    outputs_by_op = _group_by(task_outputs, get_key)

    # Drive the operator set from task_outputs: every operator that does work
    # emits outputs (scans included), whereas source operators emit no inputs.
    op_keys = set(outputs_by_op) | set(inputs_by_op)

    rows: List[dict] = []
    for pid, op_id in op_keys:
        inps = inputs_by_op.get((pid, op_id), [])
        outs = outputs_by_op.get((pid, op_id), [])

        # operator_begin = earliest output timestamp for the operator.
        # operator_end   = latest (timestamp + execution_time) for the operator.
        begin_dt = _min_or_none(_parse_ts(r["timestamp"]) for r in outs)
        end_candidates = []
        for r in outs:
            ts = _parse_ts(r["timestamp"])
            if ts is not None:
                end_candidates.append(
                    ts + timedelta(milliseconds=r["execution_time_ms"])
                )
        end_dt = max(end_candidates) if end_candidates else None

        operator_type = (
            outs[0]["operator_type"]
            if outs
            else (inps[0]["operator_type"] if inps else "")
        )

        rows.append(
            {
                "query_begin_ts": query_begin_ts,
                "sql_preview": sql_preview,
                "pipeline_id": pid,
                "operator_id": op_id,
                "operator_type": operator_type,
                "operator_begin": _fmt_ts(begin_dt),
                "operator_end": _fmt_ts(end_dt),
                "num_tasks": _count_tasks(outs) if outs else _count_tasks(inps),
                "sum_input_num_rows": sum(r["num_rows"] for r in inps),
                "sum_input_size_bytes": sum(r["size_bytes"] for r in inps),
                "sum_output_num_rows": sum(r["num_rows"] for r in outs),
                "sum_output_size_bytes": sum(r["size_bytes"] for r in outs),
                "sum_output_peak_allocated_bytes": sum(
                    r["peak_allocated_bytes"] for r in outs
                ),
                "sum_execution_time_ms": sum(r["execution_time_ms"] for r in outs),
                "max_execution_time_ms": (
                    max(r["execution_time_ms"] for r in outs) if outs else 0.0
                ),
            }
        )

    # Within a query: pipeline_id, then operator run order (operator_begin),
    # then operator_id as a final tiebreak. None timestamps sort last.
    rows.sort(
        key=lambda r: (
            r["pipeline_id"],
            r["operator_begin"] is None,
            r["operator_begin"] or "",
            r["operator_id"],
        )
    )
    return rows
