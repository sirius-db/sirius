"""Build the per-(query, pipeline) aggregate table.

Produces one row per (query, pipeline_id) with the column set agreed in the
plan. See README.md for the rationale on which columns are sum'd vs min/max.
"""

from typing import Dict, Iterable, List, Tuple


AGG_COLUMNS = [
    "query_begin_ts",
    "sql_preview",
    "pipeline_id",
    "pipeline_begin",
    "pipeline_end",
    "num_tasks",
    "operator_types",
    # task_inputs
    "sum_input_num_rows",
    "sum_input_size_bytes",
    # task_outputs
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


def _group_by_pipeline(rows: Iterable[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for r in rows:
        grouped.setdefault(r["pipeline_id"], []).append(r)
    return grouped


def _min_or_none(values):
    vs = [v for v in values if v is not None]
    return min(vs) if vs else None


def _max_or_none(values):
    vs = [v for v in values if v is not None]
    return max(vs) if vs else None


def build_rows(
    query_begin_ts: str,
    sql_preview: str,
    task_inputs: List[dict],
    task_outputs: List[dict],
    memory_reservations: List[dict],
    memory_history: List[dict],
) -> List[dict]:
    """Build aggregate rows for a single query, one per pipeline_id seen."""
    inputs_by_pid = _group_by_pipeline(task_inputs)
    outputs_by_pid = _group_by_pipeline(task_outputs)
    reservations_by_pid = _group_by_pipeline(memory_reservations)
    history_by_pid = _group_by_pipeline(memory_history)

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

        row = {
            "query_begin_ts": query_begin_ts,
            "sql_preview": sql_preview,
            "pipeline_id": pid,
            "pipeline_begin": pipeline_begin,
            "pipeline_end": pipeline_end,
            "num_tasks": len(outs),
            "operator_types": operator_types,
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
            "sum_reserved_bytes": sum(r["reserved_bytes"] for r in res),
            "min_memory_available": _min_or_none(r["memory_available"] for r in res),
            "max_total_reserved": _max_or_none(r["total_reserved"] for r in res),
            "max_max_pool": _max_or_none(r["max_pool"] for r in res),
            "sum_history_input_basis": sum(r["input_basis"] for r in hist),
            "sum_history_output_bytes": sum(r["output_bytes"] for r in hist),
            "sum_history_reservation_bytes": sum(r["reservation_bytes"] for r in hist),
            "sum_history_peak_bytes": sum(r["peak_bytes"] for r in hist),
            "sum_history_peak_bytes_to_materialize_input": sum(
                r["peak_bytes_to_materialize_input"] for r in hist
            ),
        }
        out.append(row)
    return out
