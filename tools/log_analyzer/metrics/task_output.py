"""Parse task-output log lines ("produced N batches").

Source line example:
  [2026-05-20 14:25:03.674] [trace] [gpu_pipeline_task.cpp:115]
  [GPU:0] Pipeline 1: operator TABLE_SCAN (id=12) task=2 produced 1
  batches, num rows: 20000  , size: 871416 bytes (0.83 MB).
  execution time: 0.01 ms, peak allocated: 872192 bytes (0.83 MB)

The "[GPU:N]" tag and `task=` field were added with the multi-GPU
executor split; both are optional so older logs still parse (gpu_id /
task_id are None in that case).

CSV columns: timestamp, gpu_id, pipeline_id, task_id, operator_type,
             operator_id, num_batches, num_rows, size_bytes,
             execution_time_ms, peak_allocated_bytes
"""

from typing import List

from .. import patterns
from ..validators import FormatWarnings


COLUMNS = [
    "timestamp",
    "gpu_id",
    "pipeline_id",
    "task_id",
    "operator_type",
    "operator_id",
    "num_batches",
    "num_rows",
    "size_bytes",
    "execution_time_ms",
    "peak_allocated_bytes",
]


def parse(lines: List[str], warnings: FormatWarnings) -> List[dict]:
    rows: List[dict] = []
    for line in lines:
        if patterns.TASK_OUTPUT_ANCHOR not in line:
            continue
        if " produced " not in line or " operator " not in line:
            continue
        # The parquet_scan_task.cpp variant is filtered out by the anchor
        # (it has no "peak allocated:" segment).
        m = patterns.TASK_OUTPUT_RE.search(line)
        if not m:
            warnings.record_drift("task_output", line)
            continue
        gpu_id = m.group("gpu_id")
        task_id = m.group("task_id")
        rows.append(
            {
                "timestamp": m.group("ts"),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
                "pipeline_id": int(m.group("pipeline_id")),
                "task_id": int(task_id) if task_id is not None else None,
                "operator_type": m.group("op_type"),
                "operator_id": int(m.group("op_id")),
                "num_batches": int(m.group("num_batches")),
                "num_rows": patterns.sum_space_separated_ints(m.group("num_rows")),
                "size_bytes": int(m.group("size_bytes")),
                "execution_time_ms": float(m.group("execution_time_ms")),
                "peak_allocated_bytes": int(m.group("peak_allocated_bytes")),
            }
        )
    return rows
