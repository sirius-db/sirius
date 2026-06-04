"""Parse memory-history-record log lines.

Source line example:
  [2026-05-20 14:25:03.674] [trace] [gpu_pipeline_task.cpp:425]
  [GPU:0] Pipeline 1: memory history record - task=2,
  input_basis=100663296, output_bytes=871416, reservation_bytes=301989888,
  peak_bytes=0, peak_bytes_to_materialize_input=100663296

The "[GPU:N]" tag and `task=` field were added with the multi-GPU
executor split; both are optional so older logs still parse (gpu_id /
task_id are None in that case).

CSV columns: timestamp, gpu_id, pipeline_id, task_id, input_basis,
             output_bytes, reservation_bytes, peak_bytes,
             peak_bytes_to_materialize_input
"""

from typing import List

from .. import patterns
from ..validators import FormatWarnings


COLUMNS = [
    "timestamp",
    "gpu_id",
    "pipeline_id",
    "task_id",
    "input_basis",
    "output_bytes",
    "reservation_bytes",
    "peak_bytes",
    "peak_bytes_to_materialize_input",
]


def parse(lines: List[str], warnings: FormatWarnings) -> List[dict]:
    rows: List[dict] = []
    for line in lines:
        if patterns.MEM_HISTORY_ANCHOR not in line:
            continue
        m = patterns.MEM_HISTORY_RE.search(line)
        if not m:
            warnings.record_drift("memory_history", line)
            continue
        gpu_id = m.group("gpu_id")
        task_id = m.group("task_id")
        rows.append(
            {
                "timestamp": m.group("ts"),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
                "pipeline_id": int(m.group("pipeline_id")),
                "task_id": int(task_id) if task_id is not None else None,
                "input_basis": int(m.group("input_basis")),
                "output_bytes": int(m.group("output_bytes")),
                "reservation_bytes": int(m.group("reservation_bytes")),
                "peak_bytes": int(m.group("peak_bytes")),
                "peak_bytes_to_materialize_input": int(
                    m.group("peak_bytes_to_materialize_input")
                ),
            }
        )
    return rows
