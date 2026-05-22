"""Parse task-input log lines ("executing on N batches").

Source line example:
  [2026-05-20 14:25:03.674] [trace] [gpu_pipeline_task.cpp:115]
  Pipeline 1: operator TABLE_SCAN (id=12) executing on 1 batches,
  num rows: 20000  , size: 871416 bytes (0.83 MB).

Multi-batch rows: `num rows` is space-separated (e.g. "0  0  1  0").
We sum the values into a single num_rows total; the size is already a
total in bytes.

CSV columns: timestamp, pipeline_id, operator_type, operator_id,
             num_batches, num_rows, size_bytes
"""

from typing import List

from .. import patterns
from ..validators import FormatWarnings


COLUMNS = [
    "timestamp",
    "pipeline_id",
    "operator_type",
    "operator_id",
    "num_batches",
    "num_rows",
    "size_bytes",
]


def parse(lines: List[str], warnings: FormatWarnings) -> List[dict]:
    rows: List[dict] = []
    for line in lines:
        if patterns.TASK_INPUT_ANCHOR not in line:
            continue
        # The anchor matches both task_input and task_output lines; reject
        # task_output lines (those carry "produced") and any line without the
        # "operator " prefix.
        if " produced " in line or " operator " not in line:
            continue
        m = patterns.TASK_INPUT_RE.search(line)
        if not m:
            warnings.record_drift("task_input", line)
            continue
        rows.append(
            {
                "timestamp": m.group("ts"),
                "pipeline_id": int(m.group("pipeline_id")),
                "operator_type": m.group("op_type"),
                "operator_id": int(m.group("op_id")),
                "num_batches": int(m.group("num_batches")),
                "num_rows": patterns.sum_space_separated_ints(m.group("num_rows")),
                "size_bytes": int(m.group("size_bytes")),
            }
        )
    return rows
