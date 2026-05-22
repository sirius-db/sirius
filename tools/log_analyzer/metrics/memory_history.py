"""Parse memory-history-record log lines.

Source line example:
  [2026-05-20 14:25:03.674] [trace] [gpu_pipeline_task.cpp:425]
  Pipeline 1: memory history record - input_basis=100663296,
  output_bytes=871416, reservation_bytes=301989888, peak_bytes=0,
  peak_bytes_to_materialize_input=100663296

CSV columns: timestamp, pipeline_id, input_basis, output_bytes,
             reservation_bytes, peak_bytes, peak_bytes_to_materialize_input
"""

from typing import List

from .. import patterns
from ..validators import FormatWarnings


COLUMNS = [
    "timestamp",
    "pipeline_id",
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
        rows.append(
            {
                "timestamp": m.group("ts"),
                "pipeline_id": int(m.group("pipeline_id")),
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
