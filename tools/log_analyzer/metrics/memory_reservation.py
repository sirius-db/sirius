"""Parse memory-reservation log lines.

Source line example:
  [2026-05-20 14:25:03.673] [trace] [gpu_pipeline_executor.cpp:94]
  [GPU:0] GPU Pipeline Executor: Acquiring memory reservation for
  pipeline 1 of 301989888 bytes for task 1027. Memory available:
  7818543104, total reserved: 0, max: 7818543104

The "[GPU:N]" tag was added with the multi-GPU executor split; older
logs omit it and `gpu_id` is None in that case.

CSV columns: timestamp, gpu_id, pipeline_id, task_id, reserved_bytes,
             memory_available, total_reserved, max_pool
"""

from typing import List

from .. import patterns
from ..validators import FormatWarnings


COLUMNS = [
    "timestamp",
    "gpu_id",
    "pipeline_id",
    "task_id",
    "reserved_bytes",
    "memory_available",
    "total_reserved",
    "max_pool",
]


def parse(lines: List[str], warnings: FormatWarnings) -> List[dict]:
    rows: List[dict] = []
    for line in lines:
        if patterns.MEM_RESERVATION_ANCHOR not in line:
            continue
        m = patterns.MEM_RESERVATION_RE.search(line)
        if not m:
            warnings.record_drift("memory_reservation", line)
            continue
        gpu_id = m.group("gpu_id")
        rows.append(
            {
                "timestamp": m.group("ts"),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
                "pipeline_id": int(m.group("pipeline_id")),
                "task_id": int(m.group("task_id")),
                "reserved_bytes": int(m.group("reserved_bytes")),
                "memory_available": int(m.group("memory_available")),
                "total_reserved": int(m.group("total_reserved")),
                "max_pool": int(m.group("max_pool")),
            }
        )
    return rows
