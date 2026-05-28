"""Parse downgrade-summary log lines (one row per satisfied downgrade request).

Source line example:
  [2026-05-28 10:36:55.124] [debug] [downgrade_executor.cpp:348]
  [downgrade] [GPU:0] request monitor done: 3 batches, 2112 bytes in
  6.04 ms (0.3 MB/s) | repos: 3/2112 batches/bytes, pipeline_queue: 0/0
  | to_host: 3/2112 batches/bytes, to_disk: 0/0 batches/bytes

Each row is emitted by `downgrade_executor` after a single downgrade request
finishes. The "source" identifies WHERE the data was downgraded *from*:

  source_tier=GPU,  source_device_id=N   → evicted from GPU N (typical case)
  source_tier=HOST, source_device_id=-1  → evicted from host (typically to disk)
  source_tier=DISK, source_device_id=N   → reserved but no current emitter

The "to_host" / "to_disk" pair tells you where the data WENT.

`is_monitor` distinguishes background-pressure requests (the downgrade
executor's monitor_loop) from caller-initiated requests (a pipeline task
asking for free memory because it OOM'd or expected to OOM).

The "repos vs pipeline_queue" pair attributes the freed bytes to either
batches resident in the data repositories or batches still queued for an
upstream pipeline that hadn't run yet.

CSV columns: timestamp, source_tier, source_device_id, is_monitor,
             total_batches, total_bytes, duration_ms, throughput_mbs,
             repos_batches, repos_bytes,
             pipeline_queue_batches, pipeline_queue_bytes,
             to_host_batches, to_host_bytes,
             to_disk_batches, to_disk_bytes
"""

from typing import List

from .. import patterns
from ..validators import FormatWarnings


COLUMNS = [
    "timestamp",
    "source_tier",
    "source_device_id",
    "is_monitor",
    "total_batches",
    "total_bytes",
    "duration_ms",
    "throughput_mbs",
    "repos_batches",
    "repos_bytes",
    "pipeline_queue_batches",
    "pipeline_queue_bytes",
    "to_host_batches",
    "to_host_bytes",
    "to_disk_batches",
    "to_disk_bytes",
]


def parse(lines: List[str], warnings: FormatWarnings) -> List[dict]:
    rows: List[dict] = []
    for line in lines:
        if patterns.DOWNGRADE_SUMMARY_ANCHOR not in line:
            continue
        # The anchor catches every "[downgrade] [..." line; the WARN line
        # ("downgrade request not satisfied and disk memory space is not
        # configured") and the error lines ("convert failed") don't match the
        # strict regex. Only treat the line as drift if it looks like a
        # summary (contains " request " AND " done: ").
        if " request " not in line or " done: " not in line:
            continue
        m = patterns.DOWNGRADE_SUMMARY_RE.search(line)
        if not m:
            warnings.record_drift("downgrade", line)
            continue
        rows.append(
            {
                "timestamp": m.group("ts"),
                "source_tier": m.group("source_tier"),
                "source_device_id": int(m.group("source_device_id")),
                "is_monitor": m.group("monitor") is not None,
                "total_batches": int(m.group("total_batches")),
                "total_bytes": int(m.group("total_bytes")),
                "duration_ms": float(m.group("duration_ms")),
                "throughput_mbs": float(m.group("throughput_mbs")),
                "repos_batches": int(m.group("repos_batches")),
                "repos_bytes": int(m.group("repos_bytes")),
                "pipeline_queue_batches": int(m.group("pipeline_queue_batches")),
                "pipeline_queue_bytes": int(m.group("pipeline_queue_bytes")),
                "to_host_batches": int(m.group("to_host_batches")),
                "to_host_bytes": int(m.group("to_host_bytes")),
                "to_disk_batches": int(m.group("to_disk_batches")),
                "to_disk_bytes": int(m.group("to_disk_bytes")),
            }
        )
    return rows
