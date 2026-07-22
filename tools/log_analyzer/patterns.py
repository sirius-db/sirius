"""Central registry of log-line anchors and strict regexes.

When the Sirius log format drifts, this file is the single place to update.
Each pattern has two parts:

  * LOOSE_PREFIX: a static substring used to cheaply detect "this line is
    probably trying to be one of ours". If we see the loose prefix but the
    strict regex fails, that signals format drift and we emit a warning.
  * RE: the strict regex that captures fields.

SHAPE_VERSION should be bumped whenever a pattern is changed so the skill
that consumes this data can detect mismatched parser versions.
"""

import re

SHAPE_VERSION = "1.6"

# Timestamp at the start of every log line, e.g. "[2026-05-20 14:25:02.368]"
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]")

# --- Level tags used by the trace-level check --------------------------------
TRACE_TAG = "[trace]"
DEBUG_TAG = "[debug]"

# --- Query boundaries (info-level) -------------------------------------------
# Example: "[2026-06-10 19:52:14.000] [info] [:] [host_pool] HOST:0 QueryBegin allocated=5242880 bytes peak=307232768 bytes free_blocks=5115"
# HOST:{device_id} was added in the DRY-cleanup refactor (commit e863fd4c).
# The device_id varies by system so we use helper predicates (below) instead
# of hardcoding it in anchor strings.
#
# Strict regex to extract stats from the host-pool boundary lines.
# HOST:{device_id} group is optional so older logs (pre-e863fd4c) still parse.
HOST_POOL_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[info\] \[[^\]]+\] \[host_pool\] "
    r"(?:HOST:(?P<host_device_id>-?\d+) )?"
    r"(?P<tag>QueryBegin|QueryEnd) "
    r"allocated=(?P<allocated_bytes>\d+) bytes "
    r"peak=(?P<peak_bytes>\d+) bytes "
    r"free_blocks=(?P<free_blocks>\d+)"
)


# Predicates used by segmenter.py and parse_logs.py to detect query-boundary
# lines without hardcoding the HOST device_id.
def is_query_begin_line(line: str) -> bool:
    return "[host_pool]" in line and "QueryBegin allocated=" in line


def is_query_end_line(line: str) -> bool:
    return "[host_pool]" in line and "QueryEnd allocated=" in line


# --- GPU device memory pool stats at query boundaries (info-level) -----------
# Example: "[2026-06-10 10:00:00.000] [info] [:] [gpu_pool] GPU:0 QueryBegin allocated=1234567890 bytes peak=1234567890 bytes reserved=1234567890 bytes"
# One line per configured GPU; tag is QueryBegin or QueryEnd.
GPU_POOL_ANCHOR = "[info] [:] [gpu_pool]"
GPU_POOL_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[info\] \[[^\]]+\] \[gpu_pool\] "
    r"GPU:(?P<gpu_id>\d+) (?P<tag>QueryBegin|QueryEnd) "
    r"allocated=(?P<allocated_bytes>\d+) bytes "
    r"peak=(?P<peak_bytes>\d+) bytes "
    r"reserved=(?P<reserved_bytes>\d+) bytes"
)


def is_pool_boundary_line(line: str) -> bool:
    """True for the per-HOST / per-GPU memory-pool stat lines emitted at query
    boundaries (both QueryBegin and QueryEnd).

    These lines sit between the `[host_pool] QueryBegin` anchor and the
    `QueryBegin: SQL:` line. There is one per HOST node and one per GPU, so on
    multi-GPU systems the block spans many lines — the segmenter skips over it
    to find the SQL line rather than assuming a fixed lookahead. The SQL line
    itself is not matched here (it carries neither pool tag).
    """
    return ("[host_pool]" in line or "[gpu_pool]" in line) and (
        "QueryBegin " in line or "QueryEnd " in line
    )


# Example: "[2026-06-11 ...] [info] [sirius_context.cpp:246] QueryBegin: SQL: select ..."
# "SQL: " was added to distinguish the SQL line from the pool-stat lines that
# also begin with "QueryBegin" (e.g. "[host_pool] HOST:-1 QueryBegin allocated=").
QUERY_SQL_ANCHOR = "QueryBegin: SQL: "
QUERY_SQL_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[info\] \[[^\]]+\] QueryBegin: SQL: (?P<sql>.*)$"
)

# --- Pipeline Overview block -------------------------------------------------
PIPELINE_OVERVIEW_HEADER = "=== Pipeline Overview ==="
PIPELINE_OVERVIEW_END_MARKERS = (
    "=== Query Plan DAG ===",  # the block immediately following in current logs
)

# "Pipeline #8: HASH_JOIN (id=13) -> FILTER (id=14) -> ..."
PIPELINE_HEADER_RE = re.compile(r"^Pipeline #(\d+): (.+)$")
# Single operator within a pipeline header chain: "HASH_JOIN (id=13)"
OPERATOR_RE = re.compile(r"(\w+) \(id=(\d+)\)")
# "  Input: <- Pipeline #7 [on: HASH_JOIN (id=13), port: default, barrier: FULL]"
PIPELINE_INPUT_RE = re.compile(
    r"^\s*Input:\s*<-\s*Pipeline #(\d+) \[on: (\w+) \(id=(\d+)\), port: (\w+), barrier: (\w+)\]"
)
# "  Dependencies: Pipeline #3, Pipeline #7"
PIPELINE_DEPS_RE = re.compile(r"^\s*Dependencies:\s*(.+)$")
# "  Output: -> HASH_JOIN [port: build, barrier: FULL]"
PIPELINE_OUTPUT_RE = re.compile(
    r"^\s*Output:\s*->\s*(\w+) \[port: (\w+), barrier: (\w+)\]"
)

# --- Metric: memory reservation ----------------------------------------------
# "[ts] [trace] [gpu_pipeline_executor.cpp:94] [GPU:0] GPU Pipeline Executor: Acquiring memory reservation for pipeline 1 of 301989888 bytes for task 2. Memory available: 50986745856, total reserved: 0, max: 50986745856"
#
# The "[GPU:N]" tag was added with the multi-GPU executor split; older logs
# omit it. The optional group keeps the parser backward-compatible.
MEM_RESERVATION_ANCHOR = "Acquiring memory reservation for pipeline"
MEM_RESERVATION_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[trace\] \[gpu_pipeline_executor\.cpp:\d+\] "
    r"(?:\[GPU:(?P<gpu_id>\d+)\] )?"
    r"GPU Pipeline Executor: Acquiring memory reservation for pipeline (?P<pipeline_id>\d+) "
    r"of (?P<reserved_bytes>\d+) bytes for task (?P<task_id>\d+)\. "
    r"Memory available: (?P<memory_available>\d+), total reserved: (?P<total_reserved>\d+), "
    r"max: (?P<max_pool>\d+)"
)

# --- Metric: task input (executing on N batches) -----------------------------
# "[ts] [trace] [gpu_pipeline_task.cpp:115] [GPU:0] Pipeline 1: operator TABLE_SCAN (id=5) task=2 executing on 1 batches, num rows: 178176  , size: 9621520 bytes (9.18 MB). "
#
# Note: a different log line "Pipeline N: operator X (id=N) task=T executing
# on non-pipelineable data." also matches the word "executing on". We use the
# more specific "batches, num rows:" anchor so those lines are silently
# ignored instead of triggering format-drift warnings.
#
# The "[GPU:N]" tag and `task=` field were added with the multi-GPU executor
# split; both are optional so older logs still parse.
TASK_INPUT_ANCHOR = "batches, num rows:"
TASK_INPUT_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[trace\] \[gpu_pipeline_task\.cpp:\d+\] "
    r"(?:\[GPU:(?P<gpu_id>\d+)\] )?"
    r"Pipeline (?P<pipeline_id>\d+): operator (?P<op_type>\w+) \(id=(?P<op_id>\d+)\) "
    r"(?:task=(?P<task_id>\d+) )?"
    r"executing on (?P<num_batches>\d+) batches, num rows: (?P<num_rows>[\d\s]+?)\s*, "
    r"size: (?P<size_bytes>\d+) bytes"
)

# --- Metric: task output (produced N batches) --------------------------------
# "[ts] [trace] [gpu_pipeline_task.cpp:115] [GPU:0] Pipeline 1: operator TABLE_SCAN (id=5) task=2 produced 1 batches, num rows: 93150  , size: 3912308 bytes (3.73 MB). execution time: 8.87 ms, peak allocated: 10336768 bytes (9.86 MB)"
#
# Note: a different log line from parquet_scan_task.cpp produces a similar
# but distinct format ("produced N batches with num rows: M, execution time: ..."
# — no `size:` field, no `peak allocated:`). We anchor on `peak allocated:` so
# only the full gpu_pipeline_task.cpp:115 form is captured here.
#
# The "[GPU:N]" tag and `task=` field were added with the multi-GPU executor
# split; both are optional so older logs still parse.
TASK_OUTPUT_ANCHOR = "peak allocated:"
TASK_OUTPUT_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[trace\] \[gpu_pipeline_task\.cpp:\d+\] "
    r"(?:\[GPU:(?P<gpu_id>\d+)\] )?"
    r"Pipeline (?P<pipeline_id>\d+): operator (?P<op_type>\w+) \(id=(?P<op_id>\d+)\) "
    r"(?:task=(?P<task_id>\d+) )?"
    r"produced (?P<num_batches>\d+) batches, num rows: (?P<num_rows>[\d\s]+?)\s*, "
    r"size: (?P<size_bytes>\d+) bytes \([\d.]+ MB\)\. "
    r"execution time: (?P<execution_time_ms>[\d.]+) ms, "
    r"peak allocated: (?P<peak_allocated_bytes>\d+) bytes"
)

# --- Metric: memory history record -------------------------------------------
# "[ts] [trace] [gpu_pipeline_task.cpp:425] [GPU:0] Pipeline 1: memory history record - task=2, input_basis=100663296, output_bytes=576, reservation_bytes=301989888, peak_bytes=0, peak_bytes_to_materialize_input=100663296"
#
# The "[GPU:N]" tag and `task=` field were added with the multi-GPU executor
# split; both are optional so older logs still parse.
MEM_HISTORY_ANCHOR = "memory history record"
MEM_HISTORY_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[trace\] \[gpu_pipeline_task\.cpp:\d+\] "
    r"(?:\[GPU:(?P<gpu_id>\d+)\] )?"
    r"Pipeline (?P<pipeline_id>\d+): memory history record - "
    r"(?:task=(?P<task_id>\d+), )?"
    r"input_basis=(?P<input_basis>\d+), "
    r"output_bytes=(?P<output_bytes>\d+), "
    r"reservation_bytes=(?P<reservation_bytes>\d+), "
    r"peak_bytes=(?P<peak_bytes>\d+), "
    r"peak_bytes_to_materialize_input=(?P<peak_bytes_to_materialize_input>\d+)"
)


# --- Metric: downgrade summary (per-request) ---------------------------------
# Emitted once per satisfied downgrade request by downgrade_executor.cpp:348.
# Two label variants:
#   "request done: ..."          (caller-initiated request_free_memory / request_downgrade)
#   "request monitor done: ..."  (background monitor_loop pressure-driven request)
# The source label is "{tier}:{device_id}" where tier is GPU|HOST|DISK and
# device_id is an int (HOST has device_id=-1 because it's not GPU-specific).
#
# Example:
#   [2026-05-28 10:36:55.124] [debug] [downgrade_executor.cpp:348] [downgrade]
#   [GPU:0] request monitor done: 3 batches, 2112 bytes in 6.04 ms (0.3 MB/s)
#   | repos: 3/2112 batches/bytes, pipeline_queue: 0/0
#   | to_host: 3/2112 batches/bytes, to_disk: 0/0 batches/bytes
DOWNGRADE_SUMMARY_ANCHOR = "[downgrade] ["
DOWNGRADE_SUMMARY_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[debug\] \[downgrade_executor\.cpp:\d+\] "
    r"\[downgrade\] \[(?P<source_tier>GPU|HOST|DISK):(?P<source_device_id>-?\d+)\] "
    r"request (?P<monitor>monitor )?done: "
    r"(?P<total_batches>\d+) batches, (?P<total_bytes>\d+) bytes in "
    r"(?P<duration_ms>[\d.]+) ms \((?P<throughput_mbs>[\d.]+) MB/s\) \| "
    r"repos: (?P<repos_batches>\d+)/(?P<repos_bytes>\d+) batches/bytes, "
    r"pipeline_queue: (?P<pipeline_queue_batches>\d+)/(?P<pipeline_queue_bytes>\d+) \| "
    r"to_host: (?P<to_host_batches>\d+)/(?P<to_host_bytes>\d+) batches/bytes, "
    r"to_disk: (?P<to_disk_batches>\d+)/(?P<to_disk_bytes>\d+) batches/bytes"
)


def sum_space_separated_ints(text: str) -> int:
    """Multi-batch num_rows is space-separated (e.g. '0  0  1  0'). Sum them."""
    return sum(int(x) for x in text.split())
