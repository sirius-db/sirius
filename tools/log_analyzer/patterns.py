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

SHAPE_VERSION = "1.0"

# Timestamp at the start of every log line, e.g. "[2026-05-20 14:25:02.368]"
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]")

# --- Level tags used by the trace-level check --------------------------------
TRACE_TAG = "[trace]"
DEBUG_TAG = "[debug]"

# --- Query boundaries (info-level) -------------------------------------------
# Example: "[2026-05-20 14:25:02.368] [info] [:] [query_pool] QueryBegin allocated=5242880 bytes peak=307232768 bytes free_blocks=5115"
QUERY_BEGIN_ANCHOR = "[info] [:] [query_pool] QueryBegin allocated="
QUERY_END_ANCHOR = "[info] [:] [query_pool] QueryEnd allocated="

# Example: "[2026-05-20 14:25:02.368] [info] [:] QueryBegin: with revenue_view as ..."
QUERY_SQL_ANCHOR = "[info] [:] QueryBegin: "
QUERY_SQL_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[info\] \[:\] QueryBegin: (?P<sql>.*)$"
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
# "[ts] [trace] [gpu_pipeline_executor.cpp:94] GPU Pipeline Executor: Acquiring memory reservation for pipeline 1 of 301989888 bytes for task 2. Memory available: 50986745856, total reserved: 0, max: 50986745856"
MEM_RESERVATION_ANCHOR = "Acquiring memory reservation for pipeline"
MEM_RESERVATION_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[trace\] \[gpu_pipeline_executor\.cpp:\d+\] "
    r"GPU Pipeline Executor: Acquiring memory reservation for pipeline (?P<pipeline_id>\d+) "
    r"of (?P<reserved_bytes>\d+) bytes for task (?P<task_id>\d+)\. "
    r"Memory available: (?P<memory_available>\d+), total reserved: (?P<total_reserved>\d+), "
    r"max: (?P<max_pool>\d+)"
)

# --- Metric: task input (executing on N batches) -----------------------------
# "[ts] [trace] [gpu_pipeline_task.cpp:115] Pipeline 1: operator TABLE_SCAN (id=5) executing on 1 batches, num rows: 178176  , size: 9621520 bytes (9.18 MB). "
#
# Note: a different log line "Pipeline N: operator X (id=N) executing on
# non-pipelineable data." also matches the word "executing on". We use the more
# specific "batches, num rows:" anchor so those lines are silently ignored
# instead of triggering format-drift warnings.
TASK_INPUT_ANCHOR = "batches, num rows:"
TASK_INPUT_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[trace\] \[gpu_pipeline_task\.cpp:\d+\] "
    r"Pipeline (?P<pipeline_id>\d+): operator (?P<op_type>\w+) \(id=(?P<op_id>\d+)\) "
    r"executing on (?P<num_batches>\d+) batches, num rows: (?P<num_rows>[\d\s]+?)\s*, "
    r"size: (?P<size_bytes>\d+) bytes"
)

# --- Metric: task output (produced N batches) --------------------------------
# "[ts] [trace] [gpu_pipeline_task.cpp:115] Pipeline 1: operator TABLE_SCAN (id=5) produced 1 batches, num rows: 93150  , size: 3912308 bytes (3.73 MB). execution time: 8.87 ms, peak allocated: 10336768 bytes (9.86 MB)"
#
# Note: a different log line from parquet_scan_task.cpp produces a similar
# but distinct format ("produced N batches with num rows: M, execution time: ..."
# — no `size:` field, no `peak allocated:`). We anchor on `peak allocated:` so
# only the full gpu_pipeline_task.cpp:115 form is captured here.
TASK_OUTPUT_ANCHOR = "peak allocated:"
TASK_OUTPUT_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[trace\] \[gpu_pipeline_task\.cpp:\d+\] "
    r"Pipeline (?P<pipeline_id>\d+): operator (?P<op_type>\w+) \(id=(?P<op_id>\d+)\) "
    r"produced (?P<num_batches>\d+) batches, num rows: (?P<num_rows>[\d\s]+?)\s*, "
    r"size: (?P<size_bytes>\d+) bytes \([\d.]+ MB\)\. "
    r"execution time: (?P<execution_time_ms>[\d.]+) ms, "
    r"peak allocated: (?P<peak_allocated_bytes>\d+) bytes"
)

# --- Metric: memory history record -------------------------------------------
# "[ts] [trace] [gpu_pipeline_task.cpp:425] Pipeline 1: memory history record - input_basis=100663296, output_bytes=576, reservation_bytes=301989888, peak_bytes=0, peak_bytes_to_materialize_input=100663296"
MEM_HISTORY_ANCHOR = "memory history record"
MEM_HISTORY_RE = re.compile(
    r"\[(?P<ts>[\d\-: .]+)\] \[trace\] \[gpu_pipeline_task\.cpp:\d+\] "
    r"Pipeline (?P<pipeline_id>\d+): memory history record - "
    r"input_basis=(?P<input_basis>\d+), "
    r"output_bytes=(?P<output_bytes>\d+), "
    r"reservation_bytes=(?P<reservation_bytes>\d+), "
    r"peak_bytes=(?P<peak_bytes>\d+), "
    r"peak_bytes_to_materialize_input=(?P<peak_bytes_to_materialize_input>\d+)"
)


def sum_space_separated_ints(text: str) -> int:
    """Multi-batch num_rows is space-separated (e.g. '0  0  1  0'). Sum them."""
    return sum(int(x) for x in text.split())
