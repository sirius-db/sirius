# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""Metrics for the sort-merge benchmark: Sirius log parsers and a GPU memory sampler.

Standard library only. Every parser takes the log text and finds its lines by tag, so the spdlog
prefix (timestamp, level, thread) does not matter. Lines that do not parse are ignored.
"""

from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass

MERGE_SORT_STATS_TAG = "[merge_sort][stats]"
DOWNGRADE_MONITOR_TAG = "[downgrade][monitor]"
OOM_RETRY_TAG = "GPU Pipeline Executor: reschedule (retry"
DISCARD_SINK_TAG = "[discard_sink]"

_KEY_VALUE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(-?\d+)$")

# src/downgrade/downgrade_executor.cpp:
#   "[downgrade][monitor] [{}] request done: {} batches, {} bytes in {:.2f} ms ({:.1f} MB/s), "
#   "satisfied={} | to_host: {}/{} batches/bytes, to_disk: {}/{} batches/bytes"
_DOWNGRADE_RE = re.compile(
    r"request done: (\d+) batches, (\d+) bytes in .*?"
    r"\| to_host: (\d+)/(\d+) batches/bytes, to_disk: (\d+)/(\d+) batches/bytes"
)

# src/pipeline/gpu_pipeline_executor.cpp:
#   "GPU Pipeline Executor: reschedule (retry {}/{}) for task {} (original task {}), "
#   "resuming from operator index {}: {}"
# The reason is the exception message; every oom_reschedule_exception thrown in
# src/pipeline/gpu_pipeline_task.cpp starts its message with "OOM".
_OOM_RETRY_RE = re.compile(
    re.escape(OOM_RETRY_TAG) + r" \d+/\d+\) for task .*?operator index \d+: (.*)$"
)
_OOM_REASON_RE = re.compile(
    r"\bOOM\b|out of memory|out_of_memory|bad_alloc", re.IGNORECASE
)

# src/op/sirius_physical_discard_sink.cpp logs "[discard_sink] rows={} bytes={} order_violations={}"
# followed by further key=value tokens. A line needs these three; the others are read when present.
_DISCARD_SINK_REQUIRED_KEYS = ("rows", "bytes", "order_violations")
# 1 when the sink compared its output's order. Over several sinks it is 1 only if all of them did.
_DISCARD_SINK_ORDER_CHECKED_KEY = "order_checked"

# spdlog line: "[<timestamp>] [<level>] [<thread or file:line>] <message>". The levels kept by
# first_problem_lines; spdlog's other names are trace, debug and info.
_PROBLEM_LEVEL_RE = re.compile(r"^\[[^\]]*\] \[(?:warning|error|critical)\] ")
_DIGITS_RE = re.compile(r"\d+")


def _tagged_remainders(log_text: str, tag: str):
    """Yields, for each line containing `tag`, the text after the tag."""
    for line in log_text.splitlines():
        pos = line.find(tag)
        if pos >= 0:
            yield line[pos + len(tag) :]


def _parse_key_values(text: str) -> dict[str, int] | None:
    """Parses space-separated key=<integer> tokens; None if any token is malformed."""
    values: dict[str, int] = {}
    for token in text.split():
        match = _KEY_VALUE_RE.match(token)
        if match is None:
            return None
        values[match.group(1)] = int(match.group(2))
    return values


@dataclass
class MergeSortStats:
    op: int
    values: dict[str, int]  # every key=value token after the tag except op


def parse_merge_sort_stats(log_text: str) -> list[MergeSortStats]:
    """Returns one entry per '[merge_sort][stats] op=<id> key=value ...' line, in log order."""
    stats: list[MergeSortStats] = []
    for remainder in _tagged_remainders(log_text, MERGE_SORT_STATS_TAG):
        values = _parse_key_values(remainder)
        if values is None or "op" not in values:
            continue
        op = values.pop("op")
        stats.append(MergeSortStats(op=op, values=values))
    return stats


def sum_merge_sort_stats(stats: list[MergeSortStats]) -> dict[str, int]:
    """Sums counters; round_rows_min = min of non-zero mins, round_rows_max = max."""
    totals: dict[str, int] = {}
    for entry in stats:
        for key, value in entry.values.items():
            if key == "round_rows_min":
                current = totals.get(key, 0)
                if value != 0 and (current == 0 or value < current):
                    current = value
                totals[key] = current
            elif key == "round_rows_max":
                totals[key] = max(totals.get(key, 0), value)
            else:
                totals[key] = totals.get(key, 0) + value
    return totals


@dataclass
class DowngradeTotals:
    requests: int
    batches: int
    bytes: int
    to_host_batches: int
    to_host_bytes: int
    to_disk_batches: int
    to_disk_bytes: int


def parse_downgrade_totals(log_text: str) -> DowngradeTotals:
    """Sums the '[downgrade][monitor] ... request done: ...' INFO lines."""
    totals = DowngradeTotals(0, 0, 0, 0, 0, 0, 0)
    for remainder in _tagged_remainders(log_text, DOWNGRADE_MONITOR_TAG):
        match = _DOWNGRADE_RE.search(remainder)
        if match is None:
            continue
        batches, nbytes, host_batches, host_bytes, disk_batches, disk_bytes = (
            int(group) for group in match.groups()
        )
        totals.requests += 1
        totals.batches += batches
        totals.bytes += nbytes
        totals.to_host_batches += host_batches
        totals.to_host_bytes += host_bytes
        totals.to_disk_batches += disk_batches
        totals.to_disk_bytes += disk_bytes
    return totals


def count_oom_retries(log_text: str) -> int:
    """Counts 'GPU Pipeline Executor: reschedule (retry' WARN lines caused by OOM."""
    count = 0
    for line in log_text.splitlines():
        match = _OOM_RETRY_RE.search(line)
        if match is not None and _OOM_REASON_RE.search(match.group(1)):
            count += 1
    return count


def parse_discard_sink(log_text: str) -> dict[str, int] | None:
    """Returns every key=value of the '[discard_sink] rows=… bytes=… order_violations=… ...' line.

    A line without rows, bytes or order_violations is ignored; keys after those three
    (order_checked, ranges_checked) are returned when the line has them. With several such lines
    (one per sink) the values are summed, except order_checked, which is 1 only if every line
    says 1. None if there is no such line.
    """
    totals: dict[str, int] | None = None
    for remainder in _tagged_remainders(log_text, DISCARD_SINK_TAG):
        values = _parse_key_values(remainder)
        if values is None or any(
            key not in values for key in _DISCARD_SINK_REQUIRED_KEYS
        ):
            continue
        if totals is None:
            totals = values
            continue
        for key in totals.keys() | values.keys():
            if key == _DISCARD_SINK_ORDER_CHECKED_KEY:
                totals[key] = min(totals.get(key, 0), values.get(key, 0))
            else:
                totals[key] = totals.get(key, 0) + values.get(key, 0)
    return totals


def first_problem_lines(
    paths: list[str],
    max_lines: int = 200,
    max_per_message: int = 5,
    max_line_chars: int = 2000,
) -> list[str]:
    """Returns the first `max_lines` log lines at warning level or above, in file order.

    The files are streamed line by line and reading stops once `max_lines` are kept, so a log of
    many GiB is never held in memory. A message that repeats (same text once every number is
    masked, e.g. a warning logged per batch) is kept `max_per_message` times only, so it cannot
    fill the result. Lines longer than `max_line_chars` are cut. Unreadable files are skipped.
    """
    kept: list[str] = []
    seen: dict[str, int] = {}  # one entry per kept message shape, so at most max_lines
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as log_file:
                for line in log_file:
                    if _PROBLEM_LEVEL_RE.match(line) is None:
                        continue
                    line = line.rstrip("\n")[:max_line_chars]
                    # The line without its timestamp, numbers masked.
                    shape = _DIGITS_RE.sub("N", line.split("] ", 1)[-1])
                    count = seen.get(shape, 0)
                    if count >= max_per_message:
                        continue
                    seen[shape] = count + 1
                    kept.append(line)
                    if len(kept) >= max_lines:
                        return kept
        except OSError:
            continue
    return kept


class GpuMemorySampler:
    """Keeps the highest `memory.used` printed by one long-lived
    `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i <index> --loop-ms=<ms>`
    process; stop() ends it and returns the peak in MiB (None if nvidia-smi is unavailable).
    """

    def __init__(self, device_index: int, interval_s: float = 1.0):
        self._device_index = device_index
        self._interval_s = interval_s
        self._peak_mib: int | None = None
        self._first_line = threading.Event()
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._process is not None:
            return
        self._peak_mib = None
        self._first_line.clear()
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self._device_index),
                    f"--loop-ms={max(1, round(self._interval_s * 1000))}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return
        self._thread = threading.Thread(
            target=self._read, args=(self._process.stdout,), name="gpu-memory-sampler"
        )
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> int | None:
        if self._process is not None:
            # nvidia-smi prints its first sample when it starts; wait for it so that a run
            # shorter than the interval still has one.
            self._first_line.wait(timeout=10)
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._thread.join()
            self._process.stdout.close()
            self._process = None
            self._thread = None
        return self._peak_mib

    def _read(self, stdout) -> None:
        for line in stdout:
            try:
                used_mib = int(line.split()[0])
            except (IndexError, ValueError):
                used_mib = None
            if used_mib is not None and (
                self._peak_mib is None or used_mib > self._peak_mib
            ):
                self._peak_mib = used_mib
            self._first_line.set()
        # The process ended (or never printed): stop() must not wait for a sample.
        self._first_line.set()
