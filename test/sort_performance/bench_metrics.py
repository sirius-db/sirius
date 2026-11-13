# =============================================================================
# Copyright 2025, Sirius Contributors.
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

import os
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

_DISCARD_SINK_KEYS = ("rows", "bytes", "order_violations")


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
    """Returns rows, bytes and order_violations of the '[discard_sink] ...' line.

    With several such lines (one per sink) the values are summed; None if there is none.
    """
    totals: dict[str, int] | None = None
    for remainder in _tagged_remainders(log_text, DISCARD_SINK_TAG):
        values = _parse_key_values(remainder)
        if values is None or any(key not in values for key in _DISCARD_SINK_KEYS):
            continue
        if totals is None:
            totals = dict.fromkeys(_DISCARD_SINK_KEYS, 0)
        for key in _DISCARD_SINK_KEYS:
            totals[key] += values[key]
    return totals


class LogTail:
    """Remembers a file offset; read_new() returns text appended since the last call (handles a missing file).

    Only complete lines are returned: a trailing line still being written stays in the file for
    the next call, so a parser never sees half a line. A file that shrank (truncated or replaced)
    is read again from its start.
    """

    def __init__(self, path: str):
        self._path = path
        self._offset = 0

    def read_new(self) -> str:
        try:
            with open(self._path, "rb") as log_file:
                size = os.fstat(log_file.fileno()).st_size
                if size < self._offset:
                    self._offset = 0
                log_file.seek(self._offset)
                data = log_file.read()
        except OSError:
            return ""
        end = data.rfind(b"\n") + 1
        self._offset += end
        return data[:end].decode("utf-8", errors="replace")


class GpuMemorySampler:
    """Background thread polling `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i <index>`
    every interval_s; stop() returns the peak in MiB (None if nvidia-smi is unavailable).
    """

    def __init__(self, device_index: int, interval_s: float = 0.2):
        self._device_index = device_index
        self._interval_s = interval_s
        self._peak_mib: int | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._peak_mib = None
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="gpu-memory-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> int | None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
        return self._peak_mib

    def _run(self) -> None:
        while True:
            used_mib = self._sample()
            if used_mib is not None and (
                self._peak_mib is None or used_mib > self._peak_mib
            ):
                self._peak_mib = used_mib
            # Checked after the sample so that the last sample is taken at stop().
            if self._stop_event.is_set():
                return
            self._stop_event.wait(self._interval_s)

    def _sample(self) -> int | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self._device_index),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        try:
            return int(result.stdout.split()[0])
        except (IndexError, ValueError):
            return None
