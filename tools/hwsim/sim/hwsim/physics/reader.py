"""Defensive reader for an nsys sqlite export (nsys 2025.6.3 schema).

Schema facts per tools/hwsim/docs/nsys-extraction.md section 2.1. Everything
here is written to survive first contact with a real export:

- required tables/columns are checked up front with a clear error listing what
  *was* found;
- optional tables (memcpy/memset/sync/enums/session-start/gpu-metrics) degrade
  to diagnostics notes, never crashes;
- enum labels fall back to the standard CUPTI numeric values when the ENUM_*
  tables are absent;
- NVTX label parsing is done in Python (regex), not SQL string slicing.

All timestamps are integer ns relative to session start (nsys convention).
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class NsysReadError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Row containers
# --------------------------------------------------------------------------


@dataclass
class KernelRow:
    start: int
    end: int
    stream: int
    device: int
    correlation: int
    name: str
    launch_start: int = -1  # from the RUNTIME row (same correlationId)
    launch_end: int = -1
    global_tid: int = -1

    @property
    def dur_ns(self) -> int:
        return self.end - self.start


@dataclass
class MemcpyRow:
    start: int
    end: int
    stream: int
    device: int
    correlation: int
    bytes: int
    direction: str  # 'Host-to-Device' | 'Device-to-Host' | 'Device-to-Device'...
    src_kind: str  # 'Pageable' | 'Pinned' | 'Device' ...
    dst_kind: str
    launch_start: int = -1
    launch_end: int = -1
    global_tid: int = -1

    @property
    def dur_ns(self) -> int:
        return self.end - self.start

    @property
    def channel(self) -> str:
        return f"{self.direction}|{self.src_kind}|{self.dst_kind}"


@dataclass
class HostSpanRow:
    """A host-side runtime API span of interest (sync or launch)."""

    start: int
    end: int
    global_tid: int
    api: str
    kind: str  # 'sync' | 'launch'


@dataclass
class TaskRangeRow:
    pipeline_id: int
    task_id: int
    start: int
    end: int
    global_tid: int


@dataclass
class OpRangeRow:
    pipeline_id: int
    op_name: str
    op_id: int
    is_sink: bool
    start: int
    end: int
    global_tid: int


@dataclass
class NsysData:
    path: str
    session_start_utc_ns: Optional[int]
    query_windows: List[Tuple[int, int]]
    task_ranges: List[TaskRangeRow]
    op_ranges: List[OpRangeRow]
    pipeline_spans: List[Tuple[int, int, int]]  # (pipeline_id, start, end)
    clock_beacons: List[Tuple[int, int]]  # (nsys_ns, epoch_ns) from marks
    kernels: List[KernelRow]
    memcpys: List[MemcpyRow]
    host_spans: List[HostSpanRow]
    gpu_metrics: List[Tuple[int, str, float]]  # (timestamp, metricName, value)
    notes: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Fallback enum maps (standard CUPTI values) — used only when the ENUM_*
# tables are missing from the export.
# --------------------------------------------------------------------------

_FALLBACK_MEMCPY_OPER = {
    0: "Unknown",
    1: "Host-to-Device",
    2: "Device-to-Host",
    3: "Host-to-Array",
    4: "Array-to-Host",
    5: "Array-to-Array",
    6: "Array-to-Device",
    7: "Device-to-Array",
    8: "Device-to-Device",
    9: "Host-to-Host",
    10: "Peer-to-Peer",
}
_FALLBACK_MEM_KIND = {
    0: "Unknown",
    1: "Pageable",
    2: "Pinned",
    3: "Device",
    4: "Array",
    5: "Managed",
    6: "Device Static",
    7: "Managed Static",
}

_SYNC_API_PREFIXES = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cuStreamSynchronize",
    "cuCtxSynchronize",
    "cudaMemcpy_v",  # synchronous memcpy variants block the host too
)
_LAUNCH_API_PREFIXES = ("cudaLaunchKernel", "cuLaunchKernel")

# NVTX label formats (source of truth: nsys-extraction.md section 2.2a)
_TASK_RE = re.compile(r"^Pipeline (\d+) Task (\d+) \[")
_OP_RE = re.compile(r"^Pipeline (\d+): (.+?) \(id=(\d+)\)( sink)?$")
_PIPE_SPAN_RE = re.compile(r"^Pipeline (\d+):.*->")
_BEACON_RE = re.compile(r"^sirius::clock_sync:(\d+)$")

REQUIRED_TABLES = (
    "NVTX_EVENTS",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "StringIds",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _open_ro(path: str) -> sqlite3.Connection:
    if not os.path.isfile(path):
        raise NsysReadError(f"nsys sqlite not found: {path}")
    with open(path, "rb") as f:
        magic = f.read(16)
    if not magic.startswith(b"SQLite format 3"):
        raise NsysReadError(
            f"{path} is not a sqlite database (did you pass the .nsys-rep? "
            "run `nsys export --type sqlite <rep>` first)"
        )
    uri = f"file:{path}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _tables(conn: sqlite3.Connection) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }


def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _load_enum(
    conn: sqlite3.Connection, tables: set, name: str, fallback: Dict[int, str]
) -> Tuple[Dict[int, str], Optional[str]]:
    if name not in tables:
        return dict(fallback), f"{name} table missing; using standard CUPTI values"
    cols = _columns(conn, name)
    label_col = "label" if "label" in cols else ("name" if "name" in cols else None)
    if label_col is None:
        return dict(fallback), f"{name} has no label column; using CUPTI values"
    return {
        int(r[0]): str(r[1])
        for r in conn.execute(f"SELECT id, {label_col} FROM {name}")
    }, None


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------


def read_nsys(path: str) -> NsysData:
    conn = _open_ro(path)
    try:
        return _read(conn, path)
    finally:
        conn.close()


def _read(conn: sqlite3.Connection, path: str) -> NsysData:
    notes: List[str] = []
    counts: Dict[str, int] = {}
    tables = _tables(conn)

    missing = [t for t in REQUIRED_TABLES if t not in tables]
    if missing:
        raise NsysReadError(
            f"required tables missing from {path}: {', '.join(missing)}. "
            f"Found {len(tables)} tables: {', '.join(sorted(tables)[:20])}... "
            "Expected an nsys 2025.x sqlite export with --trace=cuda,nvtx."
        )

    # ---- session clock anchor ------------------------------------------
    session_start_utc: Optional[int] = None
    if "TARGET_INFO_SESSION_START_TIME" in tables:
        cols = _columns(conn, "TARGET_INFO_SESSION_START_TIME")
        if "utcEpochNs" in cols:
            row = conn.execute(
                "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME LIMIT 1"
            ).fetchone()
            if row and row[0] is not None:
                session_start_utc = int(row[0])
        else:
            notes.append(
                "TARGET_INFO_SESSION_START_TIME has no utcEpochNs column "
                f"(has: {cols}); clock anchor unavailable"
            )
    else:
        notes.append(
            "TARGET_INFO_SESSION_START_TIME missing; clock-domain alignment "
            "will be structural-only"
        )

    # ---- NVTX ------------------------------------------------------------
    nvtx_cols = _columns(conn, "NVTX_EVENTS")
    text_expr = "e.text"
    join_sid = ""
    if "textId" in nvtx_cols:
        text_expr = "COALESCE(e.text, s.value)"
        join_sid = "LEFT JOIN StringIds s ON s.id = e.textId"
    dom_expr = "COALESCE(e.domainId, 0)" if "domainId" in nvtx_cols else "0"
    gtid_expr = "e.globalTid" if "globalTid" in nvtx_cols else "NULL"

    query_windows: List[Tuple[int, int]] = []
    task_ranges: List[TaskRangeRow] = []
    op_ranges: List[OpRangeRow] = []
    pipeline_spans: List[Tuple[int, int, int]] = []
    beacons: List[Tuple[int, int]] = []
    n_nvtx = 0
    for start, end, etype, dom, text, gtid in conn.execute(
        f"SELECT e.start, e.end, e.eventType, {dom_expr}, {text_expr}, {gtid_expr} "
        f"FROM NVTX_EVENTS e {join_sid} "
        "WHERE e.eventType IN (34, 59, 60)"
    ):
        n_nvtx += 1
        if text is None or dom != 0:
            continue
        gtid = int(gtid) if gtid is not None else -1
        if etype == 34:
            m = _BEACON_RE.match(text)
            if m and start is not None:
                beacons.append((int(start), int(m.group(1))))
            continue
        if start is None or end is None:
            continue  # unterminated range (aborted run) — skip defensively
        start, end = int(start), int(end)
        if etype == 60:
            m = _PIPE_SPAN_RE.match(text)
            if m:
                pipeline_spans.append((int(m.group(1)), start, end))
            continue
        # eventType 59
        if text == "sirius::query":
            query_windows.append((start, end))
            continue
        m = _TASK_RE.match(text)
        if m:
            task_ranges.append(
                TaskRangeRow(int(m.group(1)), int(m.group(2)), start, end, gtid)
            )
            continue
        m = _OP_RE.match(text)
        if m:
            op_ranges.append(
                OpRangeRow(
                    pipeline_id=int(m.group(1)),
                    op_name=m.group(2),
                    op_id=int(m.group(3)),
                    is_sink=bool(m.group(4)),
                    start=start,
                    end=end,
                    global_tid=gtid,
                )
            )
    counts["nvtx_events"] = n_nvtx
    counts["task_ranges"] = len(task_ranges)
    counts["op_ranges"] = len(op_ranges)
    counts["query_windows"] = len(query_windows)
    if not task_ranges:
        notes.append(
            "no 'Pipeline N Task M [...]' NVTX ranges found — was the capture "
            "taken with --trace=nvtx and inside profiler_start/stop?"
        )
    query_windows.sort()

    # ---- kernels (join RUNTIME for launch thread/time) --------------------
    kcols = _columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_joins, name_exprs = [], []
    if "shortName" in kcols:
        name_joins.append("LEFT JOIN StringIds sn ON sn.id = k.shortName")
        name_exprs.append("sn.value")
    if "demangledName" in kcols:
        name_joins.append("LEFT JOIN StringIds dn ON dn.id = k.demangledName")
        name_exprs.append("dn.value")
    if not name_exprs:
        notes.append("kernel table has neither shortName nor demangledName")
        name_exprs = ["''"]
    name_expr = (
        f"COALESCE({', '.join(name_exprs)}, '')" if len(name_exprs) > 1
        else f"COALESCE({name_exprs[0]}, '')"
    )
    stream_expr = "k.streamId" if "streamId" in kcols else "0"
    dev_expr = "k.deviceId" if "deviceId" in kcols else "0"

    kernels: List[KernelRow] = []
    for row in conn.execute(
        f"SELECT k.start, k.end, {stream_expr}, {dev_expr}, k.correlationId, "
        f"{name_expr}, r.start, r.end, r.globalTid "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        + " ".join(name_joins)
        + " LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r "
        "ON r.correlationId = k.correlationId"
    ):
        kernels.append(
            KernelRow(
                start=int(row[0]),
                end=int(row[1]),
                stream=int(row[2] or 0),
                device=int(row[3] or 0),
                correlation=int(row[4] or 0),
                name=str(row[5]),
                launch_start=int(row[6]) if row[6] is not None else -1,
                launch_end=int(row[7]) if row[7] is not None else -1,
                global_tid=int(row[8]) if row[8] is not None else -1,
            )
        )
    counts["kernels"] = len(kernels)

    # ---- memcpys ----------------------------------------------------------
    memcpys: List[MemcpyRow] = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        oper_map, note = _load_enum(
            conn, tables, "ENUM_CUDA_MEMCPY_OPER", _FALLBACK_MEMCPY_OPER
        )
        if note:
            notes.append(note)
        kind_map, note = _load_enum(
            conn, tables, "ENUM_CUDA_MEM_KIND", _FALLBACK_MEM_KIND
        )
        if note:
            notes.append(note)
        mcols = _columns(conn, "CUPTI_ACTIVITY_KIND_MEMCPY")
        stream_expr = "m.streamId" if "streamId" in mcols else "0"
        dev_expr = "m.deviceId" if "deviceId" in mcols else "0"
        src_expr = "m.srcKind" if "srcKind" in mcols else "NULL"
        dst_expr = "m.dstKind" if "dstKind" in mcols else "NULL"
        for row in conn.execute(
            f"SELECT m.start, m.end, {stream_expr}, {dev_expr}, m.correlationId, "
            f"m.bytes, m.copyKind, {src_expr}, {dst_expr}, "
            "r.start, r.end, r.globalTid "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
            "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r "
            "ON r.correlationId = m.correlationId"
        ):
            memcpys.append(
                MemcpyRow(
                    start=int(row[0]),
                    end=int(row[1]),
                    stream=int(row[2] or 0),
                    device=int(row[3] or 0),
                    correlation=int(row[4] or 0),
                    bytes=int(row[5] or 0),
                    direction=oper_map.get(int(row[6] or 0), f"copyKind:{row[6]}"),
                    src_kind=(
                        kind_map.get(int(row[7]), f"memKind:{row[7]}")
                        if row[7] is not None
                        else "Unknown"
                    ),
                    dst_kind=(
                        kind_map.get(int(row[8]), f"memKind:{row[8]}")
                        if row[8] is not None
                        else "Unknown"
                    ),
                    launch_start=int(row[9]) if row[9] is not None else -1,
                    launch_end=int(row[10]) if row[10] is not None else -1,
                    global_tid=int(row[11]) if row[11] is not None else -1,
                )
            )
    else:
        notes.append(
            "CUPTI_ACTIVITY_KIND_MEMCPY missing — no transfer physics; "
            "transfers keep v0 (conflated) scaling"
        )
    counts["memcpys"] = len(memcpys)

    # ---- host-side sync + launch API spans ---------------------------------
    host_spans: List[HostSpanRow] = []
    like_terms = " OR ".join(
        f"s.value LIKE '{p}%'" for p in _SYNC_API_PREFIXES + _LAUNCH_API_PREFIXES
    )
    for start, end, gtid, api in conn.execute(
        "SELECT r.start, r.end, r.globalTid, s.value "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id = r.nameId "
        f"WHERE {like_terms}"
    ):
        if start is None or end is None:
            continue
        kind = (
            "launch"
            if any(str(api).startswith(p) for p in _LAUNCH_API_PREFIXES)
            else "sync"
        )
        host_spans.append(
            HostSpanRow(
                start=int(start),
                end=int(end),
                global_tid=int(gtid) if gtid is not None else -1,
                api=str(api),
                kind=kind,
            )
        )
    counts["host_api_spans"] = len(host_spans)

    # ---- optional Tier B gpu metrics ---------------------------------------
    gpu_metrics: List[Tuple[int, str, float]] = []
    if "GPU_METRICS" in tables and "TARGET_INFO_GPU_METRICS" in tables:
        try:
            for ts, name, value in conn.execute(
                "SELECT g.timestamp, m.metricName, g.value "
                "FROM GPU_METRICS g "
                "JOIN TARGET_INFO_GPU_METRICS m ON m.metricId = g.metricId "
                "WHERE m.metricName LIKE '%DRAM%'"
            ):
                gpu_metrics.append((int(ts), str(name), float(value)))
        except sqlite3.Error as e:  # column names unverified pre-first-capture
            notes.append(f"GPU_METRICS present but unreadable ({e}); skipped")
    counts["gpu_metric_samples"] = len(gpu_metrics)

    return NsysData(
        path=path,
        session_start_utc_ns=session_start_utc,
        query_windows=query_windows,
        task_ranges=task_ranges,
        op_ranges=op_ranges,
        pipeline_spans=pipeline_spans,
        clock_beacons=beacons,
        kernels=kernels,
        memcpys=memcpys,
        host_spans=host_spans,
        gpu_metrics=gpu_metrics,
        notes=notes,
        counts=counts,
    )
