"""Synthetic nsys-sqlite fixture builder for physics tests.

Creates sqlite files conforming to the nsys 2025.6.3 export schema documented
in tools/hwsim/docs/nsys-extraction.md section 2.1 (tables, field names,
semantics: start/end are integer ns relative to session start; NVTX
eventType 59 = push/pop range, 60 = start/end range, 34 = mark; kernels and
memcpys correlate to their launching RUNTIME row via correlationId; kernel
names resolve through StringIds).

No GPU is required — this is how ingestion is unit-tested before the first
real capture exists.
"""

from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

_MEMCPY_KIND = {"h2d": 1, "d2h": 2, "d2d": 8}
_MEM_KIND = {
    "Unknown": 0,
    "Pageable": 1,
    "Pinned": 2,
    "Device": 3,
    "Managed": 5,
}
_MEMCPY_LABEL = {1: "Host-to-Device", 2: "Device-to-Host", 8: "Device-to-Device"}


class FixtureBuilder:
    def __init__(self, utc_epoch_ns: int = 1_754_000_000_000_000_000) -> None:
        self.utc_epoch_ns = utc_epoch_ns
        self._strings: Dict[str, int] = {}
        self.nvtx: List[tuple] = []  # (start,end,eventType,domainId,text,textId,globalTid)
        self.kernels: List[tuple] = []
        self.runtime: List[tuple] = []  # (start,end,globalTid,correlationId,nameId,ret)
        self.memcpys: List[tuple] = []
        self.syncs_present = True
        self.gpu_metrics: List[Tuple[int, int, float]] = []  # (ts, metricId, value)
        self.metric_names: Dict[int, str] = {}
        self._corr = 100

    # ---------------------------------------------------------------- utils

    def sid(self, value: str) -> int:
        if value not in self._strings:
            self._strings[value] = len(self._strings) + 1
        return self._strings[value]

    def _next_corr(self) -> int:
        self._corr += 1
        return self._corr

    # ---------------------------------------------------------------- NVTX

    def add_query_window(self, start: int, end: int, thread: int = 1001) -> None:
        self.nvtx.append((start, end, 59, 0, "sirius::query", None, thread))

    def add_task(
        self,
        pid: int,
        task_id: int,
        start: int,
        end: int,
        thread: int,
        chain: str = "GPU_SCAN -> SINK",
    ) -> None:
        text = f"Pipeline {pid} Task {task_id} [{chain}]"
        self.nvtx.append((start, end, 59, 0, text, None, thread))

    def add_op(
        self,
        pid: int,
        op_name: str,
        op_id: int,
        start: int,
        end: int,
        thread: int,
        sink: bool = False,
    ) -> None:
        text = f"Pipeline {pid}: {op_name} (id={op_id})" + (" sink" if sink else "")
        self.nvtx.append((start, end, 59, 0, text, None, thread))

    def add_pipeline_span(
        self, pid: int, start: int, end: int, desc: str = "SRC -> SINK"
    ) -> None:
        self.nvtx.append((start, end, 60, 0, f"Pipeline {pid}: {desc}", None, None))

    def add_beacon(self, nsys_ns: int, epoch_ns: int, thread: int = 1001) -> None:
        self.nvtx.append(
            (nsys_ns, None, 34, 0, f"sirius::clock_sync:{epoch_ns}", None, thread)
        )

    def add_raw_nvtx(self, row: tuple) -> None:
        self.nvtx.append(row)

    # ---------------------------------------------------------------- CUDA

    def add_kernel(
        self,
        name: str,
        launch_t: int,
        exec_start: int,
        exec_end: int,
        thread: int,
        stream: int = 7,
        device: int = 0,
        launch_dur: int = 1_000,
        with_runtime: bool = True,
    ) -> int:
        corr = self._next_corr()
        self.kernels.append(
            (
                exec_start, exec_end, device, 0, None, stream, corr,
                1, 1, 1, 256, 1, 1,  # grid/block
                32, 0, 0, 0, 0,  # regs/shmem/local
                self.sid(name), self.sid(f"void {name}<int>(int*)"),
            )
        )
        if with_runtime:
            self.runtime.append(
                (
                    launch_t, launch_t + launch_dur, thread, corr,
                    self.sid("cudaLaunchKernel_v7000"), 0,
                )
            )
        return corr

    def add_memcpy(
        self,
        nbytes: int,
        launch_t: int,
        start: int,
        end: int,
        thread: int,
        kind: str = "h2d",
        src: str = "Pinned",
        dst: str = "Device",
        stream: int = 7,
        device: int = 0,
        with_runtime: bool = True,
    ) -> int:
        corr = self._next_corr()
        self.memcpys.append(
            (
                start, end, device, 0, stream, corr, nbytes,
                _MEMCPY_KIND[kind], _MEM_KIND[src], _MEM_KIND[dst],
            )
        )
        if with_runtime:
            self.runtime.append(
                (
                    launch_t, launch_t + 800, thread, corr,
                    self.sid("cudaMemcpyAsync_v3020"), 0,
                )
            )
        return corr

    def add_sync(self, api: str, start: int, end: int, thread: int) -> None:
        self.runtime.append((start, end, thread, self._next_corr(), self.sid(api), 0))

    def add_dram_metric(
        self, samples: Sequence[Tuple[int, float]], name: str = "DRAM Read Throughput"
    ) -> None:
        mid = len(self.metric_names) + 1
        self.metric_names[mid] = name
        for ts, v in samples:
            self.gpu_metrics.append((ts, mid, v))

    # ---------------------------------------------------------------- write

    def write(
        self,
        path: str,
        include_session_start: bool = True,
        include_enums: bool = True,
        omit_tables: Sequence[str] = (),
    ) -> str:
        conn = sqlite3.connect(path)
        c = conn.cursor()
        omit = set(omit_tables)

        def make(name: str, ddl: str) -> bool:
            if name in omit:
                return False
            c.execute(f"CREATE TABLE {name} ({ddl})")
            return True

        if make("StringIds", "id INTEGER PRIMARY KEY, value TEXT"):
            c.executemany(
                "INSERT INTO StringIds VALUES (?,?)",
                [(i, v) for v, i in self._strings.items()],
            )
        if make(
            "NVTX_EVENTS",
            "start INTEGER, end INTEGER, eventType INTEGER, domainId INTEGER, "
            "text TEXT, textId INTEGER, globalTid INTEGER",
        ):
            c.executemany(
                "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?,?,?)", self.nvtx
            )
        if make(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER, "
            "greenContextId INTEGER, streamId INTEGER, correlationId INTEGER, "
            "gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, "
            "blockY INTEGER, blockZ INTEGER, registersPerThread INTEGER, "
            "staticSharedMemory INTEGER, dynamicSharedMemory INTEGER, "
            "sharedMemoryExecuted INTEGER, localMemoryPerThread INTEGER, "
            "shortName INTEGER, demangledName INTEGER",
        ):
            c.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                self.kernels,
            )
        if make(
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "start INTEGER, end INTEGER, globalTid INTEGER, "
            "correlationId INTEGER, nameId INTEGER, returnValue INTEGER",
        ):
            c.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?,?)",
                self.runtime,
            )
        if make(
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER, "
            "streamId INTEGER, correlationId INTEGER, bytes INTEGER, "
            "copyKind INTEGER, srcKind INTEGER, dstKind INTEGER",
        ):
            c.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES "
                "(?,?,?,?,?,?,?,?,?,?)",
                self.memcpys,
            )
        if include_enums and "ENUM_CUDA_MEMCPY_OPER" not in omit:
            c.execute("CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, label TEXT)")
            c.executemany(
                "INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (?,?)",
                list(_MEMCPY_LABEL.items()),
            )
            c.execute("CREATE TABLE ENUM_CUDA_MEM_KIND (id INTEGER, label TEXT)")
            c.executemany(
                "INSERT INTO ENUM_CUDA_MEM_KIND VALUES (?,?)",
                [(i, n) for n, i in _MEM_KIND.items()],
            )
        if include_session_start and "TARGET_INFO_SESSION_START_TIME" not in omit:
            c.execute(
                "CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER)"
            )
            c.execute(
                "INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?)",
                (self.utc_epoch_ns,),
            )
        if self.gpu_metrics and "GPU_METRICS" not in omit:
            c.execute(
                "CREATE TABLE GPU_METRICS "
                "(timestamp INTEGER, typeId INTEGER, metricId INTEGER, value REAL)"
            )
            c.executemany(
                "INSERT INTO GPU_METRICS VALUES (?, 0, ?, ?)", self.gpu_metrics
            )
            c.execute(
                "CREATE TABLE TARGET_INFO_GPU_METRICS "
                "(metricId INTEGER, metricName TEXT)"
            )
            c.executemany(
                "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?,?)",
                list(self.metric_names.items()),
            )
        c.execute(
            "CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT, smCount INTEGER)"
        )
        c.execute("INSERT INTO TARGET_INFO_GPU VALUES (0, 'GB300 (fixture)', 152)")
        conn.commit()
        conn.close()
        return path


def simple_capture(
    path: str,
    *,
    kernel_name: str = "gather_kernel",
    utc_epoch_ns: int = 1_754_000_000_000_000_000,
) -> FixtureBuilder:
    """One query window / one task / one op / one kernel / one prep memcpy.

    Layout (ns): query [0, 100M); task P3 T7 on thread 1001 [10M, 60M);
    prep memcpy 8 MiB launched at 12M, executes [13M, 23M) (=20% of the 50M
    task, 40%% of the 25M prep window); op HASH_JOIN id=5 [35M, 55M) with one
    kernel launched at 36M executing [37M, 47M) (half the op span).
    """
    fb = FixtureBuilder(utc_epoch_ns=utc_epoch_ns)
    fb.add_query_window(0, 100_000_000)
    fb.add_pipeline_span(3, 5_000_000, 90_000_000)
    fb.add_task(3, 7, 10_000_000, 60_000_000, thread=1001)
    fb.add_memcpy(
        8 << 20, 12_000_000, 13_000_000, 23_000_000, thread=1001, kind="h2d"
    )
    fb.add_op(3, "HASH_JOIN", 5, 35_000_000, 55_000_000, thread=1001)
    fb.add_kernel(kernel_name, 36_000_000, 37_000_000, 47_000_000, thread=1001)
    fb.write(path)
    return fb
