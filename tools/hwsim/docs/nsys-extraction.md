# nsys Extraction for the Hardware What-If Simulator (WS2)

Status: analysis complete, 2026-08-04. No profiling runs were performed for this analysis
(shared GB300 box); schema facts below were verified against the **nsys 2025.6.3** installation
on this machine (`/opt/nvidia/nsight-systems/2025.6.3`: bundled report scripts +
`libexporter.so` embedded schema strings), the existing extraction scripts in
`test/tpch_performance/` (which have run against real exports of this nsys version), and the
Sirius / quent source trees. Column lists marked *(verify)* should be sanity-checked with
`.schema <table>` against the first real export.

**TL;DR**

- **Correlation with Quent: the mechanism already exists.** Sirius emits NVTX ranges whose
  labels carry the same numeric `pipeline_id` / `task_id` / `operator_id` that Quent records.
  No code change is required to attribute kernels/memcpys to Sirius tasks; one optional
  one-line label enrichment and one optional clock beacon are proposed below.
- **Division of labor**: the simulator should take the *task graph and scheduling* (queue
  waits, task states, admission, downgrade) from an **unprofiled Quent run**, and the
  *physics* (kernel durations, transfer bytes → achieved C2C/HBM bandwidth, sync stalls,
  per-launch API overhead) from a **paired nsys run** of the same query. Physics quantities
  are rates and per-event durations, which are first-order robust to nsys's host-side
  overhead; scheduling timelines are not.
- **Recommended minimal capture** (identical to what `performance_test.py --mode nsys-profile`
  already does):

  ```bash
  nsys profile \
    --trace=cuda,nvtx \
    --sample=none --cpuctxsw=none --cudabacktrace=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --stats=false --export=sqlite \
    --output=<out> --force-overwrite=true \
    build/release/duckdb -unsigned -f <query.sql>
  ```

  Sirius exposes `CALL profiler_start()` / `CALL profiler_stop()` table functions
  (`src/sirius_extension.cpp:1632`, mapping to `cudaProfilerStart/Stop` at
  `src/sirius_extension.cpp:1527`), so the capture range brackets exactly the iterations of
  interest and excludes pool-priming init.
- **Quent side of a paired capture: `exporter: ndjson` is REQUIRED.** hwsim parses only
  the ndjson exporter format; the bundled `test/tpch_performance/tpch_telemetry_sirius.yaml`
  ships **`exporter: postcard`**, which hwsim silently cannot read (the session parses as
  empty — `python -m hwsim info` now errors on such directories, RTX validation defect 4).
  Set `sirius.telemetry.exporter: ndjson` in the capture config before the first run.

---

## 1. What we extract from nsys today (inventory)

### 1.1 Capture configurations in the tree

| Script | Capture flags | Notes |
|---|---|---|
| `test/tpch_performance/performance_test.py --mode nsys-profile` | `--trace=cuda,nvtx --sample=none --cudabacktrace=none --capture-range=cudaProfilerApi --capture-range-end=stop --stats=false --export=sqlite` | Canonical runner. One DuckDB subprocess per query; cold+hot iterations inside one `profiler_start/stop` range; writes `<bench>/sirius/q<N>/{nsys.nsys-rep, nsys.sqlite, timings.csv}` |
| `test/tpch_performance/profile_tpch_nsys.sh` | same flags | Legacy shell equivalent; capture starts at iteration 2 (hot-only) when `ITERATIONS>=2` |
| `test/tpch_performance/nsys_report.sh` | delegates to `performance_test.py` | Adds report packaging (`report.md`, `summary.json`) |

Nobody in the tree enables `osrt`, CPU sampling, `--gpu-metrics-devices`, or
`--cuda-memory-usage` today.

### 1.2 What the existing analysis scripts compute

`nsys_analyze.sh` (all SQL against the sqlite export; per query file):

| Section | Tables/fields used |
|---|---|
| Query execution window | `NVTX_EVENTS(domainId=0, eventType=59)` min/max span |
| Execution breakdown (trace/init/query/cleanup) | + `ANALYSIS_DETAILS.duration` |
| GPU hardware | `TARGET_INFO_GPU(name,id,smCount,totalMemory,computeMajor/Minor)` |
| NVTX domain summary | `NVTX_EVENTS(eventType=75)` for domain names |
| Per-operator wall time | `NVTX_EVENTS.text` grouped (call counts, total/avg/min/max) |
| Top kernels | `CUPTI_ACTIVITY_KIND_KERNEL` + `StringIds` via `shortName` |
| Theoretical occupancy + limiter | kernel `registersPerThread, blockX/Y/Z, sharedMemoryExecuted` × `TARGET_INFO_GPU(maxRegistersPerSm, maxShmemPerSm, maxWarpsPerSm, maxBlocksPerSm, threadsPerWarp)` |
| Register spill | `localMemoryPerThread > 0` |
| Memcpy breakdown w/ bandwidth | `CUPTI_ACTIVITY_KIND_MEMCPY(bytes, start, end, copyKind, srcKind, dstKind)` + `ENUM_CUDA_MEMCPY_OPER` |
| CUDA API hotspots (query window only) | `CUPTI_ACTIVITY_KIND_RUNTIME(nameId,start,end)` |
| Host-alloc / cudaMalloc-during-query red flags | RUNTIME filtered by function name, phase-split by query window |
| Kernel→operator attribution | `KERNEL.correlationId = RUNTIME.correlationId`, then RUNTIME `globalTid` + time containment inside NVTX operator range |
| Stream busy% | kernels grouped by `streamId` |
| Sync analysis | `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION(syncType,start,end)` + `ENUM_CUPTI_SYNC_TYPE` |
| Memset summary | `CUPTI_ACTIVITY_KIND_MEMSET(bytes,start,end)` |

`nsys_hotspots.sh` adds: operator "GPU efficiency" (kernel time ÷ operator wall),
per-operator top kernels, per-operator occupancy bottlenecks, **sync time attributed to
operators** (SYNCHRONIZATION→RUNTIME correlation + NVTX containment), **memcpy volume/bw
attributed to operators**, sequential-execution-chain detection, per-thread operator
concurrency. `nsys_compare.sh` diffs the aggregated `summary.json` metrics between two runs.
The `profile-analyzer` skill (`.claude/skills/profile-analyzer/SKILL.md`) documents the
workflow and — importantly for WS2 — already institutionalizes the rule that **profiled
timings must not be used as performance ground truth**.

### 1.3 Gaps in the *existing* extraction relative to simulator needs

Everything above is **aggregated for humans**. The simulator needs *per-event* records keyed
to task ids:

1. Per-kernel and per-memcpy **rows** (not GROUP BY), each attributed to
   `(pipeline_id, task_id, operator_id)` — the aggregation queries throw the task dimension away.
2. NVTX **eventType 60** (start/end ranges) is ignored: the per-pipeline lifetime range
   (`sirius_pipeline.cpp:437`, `nvtxRangeStartEx`) exports as eventType 60, not 59, so
   pipeline spans are invisible to today's scripts.
3. Per-transfer **bandwidth-vs-size** samples for fitting an α+β transfer model (the scripts
   only compute pooled bandwidth per direction).
4. The **clock anchor** (`TARGET_INFO_SESSION_START_TIME.utcEpochNs`) is never exported —
   required to align to Quent's unix-epoch-ns timestamps.
5. No I/O syscall timing (`osrt` never enabled) and no device-metrics sampling.

---

## 2. What the simulator needs from nsys (and Quent cannot provide)

Quent (see WS1 doc) provides the task graph: plan/pipeline/operator declarations, task state
machines (created → reserving → preparing → computing(op) → finalizing → exit), queue
telemetry, batch size estimates, tier placements, memory reservations. It has **no visibility
below the CUDA API**: no kernel durations, no transfer bytes/durations, no stream/device
placement, no sync stalls, no per-launch overhead. That is exactly what nsys supplies:

| Quantity (from nsys) | Simulator use / knob it calibrates | Source table |
|---|---|---|
| Per-kernel: name, grid/block, duration, stream, device, shmem/regs | `gpu_compute` scaling of task busy time; occupancy-aware scaling; kernel classification | `CUPTI_ACTIVITY_KIND_KERNEL` |
| Per-memcpy: direction, src/dst kind, bytes, duration → achieved BW | empirical **C2C** rate (Pinned↔Device H2D/D2H on this box is NVLink-C2C) and **HBM** rate (D2D); α+β (latency+bandwidth) transfer model per direction | `CUPTI_ACTIVITY_KIND_MEMCPY` |
| Per-memset: bytes, duration | HBM write rate; part of task GPU time | `CUPTI_ACTIVITY_KIND_MEMSET` |
| CPU-blocked sync intervals (`cudaStreamSynchronize` etc.) | serialization points in the replay: portions of a task that cannot overlap; distinguishes GPU-busy from CPU-waiting-on-GPU | `CUPTI_ACTIVITY_KIND_RUNTIME` (API side) + `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` (GPU side) |
| CUDA API call durations (launch overhead) | fixed per-launch host cost that does **not** scale with `gpu_compute` — matters when kernels get faster and launch overhead dominates | `CUPTI_ACTIVITY_KIND_RUNTIME` |
| Kernel/memcpy ↔ launching thread | attribution to Sirius task + overlap analysis | `RUNTIME.correlationId`, `RUNTIME.globalTid` |
| GPU init/cleanup cost outside query window | excluded from replay | RUNTIME phase split |
| (optional, Tier B) device-wide SM-active % and DRAM BW timeline | membw-vs-compute boundedness classification (§5.1) | `GPU_METRICS` + `TARGET_INFO_GPU_METRICS` |
| (optional, Tier C) `read/pread64/mmap` syscall spans | I/O timing cross-check vs Quent scan telemetry | `OSRT_API` (+`OSRT_FILE_ACCESS_EVENTS/…_DESCRIPTORS` in 2025.6) |

### 2.1 Export schema the simulator relies on (nsys 2025.6.3)

All `start`/`end` columns are **integer nanoseconds relative to session (capture) start**.
The wall-clock anchor is `TARGET_INFO_SESSION_START_TIME.utcEpochNs` — "UTC Epoch timestamp
at start of the capture (ns)" (verified in `libexporter.so`). Kernel names resolve through
`StringIds(id, value)` via `shortName` / `demangledName`.

Core tables (columns verified via the bundled report scripts and the in-tree SQL, which run
against this nsys version):

- `CUPTI_ACTIVITY_KIND_KERNEL(start, end, deviceId, contextId, greenContextId, streamId,
  correlationId, gridX/Y/Z, blockX/Y/Z, registersPerThread, staticSharedMemory,
  dynamicSharedMemory, sharedMemoryExecuted, localMemoryPerThread, shortName, demangledName, …)`
- `CUPTI_ACTIVITY_KIND_MEMCPY(start, end, deviceId, contextId, streamId, correlationId,
  bytes, copyKind → ENUM_CUDA_MEMCPY_OPER, srcKind/dstKind → ENUM_CUDA_MEM_KIND, …)`
- `CUPTI_ACTIVITY_KIND_MEMSET(start, end, deviceId, streamId, correlationId, bytes,
  memKind, value, …)`
- `CUPTI_ACTIVITY_KIND_RUNTIME(start, end, globalTid, correlationId, nameId → StringIds,
  returnValue, …)`
- `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION(start, end, deviceId, contextId, streamId,
  correlationId, syncType → ENUM_CUPTI_SYNC_TYPE, …)`
- `CUPTI_ACTIVITY_KIND_OVERHEAD(start, end, overheadType → ENUM_CUPTI_OVERHEAD_TYPE, …)`
  *(verify column name on first export)* — profiler-injected overhead events, useful for §4.3.
- `NVTX_EVENTS(start, end, eventType, domainId, text, textId → StringIds, globalTid, …)`
  — eventType **59** = push/pop range (all `nvtx3::scoped_range` sites), **60** = start/end
  range (`nvtxRangeStartEx`, the pipeline-lifetime range), **34** = mark, **75** = domain
  registration. All Sirius ranges are in the default domain (`domainId = 0`); libkvikio,
  cuFile, libcudf, CCCL register their own domains (discovered per profile via eventType 75).
- `TARGET_INFO_GPU(id, name, smCount, totalMemory, computeMajor/Minor, maxRegistersPerSm,
  maxShmemPerSm, maxWarpsPerSm, maxBlocksPerSm, threadsPerWarp, …)`
- `TARGET_INFO_SESSION_START_TIME(utcEpochNs)`
- `ANALYSIS_DETAILS(duration, …)` — total trace duration.
- Tier B: `GPU_METRICS(timestamp, typeId, metricId, value)` +
  `TARGET_INFO_GPU_METRICS(metricId, metricName)` *(verify)*.
- Tier C: `OSRT_API(start, end, globalTid, nameId → StringIds, returnValue, callchainId)`;
  2025.6 also ships `OSRT_FILE_ACCESS_EVENTS` / `OSRT_FILE_ACCESS_DESCRIPTORS` /
  `ENUM_OSRT_FILE_ACCESS_EVENT_TYPE` *(verify availability & columns — new table family)*.

### 2.2 Concrete extraction SQL

The queries below are written to run with `sqlite3 <profile>.sqlite`. They build reusable
views first; the simulator's ETL should materialize these into its own input files
(parquet/CSV), one set per `(query, iteration)`.

#### a) Parse Sirius ids out of NVTX labels (the correlation backbone)

Label formats (source of truth):

| Label format | Emitted at | eventType |
|---|---|---|
| `sirius::query` | `src/sirius_engine.cpp:136` | 59 |
| `Pipeline {pid}: {src} -> {sink}` (pipeline lifetime) | `src/pipeline/sirius_pipeline.cpp:437-447` (`mark_task_created` → finish) | **60** |
| `Pipeline {pid} Task {tid} [{op chain}]` (task execute) | `src/pipeline/gpu_pipeline_task.cpp:513-515` | 59 |
| `Pipeline {pid}: {op_name} (id={oid})` (operator invocation) | `src/pipeline/gpu_pipeline_task.cpp:180-182` (`run_one_operator`) | 59 |
| `Pipeline {pid}: {op_name} (id={oid}) sink` | `src/pipeline/gpu_pipeline_task.cpp:477-481` | 59 |
| `sirius_physical_<op>::execute/sink` (operator-internal) | each `src/op/sirius_physical_*.cpp` | 59 |
| `native_reads`, `native_h2d` | `src/op/scan/duckdb_native_decoder.cpp:819,839` | 59 |
| `sirius::native_metadata_*`, `dynfilter::*`, `sirius::pin::*`, `sirius::compression::*` | scan/dynfilter/pin paths | 59 |

```sql
-- Task-execution ranges: one row per task attempt, with numeric ids.
CREATE TEMP VIEW task_ranges AS
SELECT
  start, end, globalTid, text,
  CAST(substr(text, 10, instr(text, ' Task ') - 10)              AS INTEGER) AS pipeline_id,
  CAST(substr(text, instr(text, ' Task ') + 6,
              instr(text, ' [') - instr(text, ' Task ') - 6)     AS INTEGER) AS task_id
FROM NVTX_EVENTS
WHERE eventType = 59 AND domainId = 0
  AND text LIKE 'Pipeline % Task %[%';

-- Operator-invocation ranges (includes ' sink' variants), nested inside task ranges.
CREATE TEMP VIEW op_ranges AS
SELECT
  start, end, globalTid, text,
  CAST(substr(text, 10, instr(text, ':') - 10)                   AS INTEGER) AS pipeline_id,
  trim(substr(text, instr(text, ': ') + 2,
              instr(text, ' (id=') - instr(text, ': ') - 2))                 AS op_name,
  CAST(substr(text, instr(text, '(id=') + 4,
              instr(text, ')') - instr(text, '(id=') - 4)        AS INTEGER) AS operator_id,
  (text LIKE '% sink')                                                        AS is_sink
FROM NVTX_EVENTS
WHERE eventType = 59 AND domainId = 0
  AND text LIKE 'Pipeline %: % (id=%';

-- Pipeline lifetime spans (NOTE eventType 60 — start/end range, missed by today's scripts).
CREATE TEMP VIEW pipeline_spans AS
SELECT
  start, end,
  CAST(substr(text, 10, instr(text, ':') - 10) AS INTEGER) AS pipeline_id,
  text
FROM NVTX_EVENTS
WHERE eventType = 60 AND domainId = 0 AND text LIKE 'Pipeline %:%->%';

-- Query window (all analysis scoped inside it).
CREATE TEMP VIEW query_window AS
SELECT start AS qstart, end AS qend
FROM NVTX_EVENTS
WHERE eventType = 59 AND domainId = 0 AND text = 'sirius::query';
```

`task_id` is a process-global atomic counter (`src/creator/task_creator.cpp:545`), so
`(pipeline_id, task_id)` is unique within a run and `task_id` alone is unique per capture.

#### b) Per-kernel rows attributed to task + operator

```sql
-- Kernel → launching runtime call → (task, operator) via same-thread interval containment.
-- "Innermost containing op range" = the op range with the max start ≤ launch < end.
CREATE TEMP VIEW kernels_attributed AS
SELECT
  k.start, k.end, k.end - k.start                       AS dur_ns,
  k.deviceId, k.streamId,
  k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
  k.registersPerThread, k.staticSharedMemory, k.dynamicSharedMemory,
  k.sharedMemoryExecuted, k.localMemoryPerThread,
  sn.value                                              AS kernel_name,
  t.pipeline_id, t.task_id,
  (SELECT o.operator_id FROM op_ranges o
    WHERE o.globalTid = r.globalTid
      AND r.start >= o.start AND r.start < o.end
    ORDER BY o.start DESC LIMIT 1)                      AS operator_id,
  (SELECT o.op_name FROM op_ranges o
    WHERE o.globalTid = r.globalTid
      AND r.start >= o.start AND r.start < o.end
    ORDER BY o.start DESC LIMIT 1)                      AS op_name
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
JOIN StringIds sn                  ON sn.id = k.shortName
LEFT JOIN task_ranges t
       ON t.globalTid = r.globalTid
      AND r.start >= t.start AND r.start < t.end;
```

Attribution is by **launch time on the launching thread** (kernels execute later,
asynchronously — that is what the simulator wants: the launch belongs to the task, the
execution interval is the GPU-resource demand).

#### c) Per-memcpy rows → empirical C2C / HBM bandwidth

```sql
CREATE TEMP VIEW memcpys_attributed AS
SELECT
  m.start, m.end, m.end - m.start                        AS dur_ns,
  m.bytes,
  mo.label                                               AS direction,   -- 'Host-to-Device', ...
  ms.label                                               AS src_kind,    -- Pageable/Pinned/Device...
  md.label                                               AS dst_kind,
  m.deviceId, m.streamId,
  m.bytes * 1.0 / (m.end - m.start)                      AS bw_gb_s,     -- bytes/ns == GB/s
  t.pipeline_id, t.task_id
FROM CUPTI_ACTIVITY_KIND_MEMCPY m
JOIN ENUM_CUDA_MEMCPY_OPER mo ON mo.id = m.copyKind
LEFT JOIN ENUM_CUDA_MEM_KIND ms ON ms.id = m.srcKind
LEFT JOIN ENUM_CUDA_MEM_KIND md ON md.id = m.dstKind
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = m.correlationId
LEFT JOIN task_ranges t
       ON t.globalTid = r.globalTid AND r.start >= t.start AND r.start < t.end;

-- Empirical bandwidth curve per channel, bucketed by log2(bytes): fit t = alpha + bytes/beta.
-- On GB300: Pinned<->Device H2D/D2H == the C2C link; Device->Device == HBM (via SM or CE).
SELECT
  direction, src_kind, dst_kind,
  CAST(ln(bytes)/ln(2) AS INTEGER)               AS log2_bytes,   -- sqlite >= 3.35
  COUNT(*)                                        AS n,
  SUM(bytes) / 1e9                                AS total_gb,
  SUM(bytes) * 1.0 / SUM(dur_ns)                  AS pooled_gb_s,
  AVG(bw_gb_s)                                    AS mean_gb_s,
  MIN(dur_ns)                                     AS min_dur_ns     -- lower bound on alpha
FROM memcpys_attributed
GROUP BY direction, src_kind, dst_kind, log2_bytes
ORDER BY direction, log2_bytes;
```

Two prior findings to encode in the model (memory notes, measured on this box): the
`cudaMemcpy` D2D path is flag-capped near ~470 GB/s for large buffers while small buffers
behave very differently — so **fit the α+β model per size bucket, don't use one pooled
number**; and pageable transfers are a different regime entirely (keep `src_kind/dst_kind`
in the key).

#### d) Sync stalls (CPU blocked on GPU) per task

```sql
-- API-side blocked time: this is what serializes the host thread in the replay.
CREATE TEMP VIEW sync_stalls AS
SELECT
  r.start, r.end, r.end - r.start AS dur_ns,
  s.value                          AS api,           -- cudaStreamSynchronize, cudaEventSynchronize, ...
  t.pipeline_id, t.task_id,
  (SELECT o.operator_id FROM op_ranges o
    WHERE o.globalTid = r.globalTid AND r.start >= o.start AND r.start < o.end
    ORDER BY o.start DESC LIMIT 1) AS operator_id
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON s.id = r.nameId
LEFT JOIN task_ranges t
       ON t.globalTid = r.globalTid AND r.start >= t.start AND r.start < t.end
WHERE s.value LIKE 'cudaStreamSynchronize%'
   OR s.value LIKE 'cudaDeviceSynchronize%'
   OR s.value LIKE 'cudaEventSynchronize%'
   OR s.value LIKE 'cuStreamSynchronize%'
   OR s.value LIKE 'cudaMemcpy_v%';       -- synchronous memcpy variants block too
```

Note: `gpu_pipeline_task.cpp` deliberately calls `stream.synchronize()` at the end of the
`preparing` phase "to ensure the timing collected by Quent … is accurate" — so Quent's
`preparing` duration already includes the H2D drain; nsys tells you how much of it was
transfer vs stall.

#### e) Per-launch API overhead (does not scale with `gpu_compute`)

```sql
SELECT s.value AS api, COUNT(*) AS calls,
       SUM(r.end - r.start) AS total_ns,
       AVG(r.end - r.start) AS avg_ns
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON s.id = r.nameId
JOIN query_window w ON r.start >= w.qstart AND r.start < w.qend
WHERE s.value LIKE 'cudaLaunchKernel%' OR s.value LIKE 'cuLaunchKernel%'
GROUP BY s.value;
```

#### f) I/O syscalls (Tier C, `--trace=osrt` only)

```sql
SELECT s.value AS syscall, COUNT(*) AS calls,
       SUM(o.end - o.start) AS total_ns,
       AVG(o.end - o.start) AS avg_ns,
       o.globalTid
FROM OSRT_API o
JOIN StringIds s ON s.id = o.nameId
WHERE s.value IN ('read','pread64','preadv','readv','mmap','openat','fsync')
GROUP BY s.value, o.globalTid
ORDER BY total_ns DESC;
```

**OSRT records durations only — no byte counts** (see §5.3). Prefer the libkvikio NVTX
domain (`posix_host_read` ranges) and Sirius's own `native_reads` NVTX range for I/O phase
timing in the default (Tier A) capture; use osrt only for targeted I/O studies.

#### g) Device metrics timeline (Tier B, `--gpu-metrics-devices` only)

```sql
SELECT g.timestamp, m.metricName, g.value
FROM GPU_METRICS g
JOIN TARGET_INFO_GPU_METRICS m ON m.metricId = g.metricId
WHERE m.metricName LIKE '%DRAM%' OR m.metricName LIKE '%SM Active%'
ORDER BY g.timestamp;
```

Integrate `DRAM Read/Write Throughput` over each kernel's `[start,end)` to estimate that
kernel's achieved HBM bandwidth (valid only where a single kernel dominates the device —
check with a stream-overlap query first; see §5.1).

#### h) Clock anchor + profiler self-overhead

```sql
SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME;   -- unix-epoch ns of session t=0

SELECT e.label, COUNT(*) AS n, SUM(o.end - o.start) AS total_ns
FROM CUPTI_ACTIVITY_KIND_OVERHEAD o
LEFT JOIN ENUM_CUPTI_OVERHEAD_TYPE e ON e.id = o.overheadType   -- (verify column name)
GROUP BY e.label;
```

---

## 3. Correlation with Quent

### 3.1 Verdict: the mechanism exists — NVTX labels and Quent records share the same ids

Emission sites are pairwise adjacent in the code (same thread, same code path, microseconds
apart), so every Quent scheduling record has an NVTX twin:

| Quent record (WS1) | Emitted at | NVTX twin | Emitted at | Join key |
|---|---|---|---|---|
| `quent::task::create` with `instance_name = "task-{task_id}"`, `pipeline_uuid` | `src/pipeline/sirius_pipeline_itask.cpp:34-40` | `Pipeline {pid} Task {tid} [...]` range | `gpu_pipeline_task.cpp:513` | numeric `task_id` (+ `pipeline_id`) |
| task `computing({instance_name = "{op}({oid})"})` transition | `gpu_pipeline_task.cpp:374` | `Pipeline {pid}: {op} (id={oid})` range opens immediately after | `gpu_pipeline_task.cpp:180` (`run_one_operator`) | `(pipeline_id, operator_id)` + ordering within task |
| task `preparing(...)` transition | `gpu_pipeline_task.cpp:546` | contained in the task range; `native_h2d`/`native_reads` sub-ranges | `duckdb_native_decoder.cpp:819,839` | task containment |
| operator declaration: `pipeline_uuid` ↔ `type_name = "Pipeline Id {pid}"` | `src/telemetry/telemetry_context.cpp:209-217` | pipeline lifetime range `Pipeline {pid}: src -> sink` (eventType 60) | `sirius_pipeline.cpp:437` | numeric `pipeline_id` — this is also how Quent's `pipeline_uuid` maps to the numeric id |
| query declaration (`telemetry_query_id`) | `src/planner/query.cpp` / `sirius_engine.cpp` | `sirius::query` range | `sirius_engine.cpp:136` | one per query; iteration ordering |

So the attribution chain for every kernel/memcpy is:

```
CUPTI kernel/memcpy row
  --correlationId-->  CUPTI_ACTIVITY_KIND_RUNTIME row (launch, has globalTid + timestamp)
  --globalTid + interval containment-->  NVTX task range  --parse-->  (pipeline_id, task_id)
  --innermost op range-->  operator_id
  --task_id == "task-{N}" / pipeline_id == "Pipeline Id {N}"-->  Quent task / pipeline records
```

**No NVTX additions are required.** Two optional, low-cost enrichments:

1. *(nice-to-have)* Add `task_id` to the operator-invocation label in `run_one_operator`
   (`src/pipeline/gpu_pipeline_task.cpp:180`):
   `std::format("Pipeline {} Task {}: {} (id={})", pipeline->get_pipeline_id(), task_id, op.get_name(), op.get_operator_id())`
   — `task_id` is already a parameter of that function. This makes operator attribution a
   direct parse instead of a nesting join (robust if task ranges are ever disabled).
2. *(clock beacon, see §3.2)* A pair of `nvtx3::mark()` calls with an embedded wall-clock
   payload in `sirius_engine.cpp` next to the `sirius::query` range.

One correctness caveat for the ETL: a task that **downgrades/re-executes** appears as
multiple `gpu_pipeline_task::execute` NVTX ranges with the same `task_id`; Quent likewise
records repeated state transitions. Treat `(task_id, attempt ordinal by start time)` as the
event key, not `task_id` alone.

### 3.2 Clock-domain alignment

**Quent's clock** (verified in the vendored quent source,
`~/.cargo/git/checkouts/quent-*/2a5ca83/crates/time/src/lib.rs`): timestamps are
`u64` **unix-epoch nanoseconds**, computed as a one-time `SystemTime::now()` (CLOCK_REALTIME)
anchor captured at the first `timestamp()` call in the process, advanced by
`Instant::now()` elapsed (CLOCK_MONOTONIC on Linux). Monotone by construction; immune to
NTP steps after process start, but *not* to the anchor being taken at a different instant
than nsys's.

**nsys's clock**: exported `start`/`end` are ns **relative to session start**; on this
aarch64 box the target timestamps come from the ARM generic counter / CLOCK_MONOTONIC_RAW
family (the exporter's internal time-correlation chains include `TargetCntVct`,
`TargetMonotonicRawNs`, `TargetUtcNs`). The absolute anchor is
`TARGET_INFO_SESSION_START_TIME.utcEpochNs`.

**Alignment procedure** (recommended, no code change):

1. First-order: `quent_ns ≈ nsys_ns + utcEpochNs`. Residual error = (difference between the
   two CLOCK_REALTIME snapshots) + (CLOCK_MONOTONIC vs MONOTONIC_RAW slew drift, ~ppm —
   microseconds over a multi-second query, but the constant offset can be milliseconds).
2. Robust refinement: least-squares fit `quent_t = a + b · nsys_t` over **matched event
   pairs** — Quent `computing({op}({oid}))` transition timestamps vs the NVTX
   `Pipeline {pid}: {op} (id={oid})` range `start`, matched by
   `(pipeline_id, operator_id, occurrence ordinal within task)`. These fire back-to-back on
   the same thread (`gpu_pipeline_task.cpp:374` then `:180`), so pair residuals are
   microseconds. A TPC-H query yields hundreds–thousands of pairs; expect `b ≈ 1 ± 1e-5` and
   sub-100 µs alignment. Reject outlier pairs (>3σ) before the fit.
3. Optional hardening — **sync beacon**: in `sirius_engine.cpp` immediately after opening
   the `sirius::query` range (line 136), emit
   `nvtx3::mark(std::format("sirius::clock_sync:{}", duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()))`
   and again just before the range closes. Each mark gives an exact
   (nsys_session_ns ↔ CLOCK_REALTIME ns) sample; two per query give offset + drift directly
   and validate the regression. Payload lives in `NVTX_EVENTS.text` (eventType 34).
   This is the only place a beacon is needed; do not scatter beacons per task.

Note: since the simulator takes *scheduling* from an unprofiled Quent run and *physics* from
a separate nsys run (§4.4), cross-run correlation is by **structural key**
(query, pipeline_id, operator_id, task ordinal), not by timestamp — clock alignment is only
needed when analyzing the profiled run's own Quent output (e.g., the overhead-skew analysis
in §4.3, or if WS6 chooses to replay directly from a single combined run).

---

## 4. Profiling overhead

### 4.1 Cost of nsys options

| Option | Cost | Verdict for simulator-input runs |
|---|---|---|
| `--trace=nvtx` | ~100 ns per range push/pop | Keep (required — correlation backbone) |
| `--trace=cuda` | ~1 µs interception per CUDA API call; grows with launch/memcpy count. This is the dominant term on launch-heavy queries | Keep (required — it *is* the data) |
| `--capture-range=cudaProfilerApi` | Confines collection to the query window | Keep |
| `--sample=none` | Disables CPU IP/backtrace sampling (default `process-tree` sampling is **expensive**: per-thread 1 kHz interrupts + unwinds) | Keep off |
| `--cpuctxsw=none` | Context-switch tracing off | Keep off (add explicitly; it can default on with sampling) |
| `--cudabacktrace=none` | CUDA API backtraces are very expensive (unwind per call) | Keep off |
| `--trace=osrt` | Hooks ~hundreds of libc/OS functions above a 1 µs threshold; skews syscall-heavy (I/O, mutex) phases | Off by default; Tier C only |
| `--gpu-metrics-devices=… --gpu-metrics-frequency=10000` | Background HW-counter sampling; low single-digit % and roughly uniform (device-side) | Tier B; on for calibration runs |
| `--cuda-memory-usage=true` | Tracks every alloc/free | Off (Quent covers memory) |
| `--stats=false`, `--export=sqlite` | Post-processing only, zero runtime cost | Keep; export can also be done later via `nsys export --type sqlite x.nsys-rep` |

### 4.2 Recommended capture tiers

- **Tier A (default simulator input)** — the command in the TL;DR. Identical to today's
  `--mode nsys-profile` flags plus an explicit `--cpuctxsw=none`.
- **Tier B (kernel-classification calibration)** — Tier A +
  `--gpu-metrics-devices=all --gpu-metrics-frequency=10000`. Run once per (dataset, config)
  to classify kernels (§5.1); not needed on every capture.
- **Tier C (I/O studies)** — Tier A + `--trace=cuda,nvtx,osrt`. Only when validating the
  I/O throttler (WS3) against syscall timing; expect measurable skew in scan phases.

### 4.3 Protocol to quantify overhead

Goal: a per-query, per-phase overhead table, because **the simulator inherits any skew**
(if profiling inflates only host-side orchestration, a replay calibrated from profiled
gaps would over-predict CPU-bound behavior).

1. **Fixed setup**: same dataset, Sirius YAML, query set; Quent enabled in *all* runs
   (`telemetry.enable_quent: true` — its cost is a constant across arms). GPU idle per the
   shared-box rule (check `nvidia-smi` first).
2. **Arms**, ≥5 hot iterations each, interleaved A→B→A ordering to expose drift:
   - **A**: no nsys (`performance_test.py --engine gpu --iterations 5` → `csv/runtimes.csv`)
   - **B**: Tier A capture (`--mode nsys-profile --iterations 5` → per-query `timings.csv`)
   - **A′**: repeat of A — bounds run-to-run variance. Note: at SF1000 several TPC-H queries
     swing 13–28 % between identical runs; the overhead gate below is only meaningful on
     queries whose A vs A′ medians agree within a few percent (use the stable-query list, or
     raise iteration counts).
   - Optionally **C**: Tier B and/or Tier C to price gpu-metrics and osrt separately.
3. **Wall-clock overhead**: `median(B)/median(A) − 1` per query, reported next to
   `|median(A′)/median(A) − 1|` (the noise floor). Gate: accept Tier A for simulator input
   if per-query overhead < ~5 % and above the noise floor by a clear margin only where expected.
4. **Phase skew** (the part that matters): Quent runs in both arms, so compare per-state
   totals — `queued`, `reserving`, `preparing`, `computing(op)`, `finalizing` — per pipeline
   between A and B. Report a skew matrix (state × query, Δ%). Complementary in-trace checks
   on B: `CUPTI_ACTIVITY_KIND_OVERHEAD` totals (§2.2h) and NVTX per-operator distribution
   vs Quent's per-operator distribution in A. Expectation: overhead concentrates in
   API-call-dense phases (`computing` on launch-heavy operators, `preparing` on
   memcpy-heavy ones) and is *not* uniform — quantify, don't assume.
5. **GPU-side purity check**: kernel durations themselves (device time) should be unaffected
   by host-side tracing; verify by comparing kernel-duration distributions of the same
   kernel across B iterations (and against Tier B, where gpu-metrics sampling *can* add a
   small device cost).

### 4.4 How the simulator should consume this (overhead management by design)

- **Scheduling ground truth** (task states, queue waits, admission, downgrades, phase
  timestamps): from **arm A** (Quent only, no nsys).
- **Physics** (per-kernel ns, per-memcpy bytes+ns → bandwidth curves, sync-stall shares,
  per-launch API cost): from **arm B**, expressed as *rates and per-event durations
  normalized per byte/row*, then joined to arm A's task graph by structural key
  (query, pipeline_id, operator_id, task ordinal). Kernel and memcpy durations are measured
  on the device and are overhead-clean; only host-side *gaps* are contaminated, and those
  come from arm A anyway.
- Persist the measured overhead table with every calibration bundle so predictions can be
  caveated when a query's inputs came from a high-overhead capture.

---

## 5. Gaps — what neither Quent nor nsys provides

### 5.1 Per-kernel membw-bound vs compute-bound classification (top gap)

nsys gives kernel durations but no per-kernel memory counters, so `gpu_mem_bandwidth` vs
`gpu_compute` sensitivity per kernel is not directly measurable. Three-pronged mitigation:

1. **Tier B gpu-metrics integration**: integrate the device-wide DRAM read+write throughput
   timeline over each kernel's execution interval → achieved GB/s per kernel occurrence;
   compare to HBM peak for a boundedness score. Caveat: Sirius runs multiple streams
   (pool sized by `pipeline.num_threads`), so restrict to intervals where one kernel owns
   the device (detectable in SQL by non-overlap of kernel rows), and treat overlapped
   intervals as unclassified.
2. **ncu spot checks as calibration** (offline, not on simulator-input runs): profile the
   top ~20 kernels by cumulative time with
   `ncu --set roofline` / `--metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed`
   on a small scale factor. ncu replays kernels and is orders of magnitude slower — use it
   once per kernel *family* to build a static classification table keyed by kernel name.
3. **Name-based priors** for the cuDF/CUB kernel families as a fallback: gather/scatter,
   `copy_if`, radix-sort passes, materialization/`concat` → membw-bound; hash probe/build →
   latency/membw mixed; decompression kernels → **SM-bound on this box** (measured: the
   "Preparing" decompress path is compute-limited, see memory note *decompress is SM-bound* —
   a warning against assuming transfers/decode scale with `c2c_bandwidth`).

The classification feeds the simulator as a per-kernel-name sensitivity vector
(∂t/∂gpu_compute vs ∂t/∂gpu_mem_bandwidth), validated by WS4/WS5 throttle runs.

### 5.2 SM-driven / zero-copy C2C traffic is invisible

`CUPTI_ACTIVITY_KIND_MEMCPY` only records explicit `cudaMemcpy*` operations. On
Grace-Blackwell, kernels can directly load/store host memory over C2C (coherent path); that
traffic appears as kernel time, not as transfers, so scaling `c2c_bandwidth` would silently
miss it. Audit which Sirius paths do this (pinned-host reads inside decode kernels are the
candidates); Tier B C2C/NVLink metrics (if exposed in `TARGET_INFO_GPU_METRICS` on this
target) or ncu `pcie/c2c` counters are the only measurement paths.

### 5.3 I/O bytes and achieved storage bandwidth

- OSRT gives syscall **durations only** — no byte counts, no file offsets (the new
  `OSRT_FILE_ACCESS_*` tables in 2025.6 should be checked, but plan without them).
- kvikio's NVTX domain (`posix_host_read`, `task`) gives read spans, again duration-only.
- Sirius's `native_reads` NVTX range (`duckdb_native_decoder.cpp:819`) brackets a batch of
  coalesced reads whose `total_read` bytes are computed right above it — **proposed
  enhancement**: include the byte count in the range label
  (`std::format("native_reads:{}", total_read)`), giving per-batch achieved read bandwidth
  in every Tier A capture for free.
- Until then: Quent's scan/batch byte estimates + an `iostat -x 1` sidecar during
  simulator-input runs supply the io_bandwidth calibration; nsys supplies only the timing
  shape of the read phase.

### 5.4 Host-side cost decomposition

With `--sample=none` there is no CPU profiling, so host gaps between GPU ops (decode
orchestration, plan glue, spdlog, allocator work) can't be split into cpu-compute-sensitive
vs cpu-membw-sensitive parts. Mitigation: model host segments as scaling with `cpu_compute`
only, and run an occasional sampling-enabled capture (Tier D: `--sample=cpu`, small SF,
not used as simulator input) to estimate the memory-stall share of host time if WS6 finds
host segments dominate any what-if scenario.

### 5.5 Covered elsewhere (no action)

- **GPU/host memory occupancy over time** (drives back-pressure/downgrade in the sim):
  not in nsys (RMM suballocations invisible; `cudaMalloc` appears only at pool growth) —
  Quent's `memory_context` reservation/pool telemetry covers this (WS1).
- **How kernel duration responds to SM/clock scaling** (the actual `gpu_compute` transfer
  function): no trace provides it; that is precisely what WS5's compute throttler + WS7/8
  validation measure.

---

## Appendix: first-profile validation checklist

On the first Tier A capture, verify assumptions this doc could not check without a run:

```bash
sqlite3 q1.sqlite ".tables"
sqlite3 q1.sqlite ".schema CUPTI_ACTIVITY_KIND_KERNEL"      # column list (verify marks above)
sqlite3 q1.sqlite ".schema CUPTI_ACTIVITY_KIND_OVERHEAD"    # overheadType column name
sqlite3 q1.sqlite "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME;"
sqlite3 q1.sqlite "SELECT eventType, COUNT(*) FROM NVTX_EVENTS GROUP BY eventType;"  # expect 59, 60, 75 (+34 if beacons added)
sqlite3 q1.sqlite "SELECT text FROM NVTX_EVENTS WHERE eventType=60 LIMIT 3;"          # pipeline spans present
sqlite3 q1.sqlite "SELECT text FROM NVTX_EVENTS WHERE text LIKE 'Pipeline % Task %' LIMIT 3;"
sqlite3 q1.sqlite "SELECT log2(4096);"   # confirms sqlite >= 3.35 for the bucketing SQL
```
