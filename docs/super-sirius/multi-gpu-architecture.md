# Multi-GPU Architecture

How Sirius executes SQL across every GPU on a single node — the components, data residency model, scheduling rules, and concurrency invariants.

## The Mental Model

Sirius treats a multi-GPU host as a set of cooperating execution units that share host memory and disk but have private device memory. The engine's job is to:

1. **Place data** so that each row lives in the GPU (or NUMA host region, or local disk) closest to where it will be consumed
2. **Place tasks** so each task runs on the GPU that already holds its inputs
3. **Move data when locality fails** — across GPUs via peer DMA (or host-staging when peer DMA is broken), to host or disk on memory pressure, and back to GPU when the next task needs it

The user writes plain SQL. The Sirius optimizer extension intercepts the physical plan and routes supported operators to the multi-GPU execution path; unsupported plans fall back to DuckDB's CPU engine transparently.

```sql
-- Plain SQL — transparently routed to multi-GPU execution
LOAD 'sirius.duckdb_extension';
SELECT l_returnflag, SUM(l_quantity) FROM lineitem GROUP BY l_returnflag;
```

The engine assumes a single process pinning all visible GPUs (`CUDA_VISIBLE_DEVICES` controls which GPUs Sirius can use). There is no notion of distributed multi-node execution in this codebase.

## Tier Hierarchy

Every byte of data lives in exactly one **memory tier** at any moment. Tiers, fastest to slowest:

| Tier | Backed by | Capacity per host | Typical purpose |
|------|-----------|-------------------|-----------------|
| `GPU` | `cuda_async_memory_resource` (one pool per device) | Bounded by GPU device memory × `reservation_fraction_per_gpu` (default 0.75) | Active query data |
| `HOST` | NUMA-local pinned host memory | Sized via `set_per_host_capacity()` × `reservation_fraction_per_host` | Downgrade target when GPU is full |
| `DISK` | On-disk file pool via cucascade's `idisk_io_backend` | Configured via `set_disk_mounting_point(gpu_id, capacity, path)` | Last-resort downgrade target |

`HOST` is partitioned by NUMA node — each GPU has a paired host region on its NUMA-local memory controller for fast downgrade. This is configured via `cucascade::memory::reservation_manager_configurator::use_host_per_gpu()` at startup.

Conceptually:

```
        ┌─────────┐  ┌─────────┐
        │  GPU 0  │  │  GPU 1  │      ← Tier::GPU
        └────┬────┘  └────┬────┘
             │            │
        ┌────┴────┐  ┌────┴────┐
        │ Host    │  │ Host    │
        │ (NUMA0) │  │ (NUMA1) │      ← Tier::HOST (per-GPU NUMA-local)
        └────┬────┘  └────┬────┘
             └─────┬──────┘
              ┌────┴────┐
              │  Disk   │              ← Tier::DISK (shared pool)
              └─────────┘
```

A `cucascade::memory::memory_space` is the in-memory representation of one (tier, gpu_id) pair. The engine queries spaces via `manager.get_memory_space(Tier, gpu_id)` and routes allocations through that space's `device_async_resource_ref`. Each space owns its allocator and a `reservation_aware_resource_adaptor` that tracks per-thread byte budgets so OOM is detectable before the driver fails.

## Component Diagram

A single `SiriusContext` per process owns the entire multi-GPU machinery. Its key fields:

```
SiriusContext (src/sirius_context.{hpp,cpp})
│
├─ sirius_memory_reservation_manager   ← extends cucascade::memory_reservation_manager
│  │                                     manages all (tier, gpu) memory_spaces
│  └─ For each gpu_id:
│     ├─ Tier::GPU  memory_space  →  cuda_async_memory_resource pool
│     ├─ Tier::HOST memory_space  →  NUMA-local pinned-host pool
│     └─ (Tier::DISK shared by all GPUs)
│
├─ gpu_ioctxs_  : unordered_map<int, shared_ptr<sirius_ioctx>>
│                  one io_uring reactor per GPU for parquet I/O
│
├─ datasource_registry_  : scheme → datasource factory
│                           kFileScheme registered at init; strict (throws on unknown)
│
├─ sirius_scan_manager
│  │  owns PinTableFunction; manages pinned_entry records
│  └─ pinned_entry
│     ├─ data_batches_by_column  (per-column DataBatch chunks)
│     └─ chunk_memory_spaces     (parallel vector — owning memory_space per chunk)
│
├─ gpu_pipeline_executor + task_creator + task_scheduler
│                                Phase 2 wire_data_repositories Phase-2 split:
│                                converter emits pure-data repository_wiring,
│                                engine calls materialize_repository_wiring()
│
└─ downgrade_executor (one per GPU's HOST-tier path)
   │  monitors memory pressure; downgrades batches GPU→HOST→DISK
   └─ Phase 22.2 K.6: only runs cudaSetDevice for tier == GPU
```

Ownership goes one direction: `SiriusContext` owns everything below it. Connections register `sirius_state` (a `shared_ptr<SiriusContext>`) into DuckDB's `ClientContext::registered_state` at `OnConnectionOpened`, and remove it at `OnConnectionClosed`.

## Per-GPU Initialization

`SiriusContext::initialize()` is the single point where multi-GPU state comes online. The sequence (`src/sirius_context.cpp`):

1. **Discover GPUs.** `topology_discovery` enumerates devices visible to the process (respects `CUDA_VISIBLE_DEVICES`), and records each GPU's NUMA node.
2. **Build memory spaces.** A `reservation_manager_configurator` is configured with per-GPU usage limits, per-host capacities, optional disk mounts, and NUMA pairings. `builder.build()` produces `memory_space_config`s, which `sirius_memory_reservation_manager` consumes to construct all tier × gpu spaces.
3. **Install per-GPU device resource refs.** For each GPU, `sirius_memory_reservation_manager`'s constructor sets that GPU's `cuda_async_memory_resource` as cudf's `current_device_resource_ref` (saving the previous ref for restoration on shutdown). This ensures cudf operations on each GPU allocate through that GPU's reservation-tracked pool.
4. **Construct per-GPU `uring_ioctx`.** One `sirius_ioctx` per GPU, each wrapping a `cucascade::uring_reactor` and bound to that GPU's `cudaSetDevice` via an RAII wrapper. Stored in `gpu_ioctxs_`. Used by all parquet readers.
5. **Register the file scheme.** `datasource_registry_.register_scheme("file", make_factory(gpu_ioctxs_.at(lowest_gpu_id)))`. Any reader resolving a `file://` URL gets a `sirius_datasource` backed by the per-GPU ioctx.
6. **Restore cudf device-resource refs on shutdown.** `sirius_memory_reservation_manager`'s destructor first synchronizes each managed GPU (`cudaDeviceSynchronize()`) so pending `cudaFreeAsync` operations against the soon-to-be-destroyed pool complete, then restores cudf's previous device resource ref. The sync step is critical — without it, tests that leave async deallocations un-synchronized can corrupt the driver's per-device pool list and crash the next manager construction on the same device.

After `initialize()`, the engine has: per-GPU memory pools, per-GPU I/O reactors, a strict datasource factory, and a manager that knows how to translate `(Tier, gpu_id)` into an allocator.

## Data Residency: Pin Tables

Tables that benefit from being resident in GPU memory get **pinned** via the SQL surface:

```sql
CALL pin_table('lineitem');                    -- distribute chunks across GPU memory spaces (default)
CALL pin_table('lineitem', tier => 'host');    -- pin into host memory only
```

The pin pipeline (`src/scan_manager/`):

1. **Open the parquet files.** `PinTableFunction` resolves the file scheme through `datasource_registry_`, gets per-GPU `sirius_datasource` instances backed by `uring_ioctx`.
2. **Enumerate chunks.** Reads parquet metadata to discover row-group chunks.
3. **Distribute round-robin.** `chunk_memory_spaces[i] = gpu_spaces[i % gpu_spaces.size()]` — chunk `i` is assigned to the `i`-th GPU's memory_space. This is the **PIN-MGPU-01 invariant**.
4. **Read each chunk into its target space.** Each chunk's read goes through the `uring_ioctx` whose GPU owns the target memory_space — so the I/O lands directly in the right GPU's pinned memory region with no cross-GPU copies.
5. **Record the pinned entry.** `pinned_entry { data_batches_by_column, chunk_memory_spaces }` is stored on the `sirius_scan_manager`. The `chunk_memory_spaces` vector is parallel to `data_batches_by_column[col_idx]` — `chunk_memory_spaces[i]` is the owning space for chunk `i` regardless of which column you're looking at.

Repeat invocations of `pin_table('lineitem', ...)` are idempotent — duplicates dropped, existing `chunk_memory_spaces` preserved (Phase 22 Pitfall 3 invariant: any merge must verify `chunk_memory_spaces` integrity).

When the new HOST-tier pinning path is used (`2e197c6` upstream feature, integrated in Phase 24), `pin_table` constructs a `cucascade::host_data_representation` and stores it in `pinned_entry`'s host-mode slot; subsequent scans go through a host-mode `cached_split_provider` that slices the host chunks per query and converts back to GPU only when a scan task starts. The GPU-tier and HOST-tier pinning paths coexist as parallel code paths — both maintain the `chunk_memory_spaces` invariant.

## Scan-Time: Routing to the Right GPU

When a query selects from a pinned table, the `cached_split_provider` walks `pinned_entry`'s chunks. For each chunk:

1. **Look up the chunk's owning memory_space** from `chunk_memory_spaces[i]`.
2. **Create a scan task** carrying the chunk plus the memory_space.
3. **Annotate the task with `preferred_device_id`** = that memory_space's GPU id.

The pipeline executor consumes tasks with their `preferred_device_id` set, and the scheduler routes each task to that GPU's worker thread. The result: each chunk is processed on the same GPU it's resident on, no cross-GPU transfer at scan time.

## Task Scheduling: SCHED-RR and Locality

For tasks that aren't bound to a specific chunk (e.g., downstream operators consuming many input batches), the scheduler computes a **locality score**:

```
locality(task, gpu_id) = bytes of task input data already on gpu_id
```

The task goes to the GPU with the highest score, falling back to the task's NUMA-paired GPU if locality is tied or all data is on HOST. This is the **SCHED-RR** (round-robin with locality) distribution policy. Source-pipeline tasks (those at the start of a pipeline) are distributed using strict round-robin; downstream tasks use locality.

Two-level `preferred_device_id`:

- `gpu_pipeline_task_local_state::_preferred_device_id` — per-task override (winner)
- `sirius_pipeline_task_global_state::_preferred_device_id` — pipeline-level default

`task->get_preferred_device_id()` checks local first, falls back to global. Operators that materialize cross-GPU data (e.g., partition-then-merge) propagate the right device id through the task tree.

See [`pipeline-execution.md`](pipeline-execution.md) "Per-task-device contract under SCHED-RR" for the deeper contract.

## Cross-GPU Data Movement

When an operator must consume data on GPU A that lives on GPU B (e.g., a hash join's probe side has chunks scattered across all GPUs), the engine performs a **GPU↔GPU transfer** via `cucascade::convert_gpu_to_gpu` (in `cucascade/src/data/representation_converter.cpp`).

The transfer chooses one of two paths empirically:

1. **Direct peer DMA** (`cudaMemcpyPeerAsync`) — fastest, used when `probe_peer_dma_works(src, dst)` returns true. Real peer access requires both GPUs to have driver-level P2P enabled AND the hardware to actually honor it.
2. **Host-staging** (`cudaMemcpyAsync(DtoH)` → host buffer → `cudaMemcpyAsync(HtoD)`) — fallback for hardware where peer DMA is empirically broken (e.g., the consumer-grade RTX 6000 Ada we use for development, which advertises P2P but silently fails DMA in both directions).

The probe runs once at startup per (src, dst) pair: allocate small buffers on each device, attempt a `cudaMemcpyPeerAsync` and a roundtrip read-back. If the bytes don't match, mark the pair as host-stage-required.

**Concurrency invariants for cross-GPU transfers:**

- **Same-stream invariant** (Phase 22 Cluster B): Both the DtoH and HtoD copies in the host-staging path must execute on the **same** `target_stream`. Using different streams was the cause of a race that intermittently corrupted output at SF100 Q11 — closed by collapsing producer + DtoH leg + HtoD leg onto a single stream in `alloc_and_peer_copy_async`.
- **Device-context propagation** (Phase 23 Plan 23-06): The HtoD `cudaMemcpyAsync` in `alloc_and_peer_copy_async` is wrapped in an `rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}}` so the destination device's CUDA context is active during the copy. The outer `convert_gpu_to_gpu`'s `target_guard` does not propagate through `reconstruct_column_p2p` → `alloc_and_peer_copy_async`; the inner guard fixes a `cudaErrorInvalidValue` on broken-peer-DMA hardware.
- **Probe device-context restore** (Phase 23 Plan 23-07): `run_p2p_probe_locked` ends with a paired `cudaSetDevice` to restore the caller's device context, not a hardcoded `cudaSetDevice(0)` which would clobber the caller's RAII guard.

## Multi-GPU-Safe Parquet I/O

The I/O path is **kvikio-free** (GATE-22.1-A invariant). The reasoning: cudf's bundled `file_source` factory uses kvikio, which binds the file handle to whichever CUDA context was active at construction time. In multi-GPU execution that's a hidden source of corruption — a file_handle bound to GPU 0 will silently funnel reads through GPU 0 even when the consumer is on GPU 1.

The Sirius path:

1. **All file reads go through `sirius_ioctx::make_datasource(io_object)`.** Never `cudf::io::datasource::create(path)` and never `cudf::io::source_info{path}`. A grep gate enforces this in `src/`.
2. **`sirius_ioctx` is per-GPU.** Construction binds `cudaSetDevice(gpu_id)` via RAII; the underlying `uring_reactor` runs in a worker thread that holds that device's context.
3. **Schemes are resolved through `datasource_registry_`.** A scan task that wants to read `file:///path/to/x.parquet` passes the URL to the registry, which returns a factory bound to the per-GPU `sirius_ioctx`. No silent fallback.
4. **Pin-table chunk reads route through the owning GPU's ioctx.** Chunk `i` reads through `gpu_ioctxs_.at(chunk_memory_spaces[i]->get_device_id())`. The read lands directly in the target GPU's pinned memory region.

The 7 historical sites that bypassed this contract (kvikio-fallback in `sirius_gpu_parquet_scan_operator`, `sirius_extension`, `iceberg_metadata_reader`, `datasource_factory`, `parquet_split_provider`) were all migrated to `sirius_ioctx::make_datasource` over Phase 22.1. The unit-test fallback in `parquet_split_provider:295` (D-08 site #7) was deleted outright.

## Memory Pressure: Reservations and Downgrade

Every allocation through a `memory_space`'s default allocator goes through a `reservation_aware_resource_adaptor` that:

1. **Tracks bytes per-thread.** Per-thread budget is established via `attach_reservation_to_tracker(stream, reservation)`. The reservation tells the adaptor how many bytes this stream can claim before raising `rmm::out_of_memory`.
2. **Throws on OOM instead of segfaulting.** `do_allocate` calls `upstream->allocate`, which throws `rmm::out_of_memory` when the pool can't satisfy. The OOM unwinds back to `gpu_pipeline_task::execute`, which catches it and raises `oom_reschedule_exception`.
3. **Records OOM peak to `pipeline_memory_history`.** Subsequent estimates of the same operator's peak memory consult the history; the scheduler uses these estimates to size reservations and to decide whether to schedule a task or wait for a downgrade.

On OOM the engine triggers a **downgrade**:

- The `downgrade_executor` monitors per-tier memory pressure. When a tier crosses its threshold, it picks candidate batches (LRU within the tier) and converts them to the next-cheaper tier (`GPU` → `HOST`, `HOST` → `DISK`).
- Downgrades are stream-ordered: `cudaFreeAsync` on the source tier, then `cudaMallocFromPoolAsync` + memcpy on the target tier. The new tier's `memory_space` becomes the batch's owning space.
- Phase 22.2 K.6 fix: the executor only calls `cudaSetDevice` when its `_space_id.tier == GPU`. HOST-tier downgrade workers operate on host memory only — calling `cudaSetDevice(-1)` for a HOST-tier `space_id` was the K.6 SIGSEGV root cause.

After a downgrade frees enough space, the rescheduled task retries. The reservation tracking ensures the same task doesn't try to allocate more bytes than its reservation; if the reservation itself can't grow, the task fails out of the pipeline.

## Concurrency Invariants (Quick Reference)

| Invariant | Where enforced | Why |
|-----------|---------------|-----|
| Same-stream for DtoH+HtoD in `alloc_and_peer_copy_async` | `cucascade/src/data/representation_converter.cpp` | Prevents Phase 22 Cluster B race seen at SF100 Q11 |
| `dst_guard` around HtoD memcpy in `alloc_and_peer_copy_async` | Same file (Phase 23 fix) | Outer `target_guard` doesn't propagate through `reconstruct_column_p2p`; broken-peer-DMA hardware needs the inner guard |
| `run_p2p_probe_locked` restores caller's device context on exit | `cucascade/src/memory/common.cpp` (Phase 23 fix) | Probe was hardcoding `cudaSetDevice(0)`, clobbering caller's RAII guard |
| `cudaDeviceSynchronize` per GPU before `cudaMemPoolDestroy` | `src/memory/sirius_memory_reservation_manager.cpp` (post-Phase-24 fix) | Pending `cudaFreeAsync` against a soon-destroyed pool corrupts the driver's per-device pool list |
| Per-GPU `sirius_ioctx` binds device via RAII in `uring_reactor` setup | `src/io/uring/uring_reactor.cpp` | Reactor thread must hold the right device context for all uring submissions |
| `_per_thread_init` in `downgrade_executor` gated on `tier == GPU` | `src/downgrade/downgrade_executor.cpp` (Phase 22.2 K.6) | HOST-tier workers must not call `cudaSetDevice(-1)` |
| `chunk_memory_spaces[i]` parallel to `data_batches_by_column[col][i]` | `src/include/scan_manager/sirius_scan_manager.hpp` | Pin-table merge must preserve owning-space per chunk (Phase 22 Pitfall 3) |
| HYG-02 invariant: 0 new `rmm::cuda_stream_default` in `src/` outside `legacy/` | grep gate | Default-stream usage breaks per-task-device contract under SCHED-RR |
| GATE-22.1-A: 0 `cudf::io::datasource::create(path)` / `source_info{path}` in `src/` | grep gate | Any file-path datasource silently uses kvikio |

## Hardware Caveats

- **Consumer-grade GPUs (e.g., RTX 6000 Ada Generation)** may advertise P2P peer access via `cudaDeviceCanAccessPeer` but silently fail actual DMA transfers. The empirical probe (`probe_peer_dma_works`) catches this at startup and routes affected pairs through host-staging.
- **NUMA topology discovery** runs at startup. The `topology_discovery` component reads `/sys/class/drm/card*/device/numa_node` (and equivalents) to determine each GPU's NUMA node, then pairs each GPU with its NUMA-local host region. Without this, host-tier downgrade traffic crosses the NUMA boundary, halving effective bandwidth.
- **`CUDA_VISIBLE_DEVICES`** is the canonical way to scope which GPUs Sirius uses. Setting it to `0` runs the engine in single-GPU mode (no distribution, no cross-GPU transfers); setting it to `0,1` enables full multi-GPU.
- **Single-process scope.** Sirius runs as a single OS process. Multi-process / multi-node execution is out of scope; the user is responsible for partitioning at a higher layer if needed.

## Key Source Files

| Path | Role |
|------|------|
| `src/sirius_context.{hpp,cpp}` | `SiriusContext`, per-GPU initialization, datasource registry |
| `src/memory/sirius_memory_reservation_manager.{hpp,cpp}` | Extends `cucascade::memory_reservation_manager`; sets cudf device resource refs per GPU; synchronizes on destruction |
| `src/include/scan_manager/sirius_scan_manager.hpp` | `pinned_entry`, `chunk_memory_spaces` invariant |
| `src/scan_manager/parquet_split_provider.cpp` + `cached_split_provider.cpp` | Per-chunk memory_space lookup, GPU-mode + host-mode split providers |
| `src/io/datasource_factory.{hpp,cpp}` | Strict scheme registry, kvikio-free file resolution |
| `src/io/uring/uring_reactor.cpp` | Per-GPU `uring_reactor` with RAII `cudaSetDevice` |
| `src/op/scan/sirius_gpu_parquet_scan_operator.cpp` | Multi-GPU-aware GPU parquet scan |
| `src/op/scan/duckdb_scan_executor.cpp` | NUMA-preference reservation requests |
| `src/include/pipeline/gpu_pipeline_task.hpp` | `preferred_device_id` two-level lookup |
| `src/pipeline/gpu_pipeline_executor.cpp` + `task_scheduler.cpp` | SCHED-RR distribution, locality scoring |
| `src/downgrade/downgrade_executor.cpp` | Per-tier downgrade workers, K.6-gated `cudaSetDevice` |
| `cucascade/src/data/representation_converter.cpp` | `convert_gpu_to_gpu` / `alloc_and_peer_copy_async` with peer-DMA probe + dst_guard |
| `cucascade/src/memory/common.cpp` | `probe_peer_dma_works`, `run_p2p_probe_locked` |

## Related Documentation

- [Architecture Overview](architecture-overview.md) — component diagram, thread model, ownership hierarchy (covers single-GPU too)
- [Pipeline Execution](pipeline-execution.md) — Per-task-device contract under SCHED-RR (the deeper task-routing contract)
- [Memory Management](memory-management.md) — cuCascade tiers, reservations, downgrade executor (mechanics of memory_spaces)
- [Scan](scan.md) — scan path, pin tables, cache, parquet/iceberg readers
