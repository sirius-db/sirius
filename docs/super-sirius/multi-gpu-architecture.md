# Multi-GPU Architecture

How Sirius executes SQL across every GPU on a single node — the components, data residency model, scheduling rules, and concurrency invariants.

## The Mental Model

Sirius treats a multi-GPU host as a set of cooperating execution units that share host memory, disk, and optional remote object-store spill spaces but have private device memory. The engine's job is to:

1. **Place data** so that each row lives in the GPU (or NUMA host region, or local disk) closest to where it will be consumed
2. **Place tasks** so each task runs on the GPU that already holds its inputs
3. **Move data when locality fails** — across GPUs via peer-copy calls or explicit host staging, to host, disk, or an enabled remote spill space on memory pressure, and back to GPU when the next task needs it

The user writes plain SQL. The Sirius optimizer extension intercepts the physical plan and routes supported operators to the multi-GPU execution path; unsupported plans fall back to DuckDB's CPU engine transparently.

```sql
-- Plain SQL — transparently routed to multi-GPU execution
LOAD 'sirius.duckdb_extension';
SELECT l_returnflag, SUM(l_quantity) FROM lineitem GROUP BY l_returnflag;
```

The engine assumes a single process pinning the configured subset of visible GPUs
(`CUDA_VISIBLE_DEVICES` bounds which GPUs Sirius can use; `sirius.topology` selects from that
set). There is no notion of distributed multi-node execution in this codebase.

## Tier Hierarchy

Every byte of data lives in exactly one **memory tier** at any moment. Tiers, fastest to slowest:

| Tier | Backed by | Capacity per host | Typical purpose |
|------|-----------|-------------------|-----------------|
| `GPU` | `cuda_async_memory_resource` (one pool per device) | Bounded by GPU device memory × `usage_limit_fraction` | Active query data |
| `HOST` | NUMA-local pinned host memory | Set by config: `memory.host.capacity_bytes` (per NUMA host memory resource) | Downgrade target when GPU is full |
| `DISK` | Raw and `.hpln` spill files via a per-space `std::shared_ptr<cucascade::io::ioctx>` ([I/O ownership](memory-management.md#disk-io-and-ownership)) | Reservation bound via `set_disk_mounting_point(disk_id, capacity, path)` | Local spill target, preferred over REMOTE |
| `REMOTE` | S3-compatible spill objects via an attached REST `ioctx`, shared across GPUs | `memory.remote.capacity_bytes` (default 1 PiB): in-flight reservation bound, **not a stored-object quota** | Optional spill target after DISK; eligible only with an attached `ioctx` |

`HOST` is partitioned by NUMA node — each GPU has a paired host region on its NUMA-local memory controller for fast downgrade. This is configured via `cucascade::memory::reservation_manager_configurator::use_host_per_gpu()` at startup.

`REMOTE` is disabled by default (empty `sirius.memory.remote.uri`). It is a spill tier, not
distributed query execution. GPU/HOST downgrade workers consider usable REMOTE spaces after
DISK targets, or without a DISK tier; there is **no DISK→REMOTE demotion or direct DISK↔REMOTE
converter**. See [Remote Memory](configuration.md#remote-memory-siriusmemoryremote) for high-level
`memory.remote`, low-level `space.remote[]`, and initialization-time credentials.

Conceptually (the lower branches are alternative targets, not DISK→REMOTE transfers):

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │  ← Tier::GPU
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     └─────┬──────┘            └─────┬──────┘
           │                         │
      ┌────┴────┐               ┌────┴────┐
      │ Host    │               │ Host    │
      │ (NUMA0) │               │ (NUMA1) │      ← Tier::HOST (per-NUMA, shared by sibling GPUs)
      └────┬────┘               └────┬────┘
           └─────────────┬───────────┘
                         ├───────────────────┐
                    ┌────┴────┐         ┌────┴─────┐
                    │  Disk   │         │  Remote  │
                    └─────────┘         └──────────┘
                    Tier::DISK          Tier::REMOTE (optional, attached ioctx only)
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
│     └─ (Tier::DISK and optional Tier::REMOTE shared by all GPUs)
│
├─ sirius_scan_manager
│  ├─ _io_ctx          local ioctx: uring (cuFile is single-GPU only)
│  ├─ _ioctx_registry  path → backend type + factory (uring / restful / cufile)
│  ├─ _routed_io_ctxs  one ioctx per routed backend type, built on first use
│  │
│  │  manages pinned_entry records; populated by pin_table
│  └─ pinned_entry
│     ├─ data_batches_by_column  (per-column DataBatch chunks — GPU tier, uncompressed)
│     ├─ chunk_memory_spaces     (parallel vector — owning memory_space per chunk, GPU tier)
│     ├─ host_chunks             (per-chunk idata_representation — HOST tier; may mix
│     │                           host_data_representation and compressed_host_representation)
│     ├─ device_chunks           (device_pin_chunk — GPU-tier compression-enabled storage;
│     │                           takes priority over data_batches_by_column when non-empty)
│     └─ column_storage          (chunk-major carrier matrix — see compressed docs)
│
├─ gpu_pipeline_executor + task_creator + task_scheduler
│                                Phase 2 wire_data_repositories Phase-2 split:
│                                converter emits pure-data repository_wiring,
│                                engine calls materialize_repository_wiring()
│
└─ downgrade_executor (one per GPU's HOST-tier path)
   │  monitors memory pressure; GPU→HOST, then DISK or usable REMOTE targets
   └─ Phase 22.2 K.6: only runs cudaSetDevice for tier == GPU
```

Ownership goes one direction: `SiriusContext` owns everything below it. Connections register `sirius_state` (a `shared_ptr<SiriusContext>`) into DuckDB's `ClientContext::registered_state` at `OnConnectionOpened`, and remove it at `OnConnectionClosed`.

## Per-GPU Initialization

`SiriusContext::initialize()` is the single point where multi-GPU state comes online. The sequence (`src/sirius_context.cpp`):

1. **Discover GPUs.** `topology_discovery` enumerates devices visible to the process (respects `CUDA_VISIBLE_DEVICES`), and records each GPU's NUMA node. A NUMA node the OS cannot resolve stays `-1` ("unknown") — the sentinel is carried through rather than normalized to node 0, and consumers fall back explicitly (e.g. the first host space) when they meet it.
2. **Build the topology index.** After the memory manager is populated, `initialize()` builds one immutable `sirius::memory::topology_index` (`src/memory/topology_index.hpp`) — the single NUMA↔GPU map — and injects it (as `shared_ptr<const>`) into the `task_creator`, the `sirius_scan_manager`, the NUMA-aware small pinned host resource, and the per-GPU downgrade configs (`numa_node_of(device_id)`). `terminate()` resets it. Locality decisions all route through `topology_index::gpus_of()` / `numa_node_of()` rather than component-private maps.
3. **Build memory spaces.** A `reservation_manager_configurator` is configured with per-GPU usage limits, per-host capacities, optional disk mounts and remote URI prefixes, and NUMA pairings. `builder.build()` produces `memory_space_config`s, which `sirius_memory_reservation_manager` consumes to construct all tier × gpu spaces. Once the manager exists, Sirius builds and attaches each REMOTE space's REST `ioctx` using `set_object_store_config()` / standard AWS environment credentials and the REST transport/TLS policy. Missing credentials log a warning and leave the space without a context, so downgrade skips it. A later SQL `CREATE SECRET` does not attach credentials or enable REMOTE spilling.
4. **Grant actual pool access, then install per-GPU device resource refs.** The manager calls `sirius::memory::try_grant_pool_access` for each ordered pair of distinct configured GPU spaces before installing any resource reference. Only ordinary RMM async pools have established allocation properties; unknown resources remain unavailable. Unsupported or failed byte verification is nonfatal, while a CUDA error aborts construction. No ordinary `cudaDeviceEnablePeerAccess` call or device-default-pool grant is made. For each GPU, `sirius_memory_reservation_manager`'s constructor sets that GPU's `cuda_async_memory_resource` as cudf's `current_device_resource_ref` (saving the previous ref for restoration on shutdown). This ensures cudf operations on each GPU allocate through that GPU's reservation-tracked pool.
5. **Construct the scan manager and its local ioctx.** The scan manager creates one `uring_ioctx` for local files. Its reactor pool accepts device-read requests for any visible GPU.
6. **Register the path-routed backends.** The scan manager's `io_context_registry` holds one entry per backend type (`uring` / `restful` / `cufile`), each a path checker and a factory, with no catch-all. A REST ioctx for `s3://` is built on first use and shared by all GPUs; its workers drive HTTP/1.1 over io_uring, with kTLS or user-space TLS for HTTPS endpoints.
7. **Restore cudf device-resource refs on shutdown.** `sirius_memory_reservation_manager`'s destructor first synchronizes each managed GPU (`cudaDeviceSynchronize()`) so pending `cudaFreeAsync` operations against the soon-to-be-destroyed pool complete, then restores cudf's previous device resource ref. The sync step is critical — without it, tests that leave async deallocations un-synchronized can corrupt the driver's per-device pool list and crash the next manager construction on the same device.

After `initialize()`, the engine has per-GPU memory pools, the default backend's I/O reactor pool (other backends' pools are built on first use), path-based datasource routing, and a manager that translates `(Tier, gpu_id)` into an allocator.

## Data Residency: Pin Tables

Tables that benefit from being resident in GPU memory get **pinned** via the SQL surface:

```sql
CALL pin_table('lineitem');                    -- distribute chunks across GPU memory spaces (default)
CALL pin_table('lineitem', tier => 'host');    -- pin into host memory only
```

The pin pipeline (`src/pin_table.cpp`, `src/scan_manager/`):

1. **Open the parquet files.** `PinTableFunction` reads every file of the pin through the scan manager's ioctx.
2. **Enumerate and coalesce.** Parquet metadata yields row groups. The coalescer bundles them into batches sized to `scan_task_batch_size`. A large file spans several batches; small files share one.
3. **Distribute round-robin per batch.** `round_robin_strategy` assigns each batch the next GPU memory space from an atomic cursor. The unit is a batch, so one file's row groups can land on different GPUs. Each pin call starts the round-robin cursor at 0. Identical batch sequences therefore produce the same device sequence. Batch boundaries depend on the projected column set, so pinning the same table with different columns can slice it differently. The merge path rejects different batch layouts.
4. **Materialize each batch into its target space.** The pin path makes the target GPU current (`cuda_set_device_raii`), takes a stream from that GPU's memory space, and passes the space's allocator to `read_parquet`. The decoded columns land on the owning GPU.
5. **Record the pinned entry.** `pinned_entry { data_batches_by_column, chunk_memory_spaces }` is stored on the `sirius_scan_manager`. The `chunk_memory_spaces` vector is parallel to `data_batches_by_column[col_idx]` — `chunk_memory_spaces[i]` is the owning space for chunk `i` regardless of which column you're looking at.

Repeat invocations of `pin_table('lineitem', ...)` are idempotent — duplicates dropped, existing `chunk_memory_spaces` preserved (Phase 22 Pitfall 3 invariant: any merge must verify `chunk_memory_spaces` integrity).

When the HOST-tier pinning path is used (`2e197c6` upstream feature, integrated in Phase 24), `pin_table` builds one `cucascade::host_data_representation` per batch on the host space NUMA-local to the GPU that produced it, and stores them in `pinned_entry::host_chunks`. It falls back to the first host space when the GPU's NUMA node is unknown or no matching host space exists; subsequent scans go through the cached provider in host mode, which slices the host chunks per query and converts back to GPU only when a scan task starts. When pin-table compression is enabled, individual host chunks may be `compressed_host_representation` instead (and GPU-tier compressed storage lives in `device_chunks`), with per-chunk dispatch at serve time — see [Compressed Pinning](compressed-pinning.md).

The two tiers record placement differently. A GPU-tier entry fills `chunk_memory_spaces`. A HOST-tier entry leaves it empty and keeps per-chunk placement in each `host_data_representation`. `pinned_entry::memory_space` is metadata for a host entry, and the MVCC mask path expands it into a uniform vector. Serve-time validation of `chunk_memory_spaces` runs on the GPU tier only.

## Scan-Time: Routing to the Right GPU

When a query selects from a pinned table, `cached_databatch_provider` walks `pinned_entry`'s chunks. For each chunk:

1. **Attach the chunk's recorded memory space** to the emitted batch: `chunk_memory_spaces[i]` on the GPU tier, the chunk's host representation on the HOST tier. The provider sets no device id.
2. **`task_creator` derives `preferred_device_id`** from that memory space. A GPU-tier space yields its own device id. A HOST-tier space yields a NUMA-local GPU, round-robin across the GPUs on that node.
3. **The scheduler routes the task** to that GPU's worker thread.

GPU-tier chunks run on their owning GPU without a peer copy. HOST-tier chunks are copied to a NUMA-local GPU when the task starts. `task_creator` applies this rule last, after upstream-split, partition-pin, and byte-locality preferences.

## Task Scheduling: Ready Devices and Locality

For tasks that aren't bound to a specific chunk (e.g., downstream operators consuming many input batches), the task creator computes a **locality score**:

```
locality(task, gpu_id) = bytes of task input data already on gpu_id
```

The task creator records the selected device as `preferred_device_id`, falling back to the task's NUMA-paired GPU if locality is tied or all data is on HOST. The scheduler then matches tasks to GPU executors that have signaled readiness: it first looks for a task preferring that exact GPU, then for a task with no preference. A preference is binding; a preference-less task may be claimed by any ready GPU. There is no round-robin ordering guarantee in the scheduler matcher.

The task creator resolves a task's `preferred_device_id` in priority order:

1. **Upstream input-data preference** — a fresh-read scan split carries the device the scan manager stamped onto it.
2. **Partition device pin** (see below) — partitioned operator inputs.
3. **Data-locality by bytes** — the GPU already holding the most of the task's input bytes.
4. **NUMA-affinity** — when all input lives on HOST, a GPU on the same NUMA node (round-robin when that NUMA hosts several GPUs).

Two-level `preferred_device_id`:

- `gpu_pipeline_task_local_state::_preferred_device_id` — per-task override (winner)
- `sirius_pipeline_task_global_state::_preferred_device_id` — pipeline-level default

`task->get_preferred_device_id()` checks local first, falls back to global.

See [`pipeline-execution.md`](pipeline-execution.md) "Per-task-device contract under SCHED-RR" for the deeper contract.

### Partition device pin for cuco-backed operators

BUILD_PROBE hash joins retain a per-partition GPU hash table that is **only valid on the GPU it was built on**. A stream bound to GPU A that touches a cuco counter built under GPU B trips `cudaErrorInvalidValue` in cuco's `counter_storage`. The device a partition runs on is therefore a **correctness constraint, not just a locality preference**: every task of a given partition (its build and all its probes) must land on the same GPU.

Where each partition runs is an explicit **partition placement** (`src/op/partition_placement.hpp`): one device id per partition of an exchange. The partition-sizing consumer (hash join, `grouped_aggregate_merge`, dense count join, nested loop join) chooses it together with the partition count and returns it in `partition_strategy::placement`. The consumer's input policy (`bind_partition_input` and `bind_partition_inputs` in `consumer_partition_policy.cpp`) installs it on the consumer, and `partition_exchange` gives every bound PARTITION the resulting layout, which shares it with every operator the partitions sink into (a join's CONCATs). Each of those operators stamps the partition's device onto the `partitioned_operator_data` it emits (`set_preferred_device_id`). The task creator honors that stamp like any other producer preference, so every task of a pinned partition lands on one GPU.

Nested-loop joins use `unpinned(1)`: their CONCAT tasks and join tasks can follow input locality. Single-partition STANDARD hash joins, grouped-aggregate merges, dense-count joins, and the empty-build fallback also use `unpinned(1)` because they retain no GPU state between tasks. Multiple partitions keep round-robin placement to distribute work. BUILD_PROBE joins remain pinned even with one partition.

The default policy for pinned partitions is `partition_placement::round_robin(P, active_gpu_ids)` — partition `p` on `active_gpu_ids[p % n]`. A broadcast join uses `one_per_device` (one slot per GPU, which the probe sink inverts with `first_partition_for_device` to keep each probe batch on its current GPU), and a single-partition BUILD_PROBE join picks its GPU with `select_gpu_subset(ids, 1, operator_id)` so several small joins spread across GPUs. The consumer's GPU list is the query's **admitted GPU set**, handed to every partition consumer by `sirius_pipeline_converter::configure_partition_consumers` from `pipeline_build_context::active_gpu_ids()` — the device ids that actually have a GPU executor, narrowed per query by admission (`topology.gpus_per_query`). The consumer input policy (`install_placement` in `consumer_partition_policy.cpp`) rejects a placement naming a GPU outside that set: a task that reached the scheduler pinned to a device without an executor would remain queued indefinitely because preferences are binding. The admitted set is fixed for the life of a query, so running an exchange on fewer GPUs is expressed as a placement over a subset, never by changing the set.

The pin also survives OOM reschedule. When a partitioned task OOMs and is rebuilt with a fresh `local_state`, the per-task `preferred_device_id` is carried forward; without it the rescheduled task would demote to "no preference" and scatter. (A pin held on the pipeline-level global state already survives reconstruction, so only the local-state pin needs to be copied.)

## Cross-GPU Data Movement

When an operator must consume data on GPU A that lives on GPU B (e.g., a hash join's probe side has chunks scattered across all GPUs), `lock_or_prepare_batch` (`src/pipeline/batch_lock_utils.hpp`) **clones the batch into the consumer's memory space under a shared (read) lock** via `read_only_data_batch::clone_to`. The source batch is never exclusively locked and never mutated: it stays resident on GPU B for consumers local to that device, concurrent readers proceed during the transfer, and the source drops back to the idle state as soon as the prepare completes — making it immediately downgrade-eligible. Source lifetime is ownership-driven: repositories and other tasks holding the batch keep it alive, and the consuming task releases its own pin on the original right after prepare, so a single-consumer source is freed as soon as its clone exists. The clone's allocation is charged to the consuming task's memory reservation, and the reservation estimator counts GPU inputs residing in a different memory space in `bytes_to_materialize_input`.

Host-, disk-, and remote-resident inputs intentionally keep **move semantics**: `lock_or_prepare_batch` upgrades them to the GPU in place, freeing the spilled copy — the common case is a single consumer re-materializing a downgraded batch.

The underlying byte transfer is `cucascade::convert_gpu_to_gpu` (in `cucascade/src/cudf/representation_converter_builtins.cpp`), which orders the source writer and synchronizes its copy stream before returning. The upstream converter still chooses `cudaMemcpyPeerAsync` or explicit host staging through its directional `probe_peer_dma_works` query; Sirius does not replace that upstream selection policy.

cuCascade verifies bytes in both directions using private ordinary async pools. Its first verification scans all visible ordered device pairs under a process-wide lock and caches results; this does not test allocations from Sirius's pools or grant them access. `grant_pool_peer_access` separately grants permission on the supplied actual pool after that verification. It leaves ordinary peer state and unrelated pools unchanged, restores the calling device, and preserves CUDA failures in its typed result. Neither verification nor a grant measures the physical copy route. A failed grant does not revoke existing permissions or establish that a later copy will be staged.

Sirius ordinary dynamic-filter replication and late materialization share `memory::enqueue_device_copy`. It checks the actual source pool for its destination before selecting `cudaMemcpyPeerAsync`; unavailable capability uses explicit managed HOST staging for the entire transfer, submitting each leg as a batch where supported and waiting once per leg. Source readiness is explicit: filters supply already-completed storage, and late materialization supplies recorded selection/gather events. Accumulated Bloom admission checks every required ordered pair of live replica pools before constructing its builder and declines the optional filter if capability is unavailable. CUDA errors propagate from both callers. The transfer route `peer_copy` records API selection only.

**Concurrency invariants for cross-GPU transfers:**

- **Same-stream invariant** (Phase 22 Cluster B): Both the DtoH and HtoD copies in the host-staging path must execute on the **same** `target_stream`. Using different streams was the cause of a race that intermittently corrupted output at SF100 Q11 — closed by collapsing producer + DtoH leg + HtoD leg onto a single stream in `alloc_and_peer_copy_async`.
- **Device-context propagation** (Phase 23 Plan 23-06): The HtoD `cudaMemcpyAsync` in `alloc_and_peer_copy_async` is wrapped in an `rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}}` so the destination device's CUDA context is active during the copy. The outer `convert_gpu_to_gpu`'s `target_guard` does not propagate through `reconstruct_column_p2p` → `alloc_and_peer_copy_async`; the inner guard fixes a `cudaErrorInvalidValue` on broken-peer-DMA hardware.
- **Probe device-context restore**: private-pool verification and `grant_pool_peer_access` restore the calling device, preserving the caller's RAII guard.

## Multi-GPU-Safe Parquet I/O

Multi-GPU parquet reads, local and `s3://`, use cuCascade's native I/O stack rather than cuDF's path-based file-source factory, which can bind a file handle to the CUDA context active at construction. A handle bound to GPU 0 must not silently funnel reads through GPU 0 when the consumer is on GPU 1. `sirius_config::enforce_native_backend_for_multi_gpu()` therefore forces `scan_manager_config::backend = io_backend::native` whenever more than one GPU memory space is configured, with a warning for a non-native selection. The existing `backend: cufile` selection remains single-GPU only.

The kvikIO backend and registry catch-all have been removed for every GPU count. `backend: kvikio` and the `scan_manager.kvikio` block are rejected with an error directing users to `native`; neither is an alias or an ignored block.

The native path:

1. **Managed file reads go through `sirius::io::open_datasource(io_ctx, path)`**, which normalizes the path and calls cuCascade's `cucascade::io::open_datasource`. Never `cudf::io::datasource::create(path)` and never `cudf::io::source_info{path}`.
2. **An ioctx is shared across GPUs.** The ioctx and its reactors bind no device at construction. A device read captures the caller's current CUDA device when it is submitted and carries it on the request. The reactor that serves the request makes that device current for the H2D copy and records the copy's event on it.
3. **Paths are resolved through `io_context_registry`.** The registry runs each backend's path checker and returns a backend type. Uring's checker is a filesystem stat, applied after the scan manager strips a leading `file://`. Local files use the shared uring ioctx, `s3://` the REST ioctx. An unclaimed path has no backend. Raw `http://` / `https://` input paths fail explicitly: use `s3://` and configure the HTTP(S) object-store endpoint instead. A null datasource for a claimed path means the backend could not be built or started, for example an unconfigured object store or unavailable io_uring.
4. **Pin-table placement is carried by `memory_space`.** All files of a pin go through the same ioctx. The destination GPU comes from the current-device guard and the target space's allocator, and is recorded per chunk for task creation.

Every managed read on the multi-GPU path resolves through `sirius::io::open_datasource` — the unified `sirius_gpu_scan_operator`, the split providers, `sirius_extension`, and the pin path all route through it. The parquet reader wraps `cucascade::io::datasource`s through the `datasource*` overload. Unclaimed schemes are not rerouted through cuDF's path-based factory.

## Memory Pressure: Reservations and Downgrade

Every allocation through a `memory_space`'s default allocator goes through a `reservation_aware_resource_adaptor` that:

1. **Tracks bytes per-thread.** Per-thread budget is established via `attach_reservation_to_tracker(stream, reservation)`. The reservation tells the adaptor how many bytes this stream can claim before raising `rmm::out_of_memory`.
2. **Throws on OOM instead of segfaulting.** `do_allocate` calls `upstream->allocate`, which throws `rmm::out_of_memory` when the pool can't satisfy. The OOM unwinds back to `gpu_pipeline_task::execute`, which catches it and raises `oom_reschedule_exception`.
3. **Records OOM peak to `pipeline_memory_history`.** Subsequent estimates of the same operator's peak memory consult the history; the scheduler uses these estimates to size reservations and to decide whether to schedule a task or wait for a downgrade.

On OOM the engine triggers a **downgrade**:

- The `downgrade_executor` monitors per-tier memory pressure. When a tier crosses its threshold, it picks candidate batches (LRU within the tier) and tries cheaper targets: HOST for GPU data, then DISK, then REMOTE for GPU/HOST sources. Only REMOTE spaces with an attached `ioctx` are viable targets; REMOTE does not require DISK, and DISK data is not demoted to REMOTE.
- In-memory transfers are stream-ordered. Spill conversions use the target tier's I/O context; a REMOTE write must be committed before the new representation becomes visible. The new tier's `memory_space` becomes the batch's owning space.
- Phase 22.2 K.6 fix: the executor only calls `cudaSetDevice` when its `_space_id.tier == GPU`. HOST-tier downgrade workers operate on host memory only — calling `cudaSetDevice(-1)` for a HOST-tier `space_id` was the K.6 SIGSEGV root cause.

After a downgrade frees enough space, the rescheduled task retries. The reservation tracking ensures the same task doesn't try to allocate more bytes than its reservation; if the reservation itself can't grow, the task fails out of the pipeline.

REMOTE capacity bounds reservations / in-flight conversions, not the total bytes of retained
spill objects. Normal representation release deletes its object, but crashes or failed deletion
can leave objects and multipart uploads behind. A bucket lifecycle policy with **spill-prefix
expiry and `AbortIncompleteMultipartUpload` is required**; choose expiry longer than the lifetime
of a live spill. There is no crash-leftover startup/shutdown sweep.

## Concurrency Invariants (Quick Reference)

| Invariant | Where enforced | Why |
|-----------|---------------|-----|
| Same-stream for DtoH+HtoD in `alloc_and_peer_copy_async` | `cucascade/src/cudf/representation_converter_builtins.cpp` | Prevents Phase 22 Cluster B race seen at SF100 Q11 |
| `dst_guard` around HtoD memcpy in `alloc_and_peer_copy_async` | Same file (Phase 23 fix) | Outer `target_guard` doesn't propagate through `reconstruct_column_p2p`; broken-peer-DMA hardware needs the inner guard |
| Private-pool verification and `grant_pool_peer_access` restore the calling device | `cucascade/src/memory/common.cpp` | Preserve the caller's RAII device scope |
| `cudaDeviceSynchronize` per GPU before `cudaMemPoolDestroy` | `src/memory/sirius_memory_reservation_manager.cpp` (post-Phase-24 fix) | Pending `cudaFreeAsync` against a soon-destroyed pool corrupts the driver's per-device pool list |
| A device read carries the caller's device id; the reactor sets that device for the H2D copy | `cucascade/include/cucascade/io/templated_ioctx.hpp`, `cucascade/src/io/uring/uring_reactor.cpp` | The ioctx is shared across GPUs. Copying without setting the device lands the bytes on whichever GPU the reactor thread happens to have current |
| `_per_thread_init` in `downgrade_executor` gated on `tier == GPU` | `src/downgrade/downgrade_executor.cpp` (Phase 22.2 K.6) | HOST-tier workers must not call `cudaSetDevice(-1)` |
| `chunk_memory_spaces[i]` parallel to `data_batches_by_column[col][i]` | `src/scan_manager/sirius_scan_manager.hpp` | Pin-table merge must preserve owning-space per chunk (Phase 22 Pitfall 3) |
| All tasks of a pinned partition run on the one admitted GPU its consumer's `partition_strategy::placement` names, stamped on the `partitioned_operator_data` by its emitter; pin preserved across OOM reschedule | `src/op/partition_placement.hpp`, `src/op/sirius_physical_partition.cpp`, `src/pipeline/gpu_pipeline_executor.cpp` | A cuco hash table is valid only on the GPU it was built on; cross-device access trips `cudaErrorInvalidValue`. Placements drawn from the admitted set avoid phantom pins when `num_gpus` < physical GPU count, and keep a query off devices it was not admitted onto |
| Locality-derived device preferences (operator hint, GPU-resident bytes, NUMA `gpus_of()`, cached-chunk home) clamped into `_active_gpu_ids` | `src/creator/task_creator.cpp` | These are computed from where data lives, not from the admitted set, so any of them can name an excluded device — and the scheduler treats a preference as binding |
| HYG-02 invariant: 0 new `rmm::cuda_stream_default` in `src/` outside `legacy/` | grep gate | Default-stream usage breaks per-task-device contract under SCHED-RR |
| Accumulated Bloom contributions fold into an exclusive per-GPU partial stream (`cudaStreamWaitEvent` on a transient event) instead of waiting on the host | `src/cuda/sirius_dynamic_bloom_filter.cu` (`accumulated_bloom_builder::enqueue_add`) | Every later copy out of, or free of, a partial array is ordered after all inserts into it, with no per-task host sync |
| Accumulated Bloom publication requires actual-pool grants for every ordered pair of replica GPUs | `src/op/dynamic_filter/dynamic_filter_publisher.cpp` (`try_begin_accumulation`) | The chunk pipeline issues `cudaMemcpyPeerAsync`; unavailable pool capability declines the optional filter |
| Multi-GPU parquet reads resolve to `cucascade::io::datasource`, not cuDF's path-based factory | `sirius_config::enforce_native_backend_for_multi_gpu()` forces `backend: native` when >1 GPU is configured | A path-based file source can bind to a single CUDA context; native device reads carry the caller's device |

## Hardware Caveats

- **Hardware capability and observed bytes are separate.** `cudaDeviceCanAccessPeer` alone is insufficient. The typed grant distinguishes unsupported capability, private-pool byte-verification failure, and CUDA errors; only the first two are nonfatal capability outcomes.
- **NUMA topology discovery** runs at startup. The `topology_discovery` component reads `/sys/class/drm/card*/device/numa_node` (and equivalents) to determine each GPU's NUMA node, then pairs each GPU with its NUMA-local host region. Without this, host-tier downgrade traffic crosses the NUMA boundary, halving effective bandwidth.
- **`CUDA_VISIBLE_DEVICES`** scopes which GPUs Sirius may use. With the automatic topology
  default, `CUDA_VISIBLE_DEVICES=0,1` enables both visible GPUs. Set `sirius.topology.num_gpus` to
  use only a prefix of that visible set, or use `gpu_ids` to select a non-prefix subset.
- **Single-process scope.** Sirius runs as a single OS process. Multi-process / multi-node execution is out of scope; the user is responsible for partitioning at a higher layer if needed.

## Key Source Files

| Path | Role |
|------|------|
| `src/sirius_context.{hpp,cpp}` | `SiriusContext`, per-GPU memory/topology initialization, REMOTE I/O context attachment |
| `src/memory/sirius_memory_reservation_manager.{hpp,cpp}` | Extends `cucascade::memory_reservation_manager`; grants actual managed-pool access before setting cudf device resource refs; synchronizes on destruction |
| `src/scan_manager/sirius_scan_manager.hpp` | `pinned_entry`, `chunk_memory_spaces` invariant |
| `src/memory/topology_index.hpp` | `topology_index` — the single NUMA↔GPU map injected into task creator, scan manager, downgrade configs |
| `src/pin_table.cpp` | Pin materialization; per-batch round-robin placement, target-device guard, target-space allocator |
| `src/scan_manager/split_provider.cpp` | Fresh-read split provider; resolves an ioctx per file |
| `src/scan_manager/sirius_scan_manager.cpp` | `cached_databatch_provider`; ioctx ownership and path routing |
| `src/io/path_utils.{hpp,cpp}` | `sirius::io::open_datasource`: path normalization before cuCascade's `open_datasource` |
| `cucascade/include/cucascade/io/datasource_factory.hpp` | cuCascade backend registry (`uring` / `restful` / `cufile`); resolves a path via per-backend checkers, with no catch-all |
| `cucascade/src/io/uring/uring_reactor.cpp` | cuCascade uring reactor; sets the request's device for each H2D copy |
| `src/op/scan/sirius_gpu_scan_operator.cpp` | Unified `GPU_SCAN` source operator (multi-GPU-aware) |
| `src/creator/task_creator.cpp` | Per-task `preferred_device_id` resolution, partition device pin |
| `src/pipeline/gpu_pipeline_task.hpp` | `preferred_device_id` two-level lookup |
| `src/pipeline/gpu_pipeline_executor.cpp` + `task_scheduler.cpp` | Ready-device matching, reservation-device execution, OOM-reschedule pin carry-forward |
| `src/downgrade/downgrade_executor.cpp` | Per-tier downgrade workers, usable REMOTE targets after DISK, K.6-gated `cudaSetDevice` |
| `cucascade/src/cudf/representation_converter_builtins.cpp` | `convert_gpu_to_gpu` / `alloc_and_peer_copy_async` with peer-DMA probe + dst_guard |
| `cucascade/src/memory/common.cpp` | private-pool verification and typed `grant_pool_peer_access` |

## Related Documentation

- [Architecture Overview](architecture-overview.md) — component diagram, thread model, ownership hierarchy (covers single-GPU too)
- [Pipeline Execution](pipeline-execution.md) — Per-task-device contract under SCHED-RR (the deeper task-routing contract)
- [Memory Management](memory-management.md) — cuCascade tiers, reservations, downgrade executor (mechanics of memory_spaces)
- [Scan](scan.md) — unified `GPU_SCAN` path, pin tables, cache, split providers
