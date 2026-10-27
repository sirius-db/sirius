# Memory Management

This document covers the memory tier hierarchy, reservation system, downgrade executor, and pinned host memory management.

## Memory Tier Hierarchy

Super Sirius uses cuCascade for tiered memory management across three tiers:

```
┌──────────────────────────┐
│  GPU Memory (Tier 0)     │  Fast, limited (~24GB typical)
│  Primary computation     │  Used by pipeline tasks
├──────────────────────────┤
│  Host Pinned (Tier 1)    │  Medium speed, larger (>100GB)
│  Pinned pools per NUMA   │  Used for caching, GPU↔CPU transfer
├──────────────────────────┤
│  Disk (Tier 2)           │  Slow, unlimited (~1TB default)
│  Spill files on mount    │  Last resort for extreme pressure
└──────────────────────────┘
```

Each tier has configurable thresholds:

| Parameter | GPU | Host | Purpose |
|-----------|-----|------|---------|
| `reservation_limit_fraction` | 1.0 | 1.0 | Max fraction reservable |
| `downgrade_trigger_fraction` | 0.8 | 0.9 | When to start downgrading |
| `downgrade_stop_fraction` | 0.6 | 0.8 | When to stop downgrading; configuration requires `0 < stop < trigger <= 1` |

## cuCascade Integration

**File:** `src/memory/sirius_memory_reservation_manager.hpp`

`sirius_memory_reservation_manager` inherits from `cucascade::memory::memory_reservation_manager`. It:

- Initializes all GPU memory spaces and sets cuDF device resources
- Wraps cuDF device resources and saves/restores them to prevent dangling references
- Bridges Sirius' task execution with cuCascade's tiered memory management
- On destruction, restores previous cuDF resources to avoid crashes during cleanup

### Memory Space Configuration

From `sirius_config`:

**GPU Memory Space:**
```cpp
device_id;                      // GPU device number
reservation_limit_fraction = 1.0;
downgrade_trigger_fraction = 0.8;
downgrade_stop_fraction = 0.6;
```

**Host Memory Space:**
```cpp
numa_id;                        // NUMA node affinity
reservation_limit_fraction = 1.0;
downgrade_trigger_fraction = 0.9;
downgrade_stop_fraction = 0.8;
block_size = 1MiB;              // cuCascade block size
pool_size = 128;                // blocks per pool
initial_number_pools = 4;       // pools allocated at startup
```

**Disk Memory Space:**
```cpp
disk_id;
mount_paths;                    // directories for spill files
memory_capacity = 1TB;          // total spill capacity
```

## Parquet scan working sets

Virtual-column reader scans budget selected rows before filtering, including cuDF source/row indexes, output columns and filter/cast scratch. Plain Parquet no longer needs the old per-row-group concatenation allowance; Iceberg still does. Filename estimates include characters, offsets and construction buffers, with saturating arithmetic.

Metadata-only scans reserve synthesized columns and temporary buffers, but no compressed data or data prefetch. Typed empty virtual columns allocate no scalar buffers. The coalescer and execution task use the same read mode for their estimates.

## Memory Reservations

Pipeline tasks acquire memory reservations before execution to prevent GPU OOM:

1. GPU executor's `manager_loop()` calls `memory_space.make_reservation(estimated_size)`
2. The reservation is attached to the task's local state via `set_reservation()`
3. During execution, operators allocate within the reservation
4. Reservations are released when the task completes

Multi-partition dynamic filters allocate persistent per-GPU Bloom arrays under exact non-blocking initialization leases on an untracked creator thread. Publication runs inside the final PARTITION task: its bounded scratch uses that task's allocator, reservation accounting, and recorded peak. The existing ignore-limit policy can admit scratch beyond the reservation against global capacity; a safely recoverable refusal skips the optional filter. After build tasks and retries drain, the eligible build PARTITION finalizer closes accumulation before releasing retained arrays and completed wrappers without crediting a task tracker; hash-join finalization repeats this cleanup as a fallback. Input batches may spill and re-upgrade with stable original identity; the raw Bloom arrays cannot spill. See [Dynamic Filters](dynamic-filters.md#multi-partition-build-accumulation).

### `reservation_aware_resource_adaptor`

Wraps RMM device memory resource. On each allocation:
- Checks if the reservation has sufficient capacity
- If exhausted → fails gracefully, triggering `oom_reschedule_exception`
- Enables predictable memory usage per task

### Caller reservations for HOST conversions

Conversions that land data on the HOST tier draw down a caller-owned reservation instead of double-committing host capacity: the caller obtains a reservation with `make_reservation_or_null(size)` and passes it to the reservation-taking `convert_to`/`clone_to` overloads, so the converter's allocation is charged against capacity the caller already holds. If the reservation cannot be made, the call falls back — with a warning — to the `memory_space*` overload (no reservation; the converter may OOM). Call sites: `lock_or_prepare_batch` in `src/pipeline/batch_lock_utils.hpp` and the materialized result collector (`src/op/sirius_physical_result_collector.cpp`). The `memory_space*` overloads remain the path for GPU/DISK targets and viability probes.

## Downgrade Executor

**File:** `src/downgrade/downgrade_executor.hpp`, `src/downgrade/downgrade_executor.cpp`

One `downgrade_executor` per memory space monitors pressure and moves data to lower tiers.

### Thread Model

- **Processing thread**: dequeues `downgrade_request` objects sequentially from an `interruptible_mpmc` queue
- **Monitor thread** (if `monitor_period > 0`): each cycle, if the memory space reports `should_downgrade_memory()`, it applies a **stateless viability gate** (`has_viable_downgrade_target()`) before enqueuing. A request is only fired when a lower tier could plausibly accept the data:
  - A configured DISK tier is always a viable target.
  - Without DISK, only a GPU source has a lower tier (HOST). Viability is confirmed by probing each HOST space with a chunk-sized `make_reservation_or_null` (released immediately) — the ground truth for whether downgraded data can actually land, since HOST capacity reflects both live reservations and already-stored downgraded data.
  - If no target is viable (e.g. idle GPU batches whose only lower tier is a full HOST with no DISK configured), the monitor **backs off** for that cycle without enqueuing, warning once per stall episode. The gate is re-evaluated every cycle with no latched state, so the monitor resumes automatically the instant HOST frees space or GPU pressure drops.
- **Worker thread pool** (`exec::bounded_thread_pool`): executes actual data movement concurrently

The monitor sleeps on an interruptible condition-variable wait rather than a plain sleep, so `stop()` wakes it immediately instead of blocking shutdown for up to a full `monitor_period`.

The monitor is not on the correctness-critical path: the pipeline executor drives downgrade on demand via `request_downgrade()` independently, so monitor back-off can never wedge a query. Monitor-issued requests are fire-and-forget.

### Downgrade Request Pattern

The downgrade executor uses a request-based model with tiered candidate fetching:

1. Caller invokes `request_downgrade(predicate)` which constructs a `downgrade_request` and pushes it onto the MPMC queue. Returns `std::future<size_t>`.
2. The processing thread dequeues requests **sequentially** (to avoid contention between concurrent requests competing for the same batches).
3. For each request, the processing loop fetches candidates lazily in tiered order:
   - **Tier 1 (data repositories):** Creates a `convertible_data_batch_provider` per repository and fetches idle GPU-resident batches one at a time
   - **Tier 2 (task_scheduler queue):** Creates a `convertible_gpu_pipeline_task_provider` to extract tasks with convertible data batches from the pipeline-level task queue
4. Each candidate is dispatched to the `bounded_thread_pool` and converted via `convertible_data::convert()`. The `predicate` is evaluated both **before dispatching each candidate** (a request that is already satisfied — e.g. the caller's reservation landed, or the running query freed memory — spills nothing) and after each conversion. If it returns `true`, no new candidates are dispatched (in-flight conversions finish naturally). The promise resolves with total bytes freed.

**Byte-targeted requests** (`downgrade_request::target_bytes`, set by monitor requests and `request_free_memory`) additionally stop dispatching once the *planned* bytes (freed + in-flight) cover the target, bounding overshoot to less than one batch regardless of pool width, and right-size the final pick per repository (`spill_policy.hpp`: smallest candidate that still covers the remaining deficit, ties in policy order). Together these keep a marginal overflow from evicting a whole extra multi-GB partition — the q9-class partition-spill cliff, where +0.1% of input rows doubled the spilled set and its host round-trips.

**Monitor spill sizing:** crossing the *trigger* threshold starts a monitor-issued request sized to reach the *stop* threshold. The request also observes live pressure and stops once usage reaches that threshold, preserving the trigger→stop hysteresis band.

**Pipeline integration:** When `gpu_pipeline_executor` gets a partial memory reservation (shortfall), it issues a single `request_downgrade(predicate)` where the predicate attempts `make_reservation_or_null(bytes_needed)`. The downgrade stops as soon as the reservation succeeds -- single request, no over-freeing.

### Spill Copy Granularity

**Files:** `src/data/spill_chunked_converters.hpp`, `src/data/spill_chunked_converters.cpp`, `src/data/chunked_spill_copy.hpp`

The builtin cucascade fast converter submits an entire batch's GPU→HOST copies as one monolithic batched call followed by one blocking synchronize (18–28 GB single submissions observed on q9-class partition spills). At context initialization Sirius replaces it with a chunked converter (`executor.downgrade.copy_chunk_bytes`, default 1 GiB; 0 keeps the builtin) that produces a byte-identical `host_data_representation` but flushes copies every ~chunk while the column tree is still being walked — overlapping each chunk's DMA with the next chunk's prep — and wraps each conversion in an NVTX range. The HOST→GPU restore direction deliberately stays on the builtin converter (it runs on instrumented pipeline threads and reconstructs from the self-describing `column_metadata`).

### Spill Compression

**Files:** `src/compression/spill_compression.hpp`, `src/compression/spill_compression.cpp`, `src/compression/compressed_disk_representation.hpp`, `src/compression/compression_converters.cpp`, `src/data/convertible_data_batch.hpp`

With `compression.enable_spill_compression` on, `convertible_data_batch::convert` first offers a GPU batch of at least `spill_min_batch_bytes` to the Simpatico converters when it moves to the HOST or DISK tier, and falls back to the uncompressed converters when they decline or fail, so the feature can only change how a batch is stored, never whether it spills:

| From | To | Converter |
|------|----|-----------|
| `gpu_table_representation` | `compressed_host_representation` | encode on the device, stage the payload in pinned host blocks |
| `gpu_table_representation` | `compressed_disk_representation` | encode on the device, write an `.hpln` file under the disk space's mount path through its I/O backend |
| `compressed_host_representation` | `compressed_disk_representation` | write the pinned header and payload as an `.hpln` file through the I/O backend (no decode) |
| `compressed_disk_representation` | `gpu_table_representation` | read the header, then each compressed buffer straight into device memory, through the I/O backend; decode |

The compressed host representation decodes with the converter `pin_table` already uses. Each column's plan follows from its type: fixed-width columns up to 8 GiB are bitpacked (bitpack's encode scratch holds a column at full width as 2^31 32-bit words at most; larger ones are stored as is), strings are dictionary-encoded when a sample has few distinct values and otherwise split into ans-coded chars and delta → bitpack offsets, anything else is stored as is. The encode allocates from the GPU pool the batch is being evicted from, so it runs under a reservation of `spill_encode_reserve_fraction` of the batch's size, attached to the allocation tracker for the encode's duration. Under per-thread tracking the reservation covers the column-parallel encode on the stream pool; under per-stream tracking it covers only the downgrade stream, so the encode runs on that stream alone. The batch spills uncompressed when the reservation is not grantable, when every column's plan stores it as is (the encode would only copy the batch), or when the compressed form exceeds `spill_max_compressed_fraction` of the batch. A spilled compressed batch reports its compressed size from `get_size_in_bytes()`, so code that sizes GPU work from waiting batches (the PARTITION operator's partition count, CONCAT's `concat_batch_bytes` grouping) uses `get_uncompressed_data_size_in_bytes()`, the size the task will decode to. Decoding back to the GPU reserves the batch's uncompressed size, and a decode that runs out of device memory or streams throws `rmm::out_of_memory`, which the executor reschedules like any other OOM.

### Candidate Selection Strategy

Candidates are fetched lazily via `convertible_data_provider` implementations:

1. **Data repositories** of every in-flight query, **newest query first** (descending query id: the newest query has made the least progress, so spilling it costs the least, and the oldest keeps its working set so it can finish and release memory). Within a query, repositories go in `get_repositories()` order, ascending `{operator id, port}`. Within each repository, `convertible_data_batch_provider` iterates partitions back-to-front, then batches back-to-front, filtering for idle batches in the source memory space. Batches a queued task has subscribed are skipped.
2. **Pipeline task queue** is inspected via `convertible_gpu_pipeline_task_provider`, which uses `mutable_pop_if` to temporarily extract tasks with matching data batches, starting at the **back** of the queue. Tasks are returned to the queue via RAII on all code paths.

Candidates are converted individually via `convertible_data::convert()`, which handles state locking, memory reservation, tier conversion, and failure rollback atomically.

When the preload executor is enabled, each successful conversion is recorded and published as a `batch_downgraded` event after the batch's lock is released. With no subscriber, nothing is recorded and the check costs one atomic load per request.

## Preload Executor

**Files:** `src/preload/` (`preload_executor.{hpp,cpp}`, `promotion.{hpp,cpp}`, `candidate_store.{hpp,cpp}`, `preload_policy.hpp`, `hw_caps.{hpp,cpp}`, `preload_streams.hpp`, `preload_stats.hpp`, `config.hpp`), `src/memory/memory_wait_board.hpp`

The preload executor undoes downgrade moves before the consumer asks for the data: batches spilled GPU→HOST or GPU→DISK are copied back to the GPU, and batches spilled HOST→DISK are staged on HOST. It is **off by default** (`sirius.executor.preload.enabled`, see [configuration](configuration.md#siriusexecutorpreload)); when off, no executor, board, stream pool, subscription or conversion record exists. It is advisory: a refused, failed or missed promotion costs performance only, because `lock_and_prepare_batch` still moves every input the consuming task needs.

**What it moves, and how.** HOST→GPU and DISK→GPU are `convert_to<gpu_table_representation>`, and DISK→HOST is `convert_to<host_data_representation>`: raw buffer copies, never an encode. An `encoded_data_representation` is staged DISK→HOST byte for byte through `relocate_to`, and goes to a GPU only through a registered decoding converter to `gpu_table_representation` (none is registered today); GPUs only ever receive `gpu_table_representation`.

**Where candidates come from.** Only from `batch_downgraded` events. A candidate holds a `weak_ptr` to its batch plus where it was moved from and to; a batch whose last owner dropped it is forgotten.

**Threads.** One planner thread (`preload-plan`) and pull-based lanes: one HOST→GPU lane per GPU (`preload-h2g-<gpu>-<i>`; 5 threads when the CPU and GPU share a C2C link, else 1) and one lane per DISK space (`preload-dsk-<disk>-<i>`; 5 threads with GPUDirect Storage, else 1). A DISK→GPU copy without GDS also takes one of its GPU's HOST→GPU slots, since it crosses the same link. There is no timer: the planner waits on a condition variable and runs a pass only when an event (`batch_downgraded`, `task_completed`, `memory_wait_ended_for_task`, `task_created`, `task_deployed`, `pipeline_closed`, or an idle event) or a finished promotion marks it dirty. A task that succeeds, a stalled reserve phase that ends, a pipeline that closes and a query whose repositories are erased advance the memory epochs that refusals are stamped against. Repeated idle events (`task_queue_empty`, `executor_awaiting_task`), which the scheduler re-publishes while a device waits, ask for one pass until some other event arrives.

### What gets promoted, in which order

Preload reverses the downgrade victim order (above), sharpened by the scheduler's own priorities:

- **Class A (committed): inputs of queued tasks.** Each pass visits every GPU's queue bucket in pop order (`multi_index_priority_queue::visit_in_pop_order`, which removes nothing), skips the first `H` tasks (`head_margin_tasks`, default `executor.pipeline.num_threads`), and takes the inputs of the next `lookahead_tasks` (8). The scheduler is about to run the head tasks, whose executor sizes and uploads them itself. The bucket shared by all GPUs skips `H × number of GPUs`, since every ready GPU pops it.
- **Class B (speculative): repository batches nobody has subscribed yet**, in two narrow cases only. For a consumer behind an unfinished FULL barrier, the front batches of its lowest partitions are staged DISK→HOST. For a ready consumer that drains one partition per task in ascending order, the partition after the one the task creator drains next (`repository_partition_lookahead`, default 1) moves to the GPU its placement names, front positions only (`repository_window`, default 2). Any other ready repository is left to the task creator, which wraps it in a task within one pass (it then becomes class A).
- **Order.** Oldest query first, then destination-pipeline priority, class A before B at equal priority, then partition and FIFO position. GPU targets come before HOST staging.

### Judiciousness: ceilings, windows and hold-offs

Ceilings are fractions of each space's capacity, measured as `consumed = max_memory − available`, the same measure `should_downgrade_memory()` uses:

| Ceiling | Default | Why |
|---|---|---|
| Class A on a GPU | `downgrade_stop_fraction − gpu_committed_headroom_fraction` (0.6) | The consuming task's reservation estimate excludes inputs already on its GPU, so these bytes are pre-paid |
| Class B on a GPU | `downgrade_stop_fraction − gpu_speculative_headroom_fraction` (0.5) | Extra demand: kept a step lower |
| HOST staging | `downgrade_stop_fraction − host_headroom_fraction` (0.7) | Leaves room for GPU spills, which land on HOST first |

Every ceiling is at or below `downgrade_stop`, so **preload alone never starts a downgrade**, and the hysteresis band `[stop, trigger]` stays the running tasks' working room. Each GPU also bounds the bytes preloaded but not yet consumed (`gpu_lookahead_fraction`, 0.2; class B at most `speculative_fraction`, 0.05) and the bytes being copied in at once (`max_inflight_fraction`, 0.1); a batch whose need exceeds `max_batch_fraction` (0.1) never targets a GPU.

**Hold-offs are synchronous.** The `memory_wait_board` (`src/memory/memory_wait_board.hpp`, shared by `SiriusContext` and the GPU pipeline executors only when preload is on) counts, per GPU, the tasks in a stalled reserve phase (a `waiter` the executor enters before it blocks and releases when the whole phase ends, across a partial grant and the downgrade path) and the tasks rescheduled after an OOM (an `oom_token` the rescheduled task owns until it is destroyed). While either count is non-zero, preload starts no copy into that GPU and stages nothing on its NUMA-local HOST. The planner, the lane, and the worker (before it reserves and again right before the copy) all read the board with atomic loads; event delivery lag never matters.

**The planner and the worker count the same bytes.** A planner pass adds to each space's consumed bytes only what preload's running jobs will reserve there but have not yet: a job's estimate joins a per-space unreserved count when a lane takes it and leaves as soon as its reservation exists, which `consumed` then holds. The pass loads that count before it reads `consumed`, and reads both under the executor's mutex, so no job is taken in between. A running job is therefore counted once, or briefly twice, never zero times, and the planner's room matches the worker's ceiling check.

**Gates are stamped at dispatch.** A worker refused for memory stamps the target with the memory epoch it saw when the job was taken, so any memory event that arrives during the attempt, or later, re-enables exactly one retry. A contended batch lock waits for the next queue event and is dropped after `max_lock_attempts`. Planner refusals stamp nothing: budgets are recomputed from live state on every pass.

**Per-promotion order (`promote_one`).** Board check → `make_reservation_or_null` on the target → re-check the ceiling with the reservation counted (concurrent workers can only over-refuse) → `try_to_mutable` on the batch (never blocking; failure = contended) → validate where the batch is → exact need, including the INT32→INT64 string-offset widening of the HOST/DISK→GPU converters → `grow_by` if short → board and ceiling re-check → copy. An OOM is local to the candidate: one retry with a doubled estimate, then the candidate never targets a GPU again.

**Ping-pong protection.** Preload takes queue positions past the head margin, the oldest query and the FIFO front; downgrade takes the queue back, the newest query and the FIFO back. A batch is promoted at most `max_promotions_per_batch` (2) times per query; after one bounce (a downgrade record newer than its promotion) it is never promoted speculatively again. Under sustained pressure that drives a GPU back to its trigger, a queued task's preloaded inputs can still be the GPU-resident batches nearest the queue's back, which tier 2 takes first; the `bounce` counter reports how often.

### Invariants

| # | Invariant |
|---|---|
| I1 | Preload never takes a space's consumed bytes above its preload ceiling, checked after its own reservation and after any growth of it. |
| I2 | It never blocks under a batch lock: only `try_to_read_only` / `try_to_mutable`, `make_reservation_or_null` and `grow_by`; the reservation is made before the lock. |
| I3 | The executor's mutex is a leaf: under it, preload never calls into the task queue, repositories, batch locks, memory spaces, the publisher or the allocators. |
| I4 | Candidates hold `weak_ptr`s. A strong batch reference, a repository pointer or an operator pointer is used only while holding the query's `try_acquire_work` lease. |
| I5 | No GPU→GPU moves: a GPU target is used only when the consumer's device is deterministic. |
| I6 | GPUs only receive `gpu_table_representation`. |
| I7 | Re-promotion is bounded (see ping-pong protection). |
| I8 | Advisory only: `lock_and_prepare_batch` stays authoritative. |
| I9 | Off means absent. |
| I10 | No timer and no lost wake-up: every wait is an untimed condition-variable wait, and refusals stamp the epoch seen at dispatch. |
| I11 | No resurrection: only `prepare_for_query` creates per-query state; outcomes and events for a query preload does not know update counters only. |
| I12 | Hold-offs are read from the board, never derived from event delivery. |

### Accounting handoff

A promotion reserves on the target space, attaches the reservation to the copy, and then calls `reset_stream_reservation`: the preloaded bytes stay counted on the space as plain allocations, and the consuming task's reservation estimate no longer includes them, because they are already on its GPU. Their eventual free is charged to the consuming task's reservation. A promotion that lands after the consumer sized its reservation over-counts for that task's lifetime; it never under-counts. HOST→GPU copies run on the target space's shared stream pool (the converter's choice); DISK→GPU copies use preload's own per-GPU `exclusive_stream_pool`s (`preload_streams`), which `SiriusContext` destroys only after the data repositories, since a preloaded batch's buffers are freed on that stream until a consumer rebinds them. With `per_stream_reservation: true`, HOST→GPU allocations bypass the attached reservation, so the ceiling is held less strictly (a WARN at start).

### Query lifecycle and teardown

`SiriusContext::create_query` registers the query's repositories (`prepare_for_query`); `retire_query_work` purges its candidates just before waiting for its work (`retire_query`), and every running promotion holds a lease, so that wait covers it; `release_query_state` and the best-effort failure path call `forget_query`, which logs the query's one-line `[preload] query <id>:` summary. `terminate()` stops preload after the task creator and before `query_lifecycle_.clear()`.

### Observability

`preload_executor::stats()` returns every counter (`src/preload/preload_stats.hpp`): intake, planner passes and refusals by reason, worker outcomes, promotions and bytes per kind, accuracy (`hit`, `bounce`, `mispredicted_device` and its class-B part, `wasted_unconsumed`), lane high-water marks and timings. Logs use the `[preload]` prefix: an INFO line at start with each GPU's lane cap and ceilings, an INFO summary per query, a DEBUG line per planner pass with each space's consumed bytes and budget (or the refusal that closed it), and a TRACE line per promotion.

**Measured** (one RTX 6000 Ada over PCIe, no C2C, no GDS; `test/cpp/integration/test_gpu_execution_preload.cpp` and TPC-H SF100 with a 3–4 GiB GPU limit): results are identical with preload on and off; wall time and GPU kernel time are unchanged within noise, including while about 800 HOST→GPU promotions (1.4 GB) run under a spilling aggregate. Most promotions are class A inputs of tasks that drain spilled data after a FULL barrier; under saturation the planner is mostly refused by the ceilings or held off by stalled tasks, which is the intended behavior. SF100 Q9 at a 3 GiB limit bounces about one promotion per query (17–35% of a handful); at 4 GiB nothing bounces.

## Memory Consumption History

**File:** `src/pipeline/pipeline_memory_history.hpp`

Each GPU pipeline maintains a `pipeline_memory_history` — a thread-safe ring buffer of up to 64 `task_memory_record` entries, each recording:
- `estimated_bytes` — pre-execution estimation basis (input data size)
- `peak_memory_bytes` — actual peak allocation observed during execution
- `output_bytes` — output size, or `nullopt` if the task OOM'd

### Recording

- `record(rec)` — on successful task completion
- `record_on_failure(estimated_bytes, peak)` — on OOM; keeps the **higher** peak for repeated failures with the same input size, so each retry reserves more

### Estimation

`estimate_peak_memory(estimated_bytes)` computes a weighted average of historical `peak/estimated` ratios. Records with similar estimation bases are weighted higher using a log-ratio distance function: `weight = 1 / (1 + |log(rec_est / new_est)|)`.

### Integration

`gpu_pipeline_task::get_estimated_reservation_size_info(target_space)` uses `estimate_peak_memory()` for the reservation, adding `_bytes_to_materialize_input` — the bytes needed to materialize inputs into the task's target memory space (HOST/disk upgrades plus cross-GPU clones) — and subtracting it from recorded peak to keep operator history clean of materialization overhead.

Materialization allocations are charged to the task's reservation through the default per-thread allocation tracking, so a compressed decode may use several of the thread's streams. With the non-default `per_stream_reservation: true`, allocations are charged by stream instead: converter allocations made on internal pool streams bypass the task's reservation and are only checked against space capacity, and a compressed decode runs on the task's stream alone. Frees are charged to the freeing thread or stream.

A cached scan input (a resident `scan_operator_input`) is sized by a dedicated branch before the pipelineable-input walk: it contributes its uncompressed GPU footprint when the cached batch is not already GPU-resident (since `prepare_for_processing()` uploads HOST-cached batches before execution), and zero when it is. Resident classification is deliberately unchanged by this sizing — a HOST cache hit must not be treated as a fresh read, which would apply the fresh-read decode heuristic instead.

Without history, a resident scan split is sized by `sirius_gpu_scan_operator::no_history_peak_memory_estimate` from its working set (`scan_operator_input::get_estimated_working_set_size_in_bytes`). A resident split that can reach the scan's dynamic-filter checkpoint reserves twice its batch bytes, covering the membership masks and one gather of the survivors, whether or not decompression pushdown is enabled. `scan_operator_input::cached_checkpoint_possible` decides this for both the estimate and execution: the scan has a consumer with `consumer_config::cached_placement` enabled, its ingestible can describe a cached view (only resident Parquet can), and the split has neither a visibility mask nor a stolen table. Execution tries the transfer first: `prepare_for_processing` steals a transferable table it converts (a compressed or HOST-tier chunk with no mask and no pending static filter), and execute transfers a carrier-converting one unless its transfer refuses it (for example, the scan defers columns to late materialization or its output repeats a column). The estimate runs before `prepare_for_processing` and cannot tell whether a transfer will take the table, so a resident split that will transfer may also get twice its batch bytes until the pipeline has history. A split whose decode is forecast to compact keeps the forecast envelope, because the checkpoint can then see only the compacted table, after a refused carrier-converting transfer. Decode bindings play no part: a deferring scan, whose decode consumer is null, and a permuted or duplicate output layout still reach the checkpoint. Normalizing the selected rows to the planned carriers stays covered by the resident conversion estimate. The checkpoint adds no separate floor: once the pipeline has history, the learned peak replaces this estimate, and an underestimate is retried under the raised OOM retry floor. Checked retirement of the checkpoint's GPU work makes that retry safe, although capacity limits or retry exhaustion can still fail the query.

## Memory Pool Defragmentation

**File:** `src/memory/defragmenter_oom_policy.hpp`, `src/memory/defragmenter_oom_policy.cpp`

`defragmenter_oom_policy` implements `cucascade::memory::oom_handling_policy`:

On allocation failure:
1. Check CUDA pool fragmentation via `cudaMemPoolGetAttribute()` (reserved vs. used)
2. If `reserved > used + (10× requested bytes)`: pool is fragmented
3. Trim pool with `cudaMemPoolTrimTo()` to release free blocks to driver
4. Retry allocation
5. If still fails: rethrow original exception

## Pinned Host Memory

**File:** referenced in `src/sirius_context.hpp`

`small_pinned_host_memory_resource` provides fast host memory allocation:

- Fixed-size block pools: 1 MiB blocks, 128 blocks per pool, with four pools initially
- Automatic NUMA node affinity
- Used for GPU↔CPU transfers and scan caching
- Configured via `sirius.yaml` (see [Configuration](configuration.md))

## Stream-Ordering Discipline

Device memory is freed stream-ordered (RMM async pool), and the pool may rebind the virtual
address immediately after the free executes. Batches are handed off event-ordered, not
host-synced, and read locks are host-scoped: a consumer can enqueue kernels and drop its lock
with device work still in flight. The invariant that keeps this safe:

> No owner of device memory may die (free, pool return, or dealloc-stream rebind) until every
> stream that may still read or write that memory is ordered before the free.

Three mechanisms carry it:

1. **Writer ordering before conversion reads.** `convertible_data_batch::convert` retains its stricter publication contract: a GPU batch without a writer event is rejected before rebinding or conversion. Owning `make_data_batch` helpers use `gpu_table_representation::make_written_on`, where a null stream means the actual default stream on the producing device; view helpers explicitly record it too. The direct chunked spill converter independently calls `wait_for_writer` before planning string offsets or copying bytes. At the cuCascade representation boundary, unknown writers synchronize the source device, while known writers order an event on the reader stream. Pinned views with unknown provenance retain that conservative behavior. View-backed `release_table(stream)` waits for its copy before releasing the backing owner; owned release only orders the writer.
2. **Reader events at intra-operator ownership handoffs.** The scan's `owning_table_view` records
   a reader event for zero-copy views whose owner can be replaced before control returns to the
   pipeline task (`record_reader_event`, cuCascade #184). `try_to_mutable()` then refuses, and
   `to_mutable()` waits, until those reads complete, so no downgrade can rebind or free the batch
   under an in-flight reader.
3. **Quiesce before task-owned owner death.** The pipeline task synchronizes its stream after every
   operator `execute()` and again after `publish_output`. Its exception handlers synchronize while
   the corresponding inputs and outputs are still alive, before rethrowing lets them unwind. This
   also ensures cross-task state freed by `finalize_operator` is not freed under a straggling sink
   enqueue.

Suspected violations show up as torn GPU→HOST conversions, scribbled string or selection
geometry, or wild-pointer MMU faults, almost always under concurrency plus memory pressure.
`bench/sf1000-repro/run.sh` accepts `SANITIZER=memcheck` to run the workload under
compute-sanitizer; `verify-memcheck-sf1.sh` does so for the concurrent workload against an
SF1-sized pressure config.

## Key Files

| File | Purpose |
|------|---------|
| `src/memory/sirius_memory_reservation_manager.hpp` | Memory manager, tier configuration |
| `src/downgrade/downgrade_executor.hpp` | Downgrade executor interface |
| `src/downgrade/downgrade_executor.cpp` | Processing loop, tiered candidate fetching |
| `src/memory/defragmenter_oom_policy.hpp` | Pool defragmentation policy |
| `src/memory/defragmenter_oom_policy.cpp` | Fragmentation detection and trimming |
| `src/pipeline/pipeline_memory_history.hpp` | Per-pipeline memory consumption history |
