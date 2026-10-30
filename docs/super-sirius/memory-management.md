# Memory Management

This document covers the memory tier hierarchy, reservation system, downgrade and preload executors, and pinned host memory management.

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

When anyone subscribes to `task_downgraded` (the [preload executor](#preload-executor)), tier 2 also records each task's moves and publishes one `task_downgraded` per converted task, while it still owns the task, so before the task goes back into the queue. Tier 1 publishes nothing: a batch it spilled is reported by the scheduler once a task is queued with it (see [Events](#events)). Without a subscriber nothing is recorded and the check costs one atomic load per request.

## Preload Executor

**Files:** `src/preload/` (`preload_executor.{hpp,cpp}`, `promotion.{hpp,cpp}`, `preload_streams.hpp`, `config.hpp`), `src/event/downgraded_task.hpp`

The preload executor moves the spilled inputs of queued tasks back up before the tasks run: HOST→GPU or DISK→GPU (`convert_to<gpu_table_representation>`), else DISK→HOST (`convert_to<host_data_representation>`). It promotes only `host_data_representation` and `disk_data_representation` sources and never moves a batch between GPUs. It runs only when `device_budget` or `host_budget` is > 0 ([configuration](configuration.md#siriusexecutorpreload)). It is advisory: `lock_and_prepare_batch` still moves every input a task needs, so a refused, failed or missed promotion costs performance only.

**Scope.** Its candidates are the inputs of queued tasks only. A batch tier 1 spilled from a data repository is not a candidate while it sits there; it becomes one when a task that reads it is queued.

### Events

Two `query_event_publisher` events drive it; both are published only while someone subscribes. `task_downgraded` has two producers.

| Event | Published by | When | Payload |
|---|---|---|---|
| `task_downgraded` | downgrade executor, tier 2 | once per task it converted, before the task is pushed back | query, task id, pipeline id, preferred device, queue priority, and each input moved off a GPU: weak batch pointer, batch id, the space it moved to, its size on the GPU |
| `task_downgraded` | scheduler, `schedule()` | once per task queued with inputs off the GPU, before the push | the same, for each input not on a GPU: the space it is in, and its `peak_materialization_bytes` as its size on the GPU. Each input is read through a non-blocking read-only accessor; a busy one is skipped |
| `task_popped` | scheduler matcher | right after it pops a task for GPU `g`, before dispatch | query, task id, pipeline id, `g`, and `tasks_left`: the tasks `g` can still pop (its own bucket plus the no-preference bucket; an advisory snapshot) |

A `task_downgraded` makes the task a candidate, or adds to the one it already is; a task of a query that grants no work lease is ignored. A `task_popped` removes the task from the candidates, releases the budget its inputs hold, and gives `g` one credit.

The two events come from different threads into one mailbox in no guaranteed order, so a `task_downgraded` can arrive after its task's `task_popped`. Both producers publish `task_downgraded` before the task enters the queue, so its event id is smaller than that of the task's pop. The executor therefore remembers the last 4096 popped tasks with the id of their `task_popped` event (recent-pop tombstones) and drops a `task_downgraded` whose event id is smaller than its task's pop (`stale_event`). What that window misses is cleaned up later: inputs whose batch expired are dropped when a selection walks past them, and `retire_query` drops the rest.

### Pop-time rule

Each GPU has one worker thread (`preload-<device>`) that spends each credit on at most one promotion. Pops that arrive while it is busy wait as credits, at most `executor.pipeline.num_threads` per GPU; past that the oldest is dropped. For a credit from a pop on GPU `g`:

1. **Candidates for `g`:** tasks that prefer `g` or have no preference (promoted to `g`).
2. **Gate:** a candidate is eligible if `tasks_left <= executor.pipeline.num_threads` (`g`'s queue is nearly drained) or it belongs to the popped task's pipeline (the same query and pipeline id: pipeline ids are per query).
3. **Selection:** eligible candidates in pop order (ascending queue priority); the first input that is not on a GPU, not being promoted and not skipped, and that fits, wins:
   - to `g`, if `device_budget > 0` and it fits `g`'s budget and ceiling;
   - else, if it is on DISK, to `g`'s NUMA-local HOST space (the first HOST space when the node has none or is unknown), if it fits that space's budget and ceiling.

An input's estimate is its size on the GPU from the event (before the spill, or the peak materialization bytes for an input the task was queued with) plus slack (the larger of 1 MiB and 1/64).

### Budget and ceilings

Budget and ceiling are fractions of the target space's capacity. A space's **budget** counts the bytes of running promotions into it plus promoted inputs whose task has not been popped. A promotion is charged its estimate at selection and, on success, re-charged its exact need; the charge is released when the task is popped, when that input is downgraded again, when the promotion does not succeed, or, at the latest, when the query retires. The **ceiling** bounds `consumed = max_memory − available`, the same measure `should_downgrade_memory()` uses, counting the promotion itself. Ceilings are clamped to the space's `downgrade_stop_fraction`, so preload alone never starts a downgrade.

A batch downgraded again is a candidate again: re-promotion is bounded only by the budget and the ceiling. An input whose promotion failed, or found the batch busy or gone, is skipped until it is downgraded again; one that found no room once its exact need was known keeps that need as its estimate, so it is not tried again until that much fits.

### One promotion (`promote_one`)

Under the query's work lease, in order, each refusal returning at once: reserve the estimate on the target with `make_reservation_or_null` and check the ceiling and that the space is below its downgrade trigger → `try_to_mutable` on the batch (never waits) → check that the batch still needs the move and has a promotable representation → compute the exact need (for a GPU, the peak materialization bytes plus the INT64 copy of each INT32 string-offsets column; for HOST, the DISK size), `grow_by` the reservation if short, and check the ceiling again → copy.

A GPU copy runs on a stream from preload's own per-GPU `exclusive_stream_pool` (`preload_streams`) after the source's writer event, with the reservation attached to that stream's tracker under a fail-on-overrun policy; `rmm::out_of_memory` there fails the input. The reservation is then detached (`reset_stream_reservation`): the promoted bytes stay counted on the space as plain allocations, and the consuming task's reservation estimate excludes them because they are already on its GPU. A DISK→HOST copy draws its blocks from the reservation. A fatal CUDA error is also reported to the query's lifecycle.

**Reader-side ordering invariant.** A GPU representation published *without* a writer event must be complete when its batch lock is released, because a reader that finds no event assumes completion. The pinned cuCascade's HOST→GPU and DISK→GPU converters record a writer event and host-sync, so every promotion is complete when its lock is released; if the new GPU representation has no writer event, `promote_one` synchronizes its stream before releasing the lock.

### Lifecycle and observability

`SiriusContext` starts preload after the downgrade executors. `retire_query_work` calls `retire_query`, which drops the query's candidates and their budget, before it waits for the query's work; a running promotion holds a work lease, so that wait covers it. `terminate()` stops preload after the task creator and before `query_lifecycle_.clear()`, joining a running promotion. The stream pools exist only when `device_budget > 0` and are destroyed after the data repositories, since GPU buffers a DISK→GPU promotion allocated are freed on its stream until a consumer rebinds them.

Logs use the `[preload]` prefix: an INFO line at start with each space's budget and ceiling in bytes, and an INFO line at stop with the counters (`preload_executor::counters()`): `pops`, promotions and bytes to GPU and to HOST, `gated` (candidates for the GPU, none past the gate), `no_candidate`, `no_room`, `not_needed`, `failed` and `stale_event`.

**Verified.** Unit tests in `test/cpp/preload/` drive the executor through both events, feed it `task_downgraded` from the real tier 2 and `schedule()`, and cover `promote_one`; `test/cpp/integration/test_gpu_execution_preload.cpp` runs spill-heavy queries with preload off and on and checks that the results match DuckDB's, that the engine's pops and `task_downgraded` events reach preload, and that preload promotes.

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
address immediately after the free executes. A GPU batch is published either already complete
(a producer that host-syncs first; its writer event, if any, is already signalled) or with a
writer event that a consumer must wait on (a producer that may publish early, such as preload
once a converter copies asynchronously). Read locks are host-scoped: a consumer can enqueue
kernels and drop its lock with device work still in flight. The invariant that keeps this safe:

> No owner of device memory may die (free, pool return, or dealloc-stream rebind) until every
> stream that may still read or write that memory is ordered before the free.

Four mechanisms carry it:

1. **Writer ordering before conversion reads.** `convertible_data_batch::convert` retains its stricter publication contract: a GPU batch without a writer event is rejected before rebinding or conversion (it throws, rather than assuming such a batch is complete — deliberately stricter than the reader-side wait in mechanism 4). Owning `make_data_batch` helpers use `gpu_table_representation::make_written_on`, where a null stream means the actual default stream on the producing device; view helpers explicitly record it too. The direct chunked spill converter independently calls `wait_for_writer` before planning string offsets or copying bytes. At the cuCascade representation boundary, unknown writers synchronize the source device, while known writers order an event on the reader stream. Pinned views with unknown provenance retain that conservative behavior. View-backed `release_table(stream)` waits for its copy before releasing the backing owner; owned release only orders the writer.
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
4. **Reader-side writer-event wait in `lock_and_prepare_batch`.** A GPU representation published
   *without* a writer event must be complete when its batch lock is released (every existing
   Sirius producer host-syncs its stream before publishing; preload enforces this on its own
   outputs, see [Preload Executor](#preload-executor)). A producer that may publish before
   its writes complete, such as an asynchronous preload copy on its own stream, must record a
   writer event instead. Consumers order their task stream after that event inside
   `lock_and_prepare_batch` (`order_after_writer`, a device-side `cudaStreamWaitEvent`, never the
   device-synchronizing `wait_for_writer`): under the mutable lock before the opportunistic
   deallocation-stream rebind, and again under the read lock on every return path that hands out
   a GPU batch in the target space, so nothing is rebound, freed or read ahead of the producer.

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
| `src/preload/preload_executor.hpp` | Preload executor: candidates, pop-time rule, budgets |
| `src/preload/promotion.hpp` | One promotion (`promote_one`) |
| `src/memory/defragmenter_oom_policy.hpp` | Pool defragmentation policy |
| `src/memory/defragmenter_oom_policy.cpp` | Fragmentation detection and trimming |
| `src/pipeline/pipeline_memory_history.hpp` | Per-pipeline memory consumption history |
