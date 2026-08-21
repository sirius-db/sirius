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

**File:** `src/include/memory/sirius_memory_reservation_manager.hpp`

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

## Memory Reservations

Pipeline tasks acquire a memory reservation and an execution stream for the same GPU before
execution. This pairing prevents GPU OOM while preserving the per-task device contract:

1. GPU executor's `manager_loop()` calls `memory_space.make_reservation(estimated_size)`
2. The reservation is attached to the task's local state via `set_reservation()`
3. The executor acquires a stream from its device-bound stream pool, matching the reservation's
   memory space
4. During execution, operators allocate within the reservation and use that same-device stream
5. Reservations are released when the task completes

See [Pipeline Execution](pipeline-execution.md#the-contract) for the authoritative reservation
space, requested space, and stream-device contract.

### `reservation_aware_resource_adaptor`

Wraps RMM device memory resource. On each allocation:
- Checks if the reservation has sufficient capacity
- If exhausted → fails gracefully, triggering `oom_reschedule_exception`
- Enables predictable memory usage per task

### Caller reservations for HOST conversions

Conversions that land data on the HOST tier draw down a caller-owned reservation instead of double-committing host capacity: the caller obtains a reservation with `make_reservation_or_null(size)` and passes it to the reservation-taking `convert_to`/`clone_to` overloads, so the converter's allocation is charged against capacity the caller already holds. If the reservation cannot be made, the call falls back — with a warning — to the `memory_space*` overload (no reservation; the converter may OOM). Call sites: `lock_or_prepare_batch` in `src/include/pipeline/batch_lock_utils.hpp` and the materialized result collector (`src/op/sirius_physical_result_collector.cpp`). The `memory_space*` overloads remain the path for GPU/DISK targets and viability probes.

## Downgrade Executor

**File:** `src/include/downgrade/downgrade_executor.hpp`, `src/downgrade/downgrade_executor.cpp`

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
4. Each candidate is dispatched to the `bounded_thread_pool` and converted via `convertible_data::convert()`. After each conversion, the `predicate` is evaluated. If it returns `true`, no new candidates are dispatched (in-flight conversions finish naturally). The promise resolves with total bytes freed.

**Pipeline integration:** When `gpu_pipeline_executor` gets a partial memory reservation (shortfall), it issues a single `request_downgrade(predicate)` where the predicate attempts `make_reservation_or_null(bytes_needed)`. The downgrade stops as soon as the reservation succeeds -- single request, no over-freeing.

### Candidate Selection Strategy

Candidates are fetched lazily via `convertible_data_provider` implementations:

1. **Data repositories** are iterated in repository manager order. Within each repository, `convertible_data_batch_provider` iterates partitions back-to-front, then batches back-to-front, filtering for idle batches in the source memory space.
2. **Pipeline task queue** is inspected via `convertible_gpu_pipeline_task_provider`, which uses `mutable_pop_if` to temporarily extract tasks with matching data batches. Tasks are returned to the queue via RAII on all code paths.

Candidates are converted individually via `convertible_data::convert()`, which handles state locking, memory reservation, tier conversion, and failure rollback atomically.

## Memory Consumption History

**File:** `src/include/pipeline/pipeline_memory_history.hpp`

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

`gpu_pipeline_task::get_estimated_reservation_size_info(target_space)` uses `estimate_peak_memory()` for the reservation, adding `_bytes_to_materialize_input` — the bytes needed to materialize inputs into the task's target memory space (HOST/disk upgrades plus cross-GPU clones) — and subtracting it from recorded peak to keep operator history clean of materialization overhead. Materialization allocations are charged to the task's reservation through the default per-thread allocation tracking; with the non-default `track_per_stream_reservation: true`, converter allocations made on internal pool streams bypass the task's reservation and are only checked against space capacity.

A cached scan input (a resident `scan_operator_input`) is sized by a dedicated branch before the pipelineable-input walk: it contributes its uncompressed GPU footprint when the cached batch is not already GPU-resident (since `prepare_for_processing()` uploads HOST-cached batches before execution), and zero when it is. Resident classification is deliberately unchanged by this sizing — a HOST cache hit must not be treated as a fresh read, which would apply the fresh-read decode heuristic instead.

## Memory Pool Defragmentation

**File:** `src/include/memory/defragmenter_oom_policy.hpp`, `src/memory/defragmenter_oom_policy.cpp`

`defragmenter_oom_policy` implements `cucascade::memory::oom_handling_policy`:

On allocation failure:
1. Check CUDA pool fragmentation via `cudaMemPoolGetAttribute()` (reserved vs. used)
2. If `reserved > used + (10× requested bytes)`: pool is fragmented
3. Trim pool with `cudaMemPoolTrimTo()` to release free blocks to driver
4. Retry allocation
5. If still fails: rethrow original exception

## Pinned Host Memory

**File:** referenced in `src/include/sirius_context.hpp`

`small_pinned_host_memory_resource` provides fast host memory allocation:

- Fixed-size block pools: 1 MiB blocks, 128 blocks per pool, with four pools initially
- Automatic NUMA node affinity
- Used for GPU↔CPU transfers and scan caching
- Configured via `sirius.yaml` (see [Configuration](configuration.md))

## Key Files

| File | Purpose |
|------|---------|
| `src/include/memory/sirius_memory_reservation_manager.hpp` | Memory manager, tier configuration |
| `src/include/downgrade/downgrade_executor.hpp` | Downgrade executor interface |
| `src/downgrade/downgrade_executor.cpp` | Processing loop, tiered candidate fetching |
| `src/include/memory/defragmenter_oom_policy.hpp` | Pool defragmentation policy |
| `src/memory/defragmenter_oom_policy.cpp` | Fragmentation detection and trimming |
| `src/include/pipeline/pipeline_memory_history.hpp` | Per-pipeline memory consumption history |
