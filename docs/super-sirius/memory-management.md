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
| `reservation_limit_fraction` | 0.9 | 0.9 | Max fraction reservable |
| `downgrade_trigger_fraction` | 1.0 | 0.8 | When to start downgrading |
| `downgrade_stop_fraction` | 0.7 | 0.7 | When to stop downgrading |

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
reservation_limit_fraction = 0.9;
downgrade_trigger_fraction = 1.0;
downgrade_stop_fraction = 0.7;
```

**Host Memory Space:**
```cpp
numa_id;                        // NUMA node affinity
reservation_limit_fraction = 0.9;
downgrade_trigger_fraction = 0.8;
downgrade_stop_fraction = 0.7;
block_size = 64MB;              // cuCascade block size
pool_size = 1024;               // blocks per pool
```

**Disk Memory Space:**
```cpp
disk_id;
mount_paths;                    // directories for spill files
memory_capacity = 1TB;          // total spill capacity
```

## Memory Reservations

Pipeline tasks acquire memory reservations before execution to prevent GPU OOM:

1. GPU executor's `manager_loop()` calls `memory_space.make_reservation(estimated_size)`
2. The reservation is attached to the task's local state via `set_reservation()`
3. During execution, operators allocate within the reservation
4. Reservations are released when the task completes

### `reservation_aware_resource_adaptor`

Wraps RMM device memory resource. On each allocation:
- Checks if the reservation has sufficient capacity
- If exhausted → fails gracefully, triggering `oom_reschedule_exception`
- Enables predictable memory usage per task

## Downgrade Executor

**File:** `src/include/downgrade/downgrade_executor.hpp`, `src/downgrade/downgrade_executor.cpp`

One `downgrade_executor` per memory space monitors pressure and moves data to lower tiers.

### Thread Model

- **Processing thread**: dequeues `downgrade_request` objects sequentially from an `interruptible_mpmc` queue
- **Monitor thread** (if `monitor_period_ms > 0`): polls memory space for pressure and fires monitor requests via fire-and-forget into the same queue
- **Worker thread pool** (`exec::bounded_thread_pool`): executes actual data movement concurrently

### Downgrade Request Pattern

The downgrade executor uses a request-based model instead of a retry loop:

1. Caller invokes `request_downgrade(target_bytes, predicate)` which constructs a `downgrade_request` and pushes it onto the MPMC queue. Returns `std::future<size_t>`.
2. The processing thread dequeues requests **sequentially** (to avoid contention between concurrent requests competing for the same batches).
3. For each request, `collect_all_candidates()` iterates all data repositories, collecting GPU-resident batches as `weak_ptr` up to `target_bytes` worth.
4. Candidates are dispatched to the `bounded_thread_pool` one-by-one. After each batch completes downgrade, the `predicate` is evaluated. If it returns `true`, no new batches are dispatched (in-flight batches finish naturally). The promise resolves with total bytes freed.

**Pipeline integration:** When `gpu_pipeline_executor` gets a partial memory reservation (shortfall), it issues a single `request_downgrade(target_bytes, predicate)` where the predicate attempts `make_reservation_or_null(bytes_needed)`. The downgrade stops as soon as the reservation succeeds — single request, no over-freeing.

### Candidate Selection Strategy

`collect_all_candidates()` selects data batches for downgrade:

1. Partitioned repositories first, then by descending data size
2. Two-pass within each repository:
   - **Pass 1**: Non-active partitions (less likely to be needed soon)
   - **Pass 2**: Active partitions (if pressure persists)
3. Within a repository: iterate partitions last-to-first (newer data first)

Candidates are collected as `weak_ptr` and locked just before dispatch to avoid unnecessary lifetime extension and skip already-freed batches.

### `downgrade_task`

**File:** `src/include/downgrade/downgrade_task.hpp`

A plain struct (no polymorphism) holding a `shared_ptr<data_batch>` and a reference to the reservation manager. Execution per batch:
1. Lock `weak_ptr` to `shared_ptr` (skip if already freed)
2. `try_to_lock_for_in_transit()` — prevents concurrent pipeline access
3. Acquire HOST memory reservation
4. Convert GPU representation → HOST representation via `converter_registry`
5. Release in-transit lock, restore previous batch state

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

`gpu_pipeline_task::get_estimated_reservation_size()` uses `estimate_peak_memory()` for the reservation, adding `_bytes_to_materialize_input` (bytes to pull from HOST/disk to GPU) and subtracting it from recorded peak to keep operator history clean of I/O materialization overhead.

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

- Fixed-size block pools: 64MB blocks, 1024 blocks per pool
- Automatic NUMA node affinity
- Used for GPU↔CPU transfers and scan caching
- Configured via `sirius.yaml` (see [Configuration](configuration.md))

## Key Files

| File | Purpose |
|------|---------|
| `src/include/memory/sirius_memory_reservation_manager.hpp` | Memory manager, tier configuration |
| `src/include/downgrade/downgrade_executor.hpp` | Downgrade executor interface |
| `src/downgrade/downgrade_executor.cpp` | Monitor loop, candidate selection |
| `src/include/downgrade/downgrade_task.hpp` | Downgrade task definition |
| `src/include/memory/defragmenter_oom_policy.hpp` | Pool defragmentation policy |
| `src/memory/defragmenter_oom_policy.cpp` | Fragmentation detection and trimming |
| `src/include/pipeline/pipeline_memory_history.hpp` | Per-pipeline memory consumption history |
