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

- **Monitor thread** (`monitor_loop()`): polls memory space every ~10ms for `should_downgrade_memory()` signal
- **Manager thread** (`manager_loop()`): dispatches downgrade tasks from queue to thread pool via kiosk
- **Worker thread pool** (default: 4 threads): executes actual data movement

### Downgrade Pass

`run_downgrade_pass()` selects data batches for downgrade:

**Selection strategy:**
1. Partitioned repositories first, then by descending data size
2. Two-pass within each repository:
   - **Pass 1**: Non-active partitions (less likely to be needed soon)
   - **Pass 2**: Active partitions (if pressure persists)
3. Within a repository: iterate partitions last-to-first (newer data first)

**Execution per batch:**
1. Check if batch is already on HOST tier (skip if so)
2. `try_to_lock_for_in_transit()` — prevents concurrent pipeline access
3. Acquire HOST memory reservation
4. Convert GPU representation → HOST representation via `converter_registry`
5. Release in-transit lock, restore previous batch state
6. Send completion message to `task_creator` for downstream scheduling

The downgrade executor runs **concurrently** with pipeline execution, monitoring memory pressure asynchronously.

### `downgrade_task`

**File:** `src/include/downgrade/downgrade_task.hpp`

Global state shared across all downgrade tasks:
- Reference to `sirius_memory_reservation_manager`
- Reference to `data_repository_manager`
- Reference to task completion message queue

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
- Configured via `sirius.cfg` (memory manager settings in cuCascade)

## Key Files

| File | Purpose |
|------|---------|
| `src/include/memory/sirius_memory_reservation_manager.hpp` | Memory manager, tier configuration |
| `src/include/downgrade/downgrade_executor.hpp` | Downgrade executor interface |
| `src/downgrade/downgrade_executor.cpp` | Monitor loop, candidate selection |
| `src/include/downgrade/downgrade_task.hpp` | Downgrade task definition |
| `src/include/memory/defragmenter_oom_policy.hpp` | Pool defragmentation policy |
| `src/memory/defragmenter_oom_policy.cpp` | Fragmentation detection and trimming |
