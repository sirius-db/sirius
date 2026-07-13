# Data Management

This document covers data batches, data repositories, ports, barrier semantics, and data format conversion.

## Data Batch Lifecycle

A data batch flows through the system in these stages:

```
1. Scan Phase:     scan_task creates data_batch (idle) with host/parquet representation
2. Repository:     idle batch stored in shared_data_repository
3. Consumption:    task_creator calls pop_next_data_batch(); batch is still idle
4. Preparation:    prepare_for_processing() acquires read_only_data_batch handles,
                   converting to GPU memory space if needed (lock_or_prepare_batch)
5. Execution:      GPU task reads data through read_only_data_batch accessors
6. Output:         operators produce new idle batches, pushed to downstream repositories
7. Downgrade:      if GPU memory pressure, downgrade executor upgrades to mutable,
                   converts representation to HOST, releases back to idle
8. Cleanup:        batch destroyed when shared_ptr ref count reaches zero
```

### Owned vs. view-backed GPU batches

A `gpu_table_representation` holds its data in one of two forms:

- **Owned** — a `std::unique_ptr<cudf::table>`. Built via `sirius::make_data_batch(table, space, stream)`. This is the common case (operators that allocate new output columns).
- **View-backed (`owning_table_view`)** — a non-owning `cudf::table_view` plus a type-erased (`std::any`) *owner* that keeps the viewed device memory alive. Built via `sirius::make_data_batch_from_view(view, owner, alloc_size, space, stream)`. Used to expose existing columns without copying.

The owner can be any copy-constructible object that keeps the memory alive — e.g. a `read_only_data_batch` lock on a source batch (so the source stays alive and read-only-pinned for the view's lifetime), a `shared_ptr<cudf::table>`, or a composite of both. Producers of view-backed batches include the pinned-table scan path and the PROJECTION operator's zero-copy passthrough paths (see [operators](operators.md)). Downstream code is agnostic: `get_table_view()` works for both forms, and `release_table()` materializes a view-backed batch into an owned table on demand.

### Batch State Machine

Each `data_batch` (from cuCascade) uses a 3-class reader-writer locking model. Data is only
accessible through RAII accessor objects — the idle `shared_ptr<data_batch>` grants no data access.

```
idle ←→ read_only      (shared lock; multiple concurrent readers)
idle ←→ mutable_locked (exclusive lock; one writer, no readers)
read_only ←→ mutable_locked (upgrade/downgrade)
```

- **`idle`**: No active locks. Available for locking, cloning, or tier movement.
- **`read_only`**: One or more `read_only_data_batch` shared locks active. Concurrent reads allowed.
- **`mutable_locked`**: One `mutable_data_batch` exclusive lock active. Data can be read and mutated.

Key methods on `data_batch`:
- `batch->to_read_only()` — blocking shared lock → `read_only_data_batch`
- `batch->to_mutable()` — blocking exclusive lock → `mutable_data_batch`
- `batch->try_to_read_only()` / `try_to_mutable()` — non-blocking variants returning `std::optional`
- `data_batch::to_idle(std::move(accessor))` — release lock, return `shared_ptr<data_batch>`
- `data_batch::readonly_to_mutable(std::move(ro))` — upgrade shared → exclusive
- `data_batch::mutable_to_readonly(std::move(mut))` — downgrade exclusive → shared

## Data Repositories

Data repositories are thread-safe containers managed by the `shared_data_repository_manager`:

- Keyed by `(operator_id, port_id)` pairs
- Support partitioned storage (multiple partitions per repository)
- Provide `add_data_batch()` for producers and `pop_next_data_batch()` for consumers (non-blocking; returns `nullptr` if empty)
- Track total size and per-partition sizes
- Registered centrally in `shared_data_repository_manager` for downgrade candidate selection

### `shared_data_repository_manager`

Central registry of all repositories in query execution:
- Provides `for_each_repository()` iterator for downgrade candidate selection
- Thread-safe access to all active repositories

## Port System

**File:** `src/include/op/sirius_physical_operator.hpp`

Ports connect pipelines by routing data from one operator's output to another's input:

```cpp
struct port {
    MemoryBarrierType type;                    // PIPELINE, PARTIAL, or FULL
    cucascade::shared_data_repository* repo;   // holds queued data_batch objects
    shared_ptr<sirius_pipeline> src_pipeline;  // pipeline producing data
    shared_ptr<sirius_pipeline> dest_pipeline; // pipeline consuming data
};
```

### Barrier Semantics

| Barrier | Behavior | When Used |
|---------|----------|-----------|
| `FULL` | Downstream waits until upstream pipeline is **completely finished** before consuming any data | Hash join build side — entire hash table must be built before probing |
| `PARTIAL` | Downstream can consume data **incrementally** as it arrives, but respects pipeline boundaries | CONCAT after PARTITION in streaming joins (INNER) |
| `PIPELINE` | No synchronization — data flows **immediately** | Within a single pipeline |

### `push_data_batch()`

When a sink's `sink()` method produces output batches, the default implementation pushes each batch to downstream operators:

```cpp
for (auto& batch : output_batches) {
    for (auto& [next_op, port_id] : next_port_after_sink) {
        next_op->push_data_batch(port_id, batch);
    }
}
```

`next_port_after_sink` is configured during pipeline construction by `insert_repository()`.

### Port Names

Operators access their ports by string name:
- `"default"` — primary input (most operators)
- `"build"` — build-side input (hash join only)

## Operator Data Containers

### `operator_data`

Minimal empty base class. Provides a generic extension point for any type of operator data — signaling objects, metadata-only data, or non-batch representations can derive from `operator_data` without being forced into the batch model.

### `pipelineable_operator_data`

Extends `operator_data` with batch-based data flow. Holds two optional internal stores that are
lazily populated from each other on demand:
- `_data_batches` — `std::vector<shared_ptr<data_batch>>` (idle pointers)
- `_read_only_data_batches` — `std::vector<read_only_data_batch>` (shared-locked accessors)

Key methods:
- `get_data_batches()` — returns idle batch pointers; if only `_read_only_data_batches` exist, calls `data_batch::to_idle()` on copies to populate `_data_batches`.
- `get_read_only_batches(bool leave_locked)` — acquires `to_read_only()` on each idle batch; if `leave_locked=true`, caches result in `_read_only_data_batches`.
- `prepare_for_processing(memory_space*, stream)` — **void**, throws on failure. Calls `lock_or_prepare_batch()` for each batch (converts to the target memory space if needed, then acquires a shared lock). Stores resulting `read_only_data_batch` handles in `_read_only_data_batches`. Called by the GPU pipeline executor before `execute()`.
- `remove_read_only_lock()` — releases `_read_only_data_batches` while ensuring `_data_batches` is populated first (so the data stays alive).
- Created by `get_next_task_input_data()` from port pops.
- Passed through the operator chain during `execute()`.

### `partitioned_operator_data`

Extends `pipelineable_operator_data` with a partition index (`get_partition_idx()`). Used by partition-aware operators (CONCAT, MERGE_SORT, MERGE_GROUP_BY) to track which partition the data belongs to.

### Class Hierarchy

```
operator_data                       (empty generic base)
  └── pipelineable_operator_data    (batch vector + data flow methods)
       └── partitioned_operator_data (adds partition_idx)
```

## Data Format Conversion

### `sirius_converter_registry`

**File:** `src/include/data/sirius_converter_registry.hpp`

Global singleton for converting between data representations:
- Registers builtin cuCascade converters
- Thread-safe initialization via mutex
- Used by:
  - Downgrade tasks: GPU representation → HOST representation
  - GPU pipeline tasks: HOST representation → GPU representation (`lock_or_prepare_batch`)

### Conversion Examples

| From | To | When |
|------|----|------|
| `host_data_representation` | GPU `cudf::table` | GPU task input preparation |
| GPU `cudf::table` | `host_data_representation` | Downgrade executor |

## Key Files

| File | Purpose |
|------|---------|
| `src/include/op/sirius_physical_operator.hpp` | Port struct, barrier types, push_data_batch |
| `src/op/sirius_physical_operator.cpp` | Default sink/push implementation |
| `src/include/data/sirius_converter_registry.hpp` | Format conversion registry |
| `src/include/memory/multiple_blocks_allocation_accessor.hpp` | Multi-block allocation cursor |
