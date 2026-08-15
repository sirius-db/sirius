# Architecture Overview

This document describes the high-level architecture of Super Sirius, including component ownership, thread model, and execution lifecycle.

## Component Diagram

```mermaid
graph TD
    DuckDB["DuckDB Client"] -->|"CALL gpu_execution(...)"| EXT["sirius_extension.cpp"]
    EXT --> IFACE["sirius_interface"]
    IFACE --> ENGINE["sirius_engine"]
    ENGINE -->|"build pipelines"| PLANNER["sirius_physical_plan_generator"]
    ENGINE -->|"execute"| PE["task_scheduler"]

    PE --> GPE["gpu_pipeline_executor(s)"]
    PE --> TC["task_creator"]

    TC -->|"schedule GPU tasks"| GPE

    SM["sirius_scan_manager"] -->|"prepare per-scan state"| GPE
    SM -->|"I/O backends + prefetch cache"| IO["io_context (uring / rest / kvikio)"]

    GPE -->|"unified GPU scan source"| SM
    GPE -->|"memory reservations"| MRM["sirius_memory_reservation_manager"]
    GPE -->|"consume/produce"| DRM["data_repository_manager_registry (one manager per query)"]

    DE["downgrade_executor(s)"] -->|"monitor pressure"| MRM
    DE -->|"move GPU→Host"| DRM

    subgraph SiriusContext
        MRM
        DRM
        PE
        TC
        SM
        DE
    end
```

## Ownership Hierarchy

`SiriusContext` (`src/include/sirius_context.hpp`) is a `ClientContextState` subclass, registered once per `DatabaseInstance` and shared by every connection, that owns the lifetime of all Sirius subsystems:

```
SiriusContext
├── sirius_config                       # Configuration (thread counts, memory sizes, operator params)
├── sirius_memory_reservation_manager   # GPU/Host/Disk memory management via cuCascade
├── numa_small_pinned_mr                # NUMA-aware pinned host memory allocator for cuDF
├── data_repository_manager_registry    # One shared_data_repository_manager per in-flight query
├── query_lifecycle_registry            # Per-query enqueue gate (open → quiescing → closed)
├── task_scheduler                      # Top-level executor (owns the GPU pipeline executors)
├── sirius_scan_manager                 # Scan-side preparation + I/O (io_context, prefetch cache, split providers)
├── downgrade_executor[]                # Per-memory-space monitors for GPU→Host spilling
└── task_creator                        # Creates GPU pipeline tasks based on data availability
```

There is no context-owned "current query": each execution window mints its own `query_id`, and
per-query state lives in per-query entries inside the subsystems (see
[Concurrency Model](concurrency-model.md)). The `planner::query` object is owned by the window's
`sirius_engine`.

Key lifecycle methods on `SiriusContext`:
- `initialize()` — initializes all subsystems with config
- `terminate()` — releases all resources
- `StandaloneQueryScope` / `SlotGuard` — RAII execution/plan windows: admission slot, per-query
  config snapshot, begin mutations, and (for `StandaloneQueryScope::finish()`) the mandatory
  per-query cleanup
- `QueryBegin()` / `QueryEnd()` — DuckDB query lifecycle hooks (logging only; slot ownership is
  scope-bound to the windows above)
- `create_query()` — creates a new query with pipeline metadata and registers its per-query
  task-creator and scan-manager state; ownership of the query returns to the caller's engine

Scans are not a separate executor. A unified `sirius_gpu_scan_operator` (operator type `GPU_SCAN`) is the pipeline source: it pulls splits from a `split_connector` and delegates per-split materialization to an installed `gpu_ingestible` (parquet or duckdb-native today). The `sirius_scan_manager` prepares this state per query — it builds the per-table ingestible, installs the split connector, drives a `split_provider`, and owns the I/O backends (an `io_context` over io_uring plus optional REST/kvikio paths) and the prefetching cache.

## Thread Model

Super Sirius uses multiple dedicated thread pools, each with a specific role:

```
┌─────────────────────────────────────────────────────────────────┐
│  DuckDB Query Thread (main)                                     │
│  - Parses SQL, generates logical plan                           │
│  - Calls sirius_interface → sirius_engine                       │
│  - Builds pipelines (single-threaded)                           │
│  - Calls task_scheduler.start_query()                           │
│  - Blocks on future until query completes                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Pipeline Executor Management Thread                            │
│  - Runs management_eventloop()                                  │
│  - Listens on task_request_channel for GPU executor requests    │
│  - Dequeues pipeline tasks and routes to GPU executors           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  GPU Pipeline Executor (per GPU device)                         │
│  - Manager thread: reserves a worker slot → publishes           │
│    device_ready → pops a task → dispatches to the worker pool   │
│    (never blocks on memory — register C4)                       │
│  - Worker threads: reserve memory (downgrade-on-shortfall),     │
│    then execute GPU pipeline tasks (including the unified GPU   │
│    scan source) on dedicated CUDA streams                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Task Creator Thread Pool                                       │
│  - Manager loop: pops from task_creation_queue                  │
│  - Follows hint chain to find ready operators                   │
│  - Creates GPU pipeline tasks                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Scan Manager                                                   │
│  - Worker thread pool: per-scan preparation                    │
│  - Driver thread: runs split providers sequentially, feeding    │
│    splits into each scan operator's split_connector            │
│  - I/O reactor threads: io_uring (local disk) and REST/kvikio   │
│    backends behind the io_context, plus the prefetching cache   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Downgrade Executor(s) (per memory space)                       │
│  - Monitor thread: polls memory pressure                        │
│  - Manager thread: dispatches downgrade tasks                   │
│  - Worker threads: move data GPU→Host                           │
└─────────────────────────────────────────────────────────────────┘
```

## Execution Lifecycle

A query through Super Sirius follows these steps (the admission/cleanup brackets around them are
covered in [Concurrency Model](concurrency-model.md)):

1. **Admission** — a `SiriusContext::StandaloneQueryScope` opens the execution window: it
   acquires a slot from the counted query-lifecycle gate, mints the window's `query_id`, takes
   the per-query config snapshot, opens the query in the `query_lifecycle_registry`, and creates
   the query's data-repository manager
2. **Parse & Optimize** — DuckDB parses the SQL string and produces an optimized logical plan
3. **Physical Plan Generation** — `sirius_physical_plan_generator::create_plan()` converts the DuckDB logical plan into a Sirius physical operator tree
4. **Engine Initialization** — `sirius_engine::initialize()` builds the pipeline graph:
   - Constructs `sirius_meta_pipeline` from the physical plan via `build()` + `ready()`
   - Splits operators (TABLE_SCAN, joins, aggregates, sorts) into multiple pipelines
   - Converts each TABLE_SCAN source into a unified GPU scan source with a per-table `gpu_ingestible`
   - Injects PARTITION, CONCAT, MERGE operators at pipeline boundaries
   - Wires data repositories between pipelines with barrier types
   - Computes the query's admitted GPU subset and installs it via
     `task_creator::set_active_gpu_ids(query_id, ...)`
5. **Query Preparation** — `SiriusContext::create_query()` calls
   `task_creator::prepare_for_query()` (registers this query's per-pipeline task global states
   and completion handler) and `sirius_scan_manager::prepare_for_query()` (builds each scan's
   split provider, installs its split connector, and matches any pinned-cache entries)
6. **Query Start** — `task_scheduler::start_query()` schedules the query's first scan operator
   through the task creator; the engine already owns the query's `completion_handler` and its
   future
7. **Scan Phase** — The scan manager drives split providers that pull bytes through the `io_context` (io_uring locally, or REST/kvikio backends) and the prefetching cache; the unified GPU scan source consumes splits and materializes GPU-ready batches into data repositories
8. **Pipeline Execution** — GPU executor workers acquire memory reservations and call `execute()` on every operator in the pipeline (source through sink) on CUDA streams, then call the sink's `sink()` to push results downstream
9. **Task Creation** — After each task completes, the task creator is notified to schedule downstream consumers based on data availability in ports
10. **Memory Management** — Downgrade executors monitor GPU memory pressure and spill data to host memory when thresholds are exceeded, sweeping across every in-flight query's repositories
11. **Completion** — When the final `RESULT_COLLECTOR` pipeline finishes, `completion_handler::mark_completed()` signals the future
12. **Result Extraction** — The main thread extracts the `MaterializedQueryResult` from the result collector and returns it to DuckDB
13. **Cleanup** — `StandaloneQueryScope::finish()` runs `run_mandatory_cleanup(query_id)` —
    quiesce, per-query drains, parked-plan destruction, repository erase — then releases the slot

## Key Source Files

| File | Role |
|------|------|
| `src/include/sirius_context.hpp` | Ownership hierarchy, subsystem lifecycle |
| `src/sirius_extension.cpp` | Extension registration, table functions, config |
| `src/sirius_interface.cpp` | DuckDB-facing API, query lifecycle |
| `src/sirius_engine.cpp` | Pipeline construction, execution orchestration |
| `src/planner/sirius_physical_plan_generator.cpp` | Logical-to-physical plan translation |
| `src/include/pipeline/task_scheduler.hpp` | Top-level executor (owns GPU executors) |
| `src/include/pipeline/gpu_pipeline_executor.hpp` | Per-GPU task executor |
| `src/include/creator/task_creator.hpp` | Task creation and scheduling |
| `src/include/op/scan/sirius_gpu_scan_operator.hpp` | Unified GPU scan source operator |
| `src/include/op/scan/gpu_ingestible.hpp` | Per-format split materialization (parquet, duckdb-native) |
| `src/include/scan_manager/sirius_scan_manager.hpp` | Per-scan preparation, split providers, I/O ownership |
| `src/include/io/io_context.hpp` | I/O backends (uring / rest / kvikio) + prefetch cache |
| `src/include/downgrade/downgrade_executor.hpp` | Memory spilling |
| `src/include/memory/sirius_memory_reservation_manager.hpp` | Memory management |
| `src/include/exec/query_lifecycle_registry.hpp` | Per-query enqueue gate |
| `src/include/data/data_repository_manager_registry.hpp` | Per-query repository managers + sweep fence |
| `src/include/query_id.hpp` | Window/query identity and priority-band packing |
