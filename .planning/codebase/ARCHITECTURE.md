# Architecture

**Analysis Date:** 2026-04-13

## Pattern Overview

**Overall:** Task-Based Multi-Pipeline GPU Execution Engine (Super Sirius)

Sirius is a GPU-native SQL execution engine that intercepts DuckDB's logical query plans and converts them into a multi-pipeline, task-based execution model that partitions work across GPU and CPU thread pools with tiered memory management (GPU/Host/Disk via cuCascade). The system uses namespace `sirius` and is invoked via `CALL gpu_execution('SELECT ...')`.

**Key Characteristics:**
- DuckDB extension architecture — loads as `sirius.duckdb_extension`
- Logical-to-physical plan conversion with automatic pipeline splitting
- Multi-threaded task scheduling with dedicated thread pools (GPU executors, scan executors, task creator, downgrade monitors)
- Hierarchical operator pipeline model where operators appear as both sinks and sources across pipeline boundaries
- Unified data repository system with configurable memory barrier semantics (FULL, PARTIAL, PIPELINE)
- Graceful CPU fallback for unsupported operators/data types via `src/fallback.cpp`
- GPU memory spilling via cuCascade tiered memory (GPU → Host → Disk on pressure)

> **Legacy Note:** An older code path (`gpu_processing`, `namespace duckdb`) exists in `src/operator/`, `src/plan/`, `src/legacy/` for backward compatibility. All new development targets Super Sirius.

## Layers

**Extension Layer:**
- Location: `src/sirius_extension.cpp`
- Purpose: DuckDB extension registration, table function binding/execution, SQL parsing
- Contains: `GPUExecutionBind()` (parses SQL, generates Sirius physical plan), `GPUExecutionFunction()` (delegates to sirius_interface)
- Depends on: DuckDB parser, optimizer, sirius_interface
- Used by: DuckDB query executor (as a table function)

**Interface Layer:**
- Location: `src/sirius_interface.cpp`, `src/include/sirius_interface.hpp`
- Purpose: Query lifecycle management, error handling, prepared statement execution
- Contains: `sirius_interface` class with methods: `sirius_execute_query()`, `sirius_pending_statement_internal()`, `fetch_result_internal()`, `cleanup_internal()`
- Depends on: sirius_engine, DuckDB prepared statements
- Used by: Extension layer; returns `MaterializedQueryResult` to DuckDB

**Engine & Planning Layer:**
- Location: `src/sirius_engine.cpp`, `src/include/sirius_engine.hpp`
- Purpose: Pipeline construction, operator tree traversal, data repository wiring
- Contains: `sirius_engine` class with methods: `initialize()` (builds pipelines), `execute()` (runs query), `insert_repository()` (wires ports with barrier types)
- Pipeline construction (initialize_internal):
  - Calls `sirius_physical_plan_generator::create_plan()` to convert DuckDB logical plan to Sirius physical operators
  - Builds `sirius_meta_pipeline` via recursive `build_pipelines()` calls on each operator
  - Splits operators into multiple pipelines: TABLE_SCAN → DUCKDB_SCAN/PARQUET_SCAN, HASH_JOIN → PARTITION+CONCAT, aggregates → PARTITION+MERGE, sorts → 4-phase pipeline
  - Registers `shared_data_repository` instances between pipelines
  - Finalizes pipelines by pushing sink into operators array and reversing for execution order
- Depends on: Physical plan generator, sirius_context, operators (all in src/op/)
- Used by: sirius_interface; calls pipeline_executor.start_query()

**Physical Plan Generation:**
- Location: `src/planner/sirius_physical_plan_generator.cpp`, `src/planner/sirius_plan_*.cpp`
- Purpose: Convert DuckDB logical operators to Sirius physical operators
- Contains: Mapping from DuckDB LogicalOperator types to sirius_physical_operator subclasses
- Key files:
  - `sirius_physical_plan_generator.cpp` — main dispatcher (switch on operator type)
  - `sirius_plan_get.cpp` — TABLE_SCAN with filter pushdown
  - `sirius_plan_filter.cpp` — FILTER operator
  - `sirius_plan_aggregate.cpp` — HASH_GROUP_BY, UNGROUPED_AGGREGATE, AVG decomposition
  - `sirius_plan_comparison_join.cpp` — HASH_JOIN, NESTED_LOOP_JOIN
  - `sirius_plan_order.cpp` — ORDER_BY (4-phase sort)
  - `sirius_plan_top_n.cpp` — TOP_N with merge
- Depends on: DuckDB logical plan classes
- Used by: Extension layer (binds) and sirius_engine (initialize)

**Operator Layer:**
- Location: `src/op/`, `src/include/op/`
- Purpose: Physical operator implementations with GPU and CPU fallback logic
- Categories:
  - **Scan operators** (`src/op/scan/`): DUCKDB_SCAN, PARQUET_SCAN, ICEBERG_SCAN, DUMMY_SCAN, COLUMN_DATA_SCAN
  - **Streaming operators** (`src/op/`): FILTER, PROJECTION, LIMIT
  - **GPU blocking operators** (`src/op/aggregate/`, `src/op/order/`, `src/op/partition/`, `src/op/merge/`): HASH_JOIN, HASH_GROUP_BY, ORDER_BY, TOP_N with their merge variants
  - **Control operators** (`src/op/result/`, `src/op/partition/`, `src/op/merge/`): PARTITION, CONCAT, MERGE_GROUP_BY, MERGE_SORT, RESULT_COLLECTOR
- Base class: `sirius_physical_operator` (`src/include/op/sirius_physical_operator.hpp`, `src/op/sirius_physical_operator.cpp`)
  - Core methods: `execute()` (per-operator processing on CUDA stream), `sink()` (push output to downstream), `get_next_task_hint()` (signal data availability)
  - Port management: `add_port()`, `get_input_data_batch()`, `push_data_batch()`
  - Pipeline role: `is_source()`, `is_sink()`, `can_create_more_tasks()`
- Depends on: Expression executor (GPU kernels), cuDF, RMM, cuCascade
- Used by: sirius_engine (during initialize), gpu_pipeline_executor (during execute)

**Pipeline Execution Layer:**
- Location: `src/pipeline/`, `src/include/pipeline/`
- Purpose: Task scheduling, GPU executor management, execution orchestration
- Key files:
  - `pipeline_executor.cpp`, `src/include/pipeline/pipeline_executor.hpp` — top-level executor managing all sub-executors and task routing
  - `gpu_pipeline_executor.cpp`, `src/include/pipeline/gpu_pipeline_executor.hpp` — per-GPU task executor with worker thread pool
  - `sirius_pipeline.cpp`, `src/include/pipeline/sirius_pipeline.hpp` — pipeline metadata (operators, dependencies, completion state)
  - `gpu_pipeline_task.cpp`, `src/include/pipeline/gpu_pipeline_task.hpp` — task definition with data batch collection and sink operations
  - `sirius_pipeline_converter.cpp` — converts finalized pipelines into executable GPU tasks
  - `sirius_meta_pipeline.cpp`, `src/include/pipeline/sirius_meta_pipeline.hpp` — groups pipelines sharing same sink, manages build order during construction
- Thread Model:
  - **Query thread**: DuckDB thread, calls sirius_interface, blocks on future until completion
  - **Pipeline executor management thread**: Runs `management_eventloop()`, routes tasks from task creator to GPU/scan executors
  - **GPU executor manager/workers** (per GPU device): Manager acquires kiosk tickets, requests tasks, reserves memory; workers execute on CUDA streams
  - **Scan executor manager/workers**: Manager pops scan tasks; workers execute DuckDB/Parquet scans, publish results
  - **Task creator thread pool** (default: 2): Monitors port readiness, creates downstream GPU/scan tasks based on operator hints
  - **Downgrade executor(s)**: Per-memory-space monitors poll GPU pressure every ~10ms, dispatch spill operations
- Depends on: Operators, data repositories, memory manager, task creator, cuCascade
- Used by: sirius_engine (calls start_query), query completion handler

**Task Creation Layer:**
- Location: `src/creator/task_creator.cpp`, `src/include/creator/task_creator.hpp`
- Purpose: Convert ready operators into executable tasks, follow data availability chain
- Contains: `task_creator` class with method `schedule(operator*)` — pops from task creation queue, calls `get_operator_for_next_task()` to follow hint chain, creates GPU pipeline or scan task
- Task decision logic: checks `operator->get_next_task_hint()` (READY vs WAITING_FOR_INPUT_DATA), recursively follows producer chain
- Depends on: Operators, pipeline executor, data repositories
- Used by: Scan executors and GPU executors (call `schedule()` after task completion)

**Memory Management Layer:**
- Location: `src/memory/`, `src/include/memory/`
- Purpose: GPU/Host/Disk tiered memory management via cuCascade, reservation tracking
- Key files:
  - `sirius_memory_reservation_manager.hpp`, `src/memory/sirius_memory_reservation_manager.cpp` — manages reservations across memory spaces
  - `src/include/memory/` — reservation types, memory space APIs
- Integration: GPU executor reserves memory before executing; downgrade executor monitors pressure and moves batches; RMM pools allocated at startup
- Depends on: cuCascade, RMM, DuckDB memory allocators
- Used by: GPU executors (reserve), downgrade executors (spill), scan executors (host memory for parquet I/O)

**Downgrade & Fallback Layer:**
- Location: `src/downgrade/`, `src/include/downgrade/`, `src/fallback.cpp`
- Purpose: GPU memory spilling (GPU → Host) and CPU fallback for unsupported operations
- Downgrade: `src/include/downgrade/downgrade_executor.hpp` — monitors GPU memory pressure, dispatches spill tasks
- Fallback: `src/fallback.cpp` — if operator throws `NotImplementedException` or data type not supported, reverts to DuckDB CPU execution
- Depends on: Pipeline executor, data repositories, memory manager
- Used by: GPU executors (catch OOM), extension layer (graceful fallback)

**Expression Evaluation Layer:**
- Location: `src/expression_executor/`, `src/cuda/expression_executor/`, `src/include/expression_executor/`
- Purpose: GPU-accelerated expression evaluation via cuDF AST
- Key files:
  - `src/include/expression_executor/gpu_expression_executor.hpp` — CPU-side expression evaluator
  - `src/cuda/expression_executor/` — cuDF AST translation and kernel dispatch
- Methods: `select()` (filter), `project()` (projection), aggregate functions
- Depends on: cuDF, expression AST (DuckDB), CUDA
- Used by: Operators during execute() phase

**Data Management Layer:**
- Location: `src/data/`, `src/include/data/`
- Purpose: Data batch lifecycle, repositories, port routing, format conversion
- Key components:
  - `shared_data_repository_manager` — central registry of all data repositories by (operator_id, port_id)
  - `shared_data_repository` — thread-safe queue of data batches, supports partitions
  - `data_batch` (from cuCascade) — GPU/Host/Disk representation with state machine (idle, task_created, processing, in_transit)
  - `port` (in sirius_physical_operator) — connects pipelines with barrier semantics (FULL, PARTIAL, PIPELINE)
- Sirius Converter Registry: Converts between DuckDB chunks and GPU-compatible batch formats
- Depends on: cuCascade, DuckDB chunk format
- Used by: All executors (push/pop batches), operators (input/output)

**Context & Configuration:**
- Location: `src/include/sirius_context.hpp`, `src/sirius_engine.cpp`, `src/include/sirius_config.hpp`
- Purpose: Ownership hierarchy and lifetime management of all subsystems
- `SiriusContext` (ClientContextState subclass) owns:
  - `sirius_config` — thread counts, memory limits, operator parameters
  - `sirius_memory_reservation_manager` — GPU/Host/Disk memory allocation
  - `small_pinned_host_memory_resource` — pinned host memory for async I/O
  - `shared_data_repository_manager` — all active data repositories
  - `pipeline_executor` — top-level task orchestration
  - `downgrade_executor[]` — per-memory-space spilling monitors
  - `task_creator` — task scheduling
  - Query context (pipeline hashmap per query)
- Lifecycle: `initialize()` on first query, `QueryBegin()`/`QueryEnd()` per query, `terminate()` on connection close
- Used by: Extension (registers on connection), interface (retrieves per query)

## Data Flow

**Query Execution Flow:**

1. **Bind Phase** (`src/sirius_extension.cpp`):
   - User calls `CALL gpu_execution('SELECT ...')`
   - `GPUExecutionBind()` parses, optimizes, generates Sirius physical plan via `sirius_physical_plan_generator::create_plan()`
   - Returns `SiriusTableFunctionData` with prepared statement

2. **Interface Setup** (`src/sirius_interface.cpp`):
   - `GPUExecutionFunction()` creates `sirius_interface`
   - Calls `sirius_execute_query()` → `sirius_pending_statement_internal()`
   - Creates `sirius_engine`, `sirius_physical_materialized_collector` (result sink)

3. **Pipeline Construction** (`src/sirius_engine.cpp`):
   - `engine.initialize()` builds pipeline graph:
     - Recursively calls `build_pipelines()` on each operator
     - Inserts PARTITION/CONCAT/MERGE operators at pipeline boundaries
     - Creates `shared_data_repository` instances between pipelines
     - Sets barrier types (FULL for joins, PARTIAL for streaming joins, PIPELINE for same-pipeline flow)
     - Finalizes: pushes sink into operators, reverses operator lists
   - Pipeline list stored in `new_scheduled`

4. **Execution Start** (`src/sirius_engine.cpp`, `src/pipeline/pipeline_executor.cpp`):
   - `engine.execute()` calls `pipeline_executor.start_query()`
   - Creates `completion_handler` (promise/future pair)
   - Distributes handler to GPU executor, scan executor, task creator
   - Schedules initial scan tasks
   - Main thread blocks on `future.get()`

5. **Scan Phase** (`src/op/scan/duckdb_scan_executor.cpp`):
   - Scan executor pops scan task from queue
   - Acquires kiosk ticket (worker availability synchronization)
   - Executes scan (DuckDB table function or Parquet I/O)
   - Applies caching (CACHE: compute+save, PRELOAD: load from cache)
   - Publishes output batches to data repository
   - Calls `task_creator->schedule(downstream_operator)`

6. **GPU Pipeline Execution** (`src/pipeline/gpu_pipeline_executor.cpp`):
   - GPU executor pops GPU pipeline task
   - Acquires GPU memory reservation
   - Dispatches to worker thread on CUDA stream:
     - Locks input batches, converts to GPU if needed
     - `compute_task()`: iterates **all** operators in pipeline, calls `execute()` on each
     - `publish_output()`: calls sink's `sink()` to push results downstream
     - On OOM: catches exception, retries up to 10 times with backoff
   - Schedules downstream consumers via `task_creator->schedule()`
   - If RESULT_COLLECTOR completes: calls `completion_handler->mark_completed()`

7. **Task Creation Cycle** (`src/creator/task_creator.cpp`):
   - Task creator receives `schedule(operator*)` calls
   - Calls `get_operator_for_next_task()` which:
     - Calls `operator->get_next_task_hint()` to check port readiness
     - If READY: creates task
     - If WAITING_FOR_INPUT_DATA: recursively follows producer chain
   - Creates appropriate task (scan or GPU pipeline) and routes to executor

8. **Memory Management** (during execution):
   - Downgrade executor polls GPU memory pressure every ~10ms
   - If pressure exceeds threshold: dispatches batch spill tasks (GPU → Host)
   - Batches transitioned between tiers transparently via cuCascade

9. **Result Extraction** (`src/sirius_interface.cpp`):
   - Future resolves when RESULT_COLLECTOR completes
   - `fetch_result_internal()` gets result from engine
   - Result collector returns `ColumnDataCollection` wrapped as `MaterializedQueryResult`
   - Returns to DuckDB

**Error Handling:**
- If operator throws exception: GPU executor catches, calls `completion_handler->report_error()`
- `drain_after_error()` stops task creator, drains queues, waits for executors
- Error propagates through future to main thread

**State Management:**
- Operators maintain state via ports and repositories
- Scan operators track `exhausted` (DuckDB) or `has_more_partitions` (Parquet) atomics
- Blocking operators accumulate via `sink()`, emit via `source` + `execute()` in child pipeline
- Pipeline completion determined by `pipeline_finished` atomic + source depletion check
- Data batches transition through state machine: idle → task_created → processing → idle (or in_transit → idle for downgrades)

## Key Abstractions

**Operator Hierarchy:**
- Purpose: Abstract physical operation with GPU acceleration and CPU fallback
- Base: `sirius_physical_operator` (src/include/op/sirius_physical_operator.hpp)
  - Methods: `execute()` (per-batch processing), `sink()` (output pushing), `is_source()`, `is_sink()`, `get_next_task_hint()`
  - Members: `operator_id`, `type`, ports map, input/output batches
- Categories: Scan (produce data), Streaming (pass-through), Blocking (accumulate), Control (route/collect)
- Examples:
  - `sirius_physical_filter` — evaluates expression filter, compacts rows
  - `sirius_physical_hash_join` — cuDF hash join with build/probe partitioning
  - `sirius_physical_parquet_scan` — direct Parquet file I/O with optional caching

**Pipeline:**
- Purpose: Ordered sequence of operators executing in a single batch of work
- Definition: `source` (first), `operators` list (all), `sink` (last)
- Blocking operators appear as both sink of one pipeline and source of another
- Execution: one task iterates all operators calling `execute()`, then calls sink's `sink()`
- Completion: source drained + ports empty + tasks done

**Data Repository:**
- Purpose: Thread-safe queue of data batches between pipelines
- Keyed by: (operator_id, port_id)
- Registered centrally in `shared_data_repository_manager`
- Supports partitioned storage and batch state machine
- Barrier types: FULL (synchronize), PARTIAL (incremental), PIPELINE (no sync)

**Memory Barrier:**
- Purpose: Control data flow across pipeline boundaries
- **FULL**: Downstream waits for upstream completion (hash join build)
- **PARTIAL**: Downstream can consume as data arrives (CONCAT after PARTITION in streaming joins)
- **PIPELINE**: No synchronization (within pipeline)

**Task:**
- Purpose: Schedulable unit of work (scan or GPU pipeline execution)
- Types: `scan_task` (DuckDB/Parquet I/O), `gpu_pipeline_task` (operator chain on CUDA stream)
- Contains: input data batch references, pipeline reference, sink operation
- Lifecycle: created by task creator, routed to executor, executed atomically

**Memory Reservation:**
- Purpose: Atomic GPU memory allocation before task execution
- Supports: Multiple memory spaces (GPU, Host, Disk via cuCascade)
- Interface: `reserve()` (acquire), `release()` (free) on SiriusContext memory manager
- OOM handling: Task retry with exponential backoff (up to 10 retries)

## Entry Points

**Extension Entry:**
- Location: `src/sirius_extension.cpp`, `LoadInternal()` function
- Triggers: `LOAD 'sirius.duckdb_extension'` from SQL
- Responsibilities: Register `gpu_execution` table function, set up extension callbacks

**Query Entry:**
- Location: `src/sirius_extension.cpp`, `GPUExecutionFunction()` (table function execute)
- Triggers: `CALL gpu_execution('SELECT ...')`
- Responsibilities: Parse, generate physical plan, invoke sirius_interface

**Interface Entry:**
- Location: `src/sirius_interface.cpp`, `sirius_execute_query()`
- Triggers: Called from extension table function
- Responsibilities: Lifecycle setup, delegate to sirius_engine, extract result

**Engine Entry:**
- Location: `src/sirius_engine.cpp`, `initialize()` and `execute()`
- Triggers: Called from sirius_interface
- Responsibilities: Build pipelines, start execution, collect result

**Executor Entry:**
- Location: `src/pipeline/pipeline_executor.cpp`, `start_query()`
- Triggers: Called from sirius_engine.execute()
- Responsibilities: Create completion handler, schedule initial tasks, route to sub-executors

## Error Handling

**Strategy:** Catch exceptions, report via `completion_handler::report_error()`, drain after error.

**Patterns:**

1. **Unsupported Operator** — throws `NotImplementedException`:
   - Caught in `GPUExecutionFunction()` or at executor level
   - Triggers graceful fallback to DuckDB CPU execution via `src/fallback.cpp`
   - Fallback invokes DuckDB's standard CPU executor on logical plan

2. **GPU OOM** — throws `oom_reschedule_exception`:
   - Caught in GPU executor's worker thread
   - Task retried up to 10 times with 5ms backoff
   - Downgrade executor concurrently spills data to host memory
   - If all retries exhausted: exception propagates to completion_handler

3. **General Exception**:
   - Caught anywhere in task execution
   - Calls `completion_handler->report_error(exception)`
   - Executor `drain_after_error()` shuts down task creator, drains queues, waits for workers
   - Error propagates through future to main thread
   - Main thread rethrows in `fetch_result_internal()`

4. **Data Corruption** — detected by type/validation checks:
   - Throws `sirius::invalid_input_exception`
   - Similar propagation path as general exception

## Cross-Cutting Concerns

**Logging:**
- Framework: spdlog (configured in `src/include/log/logging.hpp`)
- Levels: trace, debug, info, warn, error
- Environment: `SIRIUS_LOG_LEVEL=debug`, `SIRIUS_LOG_DIR=/path` (default: build/log)
- Macro: `SIRIUS_LOG_*()` throughout codebase
- Files log to `build/log/sirius_*.log`

**Validation:**
- Type checking: Operators validate input/output types match operator requirements
- Expression validation: Filter/projection expressions validated during planning
- Port validation: Ports verified connected during pipeline finalization
- Batch validation: Data batches validated on push/pop with size/count checks

**Authentication & Security:**
- Not applicable (in-process GPU execution, no network)
- Memory safety: Uses smart pointers (shared_ptr, unique_ptr) throughout
- Thread safety: Mutexes on shared state (repositories, memory manager), atomics for counters

**Monitoring & Observability:**
- NVTX ranges: nvtx3 markers for profiler integration (`src/sirius_engine.cpp`, operators)
- Task counting: `pipeline->mark_task_created()`, `mark_task_completed()` track per-pipeline stats
- Performance hooks: `completion_handler` tracks overall query timing
- Metrics: Row counts per operator can be parsed from logs via `tools/parse_pipeline_log.py`

---

*Architecture analysis: 2026-04-13*
