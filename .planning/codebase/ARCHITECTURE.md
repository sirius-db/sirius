# Architecture

**Analysis Date:** 2026-04-06

## Pattern Overview

**Overall:** Pipelined GPU-accelerated SQL execution engine integrated as a DuckDB extension

**Key Characteristics:**
- Task-based pipeline parallelism with multiple dedicated thread pools (GPU execution, scan, task creation, downgrade)
- Lazy pipeline construction from DuckDB's logical plan via dynamic operator splitting
- Tiered memory management (GPU/pinned host/disk) with graceful spilling via cuCascade
- Graceful fallback to DuckDB CPU execution for unsupported operations
- Data flow through typed batches via shared repositories with barrier-based synchronization

## Layers

**Extension Layer:**
- Purpose: DuckDB integration surface, query lifecycle management
- Location: `src/sirius_extension.cpp`, `src/sirius_interface.cpp`
- Contains: Table function bindings, query preparation, result collection
- Depends on: DuckDB parsing/optimization, SiriusContext, sirius_engine
- Used by: DuckDB client via `CALL gpu_execution('SELECT ...')`

**Planning Layer:**
- Purpose: Translate DuckDB's logical plan to Sirius physical operators with GPU-aware splitting
- Location: `src/planner/`, `src/include/planner/`
- Contains: `sirius_physical_plan_generator`, specialized plan builders (filter, aggregate, join, order, etc.)
- Depends on: DuckDB logical operators, operator type definitions
- Used by: sirius_engine.initialize()

**Execution Engine:**
- Purpose: Orchestrate pipeline construction, execution lifecycle, memory management
- Location: `src/sirius_engine.cpp`, `src/include/sirius_engine.hpp`
- Contains: Pipeline graph building, initialization, execution coordination
- Depends on: Physical operators, pipeline builders, task creators, memory managers
- Used by: sirius_interface

**Operator Layer:**
- Purpose: GPU-accelerated (or fallback) implementations of SQL operations
- Location: `src/op/`, `src/include/op/`, `src/cuda/operator/`
- Contains: ~30 operator types (FILTER, PROJECTION, HASH_JOIN, AGGREGATE, ORDER, etc.)
- Depends on: cuDF, expression executor, data batches, memory reservations
- Used by: GPU pipeline executor during task execution

**Pipeline Execution Layer:**
- Purpose: Multi-threaded task scheduling and execution with resource management
- Location: `src/pipeline/`, `src/include/pipeline/`
- Contains: `pipeline_executor`, `gpu_pipeline_executor`, `sirius_pipeline`, pipeline metadata
- Depends on: Operators, task creator, scan executor, downgrade executor
- Used by: sirius_engine

**Task Creation Layer:**
- Purpose: Dynamic task scheduling based on data availability in operator ports
- Location: `src/creator/`, `src/include/creator/`
- Contains: `task_creator` with hint chain following
- Depends on: Operators, GPU/scan executors, data repositories
- Used by: GPU and scan executor callbacks

**Scan Layer:**
- Purpose: Async data ingestion from DuckDB tables or Parquet files to GPU
- Location: `src/op/scan/`, `src/include/op/scan/`
- Contains: `duckdb_scan_executor`, `parquet_scan_task`, caching logic, Iceberg metadata
- Depends on: DuckDB table functions, Parquet reader, caching infrastructure
- Used by: task creator, data repositories

**Memory Management Layer:**
- Purpose: Tiered GPU/host/disk memory allocation with reservation and spilling
- Location: `src/memory/`, `src/include/memory/`, cuCascade integration
- Contains: `sirius_memory_reservation_manager`, downgrade executor, defragmentation
- Depends on: RMM, cuCascade, GPU allocator
- Used by: GPU pipeline executor, downgrade executor

**Expression Executor Layer:**
- Purpose: Evaluate DuckDB bound expressions on GPU via cuDF
- Location: `src/expression_executor/`, `src/cuda/expression_executor/`
- Contains: `GpuExpressionExecutor`, expression translators, specializations for ops
- Depends on: cuDF, DuckDB expression AST
- Used by: Operators (FILTER, PROJECTION, HASH_JOIN predicates, aggregates)

**Data Management Layer:**
- Purpose: Typed data interchange between operators and external storage
- Location: `src/data/`, `src/include/data/`
- Contains: Parquet representation converters, cached data representation, converter registry
- Depends on: Parquet metadata, cuDF, host memory management
- Used by: Scan operators, data repositories

**Context Layer:**
- Purpose: Ownership and lifecycle management of all subsystems per DuckDB connection
- Location: `src/sirius_context.cpp`, `src/include/sirius_context.hpp`
- Contains: SiriusContext (config, memory manager, executor references, query state)
- Depends on: All subsystems below
- Used by: Extension, interface, engine

## Data Flow

**Query Execution Flow:**

1. **Parse & Optimize** → DuckDB generates optimized logical plan
2. **Physical Plan Generation** → `sirius_physical_plan_generator::create_plan()` converts to Sirius operators
3. **Pipeline Construction** → `sirius_engine::initialize()` builds pipeline graph:
   - `sirius_meta_pipeline::build()` recursively walks physical plan
   - Streaming operators (FILTER, PROJECTION) added to current pipeline
   - Blocking operators (JOIN, AGGREGATE, ORDER) become sinks, spawn child pipelines
   - Pipeline boundaries inject PARTITION/CONCAT/MERGE operators
   - Data repositories created with barrier types (FULL, PARTIAL, PIPELINE)
4. **Execution Start** → `sirius_engine::execute()`:
   - Creates query context with pipeline hashmap
   - Calls `pipeline_executor.start_query()`
   - Main thread blocks on completion future
5. **Scan Phase** → `duckdb_scan_executor` workers:
   - Pop scan tasks, acquire host memory
   - Execute DuckDB table function or Parquet reads
   - Convert to GPU-compatible data batches
   - Publish to shared data repositories
   - Schedule downstream consumers via `task_creator->schedule()`
6. **GPU Execution Phase** → `gpu_pipeline_executor` workers (per GPU):
   - Acquire kiosk ticket (rate limiting)
   - Pop GPU pipeline task
   - Acquire GPU memory reservation
   - Lock/prepare input batches (transfer to GPU if needed)
   - Iterate all pipeline operators: call `execute()` on each (source → sink)
   - Call sink's `sink()` method to push results downstream
   - Schedule downstream consumers or mark query complete
   - On OOM: reschedule with backoff
7. **Task Creation Cycle** → `task_creator` threads:
   - Receive `schedule(operator*)` calls
   - Call `operator->get_next_task_hint()` to check data availability
   - If ready: create GPU task or scan task
   - If waiting: recursively follow producer chain
   - Dispatch to GPU or scan executor queue
8. **Memory Pressure Management** → `downgrade_executor` monitor threads:
   - Poll GPU memory pressure every ~10ms
   - If threshold exceeded: move batches from GPU→host
   - Publish to repositories, update downstream operators
9. **Completion** → When `RESULT_COLLECTOR` pipeline finishes:
   - `completion_handler->mark_completed()` signals future
   - Main thread wakes, extracts materialized result
   - Returns `QueryResult` to DuckDB

**State Management:**
- Operator state: Global (`GlobalOperatorState`, `GlobalSinkState`) per operator, local per thread
- Pipeline state: `sirius_pipeline` tracks dependencies, batch indexes, parent relationships
- Data movement: `shared_data_repository` holds typed data batches with producer/consumer tracking
- Memory state: Tracked via `sirius_memory_reservation_manager` with per-space downgrade executors

## Key Abstractions

**sirius_physical_operator:**
- Purpose: Base class for all GPU-executable operations
- Examples: `sirius_physical_hash_join.hpp`, `sirius_physical_grouped_aggregate.hpp`, `sirius_physical_table_scan.hpp`
- Pattern: Virtual methods for operator/sink/source states, `execute()` for streaming, `sink()` for aggregation/grouping

**sirius_pipeline:**
- Purpose: Represents a sequence of operators from source to sink
- Examples: `src/include/pipeline/sirius_pipeline.hpp`
- Pattern: Tracks operators, source, sink, dependencies, batch indexes; knows parent pipelines and order requirements

**operator_data & partitioned_operator_data:**
- Purpose: Typed containers for data batches flowing between operators
- Examples: `src/include/op/sirius_physical_operator.hpp`
- Pattern: Wraps `std::vector<std::shared_ptr<cucascade::data_batch>>`; subclass tracks partition index

**shared_data_repository:**
- Purpose: Centralized buffer for inter-pipeline data transfer with synchronization
- Examples: Created in `sirius_engine::insert_repository()` with barrier types
- Pattern: Holds data batches, tracks producer/consumer counts, notifies task creator when data available

**GpuExpressionExecutor:**
- Purpose: Evaluates DuckDB bound expressions on GPU via cuDF
- Examples: `src/include/expression_executor/gpu_expression_executor.hpp`
- Pattern: Parses expression AST, dispatches to specialized cuDF operations, handles type conversions

**SiriusContext:**
- Purpose: Per-connection ownership hierarchy
- Examples: `src/include/sirius_context.hpp`
- Pattern: Registered as `ClientContextState`, owns config, memory manager, all executors, query state

## Entry Points

**CALL gpu_execution('SELECT ...'):**
- Location: `src/sirius_extension.cpp` → `GPUExecutionBind()`, `GPUExecutionFunction()`
- Triggers: Table function bind → parse/optimize → physical plan generation
- Responsibilities: Extract SQL, prepare statement, manage result collection

**sirius_interface::sirius_execute_query():**
- Location: `src/sirius_interface.cpp`
- Triggers: Pipeline construction, execution, result extraction
- Responsibilities: Query lifecycle (begin → execute → fetch → cleanup)

**sirius_engine::execute():**
- Location: `src/sirius_engine.cpp`
- Triggers: Starts pipeline executor, waits on completion future
- Responsibilities: Coordinate GPU and scan execution

**pipeline_executor::start_query():**
- Location: `src/include/pipeline/pipeline_executor.hpp` (forward decl), implementation in executor
- Triggers: Spawns sub-executor threads, queues initial scan tasks
- Responsibilities: Distribute completion handler, manage task scheduling

**task_creator manager loop:**
- Location: `src/include/creator/task_creator.hpp`
- Triggers: Receives schedule callbacks from GPU/scan executors
- Responsibilities: Determine task readiness, dispatch to executors

**gpu_pipeline_executor worker loop:**
- Location: `src/include/pipeline/gpu_pipeline_executor.hpp`
- Triggers: Pops tasks from queue, acquires reservations
- Responsibilities: Execute all operators in pipeline, call sink(), handle OOM

**duckdb_scan_executor worker loop:**
- Location: `src/include/op/scan/duckdb_scan_executor.hpp`
- Triggers: Pops scan tasks from queue
- Responsibilities: Execute DuckDB scan, convert data, publish to repositories

## Error Handling

**Strategy:** Exception propagation with graceful cleanup and optional CPU fallback

**Patterns:**
- **GPU OOM:** `oom_reschedule_exception` caught in GPU executor, retry up to 10 times with 5ms backoff (progressive reductions possible)
- **Unsupported operators:** Throw `NotImplementedException` during planning, caught by fallback layer
- **Query errors:** Exception caught in GPU/scan executor, routed to `completion_handler->report_error()` which drains queues and propagates to main thread
- **Task execution failures:** `drain_after_error()` stops task creation, drains queues, signals completion with error
- **CPU fallback:** If enabled in config, `sirius_extension` catches plan errors and re-executes via DuckDB CPU path

## Cross-Cutting Concerns

**Logging:** spdlog-based structured logging controlled by `SIRIUS_LOG_LEVEL` (trace, debug, info, warn, error), output to `SIRIUS_LOG_DIR` or CMAKE_BINARY_DIR/log

**Validation:** Operator `verify()` called after plan generation; runtime assertions via `D_ASSERT()` on batch counts, types, operator IDs

**Authentication:** None (DuckDB handles client auth)

**Rate Limiting:** Kiosk tickets used in GPU and scan executors to bound concurrent worker threads

**Profiling:** NVTX3 annotations for kernel occupancy, pipeline range tracking; namespace `duckdb::sirius` for GPU expression executor

---

*Architecture analysis: 2026-04-06*
