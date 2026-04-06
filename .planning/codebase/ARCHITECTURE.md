# Architecture

**Analysis Date:** 2026-04-06

## Pattern Overview

**Overall:** GPU-native SQL acceleration engine using task-based pipeline execution

**Key Characteristics:**
- DuckDB extension architecture with pluggable GPU execution
- Logical plan → physical operator tree → pipeline graph transformation
- Distributed task execution across dedicated thread pools (GPU, scan, task creation, downgrade)
- Data-flow driven scheduling with per-operator memory barriers
- Graceful fallback to DuckDB CPU execution for unsupported operations

## Layers

**Extension Layer:**
- Purpose: DuckDB integration and API exposure
- Location: `src/sirius_extension.cpp`
- Contains: Table function registration (`gpu_execution`), extension initialization, buffer management
- Depends on: DuckDB C++ API, sirius_interface, gpu_buffer_manager
- Used by: DuckDB query execution engine

**Interface Layer:**
- Purpose: Query lifecycle management and result handling
- Location: `src/sirius_interface.cpp` and `src/include/sirius_interface.hpp`
- Contains: Active query context, prepared statements, pending query results, error handling
- Depends on: sirius_engine, sirius_context
- Used by: Extension layer to coordinate query execution

**Engine/Orchestration Layer:**
- Purpose: Physical plan construction and pipeline orchestration
- Location: `src/sirius_engine.cpp` and `src/include/sirius_engine.hpp`
- Contains: Physical plan ownership, pipeline graph construction, repository insertion, operator initialization
- Depends on: Physical plan generator, pipeline builders, data repository manager
- Used by: sirius_interface to execute queries

**Planning Layer:**
- Purpose: Logical-to-physical operator translation
- Location: `src/planner/sirius_physical_plan_generator.cpp` and `src/planner/sirius_plan_*.cpp`
- Contains: Operator mapping (TABLE_SCAN, JOIN, AGGREGATE, ORDER, etc.), plan builders for each operator type
- Depends on: DuckDB logical operators
- Used by: sirius_engine during initialization

**Execution Layer:**
- Purpose: Task scheduling and execution on GPU and CPU
- Location: `src/pipeline/`, `src/creator/`, `src/downgrade/`
- Contains: Pipeline executor, GPU executors, task creator, scan executor, downgrade executor
- Depends on: Sirius operators, data repositories, memory managers, thread pools
- Used by: Engine to run queries

**Operator Layer:**
- Purpose: Physical operator implementations (streaming and blocking)
- Location: `src/op/` (new Sirius), `src/legacy/operator/` (legacy), `src/include/op/` (headers)
- Contains: Individual operators (filter, projection, join, aggregate, scan, merge, partition, order, etc.)
- Depends on: Operator base class, expression executor, data batches
- Used by: Execution layer to transform data

**Expression Executor Layer:**
- Purpose: GPU-accelerated scalar expression evaluation
- Location: `src/expression_executor/`, `src/cuda/expression_executor/`
- Contains: Expression translation to cuDF operations, specializations for each expression type (cast, comparison, conjunction, function, case, between, etc.)
- Depends on: cuDF API
- Used by: Filter and projection operators

**Context & Configuration Layer:**
- Purpose: Per-connection Sirius state and configuration
- Location: `src/sirius_context.cpp`, `src/sirius_config.cpp`, `src/include/sirius_context.hpp`
- Contains: Subsystem ownership (memory manager, executor pool, repositories), config file parsing, lifecycle hooks
- Depends on: cuCascade, spdlog, DuckDB ClientContext
- Used by: All layers (registered in ClientContextState)

**Memory Management Layer:**
- Purpose: GPU/host/disk memory tier management and pressure relief
- Location: `src/memory/`, `src/include/memory/`
- Contains: Memory reservation manager, allocation accessors
- Depends on: cuCascade shared_data_repository_manager
- Used by: GPU executor (reserves), downgrade executor (monitors)

**Data Management Layer:**
- Purpose: Data batch representation and conversion between formats
- Location: `src/data/`, `src/include/data/`
- Contains: Host parquet representation, cached data representation, converter registry
- Depends on: cuDF, Arrow
- Used by: Scan operators and result collection

## Data Flow

**Request → Physical Plan:**

1. DuckDB calls `CALL gpu_execution('SELECT ...')`
2. `sirius_extension` registers table function, creates `SiriusTableFunctionData`
3. `sirius_interface` receives query, begins lifecycle
4. DuckDB generates logical plan
5. `sirius_physical_plan_generator::create_plan()` converts logical → physical operators
6. Physical plan returned as `sirius_physical_operator` tree

**Physical Plan → Pipelines:**

1. `sirius_engine::initialize()` receives physical plan
2. `sirius_meta_pipeline::build()` walks operator tree, creates initial `sirius_pipeline`
3. `initialize_internal()` finalizes pipelines:
   - Assigns operator IDs
   - Detects pipeline barriers (PARTITION, AGGREGATE, ORDER, JOIN) as splits
   - Injects PARTITION/CONCAT/MERGE operators between pipelines
   - Creates data repositories at each split
   - Sets barrier types (FULL, PARTIAL, PIPELINE)
4. Pipeline graph stored in `sirius_engine.sirius_pipelines` and `sirius_root_pipelines`

**Pipelines → Tasks:**

1. `pipeline_executor::start_query()` kicks off execution
2. `task_creator::prepare_for_query()` initializes scan global state for each pipeline source
3. Initial scan operators queued to scan executor
4. Scan executor creates `duckdb_scan_task` or `parquet_scan_task`
5. Scans pull data from DuckDB tables/Parquet files, convert to GPU format, publish to `shared_data_repository`
6. `task_creator` polls repositories, detects ready operators
7. GPU pipeline tasks created with input data from repositories
8. GPU executor threads acquire memory reservations, execute pipeline
9. Results published to downstream repositories
10. Cycle repeats: downstream operators become ready → new tasks created

**Execution:**

1. `gpu_pipeline_executor` pulls task from queue
2. Acquires memory reservation from manager
3. Iterates over `pipeline->get_operators()` (source through sink)
4. Calls each operator's `execute(input_data)` → `operator_data`
5. Sink calls `sink(output_data)` to push results to downstream repository
6. Task completion triggers `task_creator` to schedule downstream

**Memory Pressure Relief:**

1. `downgrade_executor` monitors `sirius_memory_reservation_manager` every ~10ms
2. When GPU memory threshold exceeded, spill tasks created
3. Data moved GPU → host via cuCascade
4. Downstream operators get host-based input, produce host output
5. Pipeline executor adaptively routes data

**Result Extraction:**

1. Final operator is `RESULT_COLLECTOR` (sink pipeline)
2. Collects all output data batches into `MaterializedQueryResult`
3. `completion_handler::mark_completed()` signals future
4. Main thread wakes, calls `sirius_engine::get_result()`
5. Result returned to DuckDB

**State Management:**

- **Query-level state:** Per-query pipelines, operator initialization, task counters
- **Execution-level state:** Task local state (input data, operator index for resumption), memory reservations, CUDA streams
- **Global state:** Operator global state (scan cursors, hash table state for stateful operators)

## Key Abstractions

**sirius_physical_operator:**
- Purpose: Base class for all GPU-executable operators
- Examples: `sirius_physical_filter.hpp`, `sirius_physical_hash_join.hpp`, `sirius_physical_grouped_aggregate.hpp`
- Pattern: Virtual `execute()` → `operator_data`, optional `sink()` for pipeline breakers
- Key fields: `operator_id`, `type`, `children`, `source_order`, ports for inter-pipeline data

**sirius_pipeline:**
- Purpose: Ordered chain of operators executing as an atomic unit
- Pattern: Container with `source`, `operators`, `sink`; tracks task count via atomic counters
- Lifecycle: Created during finalization, marked ready when dependencies complete, tasks scheduled
- Key: After finalization, `operators` includes source and sink; `source` and `sink` are aliases to first/last

**Data Batch & Repository:**
- Purpose: Wrapper for column data flowing between operators
- Pattern: `cucascade::data_batch` from cuDF table, moved through `shared_data_repository` (cuCascade-managed)
- Lifecycle: Created by operator `execute()`, published to repository by `sink()`, consumed by downstream `execute()`
- Tier management: GPU → Host → Disk via cuCascade (memory pressure triggers downgrade)

**Task:**
- Purpose: Schedulable unit of work
- Examples: `duckdb_scan_task`, `parquet_scan_task`, `gpu_pipeline_task`
- Pattern: Carries input data, operator references, memory reservation, execution state
- Lifecycle: Created by task_creator, executed by appropriate executor, completion triggers downstream tasks

**Pipeline Breaker (Barrier):**
- Purpose: Forces pipeline split due to data dependencies or memory constraints
- Examples: PARTITION (distributes to multiple branches), CONCAT (merges multiple inputs), SORT_SAMPLE (samples for merge sort)
- Pattern: Created with downstream pipeline, connected via data repository with barrier type
- Barrier types: FULL (wait all upstream), PARTIAL (PARTITION→CONCAT only), PIPELINE (streaming scans)

**Expression Executor:**
- Purpose: GPU evaluation of scalar expressions
- Pattern: `gpu_expression_executor` translates DuckDB expressions to cuDF operations
- Examples: Specializations for CAST, COMPARISON, CONJUNCTION, FUNCTION, CASE, BETWEEN
- Lifecycle: Instantiated per operator, called during `execute()` for filtering/projection

## Entry Points

**gpu_execution Table Function:**
- Location: `src/sirius_extension.cpp` (registered in LoadInternal)
- Triggers: `CALL gpu_execution('SELECT ...')`
- Responsibilities: Parse SQL, prepare statement data, bind parameters, invoke sirius_interface

**sirius_interface Constructor:**
- Location: `src/sirius_interface.cpp`
- Triggered by: Extension table function
- Responsibilities: Receive query string, initialize sirius_engine, begin query lifecycle

**sirius_engine::initialize():**
- Location: `src/sirius_engine.cpp`
- Triggered by: sirius_interface after DuckDB plan generation
- Responsibilities: Take physical plan, build pipeline graph, finalize pipelines, initialize operators

**pipeline_executor::start_query():**
- Location: `src/pipeline/pipeline_executor.cpp`
- Triggered by: sirius_interface after engine initialization
- Responsibilities: Create task_creator thread, start GPU/scan executors, queue initial tasks

**task_creator::create_tasks():**
- Location: `src/creator/task_creator.cpp`
- Triggered by: Pipeline executor event loop
- Responsibilities: Poll repositories for ready operators, create scan or GPU pipeline tasks

## Error Handling

**Strategy:** Graceful degradation with CPU fallback or exception propagation

**Patterns:**

1. **Unsupported Operator Fallback** — `NotImplementedException` thrown during plan generation → caught in sirius_extension → delegates to DuckDB CPU execution
   - File: `src/planner/sirius_physical_plan_generator.cpp` (switches on operator type)
   - Example: LOGICAL_WINDOW, LOGICAL_ASOF_JOIN, LOGICAL_RECURSIVE_CTE

2. **Type Fallback** — Unsupported data types (HUGEINT, nested types) → operator throws during `execute()` → caught, data downgraded to CPU
   - File: `src/planner/sirius_plan_aggregate.cpp` (HUGEINT downcast), `src/fallback.cpp` (type checking)
   - Example: HUGEINT downcast to BIGINT for cuDF

3. **Memory Fallback** — GPU memory exhausted → OOM exception → caught by pipeline_executor → task rescheduled on CPU
   - File: `src/pipeline/oom_reschedule_exception.hpp`, `src/downgrade/downgrade_executor.cpp`
   - Pattern: Downgrade executor monitors pressure, moves data to host before OOM occurs

4. **Expression Evaluation** — Unsupported expression (regex) → fallback to DuckDB → results cached for subsequent rows
   - File: `src/expression_executor/gpu_expression_executor.cpp`, `src/cuda/expression_executor/`
   - Example: REGEXP_REPLACE unsupported → delegates to DuckDB string_agg fallback

5. **Query Error Propagation** — Errors in scan/execution → caught, error stored in task state → main thread awakened with ErrorData
   - File: `src/sirius_interface.cpp` (sirius_process_error), `src/pipeline/pipeline_executor.cpp` (error handling)
   - Pattern: Try-catch wraps each executor's event loop

## Cross-Cutting Concerns

**Logging:**
- Framework: spdlog
- Location: `src/include/log/logging.hpp`
- Usage: Key decision points (fallback, barrier creation), task lifecycle, memory pressure
- Control: Environment variable `SIRIUS_LOG_LEVEL` (default: info), file in `$SIRIUS_LOG_DIR/log`

**Validation:**
- Input validation: DuckDB handles SQL parsing, Sirius validates operator support during planning
- Data validation: Each operator validates column count/type on `execute()` via DuckDB data_chunk assertions
- Barrier validation: `initialize_internal()` verifies barrier types match pipeline dependencies

**Authentication:**
- Location: Inherited from DuckDB connection (no GPU-specific auth)
- Config: GPU buffer sizes via `gpu_buffer_init()` parameters, per-extension security model

**Thread Safety:**
- Data repositories: Protected by cuCascade's atomic operations
- Pipeline state: Atomic counters (`tasks_created`, `tasks_completed`)
- Global operator state: Operator-specific locks (e.g., hash join build table lock)
- Scan global state: Per-operator global state with mutex in `scan_task_global_state`

**Observability:**
- NVTX regions: Pipeline task creation/completion, operator execution
- Metrics: Task count, memory reservation size, operator wall clock time
- Profiler integration: DuckDB QueryProfiler called for metrics collection

---

*Architecture analysis: 2026-04-06*
