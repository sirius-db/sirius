# Architecture

**Analysis Date:** 2026-04-03

## Pattern Overview

**Overall:** Layered GPU-CPU execution pipeline with graceful CPU fallback

**Key Characteristics:**
- DuckDB extension that intercepts physical plan execution and routes to GPU when possible
- Task-based pipeline execution with separate GPU and CPU streams
- Hierarchical operator-based design mirroring DuckDB's physical operator patterns
- Tiered memory management (GPU/HOST/DISK) via cuCascade library
- Modular expression evaluation dispatched to GPU via CUDA kernels

## Layers

**Extension Layer:**
- Purpose: DuckDB integration point, query registration, and buffer management
- Location: `src/sirius_extension.cpp`, `src/include/sirius_extension.hpp`
- Contains: DuckDB extension loading, function registration (gpu_execution, gpu_processing), buffer initialization
- Depends on: DuckDB extension APIs, GPU buffer manager
- Used by: DuckDB's extension loading mechanism

**Interface Layer:**
- Purpose: Query lifecycle management and execution orchestration
- Location: `src/sirius_interface.cpp`, `src/include/sirius_interface.hpp`
- Contains: Query preparation, execution state tracking, result fetching
- Key class: `sirius_interface` manages active query context and routes to GPU engine
- Depends on: DuckDB client context, sirius_engine
- Used by: Extension layer to execute queries

**Planning Layer:**
- Purpose: Convert DuckDB logical plans to Sirius physical plans
- Location: `src/planner/sirius_physical_plan_generator.cpp`, `src/planner/sirius_plan_*.cpp`
- Contains: Physical plan generation, operator selection, optimization decision logic
- Key class: `sirius_physical_plan_generator` traverses DuckDB operators and builds GPU-capable equivalents
- Plan builders: `sirius_plan_aggregate.cpp`, `sirius_plan_filter.cpp`, `sirius_plan_join.cpp`, etc. (one per operator type)
- Depends on: DuckDB logical operators, sirius physical operators
- Used by: Interface layer after logical planning

**Physical Operator Layer:**
- Purpose: Executable unit representation mirroring DuckDB's physical operators
- Location: `src/op/sirius_physical_*.cpp`, `src/include/op/sirius_physical_*.hpp`
- Contains: Base operator class and 40+ operator implementations (TABLE_SCAN, HASH_JOIN, GROUPED_AGGREGATE, FILTER, PROJECTION, ORDER_BY, etc.)
- Key class: `sirius_physical_operator` (abstract base) with type field and children tree
- Operator families: Scans (TABLE_SCAN, PARQUET_SCAN, ICEBERG_SCAN, DUCKDB_SCAN), Joins (HASH_JOIN, NESTED_LOOP_JOIN, DELIM_JOIN), Aggregates (UNGROUPED_AGGREGATE, GROUPED_AGGREGATE, MERGE), Sorts (MERGE_SORT, TOP_N), Result (RESULT_COLLECTOR, LIMIT)
- Depends on: cuDF/CUDA kernels, expression executor
- Used by: Pipeline execution layer

**Pipeline Layer:**
- Purpose: Break operator tree into executable tasks and manage parallelism
- Location: `src/pipeline/sirius_pipeline.cpp`, `src/include/pipeline/sirius_pipeline.hpp`
- Contains: Pipeline graph construction, dependency management, execution scheduling
- Key classes: `sirius_pipeline` (represents one parallelizable segment), `sirius_meta_pipeline` (whole query plan), `sirius_pipeline_build_state` (construction state machine)
- Builds: Breaks operator tree at blocking points (joins, aggregates) into pipelines
- Depends on: Physical operators, task creation
- Used by: Task creator and engine

**Execution Layer:**
- Purpose: Create and schedule parallel tasks from pipelines
- Location: `src/creator/task_creator.cpp`, `src/pipeline/gpu_pipeline_task.cpp`, `src/pipeline/gpu_pipeline_executor.cpp`
- Contains: Task instantiation per pipeline, scheduling logic, CPU thread pool management
- Key classes: `task_creator` (creates tasks from pipelines), `gpu_pipeline_task` (single executable task), `gpu_pipeline_executor` (thread pool executor)
- Execution flow: Pipelines → Tasks → Thread pool workers
- Depends on: Pipelines, physical operators, memory manager
- Used by: Engine to execute query

**Memory Layer:**
- Purpose: GPU memory lifecycle and reservation management
- Location: `src/memory/sirius_memory_reservation_manager.cpp`, `src/include/memory/sirius_memory_reservation_manager.hpp`
- Contains: Memory pool management, reservation tracking, OOM policy
- Integrates with: cuCascade (tiered memory), RMM (GPU memory)
- Depends on: RAPIDS RMM, cuCascade memory spaces
- Used by: Task creator, downgrade executor, operators

**Downgrade Layer:**
- Purpose: Automatic memory pressure response and GPU→HOST data movement
- Location: `src/downgrade/downgrade_executor.cpp`, `src/include/downgrade/downgrade_executor.hpp`
- Contains: Memory space monitoring, downgrade task scheduling, data repository management
- Key class: `downgrade_executor` runs monitor thread polling GPU memory, dispatches downgrade tasks when threshold exceeded
- Depends on: cuCascade data repositories, memory spaces
- Used by: Context to manage tiered memory automatically

**Expression Evaluation Layer:**
- Purpose: Dispatch SQL expressions to GPU or CPU evaluation
- Location: `src/expression_executor/gpu_expression_executor.cpp`, `src/include/expression_executor/gpu_expression_executor.hpp`, `src/cuda/expression_executor/`
- Contains: Expression AST traversal, CUDA kernel dispatch, CPU fallback
- Key classes: `gpu_expression_executor` (orchestrates), dispatch kernels in `src/cuda/expression_executor/`
- Supports: Arithmetic, comparison, string operations, casts, regex matching, aggregation functions
- Depends on: DuckDB expression AST, CUDA kernels
- Used by: Physical operators (FILTER, PROJECTION, AGGREGATE)

**Data Conversion Layer:**
- Purpose: DuckDB↔GPU data format transformation
- Location: `src/data/`, `src/include/data/`
- Contains: cuDF table builders, Parquet representation converters, data batch utilities
- Key class: `sirius_converter_registry` maps DuckDB types to cuDF representations
- Handles: Arrow format, Parquet metadata, columnar GPU data
- Depends on: cuDF, DuckDB data types, Parquet library
- Used by: Scan operators, expression executor

**CUDA/GPU Layer:**
- Purpose: GPU computation kernels
- Location: `src/cuda/` and subdirectories
- Contains: ~50 CUDA kernels for joins, aggregates, sorts, expressions, string operations, Iceberg delete masks
- Kernel families: `cudf/` (cuDF wrappers), `operator/` (custom kernels), `expression_executor/` (expression dispatch), `iceberg/` (delete masking)
- Uses: cuDF, RMM, NVIDIA libraries (libcudf, RMM, cuCascade)
- Depends on: CUDA 13+, cuDF headers
- Used by: Physical operators via expression executor

**Context & Config Layer:**
- Purpose: Query-wide state and configuration
- Location: `src/sirius_context.cpp`, `src/include/sirius_context.hpp`, `src/sirius_config.cpp`, `src/include/sirius_config.hpp`
- Contains: DuckDB ClientContextState subclass holding task creator, downgrade executor, memory manager, config options
- Key class: `SiriusContext` (lifecycle management), `sirius_config` (hardware topology, GPU selection)
- Manages: QueryBegin/QueryEnd lifecycle, internal query guards for Iceberg metadata lookups
- Depends on: DuckDB client context, cuCascade
- Used by: Extension to initialize and track per-connection state

## Data Flow

**Query Execution:**

1. User calls `CALL gpu_execution('SELECT ...')`
2. Extension entry point (`src/sirius_extension.cpp`) → `sirius_interface::execute_query()`
3. `sirius_interface::begin_query_internal()` creates `sirius_active_query_context`
4. Physical plan generation: `sirius_physical_plan_generator::plan()` converts DuckDB logical operators to Sirius physical operators using plan builders
5. `sirius_engine::initialize()` receives physical plan, calls `initialize_internal()` which:
   - Builds pipeline graph (breaking at blocking operators)
   - Creates initial pipelines in `sirius_root_pipelines`
   - Assigns unique operator IDs for state tracking
6. `sirius_engine::execute()` starts execution:
   - `task_creator::start()` begins with table scan pipelines
   - Task creation loop: operators produce task hints, task_creator instantiates `gpu_pipeline_task`
   - Tasks enqueued to thread pool
   - GPU pipeline executor runs tasks: pulls data batches from upstream, executes operator logic on GPU via cuDF/CUDA kernels
7. Data materialization and result collection:
   - Final operator is `sirius_physical_result_collector` which gathers results
   - Results fetched via `sirius_interface::fetch_result_internal()` → `sirius_engine::get_result()`
8. Query cleanup: `QueryEnd()` on `SiriusContext` drains downgrade tasks, clears repositories

**Fallback Flow:**

- If operator is unsupported or data exceeds GPU memory → `fallback.cpp` routes to DuckDB CPU execution
- Downgrade executor monitors GPU memory pressure → migrates data to HOST tier automatically

**State Management:**

- Per-query state: `sirius_active_query_context` (prepared statement, engine, progress bar)
- Per-connection state: `SiriusContext` (task_creator, downgrade_executor, memory manager)
- Per-operator state: Global (sink_state, source_state) and local (per-thread) operator states
- Per-pipeline state: Source, sink, operators, dependencies
- Data state: Batches flow through `operator_data` and `partitioned_operator_data` containers via repositories

## Key Abstractions

**sirius_physical_operator:**
- Purpose: Base class for all executable operators
- Examples: `sirius_physical_table_scan`, `sirius_physical_hash_join`, `sirius_physical_grouped_aggregate`, `sirius_physical_filter`
- Pattern: Subclass per operator type, each implements GPU and fallback paths
- Methods: `get_global_sink_state()`, `get_local_sink_state()`, `build_pipelines()`, `execute()`, `finalize()`

**sirius_engine:**
- Purpose: Query executor managing operator tree, pipelines, and task scheduling
- Key state: `sirius_owned_plan` (root operator), `sirius_pipelines` (all), `sirius_root_pipelines` (entry points), `sirius_scheduled` (queued)
- Methods: `initialize()`, `initialize_internal()`, `execute()`, `prefetch_iceberg_metadata()`, `insert_repository()`

**sirius_pipeline:**
- Purpose: Single parallelizable segment of execution
- Contains: Source operator, sink operator, middle operators
- Pattern: Pipelines split at synchronization points (PARTITION, CONCAT, MERGE_SORT, etc.)
- Methods: `get_source()`, `get_sink()`, `get_operators()`, `schedule()`, `reset()`

**task_creator:**
- Purpose: Converts pipelines to executable tasks and manages task dispatch
- Thread pool: Configurable worker count (default = CPU cores)
- Pattern: Task creation queue, priority for table scan pipelines, operator hints for scheduling
- Methods: `start()`, `start_thread_pool()`, `stop_thread_pool()`, `schedule_task_creation()`

**gpu_pipeline_task:**
- Purpose: Single executable unit for one pipeline iteration
- Contains: Operator references, source/sink state, data batch input
- Methods: `execute()` (runs all operators in pipeline), `has_output()`, `get_output()`

**downgrade_executor:**
- Purpose: Monitors memory pressure and triggers data migrations
- Pattern: Monitor thread checks memory_space pressure, manager thread dispatches tasks to pool
- Methods: `should_downgrade_memory()`, `drain()`, `stop()`

**gpu_expression_executor:**
- Purpose: Evaluate expressions on GPU
- Pattern: Traverses expression AST, dispatches to CUDA kernels via type-specific dispatch functions
- Methods: `execute()`, `execute_expression()`, returns cuDF table

## Entry Points

**gpu_execution:**
- Location: `src/sirius_extension.cpp` (registered as table function)
- Triggers: `CALL gpu_execution('SELECT ...')` via DuckDB function call
- Responsibilities: Parse query, prepare statement, route to `sirius_interface`

**gpu_processing (Legacy):**
- Location: `src/sirius_extension.cpp`
- Triggers: `CALL gpu_processing('SELECT ...')` (requires prior `CALL gpu_buffer_init()`)
- Responsibilities: Legacy execution path using GPU buffer context (namespace duckdb, not sirius)

**DuckDB Extension Loading:**
- Location: `src/sirius_extension.cpp` LoadInternal()
- Triggers: `LOAD 'build/release/extension/sirius/sirius.duckdb_extension'`
- Responsibilities: Register functions, initialize extension-wide state

## Error Handling

**Strategy:** Try GPU first, fallback to CPU on unsupported operations

**Patterns:**
- Operator::supports_gpu() checks if operator can execute on GPU
- Catch and wrap exceptions in `ErrorData` objects
- `sirius_process_error()` formats error with query context
- Memory errors → trigger defragmentation or eviction before retry
- Unsupported types/operations → degrade to DuckDB CPU operator seamlessly

## Cross-Cutting Concerns

**Logging:** 
- Framework: spdlog
- Levels: TRACE, DEBUG, INFO, WARN, ERROR configured via `SIRIUS_LOG_LEVEL` env var
- Output: `SIRIUS_LOG_DIR` (default: `${CMAKE_BINARY_DIR}/log`)
- Usage: `SIRIUS_LOG_DEBUG("message")` throughout codebase

**Validation:**
- Expression types validated against supported set (INTEGER, BIGINT, FLOAT, DOUBLE, VARCHAR, DATE, TIMESTAMP, DECIMAL)
- Operator cardinality bounds checked (libcudf int32_t row limit ~2B rows)
- Join keys validated for GPU execution
- Aggregate functions validated against cuDF support

**Authentication:**
- Inherits from DuckDB connection context (read_only flag, catalog access)
- No additional auth layer; operates within DuckDB's connection security model

---

*Architecture analysis: 2026-04-03*
