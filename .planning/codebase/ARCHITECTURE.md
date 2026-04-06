# Architecture

**Analysis Date:** 2026-04-06

## Pattern Overview

**Overall:** Sirius is a GPU-native SQL query execution engine (Super Sirius) that integrates with DuckDB as an extension. It uses a **pipeline-based execution model** with task scheduling and graceful CPU fallback.

**Key Characteristics:**
- **GPU-accelerated execution**: RAPIDS cuDF and CUDA kernels for data processing
- **Pipeline architecture**: Operators connected via pipelines; source-sink relationships
- **Task-based execution**: Tasks scheduled across CUDA streams and thread pools
- **Layered operators**: Physical operators with source/sink/regular node types
- **Fallback mechanism**: Transparent degradation to DuckDB CPU execution when needed
- **Memory-tiered**: cuCascade integration for GPU/host/disk memory management

## Layers

**Extension/Interface Layer:**
- Purpose: Bridge between DuckDB and Sirius GPU execution
- Location: `src/sirius_extension.cpp`, `src/sirius_interface.cpp`, `src/sirius_context.cpp`
- Contains: Table function registration, query routing, error handling
- Depends on: DuckDB public API, Configuration system
- Used by: DuckDB runtime calling `CALL gpu_execution(...)` or `CALL gpu_processing(...)`

**Planning Layer:**
- Purpose: Convert DuckDB logical plans to Sirius physical plans
- Location: `src/planner/sirius_physical_plan_generator.cpp`, `src/planner/sirius_plan_*.cpp`
- Contains: Operator-specific plan builders (filter, aggregate, join, projection, etc.)
- Depends on: DuckDB logical operator types, sirius physical operators
- Used by: `sirius_interface` during query preparation phase
- Key files: `src/planner/sirius_plan_filter.cpp`, `src/planner/sirius_plan_aggregate.cpp`, `src/planner/sirius_plan_comparison_join.cpp`, etc.

**Physical Operator Layer:**
- Purpose: Execute GPU operations on data batches
- Location: `src/op/sirius_physical_*.cpp`, `src/include/op/sirius_physical_*.hpp`
- Contains: Source operators (table scan, parquet scan, iceberg scan), sink operators (aggregate, join, partition), regular operators (filter, projection, limit, order, etc.)
- Depends on: Data batch API (cuCascade), CUDA kernels, Expression executor
- Used by: Pipeline executor to transform data
- **Base class**: `sirius_physical_operator` (`src/include/op/sirius_physical_operator.hpp`) with `execute()` and state management methods
- **Operator types**: Defined in `src/op/sirius_physical_operator_type.cpp` (FILTER, PROJECTION, HASH_JOIN, NESTED_LOOP_JOIN, GROUPED_AGGREGATE, etc.)

**Pipeline/Execution Layer:**
- Purpose: Organize operators into pipelines and schedule execution tasks
- Location: `src/pipeline/sirius_pipeline.cpp`, `src/pipeline/sirius_meta_pipeline.cpp`, `src/sirius_engine.cpp`
- Contains: Pipeline scheduling, task creation, execution state management
- Depends on: Physical operators, task scheduler, thread pool
- Used by: Query execution flow
- Key types: `sirius_pipeline` (single source-sink pathway), `sirius_meta_pipeline` (multiple pipelines with same sink), `sirius_engine` (query executor)

**Task Execution Layer:**
- Purpose: Create and execute tasks across GPU and CPU
- Location: `src/creator/task_creator.cpp`, `src/pipeline/gpu_pipeline_executor.cpp`, `src/pipeline/pipeline_executor.cpp`
- Contains: Task creation from operators, thread pool management, GPU stream scheduling
- Depends on: CUDA stream API, thread pool, bounded queue
- Used by: Pipeline executor for parallelism

**Expression Evaluation Layer:**
- Purpose: Evaluate SQL expressions on GPU
- Location: `src/expression_executor/gpu_expression_executor.cpp`, `src/expression_executor/gpu_expression_translator.cpp`
- Contains: Expression AST to cuDF kernel dispatch, binary/unary operators, type casting
- Depends on: DuckDB expression API, cuDF kernel selection
- Used by: Filter, projection, join condition evaluation

**Data Layer:**
- Purpose: Represent and transform data between DuckDB and GPU formats
- Location: `src/include/data/`, converter registry, batch utilities
- Contains: Data batch representation, parquet/iceberg converters, type mapping
- Depends on: Arrow/Parquet libraries, cuDF column format
- Used by: Scan operators, result collection

**Memory Management Layer:**
- Purpose: Coordinate GPU memory allocation and tiered caching
- Location: `src/gpu_buffer_manager.cpp`, `src/include/memory/`
- Contains: GPU memory reservation, OOM handling, tiered memory (GPU/host/disk via cuCascade)
- Depends on: RMM allocator, cuCascade data repository, DuckDB memory manager
- Used by: All GPU operators for allocation

**CUDA Kernel Layer:**
- Purpose: Low-level GPU computation
- Location: `src/cuda/operator/*.cu`, `src/cuda/*.cu`
- Contains: Hash join kernels, nested loop join, sort, aggregation, expression evaluation
- Depends on: CUDA runtime, cuDF libraries, RMM
- Used by: Physical operators via kernel dispatch

**Fallback Layer:**
- Purpose: Execute on CPU when GPU path is unavailable
- Location: `src/fallback.cpp`
- Contains: Fallback detection (unsupported operators/types), DuckDB CPU delegation
- Depends on: DuckDB physical operators, execution context
- Used by: Planning phase and operator-level fallback

**Configuration Layer:**
- Purpose: Runtime and compile-time configuration
- Location: `src/config.cpp`, `src/include/config.hpp`
- Contains: GPU memory policies, kernel selection flags, scan batch sizes, logging
- Depends on: None (low-level)
- Used by: All layers for behavior control

## Data Flow

**Query Execution Flow:**

1. **Query Entry** (`sirius_interface::sirius_execute_query`)
   - User calls `CALL gpu_execution('SELECT ...')`
   - Query string → `sirius_interface`

2. **Planning Phase** (`sirius_physical_plan_generator::create_plan`)
   - DuckDB logical plan → Sirius physical plan
   - Traversal of logical operators, creates matching sirius physical operators
   - Type resolution, cardinality estimation
   - Result: Tree of `sirius_physical_operator` nodes

3. **Pipeline Building** (`sirius_meta_pipeline::build`, `sirius_engine::initialize_internal`)
   - Physical plan → Pipeline graph
   - Identifies sources (scan operators) and sinks (aggregation, join build)
   - Creates `sirius_pipeline` for each source-sink path
   - Builds `sirius_meta_pipeline` hierarchy

4. **Task Scheduling** (`sirius_engine::execute`)
   - Pipeline graph → Task queue
   - `task_creator` converts operators to executable tasks
   - Tasks assigned to GPU streams and CPU thread pool
   - Dependencies resolved (join build must complete before probe)

5. **Execution** (`gpu_pipeline_executor`, `pipeline_executor`)
   - Tasks execute operator logic on data batches
   - Input batches → operator's `execute()` → output batches
   - Results flow through pipelines toward sinks

6. **Result Collection** (`sirius_physical_result_collector`)
   - Final sink operator materializes results
   - Data converted back to DuckDB format
   - QueryResult returned to user

**Fallback Path:**
- During planning: Unsupported operator types → throw `NotImplementedException` → catch in `sirius_extension` → delegate to DuckDB
- During execution: OOM or unsupported data type → fallback task queued, executes on CPU
- Transparent to user: CPU results merged with GPU results

**State Management:**
- `sirius_active_query_context`: Per-query state (prepared statement, engine, progress bar)
- `sirius_engine`: Per-query executor state (pipelines, scheduled tasks, results)
- Operator `sink_state`/`source_state`: Per-operator execution state (hash tables for joins, group partitions for aggregation)

## Key Abstractions

**sirius_physical_operator:**
- Purpose: Base class for GPU-executable operations
- Examples: `sirius_physical_filter.cpp`, `sirius_physical_hash_join.cpp`, `sirius_physical_grouped_aggregate.cpp`
- Pattern: Each operator implements `execute(input_data, stream) → output_data`
- Hierarchy: Operators can be sources (is_source()), sinks (is_sink()), or regular
- Metadata: Type, estimated cardinality, column types, child operators

**sirius_pipeline:**
- Purpose: Source + operators + sink as single logical unit
- Contains: Ordered list of operators, sink reference, batch index for ordering
- Scheduling: Single pipeline can have multiple parallel tasks from different batches
- Dependencies: May depend on other pipelines (e.g., probe after join build)

**sirius_meta_pipeline:**
- Purpose: Multiple pipelines sharing same sink (e.g., different join probe paths)
- Contains: Vector of `sirius_pipeline`, internal dependency graph
- Use case: Hash join (build pipeline, probe pipeline) or union branches
- Features: Batch index assignment, finish events for double-finalize operations

**operator_data & partitioned_operator_data:**
- Purpose: Container for data batches flowing between operators
- Holds: Vector of cuCascade data_batch pointers
- `partitioned_operator_data`: Adds partition index for partitioned operations

**task_creation_hint:**
- Purpose: Signal to pipeline executor whether task is ready or waiting
- Values: `WAITING_FOR_INPUT_DATA` (block until parent produces), `READY` (execute immediately)

## Entry Points

**`src/sirius_extension.cpp` - DuckDB Extension Registration:**
- Location: `ExtensionLoad()` function
- Triggers: DuckDB `LOAD 'sirius.duckdb_extension'`
- Responsibilities:
  - Register `gpu_execution` table function (main entry point)
  - Register `gpu_processing` table function (legacy)
  - Register configuration callbacks
  - Hook into DuckDB initialization

**`src/sirius_interface.cpp` - Query Execution Interface:**
- Location: `sirius_interface::sirius_execute_query()`
- Triggers: Table function calls with SQL string
- Responsibilities:
  - Prepare SQL statement via DuckDB planner
  - Generate physical plan via `sirius_physical_plan_generator`
  - Create sirius engine
  - Execute pipeline graph
  - Return results

**`src/sirius_engine.cpp` - Query Executor:**
- Location: `sirius_engine::execute()`
- Triggers: Execution of pipeline graph
- Responsibilities:
  - Schedule pipelines respecting dependencies
  - Create tasks for each pipeline
  - Drive execution loop
  - Collect results from sink operator

**`src/planner/sirius_physical_plan_generator.cpp` - Planning:**
- Location: `sirius_physical_plan_generator::create_plan()`
- Triggers: Query preparation phase
- Responsibilities:
  - Transform logical plan to physical plan
  - Select GPU-executable operators or raise `NotImplementedException` for fallback
  - Resolve column bindings and types

## Error Handling

**Strategy:** Layered fallback with explicit error propagation

**Patterns:**
- **Planning-time fallback**: Unsupported logical operator → `NotImplementedException` → caught in `sirius_extension` → re-execute on DuckDB CPU
- **Execution-time OOM**: Out of GPU memory → `oom_reschedule_exception` → task rescheduled with downgrade to CPU
- **Type unsupported**: NESTED_TYPES, some temporal types → fallback operator created during planning
- **Operator unsupported**: WINDOW, UNNEST, etc. → planning layer raises exception
- **Expression unsupported**: Complex expressions → expression executor falls back to cuDF or CPU

**Error propagation:**
- Exceptions bubble up through pipeline executor
- Caught at interface level in `sirius_interface::sirius_execute_query()`
- Errors formatted and attached to `QueryResult`
- User receives error message with query location info

## Cross-Cutting Concerns

**Logging:**
- Framework: spdlog
- Configuration: `SIRIUS_LOG_LEVEL` env var (trace, debug, info, warn, error)
- Macros: `SIRIUS_LOG_DEBUG()`, `SIRIUS_LOG_INFO()`, etc. in `src/include/log/logging.hpp`
- Sink: File-based (log directory at `SIRIUS_LOG_DIR`)

**Validation:**
- Operator verification: `sirius_physical_operator::verify()` checks tree invariants
- Cardinality estimation: Propagated through planning layer
- Type validation: Column types resolved at planning time via `ResolveOperatorTypes()`

**Authentication:**
- None (extension runs in same process as DuckDB)
- Inherits DuckDB catalog security

**GPU Stream Management:**
- CUDA streams allocated per task via `rmm::cuda_stream_view`
- Stream passed through execution pipeline
- Automatic synchronization at pipeline boundaries (meta-pipeline dependencies)

**NVTX Profiling:**
- Instrumentation markers via `nvtx3::scoped_range` in key functions
- Names match function/operator names for easy tracing
- Used with NVIDIA Nsys profiler for performance analysis

---

*Architecture analysis: 2026-04-06*
