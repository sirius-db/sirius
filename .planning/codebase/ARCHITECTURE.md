# Architecture

**Analysis Date:** 2026-04-21

## Pattern Overview

**Overall:** GPU-native SQL execution engine integrated as a DuckDB extension with transparent plan interception, task-based pipeline execution, and dynamic memory management.

**Key Characteristics:**
- DuckDB extension entry point that intercepts logical plans and converts them to GPU-executable physical plans
- Task-based pipeline execution model with materialized pipelines and stream-based data flow
- Central context ownership model (`SiriusContext`) managing all subsystems per connection
- Multi-threaded executor architecture with dedicated thread pools for GPU, scanning, task creation, and memory management
- Graceful CPU fallback mechanism when GPU constraints are reached or unsupported operations encountered

## Layers

**Extension Layer:**
- Purpose: Register Sirius as a DuckDB extension, expose table functions (`gpu_execution`, `gpu_buffer_init`), manage configuration
- Location: `src/sirius_extension.cpp`, `src/include/sirius_extension.hpp`
- Contains: Extension registration (`Load`, `LoadInternal`), table function bindings (`GPUExecutionBind`, `GPUExecutionFunction`), config registration (`InitialGPUConfigs`)
- Depends on: DuckDB extension API, `sirius_interface`, `sirius_prepared_statement_data`
- Used by: DuckDB runtime when queries call `CALL gpu_execution(...)` or `LOAD` the extension

**Interface Layer:**
- Purpose: Mediate between DuckDB's query execution context and Sirius's GPU engine; manage active query context lifecycle
- Location: `src/sirius_interface.cpp`, `src/include/sirius_interface.hpp`
- Contains: `sirius_interface` class managing query execution flow, `sirius_prepared_statement_data` wrapping logical plan + physical plan, `sirius_active_query_context` holding engine and prepared data
- Depends on: `sirius_engine`, DuckDB client context, `sirius_prepared_statement_data`
- Used by: `sirius_extension` for query invocation, handles query lifecycle (`begin_query_internal`, `fetch_result_internal`, `cleanup_internal`)

**Plan Generation Layer:**
- Purpose: Convert DuckDB logical plans to Sirius physical operator trees; dispatch to operator-specific builders
- Location: `src/planner/sirius_physical_plan_generator.cpp` (dispatcher), `src/planner/sirius_plan_*.cpp` (operator-specific builders)
- Contains: `sirius_physical_plan_generator::create_plan()` (entry point with type dispatcher), operator builders for each DuckDB logical operator (filter, aggregate, join, order, scan, etc.)
- Depends on: DuckDB logical operator types, `sirius_physical_operator` hierarchy
- Used by: `sirius_interface` during bind phase to generate physical plan from logical plan

**Physical Operator Layer:**
- Purpose: Define GPU-executable operator implementations; each operator knows how to execute on GPU via cuDF or custom CUDA kernels
- Location: `src/op/sirius_physical_*.cpp` implementations, `src/include/op/sirius_physical_operator.hpp` base class, `src/cuda/` for GPU kernels
- Contains: Base class `sirius_physical_operator` with virtual `execute()` method, derived operators (FILTER, PROJECTION, HASH_JOIN, UNGROUPED_AGGREGATE, TABLE_SCAN, etc.), each with CPU-side orchestration and GPU kernel dispatch
- Depends on: cuDF API, RMM, custom CUDA kernels, DuckDB expressions, data batch structures
- Used by: `sirius_engine` during execution; called via `execute()` in pipeline tasks

**Pipeline Execution Layer:**
- Purpose: Construct materialized pipelines from physical plan, manage pipeline build state, execute operators on CUDA streams
- Location: `src/include/pipeline/sirius_pipeline.hpp` (pipeline definition), `src/pipeline/sirius_pipeline_converter.cpp` (physical plan to pipeline conversion), `src/include/pipeline/pipeline_executor.hpp` (executor), `src/include/pipeline/gpu_pipeline_executor.hpp` (GPU-specific executor)
- Contains: `sirius_pipeline` (ordered operator list with source/sink/dependencies), `sirius_meta_pipeline` (groups related pipelines), `sirius_pipeline_build_state` (controlled access during construction), `pipeline_executor` (top-level orchestrator), `gpu_pipeline_executor` (per-GPU task execution)
- Depends on: `sirius_physical_operator`, CUDA streams, memory reservations
- Used by: `sirius_engine` during initialization and execution phases

**Task Execution Layer:**
- Purpose: Create and dispatch tasks (scan tasks, GPU pipeline tasks) based on data availability; schedule work across GPU and CPU scan executors
- Location: `src/include/creator/task_creator.hpp` (task creation), `src/op/scan/duckdb_scan_executor.hpp` (DuckDB table scans), `src/op/scan/parquet_scan_task.cpp` (Parquet file scans)
- Contains: `task_creator` polls for ready operators and creates tasks, scan executors pull data from storage/sources and publish to data repositories, GPU executors consume tasks
- Depends on: `sirius_physical_operator`, data repositories, scan operators
- Used by: `pipeline_executor` and `gpu_pipeline_executor` to schedule work

**Memory Management Layer:**
- Purpose: Manage GPU, host, and disk memory via cuCascade; track reservations; handle spilling during memory pressure
- Location: `src/include/memory/sirius_memory_reservation_manager.hpp`, `src/include/downgrade/downgrade_executor.hpp`
- Contains: `sirius_memory_reservation_manager` (wrapper around cuCascade's data repository manager and memory pools), `downgrade_executor` (spills GPU data to host when memory pressure detected)
- Depends on: cuCascade library, RMM
- Used by: GPU executor (reserves memory before task execution), downgrade executor (monitors pressure and spills data)

**Data Flow Layer:**
- Purpose: Manage batched data flowing between operators; convert between DuckDB and GPU formats; cache intermediate results
- Location: `src/data/` (converters), `src/include/data/` (headers), data repositories via cuCascade
- Contains: `convertible_data_batch` (data batch wrapper), converters for different data sources (DuckDB columns, Parquet, etc.), registry of converters
- Depends on: cuCascade data repositories, DuckDB column structures, Parquet reader
- Used by: Operators during execution to transform data between formats

**Context Layer:**
- Purpose: Central ownership of all subsystems within a DuckDB connection; lifecycle management
- Location: `src/include/sirius_context.hpp`, `src/sirius_context.cpp`
- Contains: `SiriusContext` owns `sirius_config`, `sirius_memory_reservation_manager`, `pipeline_executor`, `task_creator`, `downgrade_executor`, data repository manager
- Depends on: All subsystems above
- Used by: DuckDB connection lifecycle callbacks; all subsystems access context for shared resources

## Data Flow

**Plan Interception & Execution:**

1. User calls `CALL gpu_execution('SELECT ...')`
2. `sirius_extension::GPUExecutionBind()` → Parse SQL string, generate DuckDB logical plan
3. `sirius_physical_plan_generator::create_plan()` → Traverse logical plan tree, dispatch to operator-specific builders, construct physical operator tree
4. Store result in `sirius_prepared_statement_data` (holds both logical PreparedStatementData and physical operator tree)
5. `sirius_extension::GPUExecutionFunction()` → Call `sirius_interface::sirius_execute_query()`
6. `sirius_interface` → Create `sirius_engine`, call `initialize()` then `execute()`
7. `sirius_engine::initialize()` → Build pipeline graph via `sirius_meta_pipeline`, split operators into multiple pipelines, inject barriers (PARTITION, CONCAT, MERGE), wire data repositories
8. `sirius_engine::execute()` → Call `pipeline_executor::start_query()` which schedules initial scan tasks
9. Scan executors pull data from storage → publish to data repositories
10. Task creator polls for ready operators → creates GPU pipeline tasks
11. GPU executor threads dequeue tasks → call `execute()` on each operator → push results downstream via sink
12. Downgrade executor monitors GPU memory → spills to host on pressure
13. Final pipeline finishes → signals completion handler
14. Result extracted from result collector → returned to DuckDB

**Memory Management During Execution:**

- Task executor acquires memory reservation before dispatching operator tasks
- Each operator's `execute()` receives CUDA stream for any data movement
- Batches locked via `prepare_for_processing()` before operator execution
- After operator finishes, batches remain locked until next pipeline stage
- Downgrade executor monitors memory pressure periodically
- When threshold exceeded, moves GPU batches to host via data repository downgrades
- Task scheduler respects memory availability — reschedules tasks if locks fail

## Key Abstractions

**sirius_physical_operator:**
- Purpose: Represents an executable GPU operation; defines how a logical operation maps to GPU computation
- Examples: `sirius_physical_filter`, `sirius_physical_hash_join`, `sirius_physical_ungrouped_aggregate`, `sirius_physical_table_scan`
- Pattern: Hierarchy with virtual `execute(operator_data, stream)` method; some operators are blocking (need child side done first), others streaming; each has optional source and sink logic
- Entry point: Defined via `create_plan()` methods in `src/planner/sirius_plan_*.cpp` files

**sirius_pipeline:**
- Purpose: Ordered sequence of operators executed together on CUDA streams; represents a single logical execution unit
- Pattern: Operators added during construction, then finalized when pipeline becomes ready; source/sink tracked separately until finalization, then merged into single operators list
- Lifecycle: Created during `initialize()`, executed when dependencies met, finalized when all tasks complete
- Key fields: `operators` (all ops source to sink), `source`, `sink`, `dependencies` (blocking pipelines), `tasks_created/completed` (progress tracking)

**data_batch (via cuCascade):**
- Purpose: GPU-resident or host-resident data container; supports tiered memory model (GPU → host → disk)
- Pattern: Immutable during operator execution; locked via `prepare_for_processing()` to ensure memory space availability; moved between spaces by downgrade executor
- Used by: Operators consume/produce data_batch objects; repositories store and retrieve them

**memory_reservation:**
- Purpose: RAII guard ensuring GPU/host memory available for task execution
- Pattern: Acquired before task dispatch, released when task complete or batch spilled to disk
- Used by: GPU executor acquires before calling operator tasks; downgrade executor monitors available reservations

**data_repository:**
- Purpose: Named input/output port connecting pipelines; stores intermediate batches with flow control
- Pattern: Each inter-pipeline boundary has repository identified by port_id; task creator polls repositories for ready data, creates downstream tasks
- Used by: Scan operators publish results, pipeline operators consume from upstream repositories

## Entry Points

**Table Function Entry:**
- Location: `src/sirius_extension.cpp` (`SiriusExtension::GPUExecutionBind`, `SiriusExtension::GPUExecutionFunction`)
- Triggers: `CALL gpu_execution('SELECT ...')`
- Responsibilities: Parse query string, bind DuckDB logical plan, generate physical plan, execute via `sirius_interface`

**Extension Load Entry:**
- Location: `src/sirius_extension.cpp` (`SiriusExtension::Load`, `LoadInternal`)
- Triggers: `LOAD` command or automatic on connection creation (if pre-loaded)
- Responsibilities: Register table functions, initialize config options, register optimizer extension for transparent execution

**Transparent Execution Entry:**
- Location: `src/transparent/sirius_optimizer_extension.cpp` (`sirius_optimizer_hook`)
- Triggers: Automatically when `gpu_execution` config is true (after optimizer runs)
- Responsibilities: Check if logical plan is supported, if so convert to physical plan and execute, else fallback to DuckDB

**Query Execution Entry:**
- Location: `src/sirius_interface.cpp` (`sirius_execute_query`)
- Triggers: From `sirius_extension::GPUExecutionFunction` or transparent execution
- Responsibilities: Create engine, initialize with physical plan, execute, return results

## Error Handling

**Strategy:** Graceful fallback to DuckDB CPU execution when GPU constraints or unsupported operations encountered.

**Patterns:**
- **Plan Generation Failure:** If `create_plan()` throws `NotImplementedException` (unsupported operator), caught in `GPUExecutionBind` with `Config::ENABLE_DUCKDB_FALLBACK` check; query retried on CPU via `run_internal_cpu_fallback_query()`
- **Execution Failure:** If operator `execute()` throws or data batch fails to lock (out of GPU memory), task is rescheduled; if consistent failure, fallback mechanism triggered
- **Type Unsupported:** When DuckDB type not mappable to GPU type (e.g., nested types), filter operator created post-scan to filter in GPU (if table function doesn't support pushdown)
- **Memory Pressure:** When GPU memory exceeds threshold, downgrade executor moves batches to host; if host exhausted, spills to disk via cuCascade

## Cross-Cutting Concerns

**Logging:**
- Centralized via `src/include/log/logging.hpp` using spdlog
- Environment variables: `SIRIUS_LOG_DIR`, `SIRIUS_LOG_LEVEL` (trace/debug/info/warn/error/critical/off)
- Config options: `sirius_log_level`, `sirius_log_dir`, `sirius_log_flush_seconds`
- Used throughout codebase via `SIRIUS_LOG_DEBUG()`, `SIRIUS_LOG_INFO()`, etc.

**Validation:**
- Operator type checks via `D_ASSERT` macros in debug builds
- Expression validity checked during `create_plan()` with fallback if unsupported
- Data type conversion validation in converters with type mismatch detection
- Memory reservation checks before task execution ensure GPU memory available

**Profiling & Observability:**
- NVIDIA NVTX markers for profiler integration (`nvtx3::scoped_range`)
- Per-operator NVTX ranges during `execute()` calls
- Pipeline completion tracking via atomic counters (`tasks_created`, `tasks_completed`)
- Performance metrics logged at query completion (time, throughput)
- Profiler control functions (`profiler_start`, `profiler_stop`) for nsys capture ranges

**Configuration:**
- Global `Config` class in `src/config.cpp` holds flags and parameters
- Per-connection `sirius_config` in `SiriusContext` holds operator parameters
- `SET` commands can modify config at runtime (e.g., `SET use_pin_memory = true`)
- Config options registered in `InitialGPUConfigs()` with type and setter callbacks

**Thread Safety:**
- `sirius_engine` finalization of pipeline operators not thread-safe (runs on DuckDB query thread)
- Pipeline executor uses task queues for thread-safe work distribution
- Atomic counters for pipeline progress (`tasks_created`, `tasks_completed`)
- Mutex + condition variable for query completion signaling (`query_finish_mutex`, `query_finish_cv`)
- Data repositories use internal synchronization for concurrent reads

---

*Architecture analysis: 2026-04-21*
