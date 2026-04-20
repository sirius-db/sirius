# Architecture

**Analysis Date:** 2026-04-02

## Pattern Overview

**Overall:** Sirius implements a **custom task-based GPU execution engine** that intercepts DuckDB's physical query plans and routes supported operations to GPU execution via RAPIDS cuDF, with graceful fallback to CPU execution for unsupported cases.

**Key Characteristics:**
- Dual-mode execution: Legacy Sirius (`gpu_processing`, namespace `duckdb`) and New Sirius (`gpu_execution`, namespace `sirius`)
- Pipeline-based parallel execution with stream-per-thread GPU scheduling
- Three-tier memory management (GPU/host/disk) via cuCascade integration
- Task-driven architecture with dynamic task creation, GPU pipeline execution, and memory downgrading

## Layers

**DuckDB Integration Layer:**
- Purpose: Bridge between DuckDB's query execution and Sirius GPU engine
- Location: `src/sirius_extension.cpp`, `src/include/sirius_interface.hpp`, `src/sirius_interface.cpp`
- Contains: Extension registration, table function bindings (`gpu_processing`, `gpu_execution`), query result management
- Depends on: DuckDB core API, physical plan interfaces
- Used by: DuckDB query executor

**Physical Plan Generation Layer:**
- Purpose: Convert DuckDB's logical operator trees to Sirius-specific physical operators
- Location: `src/planner/` (new), `src/include/planner/`
- Contains: `sirius_physical_plan_generator`, plan builders for each operator type (`sirius_plan_filter.cpp`, `sirius_plan_aggregate.cpp`, etc.)
- Depends on: DuckDB LogicalOperator interface, operator type definitions
- Used by: Extension layer to prepare executable plans

**Execution Engine Layer:**
- Purpose: Orchestrate query execution across thread coordinator, task creator, scan executor, pipeline executor, and downgrade executor
- Location: `src/sirius_engine.cpp`, `src/include/sirius_engine.hpp`
- Contains: `sirius_engine` (orchestrator), pipeline collection and scheduling, operator ID management
- Depends on: Physical operators, pipelines, repositories, memory management
- Used by: Interface layer to execute prepared plans

**Pipeline Execution Framework:**
- Purpose: Manage operator chains as independent parallel execution units
- Location: `src/pipeline/`, `src/include/pipeline/`
- Contains: `sirius_pipeline`, `sirius_meta_pipeline`, pipeline task states, GPU task queues, execution handlers
- Depends on: Physical operators, data repositories, CUDA execution
- Used by: Engine, task creator, GPU executors

**Physical Operator Layer:**
- Purpose: GPU-accelerated implementations of query operators
- Location: `src/op/` (new Sirius), `src/include/op/`
- Contains: Scan operators (`sirius_physical_table_scan`, `sirius_physical_parquet_scan`), compute operators (filter, aggregate, join), merge operators
- Depends on: cuDF libraries, expression executor, data representations
- Used by: Pipeline executor

**Task Execution Layer:**
- Purpose: Manage creation and scheduling of GPU-bound and scan tasks
- Location: `src/creator/` (task creation), `src/downgrade/` (memory downgrade), `src/op/scan/` (scan tasks), `src/pipeline/` (GPU pipeline tasks)
- Contains: `task_creator` (schedules pipeline tasks), `downgrade_executor` (moves data across memory tiers), scan task implementations
- Depends on: Operator data, repositories, memory reservations, thread pools
- Used by: Engine during execution

**Expression Evaluation Layer:**
- Purpose: Evaluate SQL expressions on GPU
- Location: `src/expression_executor/`, `src/include/expression_executor/`, `src/cuda/expression_executor/`
- Contains: Expression translator (SQL AST → GPU code), dispatcher, GPU kernels for comparison, string ops, materialization
- Depends on: cuDF, DuckDB expression AST
- Used by: Filter, projection, join operators

**CUDA/GPU Layer:**
- Purpose: GPU kernels and cuDF wrappers
- Location: `src/cuda/`, `src/cuda/cudf/`, `src/cuda/operator/`
- Contains: cuDF wrappers (aggregate, join, orderby, groupby), operator-specific kernels, utilities
- Depends on: RAPIDS cuDF, RMM, CUDA runtime
- Used by: Physical operators

**Memory Management Layer:**
- Purpose: GPU memory allocation, caching, spilling with three-tier hierarchy
- Location: `src/memory/`, `src/include/memory/`, `cucascade/`
- Contains: `sirius_memory_reservation_manager` (memory leases), cuCascade data repositories and memory spaces
- Depends on: cuCascade library, CUDA runtime
- Used by: All layers for GPU resource allocation

## Data Flow

**Query Execution Pipeline:**

1. **Extension Entry** (`gpu_execution` table function call)
   - DuckDB calls Sirius table function with SQL query
   - `sirius_interface` receives call in `sirius_execute_query_internal()`

2. **Physical Plan Generation**
   - `sirius_physical_plan_generator::create_plan()` converts DuckDB LogicalOperator tree to `sirius_physical_operator` tree
   - Plan builders (`sirius_plan_filter`, `sirius_plan_aggregate`, etc.) handle each operator type
   - Returns root `sirius_physical_operator`

3. **Engine Initialization**
   - `sirius_engine::initialize()` receives physical plan
   - Builds pipeline structure: identifies pipeline breakers, creates `sirius_pipeline` objects
   - `initialize_internal()` assigns operator IDs, sets up repositories and ports
   - Root pipeline becomes entry point

4. **Thread Coordination**
   - Main thread in `sirius_engine::execute()` initiates execution
   - `task_creator` thread pool begins monitoring for data availability
   - Scanner threads from scan executor thread pool (DuckDB-based) start reading data

5. **Scan Phase**
   - `duckdb_scan_task` / `parquet_scan_task` / `iceberg_scan_task` from `src/op/scan/` read data storage
   - DuckDB format → GPU format via `sirius_converter_registry` converters
   - Data batches stored in `data_repository` (cuCascade `shared_data_repository`)

6. **Task Creation**
   - `task_creator` monitors repository data availability via hints from operators
   - Creates `gpu_pipeline_task` when input data ready and memory reserved
   - Task encapsulates pipeline, operator chain, and input data batch
   - Submits to `gpu_pipeline_executor` thread pool

7. **GPU Pipeline Execution**
   - GPU thread (stream-per-thread model) executes operator chain on task's input batch
   - Each operator: `Execute(data_batch)` → output stored back in repository
   - Operators use cuDF for joins, aggregates; custom kernels for expressions
   - Results available for dependent pipelines

8. **Memory Downgrading**
   - `downgrade_executor` monitors GPU memory pressure (cuCascade memory space)
   - When threshold exceeded, selects downgrade candidates and migrates data GPU→Host
   - Triggered automatically without halting pipeline execution

9. **Result Collection**
   - `sirius_physical_result_collector` (terminal operator) gathers final results
   - Materializes GPU results to DuckDB format
   - Returns to DuckDB as `QueryResult`

**State Management:**

- **Pipeline State**: `sirius_pipeline` tracks source, operators, sink; maintains dependencies via `sirius_meta_pipeline`
- **Task State**: `gpu_pipeline_task_local_state` holds input data, reservation, retry count; `gpu_pipeline_task_global_state` shared across tasks
- **Data State**: `data_repository` manages batches across tiers; operator outputs are batches (cudf::table or spilling allocation)
- **Memory State**: `sirius_memory_reservation_manager` tracks leases; `downgrade_executor` coordinates tier movement

## Key Abstractions

**sirius_physical_operator:**
- Purpose: Base class for all GPU-executable query operators
- Examples: `sirius_physical_filter`, `sirius_physical_hash_join`, `sirius_physical_grouped_aggregate`, `sirius_physical_table_scan`
- Pattern: Virtual `Execute()` and `Finalize()` methods; children stored as `vector<unique_ptr<sirius_physical_operator>>`; sink/source state management
- Location: `src/include/op/sirius_physical_operator.hpp`, `src/op/`

**sirius_pipeline:**
- Purpose: Chain of operators executed as a unit
- Examples: Filter pipeline, join build pipeline, aggregate pipeline
- Pattern: Contains source operator, operator vector, sink operator; manages dependencies via `sirius_meta_pipeline`; scheduled independently
- Location: `src/include/pipeline/sirius_pipeline.hpp`, `src/pipeline/`

**gpu_pipeline_task:**
- Purpose: Executable work unit: pipeline + input data + memory reservation
- Pattern: Encapsulates pipeline reference, input batch (via `operator_data`), retry context, memory reservation; executed by GPU thread
- Location: `src/include/pipeline/gpu_pipeline_task.hpp`, `src/pipeline/gpu_pipeline_task.cpp`

**data_repository (cuCascade):**
- Purpose: Container for operator output batches with automatic tier management
- Pattern: Created by engine per operator; stores `shared_ptr<data_batch>` entries; queries memory pressure; supports migration callbacks
- Location: Integrated from `cucascade/` (third-party), used throughout execution

**operator_data:**
- Purpose: Wrapper for vector of data batches passed between operators
- Examples: `operator_data`, `partitioned_operator_data`
- Pattern: Holds `vector<shared_ptr<data_batch>>`; provides const access interface
- Location: `src/include/op/sirius_physical_operator.hpp`

**memory_reservation (cuCascade):**
- Purpose: Lease on GPU memory to prevent oversubscription
- Pattern: Allocated by `sirius_memory_reservation_manager` before task execution; released after task completion
- Location: `src/include/memory/sirius_memory_reservation_manager.hpp`

## Entry Points

**gpu_execution Table Function:**
- Location: `src/sirius_extension.cpp`, function registered as DuckDB table function
- Triggers: User calls `CALL gpu_execution('SELECT ...')`
- Responsibilities: Parse SQL, prepare statement, create `sirius_interface`, run query, return results

**sirius_engine::execute():**
- Location: `src/include/sirius_engine.hpp`, `src/sirius_engine.cpp`
- Triggers: Called after engine initialization with physical plan
- Responsibilities: Spawn task creator and scan executor threads, monitor execution completion, aggregate results

**task_creator::start():**
- Location: `src/include/creator/task_creator.hpp`, `src/creator/task_creator.cpp`
- Triggers: Called by engine after thread pool started
- Responsibilities: Poll repositories, create GPU tasks when data and memory available, submit to executor

**gpu_pipeline_executor::execute_task():**
- Location: `src/include/pipeline/gpu_pipeline_executor.hpp`, `src/pipeline/gpu_pipeline_executor.cpp`
- Triggers: GPU thread pool pulls task from queue
- Responsibilities: Execute operator chain on GPU, handle errors, trigger completion handler

## Error Handling

**Strategy:** Operator-level fallback + graceful degradation

**Patterns:**
- **Expression Validation**: `gpu_expression_translator` checks expression support; throws `NotImplementedException` for unsupported ops (window functions, complex regex)
- **Memory Pressure**: `OomRescheduleException` allows task retry with reduced input size; `downgrade_executor` automatically migrates data
- **Data Type Checking**: Plan generation validates supported types (INTEGER, BIGINT, FLOAT, DOUBLE, VARCHAR, DATE, TIMESTAMP, DECIMAL); falls back to CPU for nested types
- **Operator Fallback**: Result collector or pipeline error triggers CPU re-execution of unsupported subtree

## Cross-Cutting Concerns

**Logging:** 
- spdlog framework, configurable via `SIRIUS_LOG_LEVEL` and `SIRIUS_LOG_DIR`
- Logged at: Physical plan generation, operator execution, task scheduling, memory allocation
- Location: `src/include/log/logging.hpp`

**Validation:** 
- Operator trees verified via `sirius_physical_operator::verify()`
- Expression translator validates AST structure before GPU code generation
- Type checking in plan generation layer

**Authentication:** 
- Not applicable (GPU execution engine layer; auth handled by DuckDB)

**CUDA Profiling:**
- NVIDIA CUDA Profiler API hooks in `sirius_extension.cpp` (`cudaProfilerStart`/`cudaProfilerStop`)
- NVTX markers in pipeline execution and operator dispatch for nsys profiling

---

*Architecture analysis: 2026-04-02*
