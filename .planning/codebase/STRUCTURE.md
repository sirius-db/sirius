# Codebase Structure

**Analysis Date:** 2026-04-13

## Directory Layout

```
/home/william/repos2/sirius/
├── src/                           # Main source code (C++/CUDA)
│   ├── include/                   # Public headers mirroring src/ structure
│   │   ├── op/                    # Operator headers
│   │   ├── pipeline/              # Pipeline execution headers
│   │   ├── planner/               # Planning headers
│   │   ├── creator/               # Task creator headers
│   │   ├── memory/                # Memory management headers
│   │   ├── downgrade/             # Memory spill headers
│   │   ├── expression_executor/   # GPU expression eval headers
│   │   ├── data/                  # Data batch/repository headers
│   │   ├── log/                   # Logging headers
│   │   └── sirius_context.hpp     # Ownership hierarchy
│   ├── planner/                   # Physical plan generation
│   │   ├── sirius_physical_plan_generator.cpp
│   │   ├── sirius_plan_*.cpp      # Per-operator plan builders
│   │   └── query.cpp              # Query metadata
│   ├── op/                        # Physical operator implementations
│   │   ├── aggregate/             # GROUP BY, aggregation merge
│   │   ├── merge/                 # MERGE_GROUP_BY, MERGE_SORT, etc.
│   │   ├── order/                 # ORDER_BY, sort phases
│   │   ├── partition/             # PARTITION, CONCAT operators
│   │   ├── result/                # RESULT_COLLECTOR
│   │   ├── scan/                  # Scan operators (DuckDB, Parquet, Iceberg)
│   │   ├── sirius_physical_*.cpp  # Individual operators
│   │   └── sirius_physical_operator.cpp
│   ├── pipeline/                  # Pipeline execution
│   │   ├── pipeline_executor.cpp          # Top-level executor
│   │   ├── gpu_pipeline_executor.cpp      # GPU executor per device
│   │   ├── gpu_pipeline_task.cpp          # GPU task definition
│   │   ├── sirius_pipeline.cpp            # Pipeline metadata
│   │   ├── sirius_meta_pipeline.cpp       # Pipeline grouping
│   │   ├── sirius_pipeline_converter.cpp  # Pipeline finalization
│   │   └── task_request.cpp
│   ├── creator/                   # Task creation
│   │   └── task_creator.cpp
│   ├── memory/                    # Memory management
│   │   └── sirius_memory_reservation_manager.cpp
│   ├── downgrade/                 # GPU memory spilling
│   │   └── downgrade_executor.cpp
│   ├── expression_executor/       # Expression evaluation
│   │   └── gpu_expression_executor*.cpp
│   ├── cuda/                      # GPU kernels and CUDA logic
│   │   ├── operator/              # Per-operator CUDA implementations
│   │   ├── expression_executor/   # cuDF AST dispatch
│   │   ├── cudf/                  # cuDF wrapper utilities
│   │   ├── iceberg/               # Iceberg delete filtering
│   │   ├── allocator.cu           # GPU memory allocation
│   │   ├── communication.cu       # GPU<->Host transfer
│   │   └── utils.cu               # GPU utilities
│   ├── data/                      # Data management
│   │   └── sirius_converter_registry.cpp
│   ├── util/                      # Utilities
│   ├── parallel/                  # Thread pool implementations
│   ├── sirius_extension.cpp       # DuckDB extension entry point
│   ├── sirius_interface.cpp       # Interface layer
│   ├── sirius_engine.cpp          # Engine and pipeline builder
│   ├── fallback.cpp               # CPU fallback
│   └── config.cpp                 # Configuration
├── docs/                          # Documentation
│   └── super-sirius/              # Architecture & design docs (read first!)
│       ├── README.md              # Index and reading order
│       ├── architecture-overview.md
│       ├── execution-flow.md
│       ├── physical-plan-generation.md
│       ├── operators.md
│       ├── expression-executor.md
│       ├── pipeline-execution.md
│       ├── task-creator.md
│       ├── scan.md
│       ├── memory-management.md
│       ├── data-management.md
│       ├── configuration.md
│       └── optimizations.md
├── test/                          # Tests
│   ├── cpp/                       # C++ unit tests (Catch2)
│   │   ├── operator/              # Operator-specific tests
│   │   ├── pipeline/              # Pipeline execution tests
│   │   ├── planner/               # Plan generation tests
│   │   ├── creator/               # Task creator tests
│   │   ├── scan/                  # Scan tests
│   │   ├── memory_management/     # Memory tests
│   │   ├── downgrade/             # Spill tests
│   │   ├── expression_executor/   # Expression tests
│   │   └── integration/           # Integration tests
│   └── sql/                       # SQL logic tests
│       └── tpch-sirius.test
├── build/                         # Build artifacts (not in git)
│   └── release/
│       ├── extension/sirius/sirius.duckdb_extension (compiled output)
│       └── extension/sirius/test/cpp/sirius_unittest (unit test binary)
├── CMakeLists.txt                 # Main build file
├── Makefile                       # Thin wrapper
├── pixi.toml                      # Pixi environment config
└── .claude/                       # Claude Code skills & documentation
    └── skills/module-discover/docs/  # Module API docs
```

## Directory Purposes

**src/include/:**
- Purpose: Public header files organized by subsystem
- Contains: Class definitions, API signatures, type definitions
- Pattern: Mirror src/ structure for easy navigation (src/planner/*.cpp ↔ src/include/planner/*.hpp)

**src/planner/:**
- Purpose: Logical-to-physical plan conversion
- Contains: Plan generator (dispatcher) and per-operator plan builders
- Key files:
  - `sirius_physical_plan_generator.cpp` — main entry point, switch on LogicalOperator type
  - `sirius_plan_get.cpp` — scan operator planning
  - `sirius_plan_filter.cpp`, `sirius_plan_projection.cpp` — streaming operator planning
  - `sirius_plan_aggregate.cpp` — GROUP BY/aggregation planning
  - `sirius_plan_comparison_join.cpp` — join planning (hash vs nested loop)
  - `sirius_plan_order.cpp`, `sirius_plan_top_n.cpp` — sorting/limiting
- Called from: `src/sirius_extension.cpp` during BIND phase
- Returns: `sirius_physical_operator` tree

**src/op/:**
- Purpose: Physical operator implementations
- Structure:
  - Subdirectories for operator categories (aggregate, merge, order, partition, result, scan)
  - `sirius_physical_*.cpp` files for individual operators
  - `sirius_physical_operator.cpp` — base class
  - `sirius_physical_operator_type.cpp` — operator type enum
- Key operators:
  - Scan: `sirius_physical_duckdb_scan.cpp`, `sirius_physical_parquet_scan.cpp`, `sirius_physical_iceberg_scan.cpp`
  - Streaming: `sirius_physical_filter.cpp`, `sirius_physical_projection.cpp`, `sirius_physical_limit.cpp`
  - Blocking: `sirius_physical_hash_join.cpp`, `sirius_physical_nested_loop_join.cpp`, `sirius_physical_grouped_aggregate.cpp`
  - Merge: `sirius_physical_grouped_aggregate_merge.cpp`, `sirius_physical_merge_sort.cpp`
  - Control: `sirius_physical_partition.cpp`, `sirius_physical_concat.cpp`, `sirius_physical_result_collector.cpp`
- Contains: `execute()`, `sink()`, `is_source()`, `is_sink()` implementations

**src/pipeline/:**
- Purpose: Multi-pipeline task execution orchestration
- Key files:
  - `pipeline_executor.cpp` — top-level orchestrator (management eventloop, sub-executor routing)
  - `gpu_pipeline_executor.cpp` — GPU task executor with worker thread pool and CUDA stream management
  - `sirius_pipeline.cpp` — pipeline metadata (operators, dependencies, completion tracking)
  - `sirius_meta_pipeline.cpp` — groups pipelines by shared sink during construction
  - `sirius_pipeline_converter.cpp` — finalizes pipelines for execution (large file, critical logic)
  - `gpu_pipeline_task.cpp` — task definition with batch collection and compute/sink phases
- Execution: Call `pipeline_executor.start_query()` from engine, which schedules initial scan tasks and manages task routing

**src/creator/:**
- Purpose: Convert ready operators into executable tasks
- Contains: `task_creator.cpp` — manager loop that pops from creation queue, follows data availability hint chain, creates GPU/scan tasks
- Called from: Scan executors and GPU executors (after task completion)
- Decides: Which operator is ready, which pipeline to create task for

**src/memory/:**
- Purpose: GPU/Host/Disk memory allocation and management
- Contains: `sirius_memory_reservation_manager.cpp` — reservation interface, memory space tracking, atomic allocation/release
- Integrates: cuCascade tiered memory, RMM GPU allocator, DuckDB host allocator
- Called from: GPU executors (reserve before task), downgrade executors (track pressure)

**src/downgrade/:**
- Purpose: GPU memory spilling (GPU → Host → Disk)
- Contains: `downgrade_executor.cpp` — monitor thread polls GPU memory pressure, dispatches spill tasks
- Watches: Data repositories for GPU-resident batches when pressure exceeds threshold
- Transitions: Batches from GPU to Host to Disk via cuCascade memory tiers

**src/expression_executor/:**
- Purpose: GPU-accelerated expression evaluation
- Contains: CPU-side expression executors (e.g., `gpu_expression_executor.cpp`)
- Methods: `select()` (filter), `project()` (projection), aggregate function dispatchers
- Delegates: To CUDA kernels in `src/cuda/expression_executor/`

**src/cuda/:**
- Purpose: GPU kernels and CUDA-specific logic
- Structure:
  - `operator/` — per-operator CUDA kernels (join, aggregation, sort, etc.)
  - `expression_executor/` — cuDF AST translation and dispatch
  - `cudf/` — cuDF wrapper utilities (batch format conversion)
  - `iceberg/` — Iceberg delete filtering on GPU
  - `*.cu` files — CUDA kernels (allocator, communication, utils, print)
- Compilation: Separable compilation enabled (CMAKE_CUDA_SEPARABLE_COMPILATION ON)

**src/data/:**
- Purpose: Data batch lifecycle and repository management
- Contains: `sirius_converter_registry.cpp` — DuckDB chunk ↔ GPU batch format conversion

**src/util/:**
- Purpose: Utility functions
- Contains: Segfault backtracing, string utilities, etc.

**src/parallel/:**
- Purpose: Thread pool abstractions
- Contains: Bounded thread pool, channel implementations, interruptible MPMC queue

**docs/super-sirius/:**
- Purpose: Comprehensive architecture and design documentation
- **READ FIRST** before modifying Super Sirius code
- Contents:
  - `README.md` — index and recommended reading order
  - `architecture-overview.md` — component diagram, thread model, ownership
  - `execution-flow.md` — end-to-end query trace with file:line references
  - `physical-plan-generation.md` — logical-to-physical mapping, pipeline splitting
  - `operators.md` — all operator types, GPU implementations, cuDF APIs
  - `expression-executor.md` — GPU expression evaluation
  - `pipeline-execution.md` — task scheduling, CUDA stream management
  - `task-creator.md` — task creation heuristics
  - `scan.md` — data ingestion (DuckDB, Parquet, Iceberg, caching)
  - `memory-management.md` — GPU memory tiers, reservations, spilling
  - `data-management.md` — data batch lifecycle, repositories, ports
  - `configuration.md` — runtime tuning parameters
  - `optimizations.md` — performance improvements with PR references

**test/cpp/:**
- Purpose: C++ unit tests (Catch2 framework)
- Structure mirrors src/:
  - `operator/` — operator-specific tests
  - `pipeline/` — pipeline construction and execution
  - `planner/` — plan generation correctness
  - `creator/` — task creation logic
  - `scan/` — scan operator tests
  - `memory_management/` — memory reservation behavior
  - `downgrade/` — spill mechanism
  - `integration/` — end-to-end query tests
- Run: `build/release/extension/sirius/test/cpp/sirius_unittest` or specific tests with tag/name filters
- Logs: `build/release/extension/sirius/test/cpp/log/`

**test/sql/:**
- Purpose: SQL logic tests (end-to-end)
- Contains: `tpch-sirius.test` — TPC-H queries for functional validation
- Run: `make test` or `build/release/test/unittest --test-dir . test/sql/tpch-sirius.test`

## Key File Locations

**Extension Entry Points:**
- `src/sirius_extension.cpp` — `LoadInternal()` function registers extension, `GPUExecutionBind()` parses SQL, `GPUExecutionFunction()` executes
- `src/sirius_extension.hpp` — Extension class definition

**Interface Layer:**
- `src/sirius_interface.cpp` — `sirius_execute_query()` main entry point
- `src/include/sirius_interface.hpp` — Interface class definition

**Engine & Planning:**
- `src/sirius_engine.cpp` — Pipeline construction and execution orchestration
- `src/include/sirius_engine.hpp` — Engine class definition
- `src/planner/sirius_physical_plan_generator.cpp` — Plan dispatcher

**Operator Base Class:**
- `src/include/op/sirius_physical_operator.hpp` — Base operator interface
- `src/op/sirius_physical_operator.cpp` — Base implementation

**Pipeline Execution:**
- `src/pipeline/pipeline_executor.cpp` — Top-level orchestrator
- `src/include/pipeline/pipeline_executor.hpp` — Executor interface
- `src/pipeline/gpu_pipeline_executor.cpp` — GPU task executor
- `src/include/pipeline/gpu_pipeline_executor.hpp` — GPU executor interface
- `src/pipeline/sirius_pipeline_converter.cpp` — Pipeline finalization (large, critical)
- `src/include/pipeline/sirius_pipeline.hpp` — Pipeline metadata

**Task Creation:**
- `src/creator/task_creator.cpp` — Task scheduling logic
- `src/include/creator/task_creator.hpp` — Task creator interface

**Memory Management:**
- `src/include/memory/sirius_memory_reservation_manager.hpp` — Memory reservation interface
- `src/memory/sirius_memory_reservation_manager.cpp` — Implementation

**Context & Configuration:**
- `src/include/sirius_context.hpp` — Ownership hierarchy
- `src/include/sirius_config.hpp` — Configuration parameters
- `src/config.cpp` — Configuration implementation

**Error Handling:**
- `src/fallback.cpp` — CPU fallback mechanism
- `src/include/sirius/exception.hpp` — Sirius exception types

**Data Management:**
- `src/include/data/` — Data batch and repository headers
- `src/data/sirius_converter_registry.cpp` — Format conversion

## Naming Conventions

**Files:**
- Source: `sirius_*.cpp` (e.g., `sirius_physical_filter.cpp`)
- Headers: `sirius_*.hpp` (e.g., `sirius_physical_filter.hpp`) in `src/include/`
- Plan builders: `sirius_plan_*.cpp` (e.g., `sirius_plan_aggregate.cpp`)
- Test files: `*_test.cpp` in `test/cpp/`
- SQL tests: `*.test` in `test/sql/`

**Classes:**
- Physical operators: `sirius_physical_<operator_name>` (e.g., `sirius_physical_hash_join`)
- Executors: `<component>_executor` (e.g., `gpu_pipeline_executor`, `duckdb_scan_executor`)
- Pipelines: `sirius_pipeline`, `sirius_meta_pipeline`
- Managers: `<component>_manager` (e.g., `shared_data_repository_manager`)

**Functions:**
- camelCase for methods (e.g., `execute()`, `get_next_task_hint()`)
- snake_case for free functions (e.g., `create_plan()`)
- Verb-first naming: `build_pipelines()`, `insert_repository()`, `push_data_batch()`

**Variables:**
- camelCase in class members (e.g., `operator_id`, `is_source`)
- snake_case for local variables
- Prefix acronyms (e.g., `gpu_executor`, `rmm_pool`)

**Types:**
- Enums: `SiriusPhysicalOperatorType`, `MemoryBarrierType`
- Structs: `port`, `sirius_config`, `gpu_pipeline_task`
- Smart pointers: `shared_ptr`, `unique_ptr`, `optional_ptr` (DuckDB's nullable pointer)

**Directories:**
- Lowercase with underscores: `expression_executor`, `memory_management`
- Component grouping: `operator/` → per-operator implementations

## Where to Add New Code

**New GPU Operator:**
1. Create header: `src/include/op/sirius_physical_<operator>.hpp`
2. Implement: `src/op/sirius_physical_<operator>.cpp` (CPU-facing interface)
3. GPU kernel: `src/cuda/operator/<operator>.cu` (cuDF/CUDA implementation)
4. Plan builder: `src/planner/sirius_plan_<operator>.cpp` (logical-to-physical mapping)
5. Register: Add case in `src/planner/sirius_physical_plan_generator.cpp` switch statement
6. Tests: Add unit test in `test/cpp/operator/test_<operator>.cpp`
7. SQL tests: Add TPC-H/custom query in `test/sql/tpch-sirius.test`

**New Streaming Operator:**
1. Create header: `src/include/op/sirius_physical_<operator>.hpp`
2. Implement: `src/op/sirius_physical_<operator>.cpp` (simpler, no batching)
3. Plan builder: `src/planner/sirius_plan_<operator>.cpp`
4. Register: Add case in plan generator
5. Tests: Add to `test/cpp/operator/`

**New Executor Component:**
- Implement in `src/` with header in `src/include/`
- Register in `SiriusContext` ownership hierarchy
- Integrate into `pipeline_executor.start_query()` task routing

**New Memory Space:**
- Extend `sirius_memory_reservation_manager` with new `memory_space` type
- Create corresponding downgrade executor logic in `src/downgrade/`

**New Configuration Parameter:**
- Add to `src/include/sirius_config.hpp`
- Implement getter/setter in `src/config.cpp`
- Use `SIRIUS_CONFIG(context).param_name` to access

**Utility Functions:**
- Shared helpers: `src/util/` or `src/include/helper/`
- GPU utilities: `src/cuda/utils.cu`
- Formatting: `src/include/print.hpp`

## Special Directories

**src/include/legacy/:**
- Purpose: Old `gpu_processing` code path (namespace duckdb)
- Committed: Yes (for backward compatibility)
- Use: Not for new features; GPU execution uses `namespace sirius` in `src/op/` instead
- Location: `src/include/legacy/operator/`, `src/operator/`, `src/plan/`, `src/legacy/`

**src/include/operator/:**
- Purpose: Legacy operator headers
- Committed: Yes
- Status: Deprecated; new operators in `src/include/op/`

**build/:/**
- Purpose: Build artifacts (CMake output)
- Generated: Yes
- Committed: No (.gitignore)
- Contents: Compiled extension (.duckdb_extension), unit test binary, object files

**.claude/skills/module-discover/docs/:**
- Purpose: Auto-generated API documentation for dependencies (cudf, rmm, duckdb, libkvikio, cucascade)
- Generated: By `/module-discover` skill (run once per dependency)
- Consumed: By `/module-context` skill (loaded before implementation tasks)
- Format: LLM-friendly markdown with function signatures, types, usage examples

**test_datasets/:**
- Purpose: Pre-generated TPC-H/TPC-DS benchmark data (Parquet format)
- Generated: By `/dataset-manager` skill
- Committed: No (.gitignore)
- Usage: `python3 test/tpch_performance/performance_test.py {SCALE_FACTOR}`

## File Dependencies

**Import Patterns:**

DuckDB headers use angle brackets (`#include <duckdb/...>`), are system includes. Local sirius headers use quotes (`#include "sirius_interface.hpp"`).

**Operator Headers Depend On:**
- `src/include/op/sirius_physical_operator.hpp` (base class)
- `src/include/expression_executor/gpu_expression_executor.hpp` (if filtering/projection)
- cuDF headers via `src/include/cudf/` wrappers
- RMM headers for memory allocation

**Pipeline Headers Depend On:**
- `src/include/op/sirius_physical_operator.hpp` (all operators)
- `src/include/data/` (repositories, batches)
- `src/include/memory/` (reservation manager)
- `src/include/creator/task_creator.hpp` (scheduling)

**Planner Headers Depend On:**
- DuckDB logical operator headers
- `src/include/op/` (all physical operator headers)
- Expression evaluation headers

**Engine Headers Depend On:**
- All above layers (operators, pipeline, planner, data, memory)
- `src/include/sirius_context.hpp` (ownership)

**Extension Headers Depend On:**
- `src/include/sirius_interface.hpp`
- DuckDB extension APIs

**Order to Explore When Understanding Code:**
1. `docs/super-sirius/README.md` (architecture overview, reading order)
2. `src/include/sirius_context.hpp` (ownership hierarchy)
3. `src/sirius_extension.cpp` (entry point)
4. `src/sirius_interface.cpp` (query lifecycle)
5. `src/sirius_engine.cpp` (pipeline construction)
6. `src/planner/sirius_physical_plan_generator.cpp` (logical-to-physical)
7. `src/include/op/sirius_physical_operator.hpp` (operator base class)
8. `src/pipeline/pipeline_executor.cpp` (execution orchestration)
9. `src/creator/task_creator.cpp` (task scheduling)
10. Individual operators in `src/op/` as needed

---

*Structure analysis: 2026-04-13*
