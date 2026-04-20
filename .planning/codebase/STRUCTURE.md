# Codebase Structure

**Analysis Date:** 2026-04-02

## Directory Layout

```
sirius_2/
├── src/                          # Sirius source code
│   ├── include/                  # Header files (mirrors src/ structure)
│   │   ├── sirius_engine.hpp
│   │   ├── sirius_interface.hpp
│   │   ├── planner/              # Physical plan generation
│   │   ├── op/                   # Physical operator definitions (new Sirius)
│   │   ├── pipeline/             # Pipeline execution framework
│   │   ├── creator/              # Task creation
│   │   ├── downgrade/            # Memory downgrade
│   │   ├── expression_executor/  # GPU expression evaluation
│   │   ├── memory/               # Memory management
│   │   ├── data/                 # Data representations
│   │   ├── legacy/               # Legacy Sirius headers
│   │   ├── parallel/             # Task threading primitives
│   │   ├── cudf/                 # cuDF utility headers
│   │   ├── helper/               # Type/util helpers
│   │   ├── log/                  # Logging
│   │   └── util/                 # Utility helpers
│   ├── op/                       # Physical operator implementations (new Sirius)
│   │   ├── scan/                 # Table scan implementations
│   │   ├── aggregate/            # GPU aggregate implementations
│   │   ├── partition/            # GPU partition kernel implementations
│   │   ├── order/                # GPU order kernel implementations
│   │   ├── merge/                # GPU merge kernel implementations
│   │   ├── result/               # Result collection
│   │   └── sirius_physical_*.cpp # Individual operator implementations
│   ├── planner/                  # Physical plan generation implementations
│   │   └── sirius_plan_*.cpp     # Plan builders for each operator
│   ├── pipeline/                 # Pipeline execution implementations
│   │   ├── gpu_pipeline_task.cpp
│   │   ├── gpu_pipeline_executor.cpp
│   │   ├── sirius_pipeline.cpp
│   │   └── ...
│   ├── creator/                  # Task creation implementations
│   │   └── task_creator.cpp
│   ├── downgrade/                # Memory downgrade implementations
│   │   ├── downgrade_executor.cpp
│   │   └── downgrade_task.cpp
│   ├── cuda/                     # GPU kernels and cuDF wrappers
│   │   ├── cudf/                 # cuDF wrapper implementations
│   │   ├── operator/             # Operator-specific kernels
│   │   ├── expression_executor/  # Expression evaluation kernels
│   │   ├── iceberg/              # Iceberg-specific kernels
│   │   └── *.cu                  # Utility kernels
│   ├── expression_executor/      # Expression evaluation implementations
│   │   ├── gpu_expression_translator.cpp
│   │   └── ...
│   ├── memory/                   # Memory management implementations
│   ├── data/                     # Data representation implementations
│   ├── parallel/                 # Parallel execution primitives
│   ├── util/                     # Utility implementations
│   ├── legacy/                   # Legacy Sirius implementations (gpu_processing)
│   ├── sirius_extension.cpp      # Extension entry point
│   ├── sirius_interface.cpp      # Query interface
│   ├── sirius_engine.cpp         # Execution engine
│   ├── sirius_context.cpp        # Context management
│   ├── config.cpp                # Configuration
│   └── fallback.cpp              # Fallback to CPU execution
├── test/                         # Test suite
│   ├── cpp/                      # C++ unit tests (Catch2)
│   │   ├── integration/          # Integration tests
│   │   ├── operator/             # Operator unit tests
│   │   └── ...
│   ├── sql/                      # SQL logic tests
│   │   └── tpch-sirius.test      # TPC-H test suite
│   └── tpch_performance/         # Performance testing
├── cucascade/                    # GPU memory management (git submodule)
│   ├── include/                  # cuCascade headers
│   │   ├── data/                 # Data batch and repository interfaces
│   │   ├── memory/               # Memory space abstractions
│   │   └── ...
│   └── src/                      # cuCascade implementation
├── duckdb/                       # DuckDB core (git submodule)
├── duckdb-python/                # Python bindings (git submodule)
├── extension-ci-tools/           # CI/build infrastructure (git submodule)
├── CMakeLists.txt                # Main CMake build file
├── extension_config.cmake        # Extension dependency configuration
├── .clang-format                 # Code formatting rules
├── .clang-tidy                   # Linting rules
├── .pre-commit-config.yaml       # Pre-commit hooks
├── pixi.toml                     # Pixi environment specification
└── CLAUDE.md                     # Project guidelines
```

## Directory Purposes

**src/:**
- Purpose: All Sirius implementation code (C++, CUDA)
- Contains: Operators, planner, pipeline, CUDA kernels, memory management
- Key files: `sirius_extension.cpp`, `sirius_engine.cpp`, `sirius_interface.cpp`

**src/include/:**
- Purpose: Header files organized by functional module
- Contains: Type definitions, class interfaces, template implementations
- Pattern: Mirrors src/ directory structure for parallel inclusion paths

**src/op/:**
- Purpose: Physical operator implementations (new Sirius, namespace sirius)
- Contains: 30+ operator types (filter, join, aggregate, scan, etc.) and their GPU implementations
- Key files: `sirius_physical_hash_join.cpp` (51KB, complex join logic), `sirius_physical_grouped_aggregate.cpp`, `sirius_physical_table_scan.cpp`
- Subdirectories: `scan/` (table scan tasks), `aggregate/` (GPU aggregate kernels), `partition/`, `order/`, `merge/` (CUDA implementations)

**src/include/op/:**
- Purpose: Physical operator type definitions and base class
- Contains: `sirius_physical_operator.hpp` (base), individual operator headers, operator_data wrapper classes
- Key: `sirius_physical_operator_type.hpp` lists all 40+ operator enum types

**src/planner/:**
- Purpose: Physical plan generation from DuckDB logical operators
- Contains: Main generator (`sirius_physical_plan_generator.cpp`) and plan builders for each operator type
- Files: `sirius_plan_filter.cpp`, `sirius_plan_aggregate.cpp`, `sirius_plan_comparison_join.cpp`, etc.
- Key: Single entry point `sirius_physical_plan_generator::create_plan()` dispatches to builders via switch statement

**src/include/planner/:**
- Purpose: Plan generator interface and declarations
- Contains: `sirius_physical_plan_generator.hpp`, query context, operator-specific plan method signatures

**src/pipeline/:**
- Purpose: Pipeline execution framework implementations
- Contains: Pipeline state machine, GPU task executor, task queues, completion handling
- Key files: `gpu_pipeline_task.cpp` (22KB, task execution), `gpu_pipeline_executor.cpp` (12KB, executor), `sirius_pipeline.cpp`, `sirius_meta_pipeline.cpp`
- Pattern: Task lifecycle: created by task_creator → queued → GPU thread executes → completion handler triggers next tasks

**src/include/pipeline/:**
- Purpose: Pipeline execution interfaces
- Contains: `sirius_pipeline.hpp`, `gpu_pipeline_task.hpp`, task state definitions, executor interfaces, queue definitions
- Key: Task state machine in `sirius_pipeline_task_states.hpp`, memory history tracking

**src/creator/:**
- Purpose: GPU pipeline task creation and scheduling
- Contains: `task_creator.cpp` (task scheduling logic)
- Key: Main loop monitors data repositories, respects memory reservations, creates tasks with hints

**src/include/creator/:**
- Purpose: Task creation interfaces
- Contains: `task_creator.hpp` (with inline doc on creation hints and scheduling)

**src/downgrade/:**
- Purpose: Memory tier migration (GPU→Host→Disk)
- Contains: `downgrade_executor.cpp`, `downgrade_task.cpp`
- Key: Background monitor thread watches GPU memory pressure, triggers automated downgrade tasks

**src/include/downgrade/:**
- Purpose: Downgrade execution interfaces
- Contains: `downgrade_executor.hpp`, `downgrade_task.hpp`

**src/cuda/:**
- Purpose: GPU kernels and cuDF wrappers
- Contains: 30+ .cu files organized by functionality
- Subdirs: `cudf/` (cuDF wrapper calls), `operator/` (join, string, expression kernels), `expression_executor/` (GPU AST dispatch), `iceberg/` (delete filtering)
- Pattern: Each .cu file is standalone; kernels wrapped by C++ operators

**src/include/cudf/:**
- Purpose: cuDF utility function signatures
- Contains: Helper function declarations for cuDF operations

**src/cuda/cudf/:**
- Purpose: cuDF wrapper implementations
- Files: `cudf_join.cu`, `cudf_aggregate.cu`, `cudf_groupby.cu`, `cudf_orderby.cu`, `cudf_duplicate_elimination.cu`
- Pattern: Call RAPIDS cuDF library functions, handle result serialization

**src/cuda/operator/:**
- Purpose: Operator-specific GPU kernels
- Files: `hash_join_inner.cu`, `hash_join_single.cu`, `hash_join_right.cu`, `nested_loop_join.cu`, `comparison_expression.cu`, `strings_matching.cu`, `materialize.cu`
- Subdirs: `unused/` (legacy/experimental kernels)

**src/expression_executor/:**
- Purpose: SQL expression evaluation on GPU
- Contains: Expression translator (SQL AST → cuDF AST), dispatcher, specializations
- Key: `gpu_expression_translator.cpp` converts bound expressions to GPU evaluable form

**src/include/expression_executor/:**
- Purpose: Expression evaluation interfaces
- Contains: Translator, executor, dispatcher, regex support headers

**src/memory/:**
- Purpose: Memory management implementation
- Contains: Reservation manager (integrates with cuCascade), memory state tracking

**src/include/memory/:**
- Purpose: Memory management interfaces
- Contains: `sirius_memory_reservation_manager.hpp`, host table utilities, allocation accessors, OOM policy

**src/data/:**
- Purpose: Data representation conversions
- Contains: Parquet converters, cached representations, data batch utilities
- Key: `host_parquet_representation.cpp` converts between DuckDB and GPU formats

**src/include/data/:**
- Purpose: Data structure definitions
- Contains: Representation interfaces, converter registry, batch utilities

**src/parallel/:**
- Purpose: Parallel execution primitives
- Contains: Task executor interface, task queue definitions, config

**src/include/parallel/:**
- Purpose: Parallel execution abstractions
- Contains: Task and executor interfaces, thread pool config, queue signatures

**src/legacy/:**
- Purpose: Legacy Sirius (`gpu_processing`, namespace duckdb) implementation
- Contains: Old physical plan generator, operators, executor (kept for backward compatibility)
- Pattern: Parallel to new Sirius but uses different namespace and architecture

**src/include/legacy/:**
- Purpose: Legacy Sirius type definitions
- Contains: Headers for old operators, context, executor

**test/cpp/:**
- Purpose: C++ unit tests (Catch2 framework)
- Contains: Operator tests, integration tests, utility tests
- Key: `test/cpp/integration/test_gpu_execution_tpch.cpp` validates TPC-H queries end-to-end

**test/sql/:**
- Purpose: SQL logic tests (DuckDB format)
- Contains: .test files with SQL statements and expected output
- Key: `test/sql/tpch-sirius.test` validates all TPC-H 22 queries

**test/tpch_performance/:**
- Purpose: Performance benchmark suite
- Contains: TPC-H data generation, query execution, result timing

**cucascade/:**
- Purpose: Third-party GPU memory management library (submodule)
- Key abstractions: `data_batch`, `data_repository`, `memory_space`, `memory_reservation`
- Used by: All memory-intensive operators for automatic tier migration

**CMakeLists.txt:**
- Purpose: Main build configuration
- Configures: CUDA version (13+), GPU architectures, dependency linking (cudf, rmm, spdlog), separable compilation

**extension_config.cmake:**
- Purpose: DuckDB extension configuration
- Specifies: Which extensions to build (sirius, json, tpcds, tpch, parquet, icu)

## Key File Locations

**Entry Points:**
- `src/sirius_extension.cpp`: DuckDB extension registration, table function binding for `gpu_processing` and `gpu_execution`
- `src/sirius_interface.cpp`: `sirius_interface` class implementing GPU query execution
- `src/sirius_engine.cpp`: `sirius_engine` orchestrator and scheduler

**Configuration:**
- `src/config.cpp`: Runtime configuration (memory limits, thread counts)
- `src/include/config.hpp`: Config declarations

**Core Logic:**
- `src/planner/sirius_physical_plan_generator.cpp`: Physical plan generation dispatcher
- `src/op/sirius_physical_operator.cpp`: Base operator class and virtual dispatch
- `src/pipeline/gpu_pipeline_task.cpp`: GPU task execution state machine
- `src/pipeline/gpu_pipeline_executor.cpp`: GPU thread pool executor
- `src/creator/task_creator.cpp`: Task creation and scheduling loop
- `src/expression_executor/gpu_expression_translator.cpp`: Expression → GPU code translation

**Testing:**
- `test/cpp/integration/test_gpu_execution_tpch.cpp`: End-to-end TPC-H integration tests
- `test/sql/tpch-sirius.test`: SQL logic test suite
- `build/release/extension/sirius/test/cpp/sirius_unittest`: Compiled unit test binary (after build)

## Naming Conventions

**Files:**
- New Sirius operators: `sirius_physical_<operator_type>.cpp` (e.g., `sirius_physical_hash_join.cpp`)
- Legacy Sirius operators: `gpu_physical_<operator_type>.cpp` (e.g., `gpu_physical_hash_join.cpp`)
- Plan builders: `sirius_plan_<operator_type>.cpp` (e.g., `sirius_plan_filter.cpp`)
- CUDA kernels: `<operation>.cu` (e.g., `hash_join_inner.cu`)
- Test files: `test_<feature>.cpp` or `*.test` (SQL logic)

**Directories:**
- Namespace-based: `src/op/` (new Sirius namespace `sirius::op`), `src/legacy/` (legacy namespace `duckdb`)
- Feature-based: `src/cuda/cudf/`, `src/cuda/operator/`, `src/cuda/expression_executor/`
- Abstract concerns: `src/pipeline/`, `src/creator/`, `src/downgrade/`, `src/memory/`

**Classes/Types:**
- New Sirius: `sirius_physical_*` (e.g., `sirius_physical_hash_join`), `sirius_*` (e.g., `sirius_engine`, `sirius_pipeline`)
- Legacy Sirius: `GPU*` (e.g., `GPUPhysicalHashJoin`), `gpu_*` (e.g., `gpu_executor`)
- Utilities: `gpu_*_impl`, `*_utils` (e.g., `gpu_aggregate_impl`, `data_batch_utils`)

**Functions:**
- Plan builders: `create_plan(LogicalOperator&)` (overloaded per operator type)
- Operators: `Execute()` (virtual), `Finalize()`, `GetChildren()`, `verify()`
- Tasks: `run()` (main execution), `execute()` (alias)
- CUDA: kernel names end with `_kernel` or are prefixed with namespace (e.g., `gpu_aggregate_kernel`)

## Where to Add New Code

**New Operator Implementation:**
1. Header: Create `src/include/op/sirius_physical_<new_op>.hpp` with class inheriting from `sirius_physical_operator`
   - Define `TYPE` constant, `Execute()`, `Finalize()`, child management
2. Implementation: `src/op/sirius_physical_<new_op>.cpp`
   - Implement virtual methods, GPU execution logic
3. Plan builder: `src/planner/sirius_plan_<new_op>.cpp`
   - Add `create_plan(LogicalNewOp&)` method to `sirius_physical_plan_generator`
4. GPU kernel: `src/cuda/operator/<new_op>.cu` (if custom kernel needed)
   - Implement CUDA kernel or cuDF wrapper
5. Generator dispatch: Update `sirius_physical_plan_generator.cpp` switch statement
6. Type enum: Add to `src/include/op/sirius_physical_operator_type.hpp`
7. Tests: Add to `test/cpp/operator/test_<new_op>.cpp` and `test/sql/`

**New Expression Function:**
1. Translator: Add case to `src/expression_executor/gpu_expression_translator.cpp` `visit()` method
2. Kernel: Implement in `src/cuda/expression_executor/gpu_dispatch_<category>.cu`
3. Specialization: Add fast path to `src/expression_executor/specializations/` if performance-critical
4. Tests: Add unit test to verify GPU result matches CPU

**New Optimization:**
1. Plan generator: Modify `sirius_physical_plan_generator::create_plan()` to apply optimization
2. Or: Create dedicated optimization pass in `src/planner/`
3. Tests: Verify correctness and performance via `test/tpch_performance/`

**Utilities/Helpers:**
- Shared helpers: `src/include/helper/` (types, utilities)
- cuDF wrappers: `src/cuda/cudf/`
- Data conversion: `src/data/`

## Special Directories

**build/:**
- Purpose: Build output directory
- Generated: Yes (created by CMake)
- Committed: No (.gitignored)
- Contents: Compiled binaries, test executable, object files

**cucascade/ (submodule):**
- Purpose: Third-party GPU memory management library
- Generated: No
- Committed: Git submodule
- Critical for: Data batch storage, repository management, memory tier abstractions

**duckdb/ (submodule):**
- Purpose: DuckDB core database engine
- Generated: No
- Committed: Git submodule
- Critical for: LogicalOperator types, expression AST, execution context

**duckdb-python/ (submodule):**
- Purpose: Python bindings for DuckDB
- Generated: No
- Committed: Git submodule
- Usage: Performance testing, Python API integration

**test_datasets/:**
- Purpose: TPC-H parquet files
- Generated: Yes (by `setup_test_datasets.sh`)
- Committed: No (.gitignored, too large)
- Usage: Benchmark and integration testing

**nsys_profiles/ & reports/:**
- Purpose: GPU profiling outputs from NVIDIA nsys
- Generated: Yes (by performance testing)
- Committed: No
- Usage: Performance analysis and optimization

---

*Structure analysis: 2026-04-02*
