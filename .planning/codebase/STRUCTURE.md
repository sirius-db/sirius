# Codebase Structure

**Analysis Date:** 2026-04-03

## Directory Layout

```
sirius/
├── src/                              # Source code (Super Sirius active engine)
│   ├── sirius_extension.cpp          # Extension entry point & DuckDB functions
│   ├── sirius_interface.cpp          # Query execution interface
│   ├── sirius_engine.cpp             # Query execution engine
│   ├── sirius_context.cpp            # Per-connection context lifecycle
│   ├── sirius_config.cpp             # Configuration management
│   │
│   ├── planner/                      # Physical plan generation
│   │   ├── sirius_physical_plan_generator.cpp    # Main plan builder
│   │   └── sirius_plan_*.cpp                     # Plan builders for each operator type
│   │
│   ├── op/                           # Physical operators (Super Sirius)
│   │   ├── sirius_physical_*.cpp            # 40+ operator implementations
│   │   ├── aggregate/                       # Grouped/ungrouped aggregate operators
│   │   ├── scan/                            # TABLE_SCAN, PARQUET_SCAN, ICEBERG_SCAN, DUCKDB_SCAN
│   │   ├── order/                           # ORDER_BY, TOP_N, MERGE_SORT
│   │   ├── partition/                       # PARTITION operator for parallel processing
│   │   ├── merge/                           # Merge operators for aggregate/sort finalization
│   │   └── result/                          # Result collection operator
│   │
│   ├── pipeline/                     # Pipeline execution framework
│   │   ├── sirius_pipeline.cpp               # Pipeline graph representation
│   │   ├── gpu_pipeline_task.cpp             # Executable task unit
│   │   ├── gpu_pipeline_executor.cpp         # Thread pool executor
│   │   ├── pipeline_executor.cpp             # CPU fallback executor
│   │   └── sirius_meta_pipeline.cpp          # Whole-query pipeline graph
│   │
│   ├── creator/                      # Task creation & scheduling
│   │   └── task_creator.cpp                  # Converts pipelines to executable tasks
│   │
│   ├── downgrade/                    # Memory pressure response
│   │   ├── downgrade_executor.cpp            # Monitors GPU memory, triggers data migration
│   │   └── downgrade_task.cpp                # Individual downgrade task
│   │
│   ├── expression_executor/          # Expression evaluation orchestration
│   │   ├── gpu_expression_executor.cpp       # Dispatches expressions to GPU
│   │   ├── gpu_expression_translator.cpp     # AST → CUDA kernel mapping
│   │   └── specializations/                  # Type-specific expression executors
│   │
│   ├── cuda/                         # CUDA kernels (GPU computation)
│   │   ├── allocator.cu                      # CUDA memory allocation wrappers
│   │   ├── utils.cu                          # GPU utility functions
│   │   ├── cudf/                             # cuDF library wrappers
│   │   │   ├── cudf_join.cu                  # Hash join, nested loop join kernels
│   │   │   ├── cudf_aggregate.cu             # GROUP BY, reduction kernels
│   │   │   ├── cudf_orderby.cu               # Sorting kernels
│   │   │   └── cudf_groupby.cu               # Group aggregation kernels
│   │   ├── operator/                         # Custom operator kernels
│   │   │   ├── hash_join_*.cu                # Join implementation variants
│   │   │   ├── comparison_expression.cu      # Expression evaluation kernels
│   │   │   ├── strings_matching.cu           # String operations
│   │   │   └── nested_loop_join.cu           # Nested loop join kernel
│   │   └── expression_executor/              # Expression dispatch kernels
│   │       ├── gpu_dispatch_select.cu        # WHERE clause evaluation
│   │       ├── gpu_dispatch_materialize.cu   # Column materialization
│   │       └── gpu_dispatch_string.cu        # String function dispatch
│   │
│   ├── data/                         # Data format conversion
│   │   ├── host_parquet_representation.cpp   # Parquet format handling
│   │   └── host_parquet_representation_converters.cpp  # Format converters
│   │
│   ├── memory/                       # Memory management
│   │   ├── sirius_memory_reservation_manager.cpp  # GPU memory reservation tracking
│   │   └── defragmenter_oom_policy.cpp            # OOM response policy
│   │
│   ├── parallel/                     # Parallelization primitives
│   │   └── task_executor.cpp                 # Task execution interface
│   │
│   ├── util/                         # Utilities
│   │   └── *.cpp                             # Helper functions for common operations
│   │
│   ├── legacy/                       # Deprecated code (gpu_processing)
│   │   ├── operator/                         # Legacy DuckDB-based operators
│   │   └── plan/                             # Legacy plan generation
│   │
│   └── include/                      # Header files (mirrors src structure)
│       ├── sirius_*.hpp                      # Main interfaces (engine, interface, extension)
│       ├── op/                               # Operator headers
│       ├── pipeline/                         # Pipeline headers
│       ├── planner/                          # Planner headers
│       ├── cuda/                             # CUDA header declarations
│       ├── expression_executor/              # Expression executor headers
│       ├── downgrade/                        # Downgrade executor headers
│       ├── creator/                          # Task creator headers
│       ├── exec/                             # Execution utilities
│       │   ├── bounded_thread_pool.hpp       # Worker thread pool
│       │   ├── interruptible_mpmc.hpp        # Thread-safe queue
│       │   └── config.hpp                    # Thread pool configuration
│       ├── memory/                           # Memory management headers
│       ├── data/                             # Data conversion headers
│       ├── helper/                           # Helper utilities
│       ├── util/                             # Common utilities
│       ├── log/                              # Logging infrastructure
│       └── legacy/                           # Legacy headers
│
├── test/                             # Test suite
│   ├── cpp/                          # C++ unit tests (Catch2)
│   │   └── *.test.cpp, *.spec.cpp    # Individual test files
│   ├── sql/                          # SQL logic tests (DuckDB format)
│   │   └── *.test                    # SQL test cases
│   └── tpch_performance/             # TPC-H performance tests
│       ├── generate_test_data.py     # Data generation script
│       └── performance_test.py       # Benchmark runner
│
├── docs/                             # Documentation
│   └── super-sirius/                 # Super Sirius architecture docs
│       └── README.md                 # Documentation index
│
├── build/                            # Build artifacts (generated)
│   └── release/                      # Release build
│       ├── extension/sirius/
│       │   ├── sirius.duckdb_extension       # Static extension
│       │   ├── sirius_loadable.duckdb_extension  # Loadable extension
│       │   └── test/cpp/sirius_unittest      # Unit test binary
│       └── test/unittest             # SQL test runner
│
├── Makefile                          # Build automation
├── CMakeLists.txt                    # CMake configuration
├── pixi.toml                         # Pixi environment/dependencies
├── extension_config.cmake            # Extension list
├── .clang-format                     # C++/CUDA formatting rules
├── .clang-tidy                       # C++ linting rules
├── .pre-commit-config.yaml           # Git hooks configuration
├── CLAUDE.md                         # Claude Code instructions
└── README.md                         # Project overview
```

## Directory Purposes

**src/**
- Purpose: All implementation code
- Contains: C++ source (.cpp), CUDA kernels (.cu)
- Key files: Entry points, operator implementations, CUDA kernels

**src/include/**
- Purpose: Header files (public API and internal interfaces)
- Contains: .hpp files mirroring src/ structure
- Key files: Class definitions, function signatures, type definitions

**src/planner/**
- Purpose: Convert DuckDB logical operators to Sirius physical operators
- Contains: Main generator and per-operator plan builders
- Key files: `sirius_physical_plan_generator.cpp` (main), `sirius_plan_*.cpp` (one per operator type)

**src/op/**
- Purpose: Physical operator implementations
- Contains: 40+ operator classes, each handling GPU execution logic
- Key files: `sirius_physical_table_scan.cpp`, `sirius_physical_hash_join.cpp`, `sirius_physical_grouped_aggregate.cpp`
- Subdirectories: Organized by operator family (scan, aggregate, merge, order, partition, result)

**src/op/scan/**
- Purpose: Table scan operators (data source layer)
- Contains: TABLE_SCAN, PARQUET_SCAN, ICEBERG_SCAN, DUCKDB_SCAN, iceberg_metadata_reader
- Key files: Pull data from DuckDB tables and convert to GPU format

**src/op/aggregate/**
- Purpose: Aggregation operators
- Contains: UNGROUPED_AGGREGATE (SUM, AVG, COUNT), GROUPED_AGGREGATE (GROUP BY), MERGE operators
- Key files: `sirius_physical_ungrouped_aggregate.cpp`, `sirius_physical_grouped_aggregate.cpp`

**src/op/order/**
- Purpose: Sorting and TOP-N operators
- Contains: ORDER_BY, TOP_N, MERGE_SORT, SORT_PARTITION, SORT_SAMPLE
- Key files: GPU sort via cuDF

**src/op/merge/**
- Purpose: Finalization operators that merge intermediate results
- Contains: Merge operators for aggregates, sorts, top-N
- Key files: Combine partial results from parallel pipelines

**src/op/partition/**
- Purpose: Repartitioning operator (pipeline breaker)
- Contains: PARTITION operator that splits one pipeline into multiple
- Key files: Routes batches to partition bins for parallel processing

**src/op/result/**
- Purpose: Result collection and output
- Contains: RESULT_COLLECTOR operator (final sink)
- Key files: Gathers all results and formats output

**src/pipeline/**
- Purpose: Pipeline execution infrastructure
- Contains: Pipeline graph, task scheduling, thread pool execution
- Key files: `sirius_pipeline.cpp`, `gpu_pipeline_task.cpp`, `gpu_pipeline_executor.cpp`

**src/creator/**
- Purpose: Task instantiation and scheduling
- Contains: Converts pipelines to executable tasks, manages task dispatch
- Key files: `task_creator.cpp` (main, runs thread pool, creates tasks)

**src/downgrade/**
- Purpose: Memory pressure response and data migration
- Contains: Monitor thread polling GPU memory, task scheduling
- Key files: `downgrade_executor.cpp` (main), `downgrade_task.cpp` (individual migration task)

**src/expression_executor/**
- Purpose: SQL expression evaluation orchestration
- Contains: AST traversal, GPU kernel dispatch, CPU fallback
- Key files: `gpu_expression_executor.cpp`, `gpu_expression_translator.cpp`
- Subdirectories: `specializations/` (type-specific executors), `regex/` (regex matching)

**src/cuda/**
- Purpose: GPU computation kernels
- Contains: ~50 CUDA kernels for joins, aggregates, sorts, expressions
- Key files: `cudf/` (cuDF wrappers), `operator/` (custom kernels), `expression_executor/` (dispatch)

**src/cuda/cudf/**
- Purpose: Wrappers around RAPIDS cuDF library operations
- Contains: Kernels for join, aggregate, sort, duplicate elimination
- Key files: `cudf_join.cu`, `cudf_aggregate.cu`, `cudf_orderby.cu`

**src/cuda/operator/**
- Purpose: Custom CUDA kernels not in cuDF
- Contains: Join variants, expression evaluation, string operations
- Key files: `hash_join_*.cu`, `comparison_expression.cu`, `strings_matching.cu`

**src/cuda/expression_executor/**
- Purpose: Expression evaluation kernel dispatch
- Contains: Type-specific and expression-specific CUDA kernels
- Key files: `gpu_dispatch_select.cu`, `gpu_dispatch_materialize.cu`

**src/data/**
- Purpose: Data format conversion (DuckDB ↔ GPU)
- Contains: Parquet handling, Arrow format, cuDF table builders
- Key files: `host_parquet_representation.cpp`, converters

**src/memory/**
- Purpose: GPU memory lifecycle management
- Contains: Reservation tracking, OOM policies, defragmentation
- Key files: `sirius_memory_reservation_manager.cpp`

**src/parallel/**
- Purpose: Task execution abstraction
- Contains: Task executor interface, thread-safe primitives
- Key files: `task_executor.cpp`

**src/util/**
- Purpose: Helper functions and utilities
- Contains: Common patterns, debugging utilities
- Key files: Various helper implementations

**src/legacy/**
- Purpose: Deprecated execution path (gpu_processing)
- Contains: Old operator and plan implementations
- Note: Do not use for new development; for historical reference only

**test/cpp/**
- Purpose: C++ unit tests
- Contains: Catch2 framework tests for operators, memory, expressions
- Generated: Test binaries in `build/release/extension/sirius/test/cpp/`

**test/sql/**
- Purpose: SQL logic tests (end-to-end)
- Contains: DuckDB SQLLogicTest format
- Run: `make test` or `build/release/test/unittest --test-dir . test/sql/*.test`

**build/**
- Purpose: Build outputs (generated, not in git)
- Contains: Compiled binaries, object files, test outputs
- Key: `sirius.duckdb_extension` (static), `sirius_loadable.duckdb_extension` (loadable)

## Key File Locations

**Entry Points:**
- `src/sirius_extension.cpp`: DuckDB extension registration and table functions (gpu_execution, gpu_processing)
- `src/sirius_interface.cpp`: Query execution orchestration
- `src/sirius_engine.cpp`: GPU execution engine

**Configuration:**
- `src/sirius_config.cpp`: Hardware detection, GPU selection, environment parsing
- `src/include/sirius_config.hpp`: Configuration structure and options
- `.clang-format`: C++/CUDA formatting rules
- `.clang-tidy`: C++ linting rules
- `pixi.toml`: Build environment (CUDA version, dependencies)

**Core Logic:**
- `src/planner/sirius_physical_plan_generator.cpp`: Plan conversion (logical → physical)
- `src/op/sirius_physical_operator.cpp`: Base operator class implementation
- `src/pipeline/sirius_pipeline.cpp`: Pipeline graph construction and execution
- `src/creator/task_creator.cpp`: Task instantiation and scheduling

**Memory Management:**
- `src/memory/sirius_memory_reservation_manager.cpp`: GPU memory tracking
- `src/downgrade/downgrade_executor.cpp`: Memory pressure monitoring
- `src/include/exec/bounded_thread_pool.hpp`: Worker thread pool

**Testing:**
- `test/cpp/`: Unit test directory (one .test.cpp per component)
- `test/sql/`: SQL logic test cases
- `test/tpch_performance/`: TPC-H benchmark scripts

## Naming Conventions

**Files:**
- Source: `sirius_*.cpp` (main engine components), `*.cpp` (operators, utilities)
- Headers: `sirius_*.hpp` (main components), `*.hpp` (mirroring source structure)
- CUDA: `*.cu` (kernels), `*.cuh` (CUDA headers, rarely used in this project)
- Tests: `*.test.cpp` (unit tests), `*.test` (SQL tests)
- Pattern: Name reflects primary class or functionality in file

**Directories:**
- Lowercase with underscores: `src/op/`, `src/pipeline/`, `src/expression_executor/`
- Organize by: Conceptual layer (op, pipeline, planner) or component family (cuda/cudf, cuda/operator)

**Classes:**
- Prefix: `sirius_physical_*` (operators), `sirius_pipeline*` (pipelines), `task_creator`, `downgrade_executor`
- Suffix: None (base classes), `_merge` (merge operators), `_scan` (scan operators)
- Namespace: `sirius` (main), `sirius::op` (operators), `sirius::pipeline` (pipelines), `sirius::planner` (planning)

**Functions:**
- Style: snake_case (`get_operator_state`, `execute_pipeline`, `create_task`)
- Verb-first: `execute_`, `create_`, `build_`, `initialize_`

**Variables:**
- Style: snake_case
- Prefixes: `p_` (pointer parameters), `_` (private/internal members)
- Suffixes: `_count`, `_idx`, `_ids`, `_map` (for containers)

## Where to Add New Code

**New GPU Operator:**
1. Header: `src/include/op/sirius_physical_<operator>.hpp` (subclass `sirius_physical_operator`)
2. Implementation: `src/op/sirius_physical_<operator>.cpp` (implement `execute()`, `get_global_sink_state()`, etc.)
3. CUDA kernels: `src/cuda/operator/<operator>.cu` (if custom logic needed; use cuDF if available)
4. Plan builder: `src/planner/sirius_plan_<operator>.cpp` (handle DuckDB logical operator → physical conversion)
5. Registration: Add to `SiriusPhysicalOperatorType` enum in `src/include/op/sirius_physical_operator_type.hpp`
6. Tests: `test/cpp/operator/test_<operator>.cpp` (unit tests), `test/sql/tpch-sirius.test` (SQL tests)

**New Expression Function (e.g., string operation):**
1. GPU kernel: `src/cuda/expression_executor/gpu_dispatch_*.cu` (add case for new function)
2. Dispatcher: Update `gpu_expression_translator.cpp` to route function to new kernel
3. Tests: Add to expression executor tests in `test/cpp/`

**New Memory Optimization:**
1. Policy: `src/memory/` (implement new allocation/eviction strategy)
2. Manager update: `src/memory/sirius_memory_reservation_manager.cpp` (integrate)
3. Tests: `test/cpp/memory/` (memory allocation tests)

**New Configuration Option:**
1. Definition: `src/include/sirius_config.hpp` (add to config struct)
2. Parsing: `src/sirius_config.cpp` (parse from env var or config file)
3. Usage: Reference in appropriate operator or executor

**Utilities & Helpers:**
- Shared helpers: `src/include/helper/`, `src/util/`
- Type conversions: `src/include/helper/types.hpp`
- Math/numeric helpers: `src/cuda/utils.cu`

## Special Directories

**build/**
- Purpose: Build artifacts
- Generated: Yes (cleaned by `rm -rf build`)
- Committed: No (in .gitignore)
- Key outputs: `build/release/extension/sirius/sirius.duckdb_extension`, `build/release/extension/sirius/test/cpp/sirius_unittest`

**log/**
- Purpose: Runtime logs
- Generated: Yes (populated during query execution)
- Committed: No (in .gitignore)
- Config: `SIRIUS_LOG_DIR` env var controls location

**test_datasets/**
- Purpose: TPC-H Parquet data for benchmarking
- Generated: Yes (by `test/tpch_performance/generate_test_data.py`)
- Committed: No (in .gitignore)
- Sizes: `tpch_parquet_sf100/`, `tpch_parquet_sf300/` etc.

**docs/super-sirius/**
- Purpose: Architecture documentation
- Generated: No (hand-written)
- Committed: Yes
- Contains: Design docs, algorithm descriptions, optimization guide

**extension-ci-tools/**
- Purpose: DuckDB extension build infrastructure (submodule)
- Generated: No (submodule)
- Committed: Yes
- Contains: Makefile templates, CMake helpers for extension builds

---

*Structure analysis: 2026-04-03*
