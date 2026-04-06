# Codebase Structure

**Analysis Date:** 2026-04-06

## Directory Layout

```
sirius/
├── src/
│   ├── sirius_extension.cpp           # DuckDB extension entry point, table function registration
│   ├── sirius_interface.cpp           # Query execution interface, bridges DuckDB → GPU execution
│   ├── sirius_engine.cpp              # Query executor, pipeline scheduling and execution
│   ├── sirius_context.cpp             # Per-connection context for GPU state
│   ├── config.cpp                     # Runtime configuration (GPU policies, batch sizes)
│   ├── gpu_buffer_manager.cpp         # GPU memory allocation and management
│   ├── cpu_cache.cpp                  # CPU-side caching layer
│   ├── fallback.cpp                   # Fallback detection (unsupported ops/types)
│   ├── extension_lock.cpp             # Synchronization for extension initialization
│   ├── sirius_config.cpp              # Configuration variable bindings
│   │
│   ├── include/                       # All public headers mirror src/ structure
│   │   ├── sirius_interface.hpp       # Query interface class
│   │   ├── sirius_engine.hpp          # Query executor class
│   │   ├── config.hpp                 # Config constants and flags
│   │   ├── fallback.hpp               # Fallback checker interface
│   │   ├── helper/                    # Utility headers (types, helpers)
│   │   ├── log/                       # Logging infrastructure
│   │   ├── util/                      # Misc utilities
│   │   ├── op/                        # Physical operator headers
│   │   ├── planner/                   # Planning headers
│   │   ├── pipeline/                  # Pipeline orchestration headers
│   │   ├── expression_executor/       # Expression evaluation headers
│   │   ├── data/                      # Data representation headers
│   │   ├── memory/                    # Memory management headers
│   │   ├── exec/                      # Execution infrastructure (threads, queues)
│   │   ├── creator/                   # Task creation headers
│   │   ├── downgrade/                 # CPU fallback headers
│   │   ├── cudf/                      # cuDF utility headers
│   │   ├── parallel/                  # Task and queue headers
│   │   └── legacy/                    # Legacy GPU engine (deprecated)
│   │
│   ├── planner/                       # Planning logic
│   │   ├── sirius_physical_plan_generator.cpp  # Main plan generator
│   │   ├── query.cpp                           # Query preprocessing
│   │   ├── sirius_plan_*.cpp                   # Operator-specific builders (filter, aggregate, join, etc.)
│   │   └── [12 operator-specific plan files]
│   │
│   ├── op/                            # Physical operators
│   │   ├── sirius_physical_operator.cpp        # Base operator class
│   │   ├── sirius_physical_operator_type.cpp   # Operator type enumeration
│   │   ├── sirius_physical_filter.cpp          # Filter operator
│   │   ├── sirius_physical_projection.cpp      # Projection operator
│   │   ├── sirius_physical_hash_join.cpp       # Hash join operator
│   │   ├── sirius_physical_nested_loop_join.cpp
│   │   ├── sirius_physical_delim_join.cpp      # Delimited join operator
│   │   ├── sirius_physical_grouped_aggregate.cpp
│   │   ├── sirius_physical_grouped_aggregate_merge.cpp
│   │   ├── sirius_physical_ungrouped_aggregate.cpp
│   │   ├── sirius_physical_ungrouped_aggregate_merge.cpp
│   │   ├── sirius_physical_table_scan.cpp      # DuckDB table scan
│   │   ├── sirius_physical_parquet_scan.cpp    # Parquet file scan
│   │   ├── sirius_physical_iceberg_scan.cpp    # Iceberg table scan
│   │   ├── sirius_physical_duckdb_scan.cpp     # Intermediate DuckDB result scan
│   │   ├── sirius_physical_column_data_scan.cpp # CTE column data scan
│   │   ├── sirius_physical_order.cpp           # Order by operator
│   │   ├── sirius_physical_partition.cpp       # Partition for sort/join
│   │   ├── sirius_physical_sort_partition.cpp
│   │   ├── sirius_physical_sort_sample.cpp
│   │   ├── sirius_physical_merge_sort.cpp      # Merge multiple sorted streams
│   │   ├── sirius_physical_top_n.cpp           # Top-N operator
│   │   ├── sirius_physical_top_n_merge.cpp
│   │   ├── sirius_physical_limit.cpp           # Limit operator
│   │   ├── sirius_physical_cte.cpp             # Common table expression
│   │   ├── sirius_physical_result_collector.cpp # Final sink, materializes results
│   │   ├── sirius_physical_dummy_scan.cpp      # Empty source
│   │   ├── sirius_physical_empty_result.cpp    # Empty result
│   │   ├── sirius_physical_concat.cpp          # Union/concatenation
│   │   ├── sirius_physical_partition_consumer_operator.cpp
│   │   └── scan/                      # Scan-specific infrastructure
│   │       ├── duckdb_scan_task.cpp
│   │       ├── duckdb_scan_executor.cpp
│   │       ├── parquet_scan_task.cpp
│   │       ├── iceberg_scan_task.cpp
│   │       ├── iceberg_metadata_reader.cpp
│   │       ├── iceberg_delete_pipeline.cpp
│   │       ├── iceberg_avro_reader.cpp
│   │       ├── equality_delete_filter.cpp
│   │       ├── positional_delete_filter.cpp
│   │       ├── prefetched_data_source.cpp
│   │       └── cached_ranges.cpp
│   │
│   ├── pipeline/                      # Execution orchestration
│   │   ├── sirius_pipeline.cpp        # Single source-sink pipeline
│   │   ├── sirius_meta_pipeline.cpp   # Multiple pipelines with same sink
│   │   ├── pipeline_executor.cpp      # CPU pipeline executor
│   │   ├── gpu_pipeline_executor.cpp  # GPU pipeline executor
│   │   ├── gpu_pipeline_task.cpp      # GPU task wrapper
│   │   ├── gpu_pipeline_queue.cpp     # GPU task queue
│   │   ├── pipeline_queue.cpp         # CPU task queue
│   │   ├── task_request.cpp           # Task request descriptor
│   │   └── [headers in include/pipeline/]
│   │
│   ├── expression_executor/           # SQL expression evaluation
│   │   ├── gpu_expression_executor.cpp     # Main executor (filter, project)
│   │   ├── gpu_expression_translator.cpp   # AST → cuDF kernel dispatch
│   │   ├── gpu_expression_executor_state.cpp
│   │   └── regex/                     # Regular expression support
│   │       └── regex_playground.hpp
│   │
│   ├── cuda/                          # GPU kernels
│   │   ├── allocator.cu               # RMM allocator setup
│   │   ├── utils.cu                   # GPU utility functions
│   │   ├── communication.cu           # GPU-CPU data transfer
│   │   ├── print.cu                   # GPU memory debugging
│   │   └── operator/                  # Specialized kernels
│   │       ├── hash_join_inner.cu     # Inner hash join kernel
│   │       ├── hash_join_single.cu    # Single-probe hash join
│   │       ├── hash_join_right.cu     # Right hash join kernel
│   │       ├── nested_loop_join.cu    # Nested loop join kernel
│   │       ├── comparison_expression.cu    # Comparison operators
│   │       ├── arbitrary_expression.cu    # Complex expressions
│   │       ├── strings_matching.cu    # String matching
│   │       ├── substring.cu           # Substring extraction
│   │       ├── strlen_from_offsets.cu
│   │       ├── empty_str_check.cu
│   │       └── materialize.cu         # Result materialization
│   │
│   ├── creator/                       # Task creation from operators
│   │   └── task_creator.cpp
│   │
│   ├── downgrade/                     # CPU fallback execution
│   │   ├── downgrade_executor.cpp     # Execute as DuckDB CPU operator
│   │   └── downgrade_task.cpp         # Task wrapper for downgrade
│   │
│   └── legacy/                        # Legacy gpu_processing (deprecated)
│       ├── gpu_executor.cpp
│       ├── gpu_context.cpp
│       ├── gpu_physical_plan_generator.cpp
│       ├── gpu_meta_pipeline.cpp
│       ├── gpu_pipeline.cpp
│       ├── gpu_table_function.cpp
│       ├── gpu_query_result.cpp
│       ├── gpu_pipeline_hashmap.cpp
│       └── operator/                  # Legacy operators (deprecated)
│
├── test/
│   ├── cpp/                           # C++ unit tests (Catch2)
│   │   ├── [test files by component]
│   │   └── log/                       # Unit test logs
│   ├── sql/                           # SQL logic tests
│   │   ├── tpch-sirius.test
│   │   └── [other SQL test files]
│   └── tpch_performance/              # TPC-H performance benchmarks
│       ├── performance_test.py
│       └── generate_test_data.py
│
├── docs/
│   ├── super-sirius/                  # Super Sirius architecture documentation
│   │   └── README.md                  # Index and reading order
│   └── logos/                         # Brand assets
│
├── CMakeLists.txt                     # Main build configuration
├── extension_config.cmake             # Extension manifest
├── Makefile                           # Build wrapper
├── pixi.toml                          # Development environment (Pixi)
├── .clang-format                      # C++ formatting rules
├── .clang-tidy                        # C++ linting rules
├── .pre-commit-config.yaml            # Pre-commit hooks
├── .codespell_words                   # Custom spell-check dictionary
├── CLAUDE.md                          # This codebase's Claude guidelines
└── README.md                          # Project overview
```

## Directory Purposes

**`src/`**
- Purpose: All source code (C++ implementations and CUDA kernels)
- Contains: Extension logic, operators, planning, execution, utilities
- Key files: Extension entry point, main executor, operator implementations

**`src/include/`**
- Purpose: All public headers, mirrors `src/` structure
- Contains: Class definitions, type declarations, function signatures
- Convention: Each `.cpp` file has corresponding `.hpp` in `include/`

**`src/planner/`**
- Purpose: Logical→physical plan translation
- Contains: Plan generator, operator-specific builders
- Key files: `sirius_physical_plan_generator.cpp` (dispatcher), `sirius_plan_*.cpp` (builders)

**`src/op/`**
- Purpose: Physical operator implementations
- Contains: Filter, projection, joins, aggregation, scans, results
- Pattern: One file per operator type, all inherit from `sirius_physical_operator`
- Subdirectory `scan/`: Infrastructure for various data source types

**`src/op/scan/`**
- Purpose: Scan operator task management
- Contains: Task creation for table scans, parquet, iceberg
- Handles: Metadata reading, delete filters, data prefetching

**`src/op/aggregate/`**
- Purpose: Grouping and aggregation utilities
- Contains: Aggregate implementation helpers, utility functions
- Used by: Grouped and ungrouped aggregate operators

**`src/op/merge/` and `src/op/order/` and `src/op/partition/`**
- Purpose: Specialized operator implementations
- Contains: Merge sort, order (top-n, limit), partition logic
- Note: May be hidden in include structure; these are utility implementations

**`src/pipeline/`**
- Purpose: Pipeline graph construction and execution
- Contains: Pipeline definition, meta-pipeline hierarchies, task executors
- Key files: `sirius_pipeline.cpp` (single pipeline), `sirius_meta_pipeline.cpp` (multi-pipeline graph)

**`src/expression_executor/`**
- Purpose: SQL expression evaluation on GPU
- Contains: Expression AST traversal, cuDF kernel dispatch, type casting
- Used by: Filter, projection, join condition operators

**`src/cuda/`**
- Purpose: GPU kernels
- Contains: Specialized operations impossible or inefficient in C++ (joins, aggregation)
- Note: `.cu` files compiled by NVCC to GPU object code

**`src/creator/`**
- Purpose: Convert operators to executable tasks
- Contains: Task creation logic, parallelism decisions
- Used by: Pipeline executor to generate work

**`src/downgrade/`**
- Purpose: CPU fallback when GPU unavailable
- Contains: Downgrade task wrapper, DuckDB execution delegation
- Used by: Execution layer when OOM or unsupported type detected

**`src/legacy/`**
- Purpose: Deprecated gpu_processing code path (old execution engine)
- Status: Kept for compatibility, all new development targets Super Sirius
- Do not modify: Unless specifically maintaining legacy mode

**`test/cpp/`**
- Purpose: C++ unit tests using Catch2 framework
- Structure: Mirrors `src/` structure (e.g., `test/cpp/op/` tests `src/op/`)
- Logs: Test output written to `build/release/extension/sirius/test/cpp/log`

**`test/sql/`**
- Purpose: End-to-end SQL logic tests
- Format: DuckDB SQL Logic Test files (`.test`)
- Run via: `build/release/test/unittest --test-dir . test/sql/tpch-sirius.test`

**`test/tpch_performance/`**
- Purpose: Performance benchmarking
- Contains: Scale factor data generation, query execution, result comparison
- Run via: Python scripts with built duckdb-python

**`docs/super-sirius/`**
- Purpose: Architecture documentation for Super Sirius
- Contains: Design docs, module descriptions, API references
- Read order: See `README.md` in directory

## Key File Locations

**Entry Points:**
- `src/sirius_extension.cpp`: DuckDB extension registration and `gpu_execution` table function
- `src/sirius_interface.cpp`: Query execution entry point (`sirius_execute_query()`)
- `src/sirius_engine.cpp`: Pipeline execution entry point (`execute()`)
- `src/planner/sirius_physical_plan_generator.cpp`: Planning entry point (`create_plan()`)

**Configuration:**
- `src/config.cpp` / `src/include/config.hpp`: Runtime flags (memory policies, batch sizes, debug options)
- `src/sirius_config.cpp`: Configuration variable bindings to DuckDB settings
- `pixi.toml`: Development environment setup and dependencies
- `CMakeLists.txt`: Build configuration, compiler flags, CUDA settings
- `extension_config.cmake`: Extension manifest (which extensions to load)

**Core Logic:**
- `src/op/sirius_physical_operator.cpp` / `src/include/op/sirius_physical_operator.hpp`: Base operator class and pipeline building
- `src/pipeline/sirius_meta_pipeline.cpp`: Pipeline graph construction and dependency resolution
- `src/sirius_engine.cpp`: Query execution and task scheduling
- `src/expression_executor/gpu_expression_executor.cpp`: Expression evaluation dispatcher

**Testing:**
- `test/cpp/`: Unit tests (grep for test files by component)
- `test/sql/tpch-sirius.test`: SQL logic tests for TPC-H queries
- `test/tpch_performance/performance_test.py`: Benchmark runner

## Naming Conventions

**Files:**
- `sirius_physical_*.cpp`: Physical operator implementations
- `sirius_plan_*.cpp`: Planning logic for specific operators
- `gpu_*.cpp`: GPU-related infrastructure (legacy mostly; new code uses sirius_ prefix)
- `*.cu`: CUDA kernel files

**Directories:**
- `include/`: Public headers (mirror src/ structure)
- `op/`: Physical operators
- `planner/`: Planning logic
- `pipeline/`: Execution orchestration
- `expression_executor/`: Expression evaluation
- `cuda/`: GPU kernels
- `legacy/`: Deprecated code
- `scan/`: Scan-specific infrastructure

**Classes/Types:**
- `sirius_physical_operator`: Base physical operator
- `sirius_pipeline`: Single source-sink pipeline
- `sirius_meta_pipeline`: Multi-pipeline graph
- `sirius_engine`: Query executor
- `sirius_interface`: Query interface
- `sirius_context`: Per-connection context
- All in `namespace sirius` (planning) or `namespace sirius::op` (operators)

**Functions:**
- `create_plan()`: Planning dispatch (multiple overloads per operator type)
- `execute()`: Operator execution (takes input data, CUDA stream)
- `get_global_sink_state()`: Allocate sink state (for aggregation, join build)
- `get_local_sink_state()`: Per-thread sink state
- `is_source()`, `is_sink()`: Operator type checking

## Where to Add New Code

**New GPU Operator:**
1. Header: `src/include/op/sirius_physical_MY_OP.hpp` (declare class inheriting from `sirius_physical_operator`)
2. Implementation: `src/op/sirius_physical_MY_OP.cpp` (implement `execute()`, state methods)
3. CUDA kernels (if needed): `src/cuda/operator/my_op.cu`
4. Planning: `src/planner/sirius_plan_my_op.cpp` (create physical operator from logical)
5. Registration: Add case in `sirius_physical_plan_generator::create_plan(LogicalOperator&)` switch statement
6. Tests: `test/cpp/op/sirius_physical_my_op_test.cpp` (unit tests) and SQL tests in `test/sql/`

**New Expression Type:**
1. Handler: Add case in `src/expression_executor/gpu_expression_translator.cpp` to dispatch to cuDF kernel
2. Kernel (if needed): `src/cuda/operator/my_expression.cu`
3. Tests: `test/cpp/expression_executor/`

**Bug Fix or Small Enhancement:**
1. Locate affected operator or module in `src/`
2. Update implementation in `.cpp` and `.hpp` as needed
3. Update unit tests in `test/cpp/` if logic changes
4. Add SQL test in `test/sql/` if user-visible behavior changes

**Utility Function:**
- Shared code: `src/helper/` (helpers, types, utils)
- Stream management: `src/util/` (CUDA stream wrappers)
- Memory: `src/include/memory/` (memory management utilities)

**Integration with New DuckDB Feature:**
1. Update `sirius_extension.cpp` if new callbacks needed
2. Add planning in `sirius_physical_plan_generator.cpp` or throw `NotImplementedException` for fallback
3. If GPU acceleration is desired, implement new operator

## Special Directories

**`build/`:**
- Purpose: Build output directory (generated, not committed)
- Contains: Compiled binaries, object files, test logs
- Key artifacts: `build/release/extension/sirius/sirius.duckdb_extension` (loadable extension)
- Test logs: `build/release/extension/sirius/test/cpp/log/`

**`.claude/`:**
- Purpose: Claude Code configuration and skills
- Contains: Skills for profiling, dataset management, benchmarking
- Status: Auto-generated, checked in
- Skills: `/profile-analyzer`, `/dataset-manager`, `/tpcds-benchmark`, `/module-context`

**`log/`:**
- Purpose: Runtime logs directory (generated, not committed)
- Contains: Sirius execution logs if `SIRIUS_LOG_DIR` not set
- Controlled by: `SIRIUS_LOG_LEVEL` environment variable

---

*Structure analysis: 2026-04-06*
