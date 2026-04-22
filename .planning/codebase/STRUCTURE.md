# Codebase Structure

**Analysis Date:** 2026-04-21

## Directory Layout

```
project-root/
├── src/                          # Primary source code
│   ├── sirius_extension.cpp      # Extension registration, table function entry points
│   ├── sirius_interface.cpp      # DuckDB-facing API, query execution wrapper
│   ├── sirius_engine.cpp         # Pipeline construction and execution orchestration
│   ├── sirius_context.cpp        # Context initialization and lifecycle
│   ├── config.cpp                # Global config and parameters
│   │
│   ├── planner/                  # Logical-to-physical plan translation
│   │   ├── sirius_physical_plan_generator.cpp   # Main dispatcher for all operators
│   │   ├── sirius_plan_filter.cpp               # LOGICAL_FILTER → FILTER
│   │   ├── sirius_plan_projection.cpp           # LOGICAL_PROJECTION → PROJECTION
│   │   ├── sirius_plan_aggregate.cpp            # LOGICAL_AGGREGATE → HASH_GROUP_BY / UNGROUPED_AGGREGATE
│   │   ├── sirius_plan_comparison_join.cpp      # LOGICAL_COMPARISON_JOIN → HASH_JOIN / NESTED_LOOP_JOIN
│   │   ├── sirius_plan_order.cpp                # LOGICAL_ORDER_BY → ORDER_BY
│   │   ├── sirius_plan_top_n.cpp                # LOGICAL_TOP_N → TOP_N
│   │   ├── sirius_plan_limit.cpp                # LOGICAL_LIMIT → STREAMING_LIMIT
│   │   ├── sirius_plan_get.cpp                  # LOGICAL_GET → TABLE_SCAN
│   │   ├── sirius_plan_cte.cpp                  # LOGICAL_MATERIALIZED_CTE → CTE
│   │   ├── sirius_plan_column_data_get.cpp      # LOGICAL_CHUNK_GET / EXPRESSION_GET → COLUMN_DATA_SCAN
│   │   ├── sirius_plan_delim_get.cpp            # LOGICAL_DELIM_GET → DELIM_SCAN
│   │   ├── sirius_plan_delim_join.cpp           # Delim join handling
│   │   ├── sirius_plan_recursive_cte.cpp        # LOGICAL_CTE_REF → CTE_SCAN
│   │   ├── sirius_plan_dummy_scan.cpp           # LOGICAL_DUMMY_SCAN → DUMMY_SCAN
│   │   ├── sirius_plan_empty_result.cpp         # LOGICAL_EMPTY_RESULT → EMPTY_RESULT
│   │   └── query.cpp                            # Query helper utilities
│   │
│   ├── op/                       # Physical operator implementations (CPU-side orchestration)
│   │   ├── sirius_physical_operator.cpp         # Base class implementation
│   │   ├── sirius_physical_filter.cpp           # FILTER operator (calls gpu_expression_executor for filtering)
│   │   ├── sirius_physical_projection.cpp       # PROJECTION operator
│   │   ├── sirius_physical_hash_join.cpp        # HASH_JOIN operator (cuDF joins)
│   │   ├── sirius_physical_nested_loop_join.cpp # NESTED_LOOP_JOIN operator
│   │   ├── sirius_physical_grouped_aggregate.cpp    # Grouped aggregation (cuDF groupby)
│   │   ├── sirius_physical_grouped_aggregate_merge.cpp # Merge step for grouped agg
│   │   ├── sirius_physical_ungrouped_aggregate.cpp   # Ungrouped aggregation
│   │   ├── sirius_physical_ungrouped_aggregate_merge.cpp # Merge for ungrouped agg
│   │   ├── sirius_physical_table_scan.cpp      # TABLE_SCAN operator (scan source)
│   │   ├── sirius_physical_parquet_scan.cpp    # PARQUET_SCAN operator
│   │   ├── sirius_physical_iceberg_scan.cpp    # Iceberg table scan
│   │   ├── sirius_physical_duckdb_scan.cpp     # DuckDB table scan (CPU fallback source)
│   │   ├── sirius_physical_order.cpp           # ORDER_BY operator
│   │   ├── sirius_physical_merge_sort.cpp      # Merge sort (partial order merging)
│   │   ├── sirius_physical_sort_partition.cpp  # Partition for sorting
│   │   ├── sirius_physical_sort_sample.cpp     # Sample for sort planning
│   │   ├── sirius_physical_top_n.cpp           # TOP_N operator
│   │   ├── sirius_physical_partition.cpp       # PARTITION operator (pipeline breaker)
│   │   ├── sirius_physical_concat.cpp          # CONCAT operator (merges partitions)
│   │   ├── sirius_physical_limit.cpp           # STREAMING_LIMIT operator
│   │   ├── sirius_physical_result_collector.cpp # RESULT_COLLECTOR (sink)
│   │   ├── sirius_physical_cte.cpp             # CTE materialization operator
│   │   ├── sirius_physical_delim_join.cpp      # Delim join operators
│   │   ├── sirius_physical_empty_result.cpp    # EMPTY_RESULT (no-op sink)
│   │   ├── sirius_physical_column_data_scan.cpp # COLUMN_DATA_SCAN (internal sources)
│   │   ├── sirius_physical_cpu_source.cpp      # CPU_SOURCE (DuckDB scan source)
│   │   ├── sirius_physical_dummy_scan.cpp      # DUMMY_SCAN source
│   │   ├── sirius_physical_operator_type.cpp   # Operator type utilities
│   │   ├── sirius_physical_partition_consumer_operator.cpp
│   │   │
│   │   ├── scan/                # Table scanning and data source operators
│   │   │   ├── duckdb_scan_executor.cpp        # DuckDB table scan execution
│   │   │   ├── duckdb_scan_task.cpp            # DuckDB scan task (pulls rows from table)
│   │   │   ├── parquet_scan_task.cpp           # Parquet file scan task
│   │   │   ├── cpu_source_task.cpp             # Generic CPU source task
│   │   │   ├── scan_utils.cpp                  # Common scan utilities
│   │   │   ├── cached_ranges.cpp               # Scan result caching
│   │   │   ├── prefetched_data_source.cpp      # Pre-fetched data source
│   │   │   ├── equality_delete_filter.cpp      # Iceberg equality delete handling
│   │   │   ├── positional_delete_filter.cpp    # Iceberg positional delete handling
│   │   │   ├── iceberg_scan_task.cpp           # Iceberg table scan
│   │   │   ├── iceberg_metadata_reader.cpp     # Iceberg metadata (delete files)
│   │   │   └── iceberg_avro_reader.cpp         # Iceberg Avro deserializer
│   │   │
│   │   ├── aggregate/          # Aggregation implementation details
│   │   │   ├── gpu_aggregate_impl.cpp          # cuDF aggregation wrapper
│   │   │   └── aggregate_op_util.cpp           # Aggregate utilities (e.g., AVG decomposition)
│   │   │
│   │   ├── partition/          # Partitioning for distributed execution
│   │   │   └── gpu_partition_impl.cpp          # cuDF partitioning (hash partition)
│   │   │
│   │   ├── order/              # Sorting and ordering implementation
│   │   │   └── gpu_order_impl.cpp              # cuDF sort/order_by wrapper
│   │   │
│   │   ├── merge/              # Merge operations
│   │   │   └── gpu_merge_impl.cpp              # Merge rows across partitions
│   │   │
│   │   └── result/             # Result collection and formatting
│   │       └── host_table_chunk_reader.cpp     # Read results from GPU tables to DuckDB chunks
│   │
│   ├── cuda/                    # GPU kernels and cuDF wrappers
│   │   ├── allocator.cu         # GPU memory allocator integration
│   │   ├── utils.cu             # CUDA utility kernels
│   │   ├── communication.cu      # GPU-Host communication utilities
│   │   ├── print.cu             # GPU table debugging output
│   │   │
│   │   ├── cudf/                # cuDF library wrappers
│   │   │   ├── cudf_aggregate.cu        # groupby(), aggregate() wrappers
│   │   │   ├── cudf_groupby.cu          # groupby-specific logic
│   │   │   ├── cudf_join.cu             # join() wrappers (hash, nested loop)
│   │   │   ├── cudf_orderby.cu          # sort_by_key() wrappers
│   │   │   ├── cudf_duplicate_elimination.cu # Duplicate removal for joins
│   │   │   └── cudf_utils.cu            # General cuDF utilities
│   │   │
│   │   ├── operator/            # Custom GPU kernels and operator implementations
│   │   │   ├── comparison_expression.cu # GPU expression evaluation (comparisons)
│   │   │   ├── arbitrary_expression.cu  # General expression evaluation on GPU
│   │   │   ├── hash_join_single.cu      # Optimized hash join single partition
│   │   │   ├── hash_join_right.cu       # Right side hash join
│   │   │   ├── nested_loop_join.cu      # Nested loop join kernel
│   │   │   ├── substring.cu             # Substring operation on GPU
│   │   │   ├── strings_matching.cu      # Pattern matching on strings
│   │   │   └── unused/                  # Legacy/experimental operators
│   │   │
│   │   └── iceberg/             # Iceberg-specific GPU operations
│   │       └── equality_delete_mask.cu  # Mask generation for deleted rows
│   │
│   ├── pipeline/               # Pipeline construction and execution
│   │   ├── sirius_pipeline_converter.cpp    # Physical plan → pipeline conversion
│   │   ├── sirius_plan_printer.cpp          # Debug printing of plans
│   │   └── (headers in src/include/pipeline/)
│   │
│   ├── creator/                # Task creation based on data availability
│   │   └── (main logic in src/include/creator/)
│   │
│   ├── transparent/            # Transparent GPU execution (auto-interception)
│   │   ├── sirius_optimizer_extension.cpp   # Optimizer hook for automatic interception
│   │   └── physical_sirius_execution.cpp    # Transparent execution wrapper
│   │
│   ├── downgrade/              # Memory pressure handling and spilling
│   │   └── downgrade_executor.cpp           # GPU→Host spilling logic
│   │
│   ├── parallel/               # Thread pool and task execution
│   │   └── task_executor.cpp                # Generic task executor for threads
│   │
│   ├── memory/                 # Memory management specifics
│   │   └── (wrapper implementations, main in include/)
│   │
│   ├── data/                   # Data conversion and batching
│   │   ├── host_parquet_representation.cpp      # Parquet data conversion
│   │   ├── host_parquet_representation_converters.cpp
│   │   └── (data batch converters)
│   │
│   ├── helper/                 # Utility functions
│   │   ├── type_conversions.cpp       # DuckDB ↔ Sirius type mapping
│   │   └── (other utilities)
│   │
│   ├── expression_executor/    # GPU expression evaluation
│   │   ├── gpu_expression_executor.cpp       # Main expression executor
│   │   ├── expression_executor_strategy.cpp  # AST vs materialized strategies
│   │   └── specializations/                  # Optimized paths for common operations
│   │
│   └── include/                # All header files (mirrors src/ structure)
│       ├── sirius_extension.hpp
│       ├── sirius_interface.hpp
│       ├── sirius_engine.hpp
│       ├── sirius_context.hpp
│       ├── config.hpp
│       ├── planner/
│       ├── op/
│       ├── cuda/
│       ├── pipeline/
│       ├── creator/
│       ├── transparent/
│       ├── downgrade/
│       ├── memory/
│       ├── data/
│       ├── expression_executor/
│       ├── helper/
│       ├── log/
│       ├── util/
│       ├── common/
│       ├── exec/
│       ├── legacy/                # Legacy GPU execution path (deprecated)
│       └── operator/              # Legacy operator definitions
│
├── test/                        # Testing infrastructure
│   ├── cpp/                     # C++ unit tests (Catch2)
│   │   ├── planner/             # Plan generator tests
│   │   ├── operator/            # Operator execution tests
│   │   ├── pipeline/            # Pipeline construction/execution tests
│   │   ├── creator/             # Task creator tests
│   │   ├── scan/                # Scan operator tests
│   │   ├── data/                # Data conversion tests
│   │   ├── memory_management/   # Memory management tests
│   │   ├── memory/              # Cache tests
│   │   ├── downgrade/           # Spilling/downgrade tests
│   │   ├── expression_executor/ # Expression evaluation tests
│   │   ├── integration/         # End-to-end tests (TPCH, TPCDS)
│   │   ├── config/              # Configuration tests
│   │   ├── debug/               # Debugging utilities
│   │   ├── helper/              # Helper function tests
│   │   ├── exec/                # Executor tests
│   │   ├── parallel/            # Thread pool tests
│   │   └── unittest.cpp         # Main test runner
│   │
│   ├── sql/                     # SQL logic tests
│   │   ├── tpch-sirius.test     # TPC-H query tests
│   │   └── (other SQL test files)
│   │
│   ├── tpch_performance/        # TPC-H performance benchmarking
│   │   ├── generate_test_data.py
│   │   └── performance_test.py
│   │
│   ├── tpcds_performance/       # TPC-DS performance benchmarking
│   │   └── (similar to tpch_performance)
│   │
│   └── answers/                 # Expected query results
│       └── tpch/
│
├── docs/                        # Documentation
│   └── super-sirius/            # Comprehensive architecture docs
│       ├── README.md            # Index and reading guide
│       ├── architecture-overview.md
│       ├── physical-plan-generation.md
│       ├── pipeline-execution.md
│       ├── task-creator.md
│       ├── operators.md
│       ├── scan.md
│       ├── memory-management.md
│       ├── data-management.md
│       ├── expression-executor.md
│       ├── optimizations.md
│       ├── configuration.md
│       └── execution-flow.md
│
├── Makefile                     # Build system (thin wrapper)
├── extension_config.cmake       # CMake config specifying extensions to load
├── .clang-format                # C++ code formatting rules
├── .clang-tidy                  # C++ linting rules
├── .pre-commit-config.yaml      # Git hooks for code quality
├── .codespell_words             # Custom dictionary for spell checking
└── pixi.toml                    # Environment setup (Pixi/conda)
```

## Directory Purposes

**src/**
- Purpose: All primary source code for Sirius
- Contains: Extension setup, query execution, operators, GPU kernels, pipeline logic

**src/planner/**
- Purpose: Logical plan → physical plan translation; operator-specific builder implementations
- Contains: `sirius_physical_plan_generator.cpp` dispatcher and 15+ operator builder files
- Key files: `sirius_plan_*.cpp` for each DuckDB operator type

**src/op/**
- Purpose: CPU-side orchestration of GPU operators; defines how each operation executes
- Contains: ~30 operator implementations + scan subdirectory with 10+ scan variants
- Key files: `sirius_physical_*.cpp` for each operator type

**src/cuda/**
- Purpose: GPU kernels and cuDF wrappers; actual GPU computation happens here
- Contains: cuDF aggregation/join/sort wrappers, custom CUDA kernels, operator specializations
- Key files: `cudf_*.cu` for main operations, `operator/*.cu` for custom kernels

**src/pipeline/**
- Purpose: Pipeline construction and execution coordination
- Contains: Physical plan → pipeline conversion, pipeline executor stubs
- Key files: `sirius_pipeline_converter.cpp`

**src/include/**
- Purpose: All header files; mirrors src/ directory structure for easy navigation
- Contains: Class definitions, inline implementations, forward declarations
- Pattern: `src/include/X/file.hpp` corresponds to `src/X/file.cpp`

**test/cpp/**
- Purpose: C++ unit tests organized by component
- Contains: ~50 test files covering planner, operators, pipelines, memory, scanning, data conversion
- Key files: `unittest.cpp` (main runner), per-component subdirectories

**test/sql/**
- Purpose: SQL logic tests; queries executed end-to-end with result validation
- Contains: TPC-H queries, TPC-DS queries, answer files for validation
- Key files: `tpch-sirius.test`

**docs/super-sirius/**
- Purpose: Comprehensive documentation of Sirius architecture and implementation
- Contains: 14 markdown files covering all major subsystems
- Key files: `README.md` (index), `architecture-overview.md`, `physical-plan-generation.md`

## Key File Locations

**Entry Points:**
- `src/sirius_extension.cpp`: Extension registration, table function bindings
- `src/sirius_interface.cpp`: Query execution wrapper; coordinates plan generation + engine
- `src/sirius_engine.cpp`: Pipeline construction and execution orchestration
- `src/transparent/sirius_optimizer_extension.cpp`: Transparent execution hook

**Configuration:**
- `src/config.cpp`: Global configuration flags and defaults
- `src/include/sirius_context.hpp`: Per-connection context ownership
- `src/sirius_extension.cpp`: Config registration with DuckDB

**Core Logic:**
- `src/planner/sirius_physical_plan_generator.cpp`: Logical → physical plan translation
- `src/include/pipeline/sirius_pipeline.hpp`: Pipeline data structure and methods
- `src/include/pipeline/pipeline_executor.hpp`: Top-level executor managing GPU+scan workers
- `src/include/creator/task_creator.hpp`: Task creation and scheduling

**Operators:**
- `src/op/sirius_physical_filter.cpp`: Filter operator
- `src/op/sirius_physical_hash_join.cpp`: Hash join operator
- `src/op/sirius_physical_grouped_aggregate.cpp`: Grouped aggregation
- `src/op/sirius_physical_table_scan.cpp`: Table scan operator

**Memory & Data:**
- `src/include/memory/sirius_memory_reservation_manager.hpp`: Memory lifecycle management
- `src/include/downgrade/downgrade_executor.hpp`: GPU→Host spilling
- `src/data/host_parquet_representation.cpp`: Parquet data conversion

**GPU Kernels:**
- `src/cuda/cudf/cudf_join.cu`: Hash join on GPU
- `src/cuda/cudf/cudf_aggregate.cu`: Groupby and aggregation on GPU
- `src/cuda/cudf/cudf_orderby.cu`: Sorting on GPU
- `src/cuda/operator/hash_join_single.cu`: Custom hash join optimization

## Naming Conventions

**Files:**
- Implementation: `sirius_physical_filter.cpp` (matches class name)
- Headers: Mirror src/ structure in `src/include/`; e.g., `src/include/op/sirius_physical_filter.hpp`
- Plan builders: `sirius_plan_<operator>.cpp`; e.g., `sirius_plan_filter.cpp`
- CUDA files: `*.cu` for GPU code, correspond to `.cpp` implementations where applicable

**Directories:**
- `src/op/` — physical operator implementations
- `src/planner/` — logical → physical plan translation
- `src/cuda/` — GPU kernels
- `src/pipeline/` — pipeline construction/execution
- `src/creator/` — task creation
- `src/memory/` — memory management
- `src/data/` — data conversion/batching
- `test/cpp/` — unit tests organized by component

**Classes:**
- `sirius_physical_<operator>` — physical operator implementations (e.g., `sirius_physical_filter`)
- `sirius_pipeline` — pipeline representation
- `sirius_engine` — main execution orchestrator
- `sirius_interface` — DuckDB-facing API
- `sirius_context` — per-connection context ownership

**Functions:**
- `create_plan(LogicalOperator&)` — plan builder entry points in planner/
- `execute(operator_data, stream)` — operator execution method
- `build_pipelines(pipeline, meta_pipeline)` — pipeline construction method
- `create_task()` — task creation in task_creator

## Where to Add New Code

**New SQL Operator Support:**
- Add logical operator case to `src/planner/sirius_physical_plan_generator.cpp` switch statement
- Create `src/planner/sirius_plan_<operator>.cpp` with `create_plan(Logical<Operator>&)` implementation
- Create `src/op/sirius_physical_<operator>.cpp` with `sirius_physical_<operator>` class and `execute()` method
- If GPU computation needed, add CUDA kernels in `src/cuda/operator/` or wrap cuDF in `src/cuda/cudf/`
- Add tests in `test/cpp/operator/test_<operator>.cpp`
- Document in `docs/super-sirius/operators.md`

**New Scan Source:**
- Create scan task in `src/op/scan/<source>_scan_task.cpp`
- Implement data conversion in `src/data/` converters
- Register converter in `src/data/sirius_converter_registry.hpp`
- Create operator in `src/op/sirius_physical_<source>_scan.cpp`
- Add tests in `test/cpp/scan/`

**Memory Management Changes:**
- Modify `src/include/memory/sirius_memory_reservation_manager.hpp` for reservation logic
- Modify `src/downgrade/downgrade_executor.cpp` for spilling thresholds/behavior
- Add tests in `test/cpp/memory_management/`

**Pipeline/Execution Changes:**
- Modify `src/include/pipeline/sirius_pipeline.hpp` for pipeline logic
- Modify `src/include/pipeline/pipeline_executor.hpp` for executor orchestration
- Modify `src/include/creator/task_creator.hpp` for task scheduling
- Add tests in `test/cpp/pipeline/`, `test/cpp/creator/`

**Configuration/Debug:**
- Add config option in `src/config.cpp` global variables
- Register config setter in `src/sirius_extension.cpp` (`InitialGPUConfigs`)
- Add tests in `test/cpp/config/`

**Utilities & Helpers:**
- Type conversions: `src/helper/type_conversions.cpp`
- Logging: Use `src/include/log/logging.hpp` (spdlog-based)
- General utilities: `src/helper/` directory

## Special Directories

**src/include/pipeline/:**
- Purpose: Pipeline classes and execution orchestrators
- Generated: No
- Committed: Yes
- Contains: `sirius_pipeline.hpp`, `sirius_meta_pipeline.hpp`, `sirius_pipeline_build_state.hpp`, `pipeline_executor.hpp`, `gpu_pipeline_executor.hpp`, etc.

**src/include/creator/:**
- Purpose: Task creation and scheduling logic
- Generated: No
- Committed: Yes
- Contains: `task_creator.hpp` and related classes

**src/include/memory/:**
- Purpose: Memory management and reservation system
- Generated: No
- Committed: Yes
- Contains: `sirius_memory_reservation_manager.hpp`, memory pool abstractions

**src/include/downgrade/:**
- Purpose: GPU memory pressure monitoring and spilling
- Generated: No
- Committed: Yes
- Contains: `downgrade_executor.hpp`

**test/cpp/integration/:**
- Purpose: End-to-end tests running full queries (TPCH, TPCDS)
- Generated: No (but creates test data files)
- Committed: Yes (source), No (generated test data)

**docs/super-sirius/:**
- Purpose: Architecture and implementation documentation
- Generated: No
- Committed: Yes
- Updated: As implementation changes; comprehensive reference

**src/legacy/**
- Purpose: Legacy GPU execution path (deprecated, pre-Super Sirius)
- Generated: No
- Committed: Yes
- Status: Maintained for reference; not active development target

---

*Structure analysis: 2026-04-21*
