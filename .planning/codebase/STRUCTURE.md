# Codebase Structure

**Analysis Date:** 2026-04-06

## Directory Layout

```
/home/bwyogatama/sirius/
├── cmake/                          # CMake helper modules
├── docs/                           # Documentation
│   ├── super-sirius/              # Super Sirius architecture docs
│   ├── gpu_execution.md           # Legacy GPU processing docs
│   └── DEVELOPMENT.md             # Development guidelines
├── src/                            # Main source code (C++/CUDA)
│   ├── include/                   # Headers mirroring src/ structure
│   │   ├── config.hpp             # Configuration enums and types
│   │   ├── sirius_context.hpp     # Per-connection subsystem ownership
│   │   ├── sirius_engine.hpp      # Pipeline orchestration
│   │   ├── sirius_interface.hpp   # DuckDB-facing API
│   │   ├── sirius_extension.hpp   # Extension registration
│   │   ├── op/                    # Physical operator headers
│   │   ├── pipeline/              # Pipeline execution headers
│   │   ├── planner/               # Physical plan generator headers
│   │   ├── creator/               # Task creator headers
│   │   ├── expression_executor/   # GPU expression eval headers
│   │   ├── memory/                # Memory management headers
│   │   ├── data/                  # Data conversion headers
│   │   ├── util/                  # Utility headers
│   │   └── helper/                # Helper type definitions
│   ├── op/                        # Physical operator implementations
│   │   ├── scan/                  # Scan operators (DuckDB, Parquet, Iceberg)
│   │   ├── aggregate/             # Grouping/aggregation implementations
│   │   ├── order/                 # Sorting implementations
│   │   ├── merge/                 # Merge implementations (for distributed sorts, aggs)
│   │   ├── partition/             # Partition operator for hash joins
│   │   ├── result/                # Result collection
│   │   ├── sirius_physical_*.cpp  # ~30 operator types
│   │   └── sirius_physical_operator.cpp  # Base class
│   ├── pipeline/                  # Pipeline execution
│   │   ├── gpu_pipeline_executor.cpp    # GPU worker executor
│   │   ├── pipeline_executor.cpp        # Top-level coordination
│   │   ├── sirius_pipeline.cpp          # Pipeline graph representation
│   │   ├── sirius_meta_pipeline.cpp     # Meta-pipeline builder
│   │   ├── gpu_pipeline_task.cpp        # GPU task type
│   │   └── task_request.cpp             # Task request types
│   ├── planner/                   # Physical plan generation
│   │   ├── sirius_physical_plan_generator.cpp  # Logical→Physical translator
│   │   ├── sirius_plan_*.cpp                    # Specialized plan builders (~20 files)
│   │   └── query.cpp                           # Query context
│   ├── creator/                   # Task creation
│   │   ├── task_creator.cpp       # Hint chain following, task dispatch
│   │   └── task_creation_hint     # Task readiness hints
│   ├── expression_executor/       # GPU expression evaluation
│   │   ├── gpu_expression_executor.cpp         # Main executor
│   │   ├── gpu_expression_translator.cpp       # AST→cuDF translation
│   │   ├── gpu_expression_executor_state.cpp   # Execution state
│   │   ├── specializations/                    # Specialized operators
│   │   │   ├── gpu_execute_*.cpp               # Cast, comparison, function, etc.
│   │   │   └── gpu_dispatch_*.cu               # CUDA dispatch kernels
│   │   └── regex/                              # Regex expression handling
│   ├── cuda/                      # CUDA kernels and GPU code
│   │   ├── operator/              # Operator kernels (joins, aggregates, sorts)
│   │   ├── expression_executor/   # GPU expression dispatch
│   │   ├── cudf/                  # cuDF wrapper kernels
│   │   ├── iceberg/               # Iceberg delete mask kernels
│   │   ├── allocator.cu           # GPU memory allocation
│   │   ├── communication.cu       # Inter-GPU communication
│   │   └── utils.cu               # GPU utility functions
│   ├── memory/                    # Memory management
│   │   ├── sirius_memory_reservation_manager.cpp  # Reservation tracking
│   │   ├── defragmenter_oom_policy.cpp            # OOM defragmentation
│   │   ├── host_table_utils.cpp                   # Host memory utilities
│   │   └── multiple_blocks_allocation_accessor.cpp # Multi-block allocation
│   ├── downgrade/                 # GPU→Host spilling
│   │   ├── downgrade_executor.cpp  # Monitor and execute spilling
│   │   └── downgrade_task.cpp      # Spill task type
│   ├── data/                      # Data conversion and representation
│   │   ├── host_parquet_representation.cpp       # Parquet metadata cache
│   │   ├── host_parquet_representation_converters.cpp  # Parquet→GPU conversion
│   │   └── sirius_converter_registry.cpp         # Converter registry
│   ├── parallel/                  # Thread pool management
│   │   └── task_executor.cpp      # Worker thread pool executor
│   ├── util/                      # Utilities
│   │   ├── stream_check_wrapper.cpp  # CUDA stream debugging
│   │   └── segfault_backtrace_handler.cpp  # Backtrace on crash
│   ├── legacy/                    # Legacy GPU execution code (being phased out)
│   │   ├── CMakeLists.txt         # Legacy build config (can be deleted)
│   │   ├── operator/              # Legacy operator impls
│   │   ├── plan/                  # Legacy plan generators
│   │   └── gpu_context.hpp        # Legacy context
│   ├── sirius_extension.cpp       # Extension registration entry point
│   ├── sirius_interface.cpp       # Query lifecycle management
│   ├── sirius_engine.cpp          # Pipeline building and execution orchestration
│   ├── sirius_context.cpp         # Subsystem ownership and lifecycle
│   ├── sirius_config.cpp          # Configuration management
│   ├── gpu_buffer_manager.cpp     # GPU buffer pool (legacy, superseded by RMM)
│   ├── cpu_cache.cpp              # Scan result caching
│   ├── fallback.cpp               # CPU fallback logic
│   ├── extension_lock.cpp         # Extension-level mutex
│   └── config.cpp                 # Runtime configuration
├── test/                          # Tests
│   ├── cpp/                       # C++ unit and integration tests
│   │   ├── operator/              # Operator-specific tests
│   │   │   ├── aggregate/         # Aggregate operator tests
│   │   │   ├── test_physical_*.cpp  # Individual operator tests
│   │   │   └── ...
│   │   ├── pipeline/              # Pipeline execution tests
│   │   ├── scan/                  # Scan executor tests
│   │   ├── memory_management/     # Memory management tests
│   │   ├── integration/           # End-to-end integration tests
│   │   ├── expression_executor/   # Expression executor tests
│   │   ├── parallel/              # Thread pool tests
│   │   ├── downgrade/             # Spilling tests
│   │   ├── data/                  # Data conversion tests
│   │   ├── config/                # Configuration tests
│   │   ├── exec/                  # Execution tests
│   │   ├── utils/                 # Test utilities (sirius_test_env, utils)
│   │   ├── integration/           # Multi-format and TPC-H integration tests
│   │   └── unittest.cpp           # Main test runner (Catch2)
│   ├── sql/                       # SQL Logic Tests
│   │   ├── tpch-sirius.test       # TPC-H test cases
│   │   ├── tpcds*.test            # TPC-DS test cases
│   │   └── ...
│   ├── answers/                   # Expected query results for SQL Logic Tests
│   │   └── tpch/, tpch-mod/       # Reference answers
│   ├── cpp/                       # C++ tests (organized by component)
│   ├── tpch_performance/          # TPC-H performance benchmarks
│   │   ├── tpch_queries/          # SQL query files
│   │   ├── generate_test_data.py  # Data generator
│   │   └── performance_test.py    # Benchmark runner
│   └── tpcds_performance/         # TPC-DS performance benchmarks
├── docs/super-sirius/             # Architecture documentation
│   ├── README.md                  # Doc index and reading order
│   ├── architecture-overview.md   # Component ownership, thread model
│   ├── execution-flow.md          # Step-by-step query execution
│   ├── pipeline-execution.md      # Pipeline scheduling details
│   ├── physical-plan-generation.md # Plan translation details
│   ├── operators.md               # Operator reference
│   ├── expression-executor.md     # Expression evaluation
│   ├── scan.md                    # Scan operators and caching
│   ├── memory-management.md       # Memory reservation and spilling
│   ├── data-management.md         # Data batches and repositories
│   ├── configuration.md           # Config options reference
│   ├── optimizations.md           # Performance optimizations
│   └── task-creator.md            # Task scheduling algorithm
├── tools/                         # Utility scripts
│   └── parse_pipeline_log.py      # Pipeline log parser for debugging
├── scripts/                       # Build/test scripts
│   └── clickbench_runner/         # ClickBench runner script
├── CMakeLists.txt                 # Root CMake build config
├── extension_config.cmake         # Extension list (sirius, json, tpcds, tpch, etc.)
├── Makefile                       # Build wrapper
├── CLAUDE.md                      # Development guidelines (this project)
├── LICENSE                        # Apache 2.0
└── README.md                      # Project overview
```

## Directory Purposes

**src/include/ ↔ src/**
- Headers in `include/` mirror the `src/` directory structure
- Each `.cpp` file in `src/` has a corresponding `.hpp` in `src/include/`
- Pattern: `src/op/sirius_physical_filter.cpp` ↔ `src/include/op/sirius_physical_filter.hpp`

**src/op/**
- Home of all physical operators (30+ implementations)
- Each operator `sirius_physical_<name>` has:
  - Header: `src/include/op/sirius_physical_<name>.hpp`
  - Implementation: `src/op/sirius_physical_<name>.cpp`
  - Optional CUDA kernel: `src/cuda/operator/<name>.cu`
- Organized into subdirectories by operator family: `scan/`, `aggregate/`, `order/`, `merge/`, `partition/`, `result/`

**src/cuda/**
- All GPU kernel code (.cu files)
- Mirrors operator structure: `src/cuda/operator/`, `src/cuda/cudf/`, `src/cuda/expression_executor/`, `src/cuda/iceberg/`
- Kernels called from CPU-side operator implementations via CUDA kernel launches

**src/pipeline/**
- Pipeline execution orchestration and task management
- `sirius_pipeline.cpp`: Pipeline graph representation (source, operators, sink, dependencies)
- `sirius_meta_pipeline.cpp`: Meta-pipeline builder (recursively walks physical plan)
- `gpu_pipeline_executor.cpp`: GPU worker threads (acquire tickets, pop tasks, execute, push results)
- `pipeline_executor.cpp`: Top-level coordination (starts executors, schedules initial tasks)

**src/planner/**
- Physical plan generation from DuckDB's logical operators
- `sirius_physical_plan_generator.cpp`: Main entry point, dispatches to specialized builders
- `sirius_plan_<type>.cpp`: Specialized plan builders (~20 files, one per operator type)
- Pattern: `LogicalOperator` → `sirius_physical_operator`, handle pipeline splitting and operator injection

**src/expression_executor/**
- GPU-side expression evaluation via cuDF
- `gpu_expression_executor.cpp`: Main API (add expressions, set inputs, execute/select)
- `gpu_expression_translator.cpp`: Walks DuckDB expression AST, builds cuDF operations
- `specializations/`: Type-specific and operator-specific implementations (cast, comparison, functions)
- `regex/`: Regular expression handling on GPU

**src/creator/**
- Task scheduling based on data availability
- `task_creator.cpp`: Manager loop implements hint chain following
- Receives schedule callbacks from GPU/scan executors
- Determines which operator is ready next based on data repositories

**src/downgrade/**
- GPU→Host memory spilling under pressure
- `downgrade_executor.cpp`: Monitor thread polls memory pressure, dispatches spill tasks
- Called when GPU memory reservation fails

**src/memory/**
- Memory reservation and allocation tracking
- `sirius_memory_reservation_manager.cpp`: Central reservation authority
- Tracks GPU/host/disk memory via cuCascade integration
- Per-memory-space downgrade executors

**src/data/**
- Data type conversions between DuckDB and GPU formats
- `host_parquet_representation.cpp`: Caches Parquet metadata
- `host_parquet_representation_converters.cpp`: Converts Parquet→GPU format
- `sirius_converter_registry.cpp`: Registry of type converters

**test/cpp/**
- C++ unit and integration tests (Catch2 framework)
- Organized by component: `operator/`, `pipeline/`, `scan/`, `memory_management/`, etc.
- `unittest.cpp`: Main test runner
- Entry point: `build/release/extension/sirius/test/cpp/sirius_unittest`

**test/sql/**
- SQL Logic Tests (DuckDB's test framework)
- `.test` files contain SQL queries and expected results
- Run via: `build/release/test/unittest --test-dir . test/sql/tpch-sirius.test`

**test/tpch_performance/ & test/tpcds_performance/**
- Performance benchmarks (Python scripts)
- `generate_test_data.py`: Generates TPC-H/TPC-DS parquet at scale factor
- `performance_test.py`: Runs queries, measures execution time, compares CPU vs GPU

**docs/super-sirius/**
- Architecture and design documentation
- Must read before modifying Super Sirius code
- References actual file paths and function signatures

## Key File Locations

**Entry Points:**
- `src/sirius_extension.cpp`: DuckDB extension registration, `CALL gpu_execution()` handler
- `src/sirius_interface.cpp`: Query preparation and lifecycle (begin → execute → fetch)
- `src/sirius_engine.cpp`: Pipeline building and execution orchestration

**Configuration:**
- `src/config.cpp`: Runtime config (memory sizes, thread counts, operator tuning)
- `src/sirius_config.cpp`: Configuration option definitions
- `src/include/config.hpp`: Configuration enums and option names

**Core Logic:**
- `src/planner/sirius_physical_plan_generator.cpp`: Plan generation dispatcher
- `src/include/op/sirius_physical_operator.hpp`: Base operator class
- `src/include/pipeline/sirius_pipeline.hpp`: Pipeline graph representation
- `src/include/pipeline/pipeline_executor.hpp`: Top-level executor interface

**Testing:**
- `test/cpp/unittest.cpp`: Catch2 test runner (main entry)
- `test/cpp/utils/sirius_test_env.cpp`: Test environment setup
- `test/cpp/integration/test_gpu_execution_tpch.cpp`: TPC-H integration tests

**Memory Management:**
- `src/include/memory/sirius_memory_reservation_manager.hpp`: Reservation API
- `src/memory/sirius_memory_reservation_manager.cpp`: Reservation implementation
- `src/downgrade/downgrade_executor.cpp`: Spilling implementation

**Data Flow:**
- `src/op/scan/duckdb_scan_executor.cpp`: Scan task execution
- `src/op/scan/parquet_scan_task.cpp`: Parquet read logic
- `src/include/data/cached_data_representation.hpp`: Inter-operator data transfer

## Naming Conventions

**Files:**
- C++ source: `snake_case.cpp`
- Headers: `snake_case.hpp`
- CUDA kernels: `snake_case.cu`
- Tests: `test_<component>.cpp` or `test_<functionality>.cpp`
- SQL logic tests: `<benchmark>.test` (e.g., `tpch-sirius.test`)

**Directories:**
- Source: `snake_case/` (e.g., `expression_executor/`, `memory_management/`)
- Tests: `snake_case/` organized by component tested (e.g., `test/cpp/operator/aggregate/`)

**Classes/Types:**
- Operators: `sirius_physical_<name>` (e.g., `sirius_physical_hash_join`)
- Executors: `<name>_executor` (e.g., `gpu_pipeline_executor`, `duckdb_scan_executor`)
- Task types: `<name>_task` (e.g., `gpu_pipeline_task`, `parquet_scan_task`)
- Managers: `<name>_manager` (e.g., `sirius_memory_reservation_manager`)

**Functions:**
- Private helpers: `snake_case_impl()` or `<verb>_<noun>()`
- Virtual operator methods: `execute()`, `sink()`, `get_operator_state()`
- Lifecycle: `initialize()`, `terminate()`, `cleanup()`

**Headers Organization:**
- Public API: `src/include/<component>/<name>.hpp`
- Forward declarations: `src/include/<component>/fwd.hpp` (if complex)
- Implementation details: `#include` in .cpp only

## Where to Add New Code

**New Operator:**
1. Header: `src/include/op/sirius_physical_<name>.hpp`
   - Extend `sirius_physical_operator`
   - Define `SiriusPhysicalOperatorType::<NAME>` in `src/include/op/sirius_physical_operator_type.hpp`
2. Implementation: `src/op/sirius_physical_<name>.cpp`
   - Implement required virtual methods (execute, sink, is_source, is_sink, etc.)
3. GPU kernel (if needed): `src/cuda/operator/<name>.cu`
4. Plan builder: `src/planner/sirius_plan_<name>.cpp`
   - Add case in `sirius_physical_plan_generator::create_plan()`
5. Tests:
   - Unit test: `test/cpp/operator/test_physical_<name>.cpp`
   - SQL logic test: Add query to `test/sql/tpch-sirius.test` or similar

**New Configuration Option:**
1. Add enum value to `config_option.hpp`
2. Add default + description in `sirius_config.cpp` (register with `Config::RegisterOption()`)
3. Add getter function in `config.hpp` or `config.cpp`
4. Document in `docs/super-sirius/configuration.md`

**New Scan Source (Table Type):**
1. Create operator: `src/op/sirius_physical_<source>_scan.cpp/hpp`
2. Create scan task: `src/op/scan/<source>_scan_task.cpp/hpp`
3. Create executor: `src/op/scan/<source>_scan_executor.cpp/hpp`
4. Register in plan generator: `sirius_plan_get.cpp`

**New Expression Type Support:**
1. Add specialization: `src/expression_executor/specializations/gpu_execute_<op>.cpp`
2. Add CUDA dispatch: `src/cuda/expression_executor/gpu_dispatch_<op>.cu`
3. Update `gpu_expression_translator.cpp` to route expression type
4. Add tests: `test/cpp/expression_executor/test_<op>.cpp`

**Utilities/Helpers:**
- Shared helpers: `src/util/<name>.cpp/hpp`
- Math/memory utilities: `src/cuda/utils.cu`
- Test utilities: `test/cpp/utils/<name>.cpp/hpp`

## Special Directories

**src/legacy/**
- Purpose: Legacy GPU processing code being phased out
- Generated: No (hand-maintained, separate from Super Sirius)
- Committed: Yes (for historical reference)
- Note: Can be deleted entirely by removing `src/legacy/CMakeLists.txt` from main CMake build

**build/release/**
- Purpose: Build outputs (generated)
- Generated: Yes
- Committed: No
- Contents:
  - `extension/sirius/sirius.duckdb_extension`: Static extension
  - `extension/sirius/sirius_loadable.duckdb_extension`: Loadable extension
  - `extension/sirius/test/cpp/sirius_unittest`: C++ test binary
  - `test/unittest`: SQL logic test runner

**test_datasets/tpch_parquet_sf1/, tpcds_parquet_sf1/**
- Purpose: Pre-generated benchmark data (Parquet format)
- Generated: Yes (via `test/tpch_performance/generate_test_data.py`)
- Committed: No (generated on-demand)

**parquet/**
- Purpose: Sample Parquet files for local testing
- Generated: No (hand-created test data)
- Committed: Yes

**.planning/codebase/**
- Purpose: GSD mapping documents (this directory)
- Generated: Yes (by Claude GSD agent)
- Committed: Yes
- Contents: ARCHITECTURE.md, STRUCTURE.md, STACK.md, etc.

---

*Structure analysis: 2026-04-06*
