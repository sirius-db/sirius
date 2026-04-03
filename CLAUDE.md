# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The main/default branch of this repository is `dev`.

## Project Overview

Sirius is a GPU-native SQL engine that integrates with DuckDB as an extension. It leverages NVIDIA CUDA-X libraries (cuDF, RMM) to accelerate SQL query execution on GPUs. Sirius intercepts DuckDB's physical plan execution and routes supported operations to GPU execution while gracefully falling back to DuckDB's CPU execution for unsupported cases.

**Key Integration Points:**
- DuckDB extension architecture: Sirius loads as a DuckDB extension (`sirius.duckdb_extension`)
- cuCascade: Third-party library for GPU memory management (tiered memory across GPU/host/disk)
- RAPIDS cuDF: GPU DataFrame library for data manipulation
- RMM: RAPIDS Memory Manager for GPU memory allocation

## Build System

### Environment Setup

**Using Pixi (Recommended):**
```bash
pixi shell                    # Activate environment with all dependencies
```

### Git Worktrees

When creating a new worktree, submodules are not automatically initialized. After creating the worktree, run:
```bash
git submodule update --init --recursive
```

### Building

```bash
# Full build (uses all cores by default)
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make

# If build consumes too much memory, reduce parallelism
CMAKE_BUILD_PARALLEL_LEVEL=8 make

# After build errors, clean build directory
rm -rf build
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
```

Build outputs:
- Static extension: `build/release/extension/sirius/sirius.duckdb_extension`
- Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
- Unit test binary: `build/release/extension/sirius/test/cpp/sirius_unittest`

### Building Python API

```bash
pixi run -e duckdb-python build-duckdb-python
```

This uses a dedicated pixi environment (`duckdb-python`) with pip, pybind11, and scikit-build-core. The task automatically points `DUCKDB_SOURCE_PATH` at the repo-level `duckdb/` submodule so the Python package links against the same DuckDB version as the C++ extension.

**Usage from Python:**
```python
import duckdb

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD 'build/release/extension/sirius/sirius.duckdb_extension'")
result = con.execute("CALL gpu_execution('SELECT ...')").fetchall()
```

## Testing

### SQL Logic Tests (End-to-End)
```bash
make test                                              # Run all SQLLogicTests
make test_debug                                        # Debug build tests

# Run specific test file
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

### C++ Unit Tests
```bash
# Build and run all unit tests
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
build/release/extension/sirius/test/cpp/sirius_unittest

# Run tests with specific tag
build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"

# Run specific test
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"
```

Test logs are saved to: `build/release/extension/sirius/test/cpp/log`

Unit tests use Catch2 framework. Test files are in `test/cpp/` organized by component.

### Performance Testing
```bash
# Requires duckdb-python to be built
python3 test/tpch_performance/generate_test_data.py {SCALE_FACTOR}
python3 test/tpch_performance/performance_test.py {SCALE_FACTOR}
```

## Code Formatting & Linting

Sirius uses pre-commit hooks for code quality:

```bash
pre-commit run -a                    # Run all hooks on all files
pre-commit install                   # Install git hooks (runs on every commit)
```

**Code style tools:**
- C++/CUDA: clang-format (style defined in `.clang-format`)
- Python: black
- CMake: cmake-format
- Spell check: codespell (custom words in `.codespell_words`)

Configuration files:
- `.clang-format`: C++/CUDA formatting rules
- `.clang-tidy`: C++ linting rules
- `.pre-commit-config.yaml`: All pre-commit hooks

## Architecture

### Super Sirius (`gpu_execution`)

The active execution engine. Uses `namespace sirius`, entry point: `CALL gpu_execution('SELECT ...')`.

- Physical plan generator: `sirius_physical_plan_generator` (`src/planner/sirius_physical_plan_generator.cpp`)
- Operators: `sirius_physical_operator` subclasses in `src/op/` (e.g., `sirius_physical_hash_join.cpp`)
- Plan builders: `src/planner/` (e.g., `sirius_plan_filter.cpp`, `sirius_plan_aggregate.cpp`)
- Engine: `src/sirius_engine.cpp`, pipelines in `src/pipeline/`
- Interface: `src/sirius_interface.cpp` (uses `sirius_interface` class)
- Task-based execution: `src/creator/`, `src/downgrade/`, `src/op/scan/`
- Extension entry point: `src/sirius_extension.cpp`
- Expression evaluation: `src/expression_executor/`
- Runtime configuration: `src/config.cpp` / `src/include/config.hpp`
- CUDA kernels: `src/cuda/` (cuDF wrappers, expression dispatch)

> **Note:** A legacy code path (`gpu_processing`, `namespace duckdb`) still exists in `src/operator/`, `src/plan/`, `src/gpu_executor.cpp` etc. All new development targets Super Sirius.

### Super Sirius Documentation

Comprehensive documentation lives in `docs/super-sirius/` — see [README](docs/super-sirius/README.md) for index and reading order. **Read these docs before modifying Super Sirius code.**

### Logging

```bash
export SIRIUS_LOG_DIR=/path/to/logs      # Default: ${CMAKE_BINARY_DIR}/log
export SIRIUS_LOG_LEVEL=debug            # Levels: trace, debug, info, warn, error
```

## Development Guidelines

### Loading Library Context for Implementation Tasks

**Before implementing new features, operators, or significant bug fixes**, always run `/module-context <task description>` first. This loads the relevant API documentation for cudf, rmm, duckdb, cucascade, and libkvikio modules so you have accurate function signatures, parameter types, and existing usage patterns. The module docs live in `.claude/skills/module-discover/docs/` and contain detailed API references extracted from the actual library headers.

This is especially important for tasks involving:
- GPU operators (joins, aggregations, sorting, filters, projections)
- Memory management (reservations, pools, streams, spilling)
- Data I/O (parquet scanning, datasources)
- Expression evaluation (AST, unary/binary ops, type casting)
- Pipeline execution (tasks, executors, data batches)

### Fallback Strategy

Sirius gracefully falls back to DuckDB CPU execution when:
- Data size exceeds GPU memory regions (caching or processing)
- Unsupported data types (nested types, some temporal types)
- Unsupported operators (window functions, ASOF JOIN, etc.)
- libcudf row count limitations (~2B rows due to int32_t row IDs)

The fallback mechanism is implemented in `src/fallback.cpp` and integrates with DuckDB's execution engine.

### Supported Features

**Data types:** INTEGER, BIGINT, FLOAT, DOUBLE, VARCHAR, DATE, TIMESTAMP, DECIMAL
**Operators:** FILTER, PROJECTION, JOIN (Hash/Nested Loop/Delim), GROUP BY, ORDER BY, AGGREGATION, TOP-N, LIMIT, CTE, TABLE SCAN
**Join types:** INNER, LEFT, RIGHT, OUTER (implemented via cudf::left_join, cudf::inner_join, etc.)

### Code Organization

- GPU kernels (`.cu` files) are in `src/cuda/` and subdirectories
- CPU-side logic (`.cpp` files) coordinates GPU execution
- Header files (`.hpp`) in `src/include/` mirror source structure
- Each operator has both a DuckDB-facing interface (`operator/`) and cuDF implementation (`cuda/operator/`)

### Adding New Operators

1. Create header in `src/include/operator/gpu_physical_<operator>.hpp`
2. Implement DuckDB integration in `src/operator/gpu_physical_<operator>.cpp`
3. Add cuDF/CUDA implementation in `src/cuda/operator/<operator>.cu`
4. Register in physical plan generator (`src/gpu_physical_plan_generator.cpp`)
5. Add tests in `test/cpp/operator/` and `test/sql/`

### CMake Notes

- Uses CUDA 13+ (specified in `pixi.toml` features)
- Requires C++20 and CUDA standard 20
- Separable compilation enabled for CUDA (`CMAKE_CUDA_SEPARABLE_COMPILATION ON`)
- GPU architectures: Turing through Blackwell (75, 80, 86, 90a, 100f, 120a, 120)
- Links against: cudf::cudf, rmm::rmm, libnuma, libconfig++, absl::any_invocable, spdlog, cuCascade

## Extension Development

This is a DuckDB extension project using the extension template. The build system integrates with DuckDB's extension infrastructure via `extension-ci-tools`.

**Key files for extension integration:**
- `Makefile`: Thin wrapper including `extension-ci-tools/makefiles/duckdb_extension.Makefile`
- `extension_config.cmake`: Specifies which extensions to load (sirius, json, tpcds, tpch, parquet, icu)
- `src/sirius_extension.cpp`: Extension registration (LoadInternal function)

**Extension API Usage:**

CLI:
```sql
LOAD 'build/release/extension/sirius/sirius.duckdb_extension';
CALL gpu_execution('SELECT ...');
-- Legacy mode (requires gpu_buffer_init first):
CALL gpu_buffer_init('1 GB', '2 GB');
CALL gpu_processing('SELECT ...');
```

Python (requires `pixi run -e duckdb-python build-duckdb-python` first):
```python
con = duckdb.connect('db.duckdb', config={"allow_unsigned_extensions": "true"})
con.execute("LOAD '/path/to/sirius.duckdb_extension'")
con.execute("CALL gpu_execution('SELECT ...')").fetchall()
```

## Claude Code Skills

Sirius includes Claude Code skills for performance analysis and dataset management. Invoke them via slash commands:

| Skill | Command | Description |
|-------|---------|-------------|
| Profile Analyzer | `/profile-analyzer` | Analyzes GPU performance from nsys profiles — kernel occupancy, memory bandwidth, operator attribution, and regression detection. |
| Dataset Manager | `/dataset-manager` | Manages TPC-H parquet datasets — generate at any scale factor, consolidate files, inspect layout, optimize row groups. |
| Optimization Advisor | `/optimization-advisor` | Maps GPU hotspots from nsys profiles to source functions, detects efficiency bottlenecks, sync overhead, and parallelism opportunities. |
| TPC-DS Benchmark | `/tpcds-benchmark` | Runs TPC-DS benchmarks on Legacy Sirius, Super Sirius, or DuckDB CPU baseline — generate data, execute queries, and compare results. |
| Module Context | `/module-context` | **Auto-loaded before implementation tasks.** Identifies which dependency modules are relevant to a task and loads their API docs (signatures, descriptions, usage examples). |
| Module Discover | `/module-discover` | Analyzes a dependency library, divides it into modules, and generates LLM-consumable API documentation. Run once per library to populate docs. |

**Useful debugging tools:**
- `tools/parse_pipeline_log.py`: Parses Sirius pipeline logs to show per-operator row counts for debugging incorrect query results.

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Downgrade Executor Redesign**

A thorough redesign of the `downgrade_executor` class and its supporting types (`downgrade_task`, `downgrade_task_local_state`, `downgrade_task_global_state`) in Sirius. The redesign shifts the unit of work from "downgrade a single data_batch" to "free a target amount of memory (or satisfy a predicate) by downgrading data_batches concurrently." This is an internal infrastructure change within the Sirius GPU-native SQL engine.

**Core Value:** The downgrade executor must reliably free GPU memory on demand — both asynchronously (fire-and-forget) and synchronously (block until done) — so that upstream components can request memory reclamation with predictable completion semantics.

### Constraints

- **Thread safety**: All public APIs must be safe to call from any thread; the monitor loop runs on its own thread
- **CUDA device affinity**: Thread pool workers must call `cudaSetDevice` on init (same as today)
- **Non-fatal failures**: Individual batch downgrade failures must not crash the executor — log and continue
- **No breaking SiriusContext**: `SiriusContext` calls `start()`, `stop()`, `drain()`, `get_space_id()` — these must continue to work
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++ 20 - Core GPU-accelerated SQL engine, all operators and expression evaluation
- CUDA 20 - GPU kernels for cuDF operations, expression execution, join/aggregate implementations
- Python 3.12+ (optional) - Performance testing, dataset generation, Python API bindings
- CMake - Build system and project configuration
- Bash - Build scripts and pixi activation
## Runtime
- Pixi 0.59+ - Environment and dependency management
- Linux 64-bit (primary), Linux ARM64 (aarch64) support
- GPU: NVIDIA CUDA 12.x or 13.x (feature-gated)
- Pixi - Conda-based environment from rapidsai and conda-forge channels
- Lockfile: Generated via pixi.lock
- Turing (75), Ampere (80, 86), Ada (90a), Hopper (100f), Blackwell (120a, 120)
- CUDA architecture selection: `CUDAARCHS` environment variable (set by pixi feature)
## Frameworks
- DuckDB 1.4.4 - SQL query engine, physical planner integration, extension API
- RAPIDS cuDF 26.02.* - GPU DataFrame library for joins, aggregations, ordering, filtering
- RAPIDS RMM - GPU memory management, device memory resources
- cuCascade (submodule) - GPU memory reservation and tiered memory management (GPU/host/disk)
- Catch2 (DuckDB bundled) - C++ unit testing framework
- DuckDB SQL Logic Tests - End-to-end query validation
- CMake 4.1.* - Primary build system
- Ninja - Build execution
- CUDA nvcc compiler - CUDA code compilation
- Clang 21.x - C++ compiler with CUDA support
- Mold - Fast linker for reduced build time
- Sccache - C++ compiler cache
- pre-commit - Git hooks for code quality
## Key Dependencies
- libcudf 26.02.* - Core GPU DataFrame operations (joins, aggregations, column selection)
- librmm - RAPIDS Memory Manager for GPU allocation/deallocation
- spdlog 1.8.* - Structured logging with daily file rotation and configurable levels
- libconfig 1.8.* - Configuration file parsing for runtime tuning
- libabseil 20260107.0+ - Standard library extensions (absl::any_invocable for task executors)
- NUMA (system package) - NUMA-aware memory management for host memory pools
- cuda-nvcc - NVIDIA CUDA compiler
- cuda-nvml-dev - CUDA device management API for GPU introspection
- cucascade - Custom GPU memory management with overflow to host memory and disk
- libcurand-dev - CUDA random number generation (for RMM initialization)
- SQLite 3.52+ - Internal storage (SQL logic test data, metadata)
- pybind11 2.6.0+ - Python binding generation for duckdb-python
- scikit-build-core 0.11.4+ - Python wheel building integration
- setuptools-scm 8.0+ - Semantic versioning from git
## Configuration
- `PIXI_PROJECT_ROOT` - Sirius project root directory (set by pixi)
- `CUDAARCHS` - GPU architecture targets (75-real through 120a-real)
- `DUCKDB_SOURCE_PATH` - Path to DuckDB source for Python build (`duckdb-python` feature)
- `SIRIUS_LOG_DIR` - Log output directory (default: build/log)
- `SIRIUS_LOG_LEVEL` - Log severity threshold: trace, debug, info, warn, error (default: info)
- Runtime config: `src/include/config.hpp` static variables with defaults in `src/config.cpp`
- `cmake/CMakePresets.json` - Build presets (release, debug, relwithdebinfo, clang variants)
- `CMakeLists.txt` - Main build definition with CUDA/C++20 requirements, link targets
- `extension_config.cmake` - DuckDB extension registration and versioning
- `.clang-format` - C++/CUDA code formatting rules
- `.clang-tidy` - C++ static analysis configuration
- `.pre-commit-config.yaml` - Auto-formatting hooks (clang-format, black, cmake-format, codespell)
- `[feature.cuda13]` / `[feature.cuda12]` in pixi.toml - CUDA version selection
- `[feature.duckdb-python]` in pixi.toml - Python binding build environment
- `ENABLE_STREAM_CHECK` CMake option - Debug utility for CUDA stream tracking
- `SIRIUS_ENABLE_LEGACY` CMake define - Legacy Sirius code path (optional, can be removed)
## Platform Requirements
- Linux x86_64 or aarch64
- NVIDIA GPU with CUDA compute capability 7.5+ (Turing)
- CUDA Toolkit 12.x or 13.x
- C++ compiler: Clang 21.x (from pixi)
- 32+ GB RAM recommended for parallel builds (CMAKE_BUILD_PARALLEL_LEVEL controls parallelism)
- NVIDIA GPU (same capability requirements)
- CUDA Runtime (distributed as libcudart)
- Host memory for CPU fallback and data staging
- Linux runtime libraries (glibc, libstdc++)
## Extension Loading
- Path: `build/release/extension/sirius/sirius.duckdb_extension`
- Linked directly into DuckDB binary at build time
- Used by CLI: `duckdb db.duckdb`
- Path: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
- Dynamically loaded at runtime via `LOAD 'path/to/sirius_loadable.duckdb_extension'`
- CLI usage: `SELECT * FROM duckdb_functions()` to list Sirius functions
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- C++ source files: `snake_case.cpp` (e.g., `sirius_interface.cpp`)
- C++ header files: `snake_case.hpp` (e.g., `sirius_interface.hpp`)
- Test files: `test_<component>.cpp` (e.g., `test_cpu_cache.cpp`, `test_config.cpp`)
- SQL test files: `<feature>-sirius.test` or `<category>.test` (e.g., `tpch-sirius.test`, `bugfix.test`)
- Python scripts: `snake_case.py` (e.g., `performance_test.py`)
- Function names: `snake_case` for most functions (e.g., `collect_bound_ref_indices()`)
- Member functions: `snake_case` (e.g., `are_conditions_supported()`)
- Private methods: Prefix with underscore not used; rely on access modifiers instead
- Example: `sirius_interface::sirius_process_error()`, `::collect_bound_ref_indices()`
- Local variables: `snake_case` (e.g., `cpu_cache_bytes`, `gpu_column`, `num_records`)
- Member variables: `snake_case` with underscore suffix for private members (e.g., `config_path_`, `db_`, `original_config_env_`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `CPU_CACHE_TEST_MEM_SF`, `PINNED_MEMORY_PARAM_KEY`)
- Template parameters: `PascalCase` (e.g., `T`, `SRC`)
- Class names: `snake_case` (e.g., `shared_test_env`, `bounded_thread_pool`, `sirius_interface`)
- Enum names: `snake_case` (e.g., `env_need`, `HASH_JOIN_MODE`)
- Struct names: `snake_case` (e.g., `sirius_active_query_context`, `SiriusTableFunctionData`)
- Type aliases: `snake_case` (e.g., `sirius_prepared_statement_data`)
- Primary namespace: `sirius` for new code (active development)
- Legacy namespace: `duckdb` for older code (gpu_processing, gpu_context)
- Sub-namespaces: `sirius::op`, `sirius::pipeline`, `sirius::exec`, `sirius::test`, `sirius::memory`
- Nested namespaces flatten in function names: `collect_bound_ref_indices()` in file scope, not as method on anonymous namespace types
## Code Style
- Tool: `clang-format` (style defined in `.clang-format`)
- Indent width: 2 spaces
- Tab width: 2 spaces
- Line length limit: 100 characters (ColumnLimit: 100)
- Pointer alignment: Left (e.g., `T* var`, not `T *var`)
- No space in empty parentheses: `func()` not `func( )`
- Control statement braces: WebKit style (opening brace on same line)
- Function braces: Opening brace on same line
- Class/struct/namespace braces: Opening brace on same line
- No split empty functions, records, or namespaces
- Tool: `clang-tidy` with modernize checks enabled (see `.clang-tidy`)
- Warnings as errors: Yes
- Header filter regex: Sirius extensions should match the cuDF convention
- Common disabled checks:
## Import Organization
- Use `#pragma once` at top of header files (preferred over traditional guards)
- Example: `#pragma once` (no include guard macros needed)
- Sort using declarations (SortUsingDeclarations: true)
- Example: `using namespace sirius;` then `using namespace sirius::op;`
- Not heavily used; imports tend to be fully qualified
- Relative imports from project root are preferred
## Error Handling
- C-style error codes
- Custom exception types (use DuckDB/std exceptions)
## Logging
- `SIRIUS_LOG_TRACE(...)` - Detailed diagnostic info
- `SIRIUS_LOG_DEBUG(...)` - Debug-level information
- `SIRIUS_LOG_INFO(...)` - Informational messages
- `SIRIUS_LOG_WARN(...)` - Warning-level issues
- `SIRIUS_LOG_ERROR(...)` - Error messages
- `SIRIUS_LOG_FATAL(...)` - Critical/fatal errors
- Macros are defined as no-ops (nvcc cannot compile spdlog chrono headers)
- Log critical info in CPU-side wrapper code instead
- Environment variables:
- Initialized in `test/cpp/unittest.cpp` via `InitGlobalLogger()`
#include "log/logging.hpp"
## Comments
- Explain WHY, not WHAT (code shows what it does)
- Non-obvious design decisions or algorithm choices
- Workarounds and known limitations
- License headers required on all source files (Apache 2.0)
- C++ doc comments use `///` for doxygen (less common in this codebase)
- Inline comments use `//` for single line, `/* */` for multi-line
- Example from test_utils.hpp:
## Function Design
- Pass by reference for mutable objects: `GPUBufferManager& manager`
- Pass by const reference for read-only large objects: `const std::vector<Column>&`
- Pass by value for small types (int, bool, enum): `bool invalidated`
- Use smart pointers for ownership: `duckdb::unique_ptr<T>`, `duckdb::shared_ptr<T>`
- Example:
- Use `duckdb::unique_ptr<T>` for new heap allocations with exclusive ownership
- Use `duckdb::shared_ptr<T>` when multiple owners are needed
- Use raw pointers for non-owning references (short-lived): `BaseQueryResult* open_result`
- Return by value for small types
- Return by const reference for large read-only objects (rare)
- Example:
## Module Design
- Header files expose public API; implementation in .cpp
- All public symbols in namespace `sirius` or its sub-namespaces
- Internal symbols use anonymous namespace or `sirius::internal` (less common)
- Not heavily used in this codebase
- Each module has its own interface file (e.g., `sirius_interface.hpp`)
- Non-copyable (delete copy constructor/assignment): Used for resource-owning classes
- Move-only (deleted copy, defaulted move): Used for unique_ptr wrappers
- RAII for resource management (memory, GPU buffer, DuckDB connection)
## Special Patterns
- Use DuckDB's smart pointers: `duckdb::unique_ptr`, `duckdb::shared_ptr` (not std::)
- Use DuckDB string type: `duckdb::string` (not std::string in API boundaries)
- Use DuckDB assert: `D_ASSERT()` (not standard assert)
- Use DuckDB cast: `expr.Cast<Type>()` (not C++ cast operators)
- CUDA kernel code goes in `src/cuda/*.cu` files
- CPU-side wrapper code in `src/op/*.cpp` or `src/*.cpp`
- Use cuDF APIs for GPU data manipulation (no direct cuDF kernel calls from CPU)
- Memory allocation via RMM: `rmm::cuda_stream`, `rmm::device_memory_resource`
- Catch2 TEST_CASE naming: `"description", "[tag1][tag2]"`
- Test namespaces: `sirius::test`, `sirius::scan_test_utils`
- Avoid test-specific exports in production headers; use inline helpers in test headers
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- DuckDB extension that intercepts physical plan execution and routes to GPU when possible
- Task-based pipeline execution with separate GPU and CPU streams
- Hierarchical operator-based design mirroring DuckDB's physical operator patterns
- Tiered memory management (GPU/HOST/DISK) via cuCascade library
- Modular expression evaluation dispatched to GPU via CUDA kernels
## Layers
- Purpose: DuckDB integration point, query registration, and buffer management
- Location: `src/sirius_extension.cpp`, `src/include/sirius_extension.hpp`
- Contains: DuckDB extension loading, function registration (gpu_execution, gpu_processing), buffer initialization
- Depends on: DuckDB extension APIs, GPU buffer manager
- Used by: DuckDB's extension loading mechanism
- Purpose: Query lifecycle management and execution orchestration
- Location: `src/sirius_interface.cpp`, `src/include/sirius_interface.hpp`
- Contains: Query preparation, execution state tracking, result fetching
- Key class: `sirius_interface` manages active query context and routes to GPU engine
- Depends on: DuckDB client context, sirius_engine
- Used by: Extension layer to execute queries
- Purpose: Convert DuckDB logical plans to Sirius physical plans
- Location: `src/planner/sirius_physical_plan_generator.cpp`, `src/planner/sirius_plan_*.cpp`
- Contains: Physical plan generation, operator selection, optimization decision logic
- Key class: `sirius_physical_plan_generator` traverses DuckDB operators and builds GPU-capable equivalents
- Plan builders: `sirius_plan_aggregate.cpp`, `sirius_plan_filter.cpp`, `sirius_plan_join.cpp`, etc. (one per operator type)
- Depends on: DuckDB logical operators, sirius physical operators
- Used by: Interface layer after logical planning
- Purpose: Executable unit representation mirroring DuckDB's physical operators
- Location: `src/op/sirius_physical_*.cpp`, `src/include/op/sirius_physical_*.hpp`
- Contains: Base operator class and 40+ operator implementations (TABLE_SCAN, HASH_JOIN, GROUPED_AGGREGATE, FILTER, PROJECTION, ORDER_BY, etc.)
- Key class: `sirius_physical_operator` (abstract base) with type field and children tree
- Operator families: Scans (TABLE_SCAN, PARQUET_SCAN, ICEBERG_SCAN, DUCKDB_SCAN), Joins (HASH_JOIN, NESTED_LOOP_JOIN, DELIM_JOIN), Aggregates (UNGROUPED_AGGREGATE, GROUPED_AGGREGATE, MERGE), Sorts (MERGE_SORT, TOP_N), Result (RESULT_COLLECTOR, LIMIT)
- Depends on: cuDF/CUDA kernels, expression executor
- Used by: Pipeline execution layer
- Purpose: Break operator tree into executable tasks and manage parallelism
- Location: `src/pipeline/sirius_pipeline.cpp`, `src/include/pipeline/sirius_pipeline.hpp`
- Contains: Pipeline graph construction, dependency management, execution scheduling
- Key classes: `sirius_pipeline` (represents one parallelizable segment), `sirius_meta_pipeline` (whole query plan), `sirius_pipeline_build_state` (construction state machine)
- Builds: Breaks operator tree at blocking points (joins, aggregates) into pipelines
- Depends on: Physical operators, task creation
- Used by: Task creator and engine
- Purpose: Create and schedule parallel tasks from pipelines
- Location: `src/creator/task_creator.cpp`, `src/pipeline/gpu_pipeline_task.cpp`, `src/pipeline/gpu_pipeline_executor.cpp`
- Contains: Task instantiation per pipeline, scheduling logic, CPU thread pool management
- Key classes: `task_creator` (creates tasks from pipelines), `gpu_pipeline_task` (single executable task), `gpu_pipeline_executor` (thread pool executor)
- Execution flow: Pipelines → Tasks → Thread pool workers
- Depends on: Pipelines, physical operators, memory manager
- Used by: Engine to execute query
- Purpose: GPU memory lifecycle and reservation management
- Location: `src/memory/sirius_memory_reservation_manager.cpp`, `src/include/memory/sirius_memory_reservation_manager.hpp`
- Contains: Memory pool management, reservation tracking, OOM policy
- Integrates with: cuCascade (tiered memory), RMM (GPU memory)
- Depends on: RAPIDS RMM, cuCascade memory spaces
- Used by: Task creator, downgrade executor, operators
- Purpose: Automatic memory pressure response and GPU→HOST data movement
- Location: `src/downgrade/downgrade_executor.cpp`, `src/include/downgrade/downgrade_executor.hpp`
- Contains: Memory space monitoring, downgrade task scheduling, data repository management
- Key class: `downgrade_executor` runs monitor thread polling GPU memory, dispatches downgrade tasks when threshold exceeded
- Depends on: cuCascade data repositories, memory spaces
- Used by: Context to manage tiered memory automatically
- Purpose: Dispatch SQL expressions to GPU or CPU evaluation
- Location: `src/expression_executor/gpu_expression_executor.cpp`, `src/include/expression_executor/gpu_expression_executor.hpp`, `src/cuda/expression_executor/`
- Contains: Expression AST traversal, CUDA kernel dispatch, CPU fallback
- Key classes: `gpu_expression_executor` (orchestrates), dispatch kernels in `src/cuda/expression_executor/`
- Supports: Arithmetic, comparison, string operations, casts, regex matching, aggregation functions
- Depends on: DuckDB expression AST, CUDA kernels
- Used by: Physical operators (FILTER, PROJECTION, AGGREGATE)
- Purpose: DuckDB↔GPU data format transformation
- Location: `src/data/`, `src/include/data/`
- Contains: cuDF table builders, Parquet representation converters, data batch utilities
- Key class: `sirius_converter_registry` maps DuckDB types to cuDF representations
- Handles: Arrow format, Parquet metadata, columnar GPU data
- Depends on: cuDF, DuckDB data types, Parquet library
- Used by: Scan operators, expression executor
- Purpose: GPU computation kernels
- Location: `src/cuda/` and subdirectories
- Contains: ~50 CUDA kernels for joins, aggregates, sorts, expressions, string operations, Iceberg delete masks
- Kernel families: `cudf/` (cuDF wrappers), `operator/` (custom kernels), `expression_executor/` (expression dispatch), `iceberg/` (delete masking)
- Uses: cuDF, RMM, NVIDIA libraries (libcudf, RMM, cuCascade)
- Depends on: CUDA 13+, cuDF headers
- Used by: Physical operators via expression executor
- Purpose: Query-wide state and configuration
- Location: `src/sirius_context.cpp`, `src/include/sirius_context.hpp`, `src/sirius_config.cpp`, `src/include/sirius_config.hpp`
- Contains: DuckDB ClientContextState subclass holding task creator, downgrade executor, memory manager, config options
- Key class: `SiriusContext` (lifecycle management), `sirius_config` (hardware topology, GPU selection)
- Manages: QueryBegin/QueryEnd lifecycle, internal query guards for Iceberg metadata lookups
- Depends on: DuckDB client context, cuCascade
- Used by: Extension to initialize and track per-connection state
## Data Flow
- If operator is unsupported or data exceeds GPU memory → `fallback.cpp` routes to DuckDB CPU execution
- Downgrade executor monitors GPU memory pressure → migrates data to HOST tier automatically
- Per-query state: `sirius_active_query_context` (prepared statement, engine, progress bar)
- Per-connection state: `SiriusContext` (task_creator, downgrade_executor, memory manager)
- Per-operator state: Global (sink_state, source_state) and local (per-thread) operator states
- Per-pipeline state: Source, sink, operators, dependencies
- Data state: Batches flow through `operator_data` and `partitioned_operator_data` containers via repositories
## Key Abstractions
- Purpose: Base class for all executable operators
- Examples: `sirius_physical_table_scan`, `sirius_physical_hash_join`, `sirius_physical_grouped_aggregate`, `sirius_physical_filter`
- Pattern: Subclass per operator type, each implements GPU and fallback paths
- Methods: `get_global_sink_state()`, `get_local_sink_state()`, `build_pipelines()`, `execute()`, `finalize()`
- Purpose: Query executor managing operator tree, pipelines, and task scheduling
- Key state: `sirius_owned_plan` (root operator), `sirius_pipelines` (all), `sirius_root_pipelines` (entry points), `sirius_scheduled` (queued)
- Methods: `initialize()`, `initialize_internal()`, `execute()`, `prefetch_iceberg_metadata()`, `insert_repository()`
- Purpose: Single parallelizable segment of execution
- Contains: Source operator, sink operator, middle operators
- Pattern: Pipelines split at synchronization points (PARTITION, CONCAT, MERGE_SORT, etc.)
- Methods: `get_source()`, `get_sink()`, `get_operators()`, `schedule()`, `reset()`
- Purpose: Converts pipelines to executable tasks and manages task dispatch
- Thread pool: Configurable worker count (default = CPU cores)
- Pattern: Task creation queue, priority for table scan pipelines, operator hints for scheduling
- Methods: `start()`, `start_thread_pool()`, `stop_thread_pool()`, `schedule_task_creation()`
- Purpose: Single executable unit for one pipeline iteration
- Contains: Operator references, source/sink state, data batch input
- Methods: `execute()` (runs all operators in pipeline), `has_output()`, `get_output()`
- Purpose: Monitors memory pressure and triggers data migrations
- Pattern: Monitor thread checks memory_space pressure, manager thread dispatches tasks to pool
- Methods: `should_downgrade_memory()`, `drain()`, `stop()`
- Purpose: Evaluate expressions on GPU
- Pattern: Traverses expression AST, dispatches to CUDA kernels via type-specific dispatch functions
- Methods: `execute()`, `execute_expression()`, returns cuDF table
## Entry Points
- Location: `src/sirius_extension.cpp` (registered as table function)
- Triggers: `CALL gpu_execution('SELECT ...')` via DuckDB function call
- Responsibilities: Parse query, prepare statement, route to `sirius_interface`
- Location: `src/sirius_extension.cpp`
- Triggers: `CALL gpu_processing('SELECT ...')` (requires prior `CALL gpu_buffer_init()`)
- Responsibilities: Legacy execution path using GPU buffer context (namespace duckdb, not sirius)
- Location: `src/sirius_extension.cpp` LoadInternal()
- Triggers: `LOAD 'build/release/extension/sirius/sirius.duckdb_extension'`
- Responsibilities: Register functions, initialize extension-wide state
## Error Handling
- Operator::supports_gpu() checks if operator can execute on GPU
- Catch and wrap exceptions in `ErrorData` objects
- `sirius_process_error()` formats error with query context
- Memory errors → trigger defragmentation or eviction before retry
- Unsupported types/operations → degrade to DuckDB CPU operator seamlessly
## Cross-Cutting Concerns
- Framework: spdlog
- Levels: TRACE, DEBUG, INFO, WARN, ERROR configured via `SIRIUS_LOG_LEVEL` env var
- Output: `SIRIUS_LOG_DIR` (default: `${CMAKE_BINARY_DIR}/log`)
- Usage: `SIRIUS_LOG_DEBUG("message")` throughout codebase
- Expression types validated against supported set (INTEGER, BIGINT, FLOAT, DOUBLE, VARCHAR, DATE, TIMESTAMP, DECIMAL)
- Operator cardinality bounds checked (libcudf int32_t row limit ~2B rows)
- Join keys validated for GPU execution
- Aggregate functions validated against cuDF support
- Inherits from DuckDB connection context (read_only flag, catalog access)
- No additional auth layer; operates within DuckDB's connection security model
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
