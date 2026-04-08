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

**Sirius Query Plan Explain**

A query plan visualization feature for Sirius, the GPU-native SQL engine. It provides a pretty-printed view of the execution pipeline structure produced by `sirius_engine::initialize_internal()`, showing operator chains within pipelines and the full DAG of pipelines connected via data repositories. Available both as automatic logging during query execution and as a dedicated `gpu_explain('SQL')` table function.

**Core Value:** Engineers can see exactly how Sirius will execute a query — which pipelines exist, what operators they contain, and how they're connected — without reading debug logs or source code.

### Constraints

- **Print timing**: Plan must be printed at end of `initialize_internal` only — after `new_scheduled` is fully populated
- **Codebase patterns**: Must follow existing namespace conventions (`sirius::pipeline`), use `SIRIUS_LOG_INFO` for logging, use DuckDB types (`duckdb::vector`, `duckdb::shared_ptr`)
- **Build system**: New `.cpp` files must be added to `CMakeLists.txt`
- **Operator casting**: Must check operator `type` field before `Cast<>()` to concrete types (existing pattern)
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++ 20 - GPU extension implementation, query planning, operators, pipeline execution
- CUDA 20 - GPU kernel implementations for data operations, expression execution, joins
- Python 3.12+ - Test utilities, dataset generation, performance testing
- CMake 4.1+ - Project configuration and build system
- Make - Build orchestration
- Ninja - Build execution backend
## Runtime
- Linux (x86-64 and ARM64 support via pixi platforms)
- CUDA Toolkit 12.x or 13.x (feature-based selection in pixi)
- GPU architectures: Turing (75), Ampere (80, 86), Hopper (90a), Blackwell (100f, 120a, 120)
- Pixi (conda-based) - Primary environment management
- Pip (within duckdb-python environment) - Python package installation
- Conda channels: `rapidsai`, `conda-forge`
## Frameworks
- DuckDB 1.4.4 - Integrated SQL engine (runs in-process as extension)
- DuckDB Extension API - Extension registration and integration point
- libcuDF 26.02.* - RAPIDS GPU DataFrame library for vectorized operations
- libRMM - RAPIDS Memory Manager for GPU memory allocation
- Catch2 - C++ unit test framework (headers in `duckdb/third_party/catch`)
- SQL logic tests - DuckDB's test harness for SQL correctness
- clang 21.x - C++ compiler (alternative to gcc)
- clang-format 21.1.8+ - Code formatting
- clang-tidy - Static analysis
- sccache - Compilation caching
- Ninja - Build system backend
## Key Dependencies
- cuCascade - GPU memory tiered memory management (CPU/GPU/disk spilling) - Git submodule in `cucascade/`
- libconfig++ - Configuration file parsing (SIRIUS_CONFIG_FILE support)
- libabseil 20260107.0+ - Google abseil C++ library (specifically `absl::any_invocable`)
- spdlog 1.8.* - Structured logging framework
- libnuma - NUMA-aware memory allocation
- libcurand-dev - CUDA random number generation
- cuda-nvcc - NVIDIA CUDA compiler
- cuda-nvml-dev - NVIDIA Management Library for device monitoring
- sqlite 3.52+ - Test data storage
- pkg-config - Library discovery
## Configuration
- `SIRIUS_LOG_DIR` - Directory for operation logs (default: `${CMAKE_BINARY_DIR}/log`)
- `SIRIUS_LOG_LEVEL` - Logging verbosity: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)
- `SIRIUS_CONFIG_FILE` - Path to sirius.cfg configuration file (default: `~/.sirius/sirius.cfg`)
- `SIRIUS_STREAM_CHECK_LIB` - Path to stream check library for CUDA stream debugging
- `DUCKDB_SOURCE_PATH` - DuckDB source directory for Python extension build
- `CMAKE_BUILD_PARALLEL_LEVEL` - Build parallelism (recommended: `$(nproc)` or reduced if memory constrained)
- Memory settings (pinned memory usage, buffer sizes)
- Scan caching levels
- Expression execution backend selection
- GPU operation optimization flags
- `CMakeLists.txt` - Main CMake build configuration
- `pixi.toml` - Pixi environment specification with CUDA version features
- `extension_config.cmake` - DuckDB extension loader configuration
- `.clang-format` - C++ formatting rules
- `.clang-tidy` - C++ linting rules
- `.pre-commit-config.yaml` - Code quality hooks (clang-format, black, cmake-format, codespell)
## Platform Requirements
- Pixi package manager (>=0.59)
- NVIDIA GPU with CUDA compute capability 7.5+ (Turing era or newer)
- 8GB+ GPU memory (larger for integration tests)
- Linux OS (x86-64 or ARM64)
- ~4GB RAM for single-threaded builds; more for parallel builds
- Static extension: `build/release/extension/sirius/sirius.duckdb_extension`
- Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
- C++ unit tests: `build/release/extension/sirius/test/cpp/sirius_unittest`
- Built against duckdb-python in `duckdb-python/` submodule
- Loaded via: `con.execute("LOAD 'path/to/sirius.duckdb_extension'")`
- Requires unsigned extension config: `allow_unsigned_extensions=true`
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- C++/CUDA files: `snake_case.cpp`, `snake_case.hpp` or `snake_case.cu`
- Examples: `sirius_interface.cpp`, `gpu_expression_executor.hpp`, `gpu_order_impl.cu`
- Operator classes: `gpu_<operation>_impl.cpp` (e.g., `gpu_aggregate_impl.cpp`, `gpu_partition_impl.cpp`)
- PascalCase for public methods (following DuckDB conventions as Sirius integrates with DuckDB)
- snake_case for helper/internal functions
- Examples: `Execute()`, `GetOperatorState()`, `AddExpression()`, `reset()`, `insert_repository()`
- Getter methods: no `get_` prefix; use direct name like `database()`, `get_memory_space()`
- snake_case for local and member variables
- Prefix/suffix for clarity: `*_idx` for indices, `*_size` for sizes, `*_count` for counters
- Examples: `root_pipeline_idx`, `num_groups`, `estimated_cardinality`, `expected_data`
- Member variables: trailing underscore for private: `config_path_`, `db_`, `had_original_config_env_`
- PascalCase with `sirius_` or `gpu_` prefix where appropriate
- Examples: `sirius_physical_filter`, `sirius_engine`, `GpuExpressionExecutor`, `gpu_type_traits<TestType>`
- Enum classes: PascalCase, e.g., `SiriusPhysicalOperatorType`, `MemoryBarrierType`
- UPPER_SNAKE_CASE for compile-time constants and configuration values
- Examples: `DEFAULT_SCAN_TASK_BATCH_SIZE`, `MAX_SORT_PARTITION_BYTES`, `LOG_LEVEL`
- Static member constants in namespace `duckdb::Config`
## Code Style
- Tool: clang-format (strict enforcement via pre-commit hooks)
- Config file: `.clang-format`
- Key settings:
- Tool: clang-tidy (integrated via pre-commit)
- Config file: `.clang-tidy`
- Checks enabled: modernize-*, performance-*, clang-analyzer-*
- Enforcement: WarningsAsErrors enabled (violations block commits)
- Notable disabled checks: modernize-use-equals-default, modernize-use-trailing-return-type (stylistic reasons)
- Python formatting: black (via pre-commit)
- CMake formatting: cmake-format with cmake-lint
- Spell check: codespell with custom words in `.codespell_words`
## Import Organization
- No explicit using aliases found, but `namespace sirius` and `namespace duckdb` are primary namespaces
- Common imports use fully qualified paths: `duckdb::`, `sirius::`, `cucascade::`
## Error Handling
- DuckDB integration: use `D_ASSERT()` for assertions (`src/sirius_interface.cpp` line 63-76)
- Exceptions: throw DuckDB exception types: `duckdb::InvalidInputException()`, `duckdb::ErrorData()`
- CUDA/GPU errors: checked via `CUDA_CHECK` macros and cuDF error handling
- Fallback strategy: exceptions trigger graceful fallback to DuckDB CPU execution via `src/fallback.cpp`
- Error context: error messages include query information and location details via `AddErrorLocation()`
## Logging
#include "log/logging.hpp"
- Macros: `SIRIUS_LOG_DEBUG("message")`, `SIRIUS_LOG_INFO("format {}", value)`
- Format: spdlog fmt-style with file:line in pattern `[%s:%#]`
- Environment variables:
- CUDA compilation units (.cu files) define logging macros as no-ops
- Line 19-26 in `logging.hpp`: `#ifdef __CUDACC__` guards prevent spdlog inclusion in CUDA code
## Comments
- Before class/struct definitions: brief description (single line)
- Complex algorithm sections: explain intent, not what the code does
- Non-obvious design choices (e.g., "GPU memory layout optimized for coalesced access")
- License headers: Apache 2.0 on all source files (Copyright 2025, Sirius Contributors)
- Used sparingly in headers for public APIs
- Example from `logging.hpp`: `@brief` for single-line descriptions
- Function signatures show parameter types clearly
## Function Design
- Range: 20-150 lines typical
- Shorter for utility functions, longer acceptable for complex operators
- Example: `sirius_engine::insert_repository()` at ~40 lines; `GpuExpressionExecutor::Execute()` handling multiple cases
- Pass vectors/large objects by reference or unique_ptr
- Small types (int, bool, enum) by value
- Pattern: const-reference for inputs, move semantics for ownership transfer
- Example from test: `make_two_column_batch<int64_t, typename Traits::type>(*space, filter_vals, data_vals, ...)`
- Return by value for small types and POD
- Return `duckdb::unique_ptr<T>` for allocated objects (DuckDB convention)
- Return `std::shared_ptr<T>` for shared lifetime (GPU buffers, data_batch)
- Return `std::optional<T>` for optional results
## Module Design
- Header-only utilities in `src/include/` with `.hpp` extension
- Implementation in `src/` with `.cpp` or `.cu` extension
- Class methods in `src/include/` declared, implemented in `src/`
- Not extensively used
- Main entry points: `src/sirius_interface.hpp`, `src/include/config.hpp`
- Test utilities: `test/cpp/operator/operator_test_utils.hpp` aggregates utility functions
## Namespacing
- `namespace sirius` - Main engine code, operators, expression executor
- `namespace sirius::op` - Physical operators (`sirius_physical_filter`, etc.)
- `namespace sirius::pipeline` - Pipeline execution infrastructure
- `namespace sirius::memory` - Memory management (reservation manager, cache)
- `namespace sirius::test` - Test utilities and fixtures
- `namespace duckdb` - DuckDB integration layer (config, legacy code)
- `namespace cucascade` - GPU cascade memory library (used for data representations)
- Used in .cpp files for private helper functions
- Pattern: unnamed namespace `{}` at file scope (e.g., `src/expression_executor/gpu_expression_executor.cpp` line 44-52)
## Type Conversions & Casts
- Avoid C-style casts `(Type)`
- Use `static_cast<>` for safe conversions
- Use `Cast<>()` method on operator base classes for type-safe downcasts
- Example: `next_op->Cast<op::sirius_physical_right_delim_join>()`
## Memory & Ownership
- Use `duckdb::make_uniq<T>()` instead of `std::make_unique<T>()`
- Use `duckdb::unique_ptr<T>` type alias
- Use `duckdb::shared_ptr<T>` for reference-counted resources
- Use `rmm::device_buffer`, `rmm::device_uvector` for GPU allocations
- Allocators passed via `rmm::device_async_resource_ref`
- Example: `auto mr = get_resource_ref(*space)` in tests
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Task-based pipeline parallelism with multiple dedicated thread pools (GPU execution, scan, task creation, downgrade)
- Lazy pipeline construction from DuckDB's logical plan via dynamic operator splitting
- Tiered memory management (GPU/pinned host/disk) with graceful spilling via cuCascade
- Graceful fallback to DuckDB CPU execution for unsupported operations
- Data flow through typed batches via shared repositories with barrier-based synchronization
## Layers
- Purpose: DuckDB integration surface, query lifecycle management
- Location: `src/sirius_extension.cpp`, `src/sirius_interface.cpp`
- Contains: Table function bindings, query preparation, result collection
- Depends on: DuckDB parsing/optimization, SiriusContext, sirius_engine
- Used by: DuckDB client via `CALL gpu_execution('SELECT ...')`
- Purpose: Translate DuckDB's logical plan to Sirius physical operators with GPU-aware splitting
- Location: `src/planner/`, `src/include/planner/`
- Contains: `sirius_physical_plan_generator`, specialized plan builders (filter, aggregate, join, order, etc.)
- Depends on: DuckDB logical operators, operator type definitions
- Used by: sirius_engine.initialize()
- Purpose: Orchestrate pipeline construction, execution lifecycle, memory management
- Location: `src/sirius_engine.cpp`, `src/include/sirius_engine.hpp`
- Contains: Pipeline graph building, initialization, execution coordination
- Depends on: Physical operators, pipeline builders, task creators, memory managers
- Used by: sirius_interface
- Purpose: GPU-accelerated (or fallback) implementations of SQL operations
- Location: `src/op/`, `src/include/op/`, `src/cuda/operator/`
- Contains: ~30 operator types (FILTER, PROJECTION, HASH_JOIN, AGGREGATE, ORDER, etc.)
- Depends on: cuDF, expression executor, data batches, memory reservations
- Used by: GPU pipeline executor during task execution
- Purpose: Multi-threaded task scheduling and execution with resource management
- Location: `src/pipeline/`, `src/include/pipeline/`
- Contains: `pipeline_executor`, `gpu_pipeline_executor`, `sirius_pipeline`, pipeline metadata
- Depends on: Operators, task creator, scan executor, downgrade executor
- Used by: sirius_engine
- Purpose: Dynamic task scheduling based on data availability in operator ports
- Location: `src/creator/`, `src/include/creator/`
- Contains: `task_creator` with hint chain following
- Depends on: Operators, GPU/scan executors, data repositories
- Used by: GPU and scan executor callbacks
- Purpose: Async data ingestion from DuckDB tables or Parquet files to GPU
- Location: `src/op/scan/`, `src/include/op/scan/`
- Contains: `duckdb_scan_executor`, `parquet_scan_task`, caching logic, Iceberg metadata
- Depends on: DuckDB table functions, Parquet reader, caching infrastructure
- Used by: task creator, data repositories
- Purpose: Tiered GPU/host/disk memory allocation with reservation and spilling
- Location: `src/memory/`, `src/include/memory/`, cuCascade integration
- Contains: `sirius_memory_reservation_manager`, downgrade executor, defragmentation
- Depends on: RMM, cuCascade, GPU allocator
- Used by: GPU pipeline executor, downgrade executor
- Purpose: Evaluate DuckDB bound expressions on GPU via cuDF
- Location: `src/expression_executor/`, `src/cuda/expression_executor/`
- Contains: `GpuExpressionExecutor`, expression translators, specializations for ops
- Depends on: cuDF, DuckDB expression AST
- Used by: Operators (FILTER, PROJECTION, HASH_JOIN predicates, aggregates)
- Purpose: Typed data interchange between operators and external storage
- Location: `src/data/`, `src/include/data/`
- Contains: Parquet representation converters, cached data representation, converter registry
- Depends on: Parquet metadata, cuDF, host memory management
- Used by: Scan operators, data repositories
- Purpose: Ownership and lifecycle management of all subsystems per DuckDB connection
- Location: `src/sirius_context.cpp`, `src/include/sirius_context.hpp`
- Contains: SiriusContext (config, memory manager, executor references, query state)
- Depends on: All subsystems below
- Used by: Extension, interface, engine
## Data Flow
- Operator state: Global (`GlobalOperatorState`, `GlobalSinkState`) per operator, local per thread
- Pipeline state: `sirius_pipeline` tracks dependencies, batch indexes, parent relationships
- Data movement: `shared_data_repository` holds typed data batches with producer/consumer tracking
- Memory state: Tracked via `sirius_memory_reservation_manager` with per-space downgrade executors
## Key Abstractions
- Purpose: Base class for all GPU-executable operations
- Examples: `sirius_physical_hash_join.hpp`, `sirius_physical_grouped_aggregate.hpp`, `sirius_physical_table_scan.hpp`
- Pattern: Virtual methods for operator/sink/source states, `execute()` for streaming, `sink()` for aggregation/grouping
- Purpose: Represents a sequence of operators from source to sink
- Examples: `src/include/pipeline/sirius_pipeline.hpp`
- Pattern: Tracks operators, source, sink, dependencies, batch indexes; knows parent pipelines and order requirements
- Purpose: Typed containers for data batches flowing between operators
- Examples: `src/include/op/sirius_physical_operator.hpp`
- Pattern: Wraps `std::vector<std::shared_ptr<cucascade::data_batch>>`; subclass tracks partition index
- Purpose: Centralized buffer for inter-pipeline data transfer with synchronization
- Examples: Created in `sirius_engine::insert_repository()` with barrier types
- Pattern: Holds data batches, tracks producer/consumer counts, notifies task creator when data available
- Purpose: Evaluates DuckDB bound expressions on GPU via cuDF
- Examples: `src/include/expression_executor/gpu_expression_executor.hpp`
- Pattern: Parses expression AST, dispatches to specialized cuDF operations, handles type conversions
- Purpose: Per-connection ownership hierarchy
- Examples: `src/include/sirius_context.hpp`
- Pattern: Registered as `ClientContextState`, owns config, memory manager, all executors, query state
## Entry Points
- Location: `src/sirius_extension.cpp` → `GPUExecutionBind()`, `GPUExecutionFunction()`
- Triggers: Table function bind → parse/optimize → physical plan generation
- Responsibilities: Extract SQL, prepare statement, manage result collection
- Location: `src/sirius_interface.cpp`
- Triggers: Pipeline construction, execution, result extraction
- Responsibilities: Query lifecycle (begin → execute → fetch → cleanup)
- Location: `src/sirius_engine.cpp`
- Triggers: Starts pipeline executor, waits on completion future
- Responsibilities: Coordinate GPU and scan execution
- Location: `src/include/pipeline/pipeline_executor.hpp` (forward decl), implementation in executor
- Triggers: Spawns sub-executor threads, queues initial scan tasks
- Responsibilities: Distribute completion handler, manage task scheduling
- Location: `src/include/creator/task_creator.hpp`
- Triggers: Receives schedule callbacks from GPU/scan executors
- Responsibilities: Determine task readiness, dispatch to executors
- Location: `src/include/pipeline/gpu_pipeline_executor.hpp`
- Triggers: Pops tasks from queue, acquires reservations
- Responsibilities: Execute all operators in pipeline, call sink(), handle OOM
- Location: `src/include/op/scan/duckdb_scan_executor.hpp`
- Triggers: Pops scan tasks from queue
- Responsibilities: Execute DuckDB scan, convert data, publish to repositories
## Error Handling
- **GPU OOM:** `oom_reschedule_exception` caught in GPU executor, retry up to 10 times with 5ms backoff (progressive reductions possible)
- **Unsupported operators:** Throw `NotImplementedException` during planning, caught by fallback layer
- **Query errors:** Exception caught in GPU/scan executor, routed to `completion_handler->report_error()` which drains queues and propagates to main thread
- **Task execution failures:** `drain_after_error()` stops task creation, drains queues, signals completion with error
- **CPU fallback:** If enabled in config, `sirius_extension` catches plan errors and re-executes via DuckDB CPU path
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

| Skill | Description | Path |
|-------|-------------|------|
| bisect | > Use this skill to find which commit introduced a bug or regression. Uses git bisect with automated build and test. Trigger when a bug appeared recently, a query started failing, performance regressed, or the user wants to compare behavior between two commits. | `.claude/skills/bisect/SKILL.md` |
| build-errors | > Use this skill when the build fails, compilation errors occur, or you see undefined references, linker errors, CUDA compilation issues, missing headers, or template instantiation failures. Analyzes errors, suggests fixes, and iteratively rebuilds until success. | `.claude/skills/build-errors/SKILL.md` |
| config-optimizer | > Use this skill to find the optimal Sirius configuration for TPC-H workloads at any scale factor. Trigger when the user wants to tune performance, optimize config parameters, find the best thread count, batch size, or cache mode, or benchmark different Sirius configurations against each other. Also use when the user mentions "config tuning", "parameter sweep", or "optimal settings". | `.claude/skills/config-optimizer/SKILL.md` |
| dataset-manager | > Use this skill to generate TPC-H test data, consolidate parquet files, inspect dataset layout, or optimize row group sizes. Trigger when the user needs test data at a specific scale factor, wants to merge small parquet files, check dataset structure, or prepare data for benchmarks. Auto-selects cudf (GPU) or pyarrow (CPU) with OOM fallback. | `.claude/skills/dataset-manager/SKILL.md` |
| module-context | Automatically identify which dependency library modules are relevant to a task and load their API documentation into context. Use PROACTIVELY before implementing features, fixing bugs, or writing new operators — analyzes the task description and loads cudf, rmm, duckdb, cucascade module docs to improve code quality. Trigger when the user asks to implement, add, fix, or modify GPU operators, pipeline components, memory management, joins, aggregations, sorting, expressions, or data I/O. | `.claude/skills/module-context/SKILL.md` |
| module-discover | Discover and document a dependency library or submodule — analyzes all uses within the codebase, divides the library into logical modules, identifies which modules are used, and generates LLM-consumable API documentation for each module. Use when the user wants to understand a library dependency, map its modules, or generate API reference docs for a submodule. | `.claude/skills/module-discover/SKILL.md` |
| optimization-advisor | > Use this skill to find exactly which source code to optimize for better GPU performance. Maps nsys profile hotspots to specific Sirius source files and functions, classifies bottlenecks as GPU-bound, CPU-bound, or sync-bound, and recommends actionable code changes. Trigger when the user wants to know what to optimize, where to focus coding effort, or wants source-level optimization guidance. This skill focuses on actionable source code targets — for generating performance reports and measurements, use profile-analyzer instead. | `.claude/skills/optimization-advisor/SKILL.md` |
| profile-analyzer | > Use this skill to understand why a Sirius query is slow, identify GPU bottlenecks, or detect performance regressions. Generates reports with kernel occupancy, memory bandwidth, operator attribution, and cross-run comparisons. Trigger when the user mentions profiling, nsys, GPU utilization, kernel analysis, performance reports, or wants to compare query timings across runs. This skill focuses on measurement and reporting — for mapping hotspots to source code fixes, use optimization-advisor instead. | `.claude/skills/profile-analyzer/SKILL.md` |
| race-check | > Use this skill when query results are non-deterministic, differ between runs, or you suspect data races, deadlocks, or thread safety issues. Uses ThreadSanitizer (CPU) and NVIDIA Compute Sanitizer (GPU) to detect and diagnose race conditions. | `.claude/skills/race-check/SKILL.md` |
| runtime-errors | > Use this skill when a Sirius query crashes, segfaults, hangs, throws an exception, or unexpectedly falls back to CPU. Also use when you see CUDA errors, std::bad_alloc, or the process gets killed. Diagnoses issues using log analysis, cuda-gdb, AddressSanitizer, and NVIDIA Compute Sanitizer. | `.claude/skills/runtime-errors/SKILL.md` |
| update-docs | > Use this skill to update Super Sirius documentation after code changes. Trigger when the user says "update docs", "refresh documentation", "sync docs with code changes", or after merging PRs that changed the Super Sirius codebase. Inspects merged PRs since the last update and patches affected doc files. | `.claude/skills/update-docs/SKILL.md` |
| validate | > Use this skill when a Sirius query returns wrong results, missing rows, extra rows, or incorrect values compared to DuckDB CPU. Pinpoints the faulty operator using per-operator row counts and data checksums. Also detects CUDA stream synchronization issues that cause garbage data. | `.claude/skills/validate/SKILL.md` |
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
