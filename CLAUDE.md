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
- Links against: cudf::cudf, rmm::rmm, libnuma, yaml-cpp, absl::any_invocable, spdlog, cuCascade

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
| Dataset Manager | `/dataset-manager` | Generates benchmark datasets (TPC-H, TPC-DS, etc.) at any scale factor in parquet or duckdb format. |
| Optimization Advisor | `/optimization-advisor` | Maps GPU hotspots from nsys profiles to source functions, detects efficiency bottlenecks, sync overhead, and parallelism opportunities. |
| Benchmark | `/benchmark` | Runs TPC-H or TPC-DS benchmarks on Super Sirius or DuckDB CPU baseline — generate data, execute queries, validate results, and compare timings. |
| Module Context | `/module-context` | **Auto-loaded before implementation tasks.** Identifies which dependency modules are relevant to a task and loads their API docs (signatures, descriptions, usage examples). |
| Module Discover | `/module-discover` | Analyzes a dependency library, divides it into modules, and generates LLM-consumable API documentation. Run once per library to populate docs. |

**Useful debugging tools:**
- `tools/parse_pipeline_log.py`: Parses Sirius pipeline logs to show per-operator row counts for debugging incorrect query results.

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Sirius data_batch API Refactoring**

A refactoring of the Sirius GPU SQL engine to adopt cucascade's new 3-class data_batch API (commit d9dc331). The old API exposed data/memory/tier directly on data_batch with manual state machine transitions (idle, task_created, in_transit, processing). The new API makes data_batch an opaque idle handle, requiring RAII accessor types — `read_only_data_batch` (shared lock) or `mutable_data_batch` (exclusive lock) — to access or mutate data. This affects ~32 files and ~94 call sites across Sirius's pipeline, operator, and downgrade subsystems.

**Core Value:** Sirius compiles cleanly against cucascade commit d9dc331 with the new 3-class data_batch API, preserving the existing execution semantics.

### Constraints

- **API compatibility**: Must use cucascade commit d9dc331 exactly — no modifications to cucascade
- **Semantic preservation**: Existing execution flow (pipeline → operator → data flow) stays the same, just using new accessor types
- **Brownfield**: This is a targeted refactoring within an active codebase — minimize changes outside the data_batch API boundary
- **Blocking pattern**: Use blocking `to_mutable()` in downgrade/convert paths (not try-based)
- **Non-blocking reads**: Use `to_read_only()` for all read-only data access on idle batches
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++ 20 - Core GPU-native SQL engine implementation
- CUDA 20 - GPU kernels and NVIDIA CUDA-X library integration
- Python 3 - Performance testing, tooling, and data generation
- CMake - Build configuration and compilation orchestration
- YAML - Configuration files for runtime settings
## Runtime
- Linux (x86_64, aarch64) - Primary deployment platform
- CUDA 12.x or 13.x - GPU execution runtime (feature-selectable)
- NVIDIA GPU - Turing through Blackwell architectures (compute capability 75-120)
- Pixi (Recommended) - Conda-based environment and dependency management
- pip - Python package installation for DuckDB Python bindings
- CMake - Cross-platform build system and dependency orchestration
- `pixi.lock` - Comprehensive pinned environment definition for reproducible builds
## Frameworks
- DuckDB 1.5.2 - Modular SQL database engine (submodule: `duckdb/`)
- Sirius Extension - Custom DuckDB extension implementing GPU acceleration
- RAPIDS cuDF 26.04 - GPU DataFrame operations (joins, aggregations, sorting)
- RMM (RAPIDS Memory Manager) - GPU memory allocation and pool management
- cuCascade - GPU memory reservation and tiered memory management (submodule: `cucascade/`)
- Ninja - Fast parallel build system
- CMake 4.x - Build configuration with version 3.30.4 minimum
- sccache - Compiler result caching for faster rebuilds
- Clang 21 / GCC - Compiler toolchain with unified standard configuration
- Catch2 - C++ unit testing framework for component tests (via DuckDB's bundled headers)
- Custom SQL logic test runner - Integrated with DuckDB test infrastructure
- pre-commit - Git hooks for code quality checks
- clang-format 20.1.4 - C++ code formatting
- clang-tidy - C++ static analysis and linting
- black - Python code formatting
- codespell - Spell checking with custom word list
- cmake-format - CMake code formatting
## Key Dependencies
- `libcudf::cudf` (26.04) - Accelerated GPU DataFrame operations (joins, groupby, aggregations, sorting)
- `rmm::rmm` - GPU memory management with fallback support
- `cuCascade::cucascade` - Tiered memory (GPU/host/disk) with reservation semantics
- `duckdb::duckdb` (1.5.2) - SQL parsing, planning, CPU fallback execution
- `spdlog::spdlog` (1.8) - High-performance logging with file sinks
- `yaml-cpp` - YAML configuration file parsing (for `sirius.yaml`)
- `libabseil` (absl) - Abseil C++ library utilities (any_invocable)
- `PkgConfig::NUMA` - NUMA-aware memory operations
- `librmm` - RMM static library link target
- `cuda-nvml-dev` - NVIDIA ML library for GPU monitoring/diagnostics
- `cuda-nvcc` - NVIDIA CUDA compiler
- `libcurand-dev` - NVIDIA random number generation on GPU
- `sqlite` (3.x) - Test infrastructure support
## Configuration
- Expression executor strategy (AST interpretation)
- GPU memory region usage (pinned memory for CPU processing/caching)
- Table scan optimization (8 CUDA streams, 64 MB memcpy threshold)
- Logging level (info) and flush interval (3 seconds)
- Scan task batch size (512 MB)
- `debug` - GCC debug build with `-g -O0`
- `release` - GCC optimized release build
- `relwithdebinfo` - GCC release with debug symbols
- `clang-*` variants - Using Clang 21 as host compiler
- `legacy-release` - Legacy code path support (optional)
- `vcpkg-*` - Static linking via vcpkg package manager
- `CMAKE_CUDA_COMPILER_LAUNCHER=sccache` - Compiler caching
- `CMAKE_CXX_COMPILER_LAUNCHER=sccache`
- `CMAKE_C_COMPILER_LAUNCHER=sccache`
- `CMAKE_LINKER_TYPE=MOLD` - Fast linker
- `DUCKDB_EXTENSION_CONFIGS` - Points to `extension_config.cmake`
- `EXTENSION_STATIC_BUILD=ON` - Static extension linking
- `SIRIUS_LOG_DIR` - Log output directory (default: `${CMAKE_BINARY_DIR}/log`)
- `SIRIUS_LOG_LEVEL` - Log verbosity (trace, debug, info, warn, error)
- `CUDAARCHS` - GPU compute capabilities (feature-selected via pixi)
- `VCPKG_CUDA_VERSION` - CUDA version selection for vcpkg build
- C++ standard: 20 (CXX_STANDARD_REQUIRED ON)
- CUDA standard: 20 (CUDA_STANDARD_REQUIRED ON)
- CUDA separable compilation: ON
- CUDA device symbol resolution: ON
## Platform Requirements
- Linux system (x86_64 or aarch64)
- Pixi >=0.59 (for environment management)
- CMake >=3.30.4
- CUDA toolkit (12.x or 13.x feature-selected)
- NVIDIA GPU (Turing 75 or newer)
- C++20 compatible compiler (GCC or Clang 21)
- NUMA-aware system libraries (libnuma)
- Python 3.x (for performance testing tools)
- pybind11 + scikit-build-core (for Python API building)
- Deployment target: Linux systems with NVIDIA GPUs
- DuckDB application environment with extension loading capability
- Sufficient GPU memory for data processing (tiered fallback to host/disk)
- Write access to log directory (default or custom via `SIRIUS_LOG_DIR`)
- Via CUDA 13: Turing (75), Ampere (80, 86), Ada (90a), Hopper (100f, 120a), Blackwell (120)
- Via CUDA 12: Turing (75), Ampere (80, 86), Ada (90a) - Hopper/Blackwell require CUDA 13+
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- All lowercase with underscores: `sirius_engine.cpp`, `sirius_physical_operator.hpp`
- CUDA kernels: `.cu` extension in `src/cuda/` directories
- Headers mirror source structure in `src/include/` using same naming
- snake_case: `initialize_test_buffer_manager()`, `calculate_test_cpu_cache_size()`
- Member functions use snake_case: `get_data_batches()`, `prepare_for_processing()`
- Local variables: snake_case: `num_records`, `gpu_column`, `cpu_cache_bytes`
- Member variables: prefixed with underscore: `_data_batches`, `_use_custom_hint`, `_custom_hint`
- Static members: PascalCase: `Config::LOG_DIR`, `Config::USE_PIN_MEM_FOR_CPU_PROCESSING`
- Constants: ALL_CAPS: `CPU_CACHE_TEST_MEM_SF`, `SIRIUS_UNITTEST_LOG_DIR`
- Enum classes: PascalCase: `TaskCreationHint`, `MemoryBarrierType`, `OrderByType`
- Struct/Class names: snake_case: `sirius_engine`, `sirius_physical_operator`, `shared_env_listener`
- DuckDB types use full namespace: `duckdb::shared_ptr<>`, `duckdb::unique_ptr<>`
## Namespace Organization
- `namespace sirius {}` - Super Sirius (new code path): Contains `sirius_engine`, operators, pipeline infrastructure, memory management, data types
- `namespace duckdb {}` - Legacy/integration layer: DuckDB extension integration, logging, configuration
- Nested namespaces follow domain:
## Code Style
- Tool: clang-format 20.1.4
- Configuration: `.clang-format` defines all style rules
- Applied automatically via pre-commit hooks (see `.pre-commit-config.yaml`)
- Run formatting: `clang-format -fallback-style=none -style=file -i <file>`
- Indentation: 2 spaces (not tabs)
- Column limit: 100 characters
- Brace style: WebKit (opening brace on same line, no space before)
- Pointer alignment: Left (e.g., `int* ptr` not `int *ptr`)
- No space after C-style casts: `(int)value` not `(int) value`
- Constructor init lists: Break before colon
- Always break template declarations
- Tool: clang-tidy with custom configuration (`.clang-tidy`)
- Enabled checks: modernize, performance, clang-analyzer
- Warnings treated as errors: `WarningsAsErrors: '*'`
- Disabled checks: use-equals-default, concat-nested-namespaces, use-trailing-return-type, use-bool-literals, use-designated-initializers (all stylistic or C++20 specific)
- Run linting: `pre-commit run clang-tidy -a`
## Import Organization
#include "sirius_engine.hpp"                      // local header
#include "log/logging.hpp"                        // local log header
#include "op/sirius_physical_table_scan.hpp"      // local operator header
#include "pipeline/sirius_pipeline_converter.hpp" // local pipeline header
#include "sirius/exception.hpp"                   // sirius namespace header
#include <vector>
#include <memory>
#include <string>
## Error Handling
- `sirius::internal_exception` - Invariant violations, internal logic errors
- `sirius::not_implemented_exception` - Feature not yet implemented
- `sirius::invalid_input_exception` - Invalid input parameters, precondition failures
- Use Catch2 assertions: `REQUIRE_THROWS_AS(expression, exception_type)`
- Example: `REQUIRE_THROWS_AS(r.required("name", name), std::runtime_error);`
- Integration code uses DuckDB's exception system
- Fallback throws `duckdb::InvalidInputException`
## Logging
- `SIRIUS_LOG_TRACE(...)` - Trace level
- `SIRIUS_LOG_DEBUG(...)` - Debug level
- `SIRIUS_LOG_INFO(...)` - Info level (default)
- `SIRIUS_LOG_WARN(...)` - Warning level
- `SIRIUS_LOG_ERROR(...)` - Error level
- `SIRIUS_LOG_FATAL(...)` - Critical/fatal level
- CUDA compilation cannot include spdlog headers
- All logging macros are no-ops in `.cu` files (see `#ifdef __CUDACC__`)
- Called in `test/cpp/unittest.cpp`: `InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, flush_seconds)`
- Log output: `${SIRIUS_LOG_DIR}/sirius.log`
- Pattern: `[%Y-%m-%d %T.%e] [%l] [%s:%#] %v` (timestamp, level, file:line, message)
- Set log level via env var: `SIRIUS_LOG_LEVEL=debug` (before initialization)
- Change at runtime: `SetGlobalLogLevel("warn")`
- Flush interval: `SetGlobalLogFlush(flush_seconds)`
## Comments
- Document public APIs with intent/usage
- Explain non-obvious logic, especially in GPU code
- Mark deprecated code or workarounds
- High-level overview comments at start of complex functions
- Use `/** ... */` for public APIs
- Common tags:
## Function Design
- Pass by const reference for input: `const std::string& query`
- Pass by reference for output/modification: `duckdb::ClientContext& context`
- DuckDB uses smart pointers: `duckdb::shared_ptr<>`, `duckdb::unique_ptr<>`
- Move semantics for heavy objects: `std::vector<...> data_batches` or `std::move(...)`
- Use `std::optional<T>` for nullable returns: `std::optional<std::vector<...>> prepare_for_processing(...)`
- Return by value for small types (enums, small structs)
- Return const reference for large read-only data
- Use `[[nodiscard]]` attribute for important return values
## Module Design
- One primary class/interface per header file
- Name matches file name
- Include guard: `#pragma once` (not `#ifndef`)
- All includes at top, organized by priority (see Import Organization)
- Public interfaces via namespace + class name
- Private implementation in `.cpp` files
- Use `namespace {}` anonymous blocks for internal helpers in `.cpp`
#pragma once
- Not commonly used; most code includes specific headers
- When used, typically in `include/` for convenience exports
## Header Organization
#pragma once
#include <vector>
#include <memory>
#include "duckdb/main/connection.hpp"
#include "duckdb/planner/physical_operator.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"
## CUDA Kernel Conventions
- Kernel functions: `__global__ void kernel_name(...)`
- Device functions: `__device__ void device_function(...)`
- Device types: Use cuDF types (`cudf::bitmask_type`, `cudf::mutable_column_view`)
- NO logging allowed - spdlog/fmt incompatible with nvcc
- Use assertions for debugging: `assert(condition)` in debug builds only
- Coordinate with CPU-side logging in wrapper functions
- Use RMM for GPU memory allocation: `rmm::cuda_stream_view stream`
- cuCascade handles CPU↔GPU transfers via data batches
- No manual `cudaMalloc`/`cudaFree` - managed via RMM
- CMake: `CMAKE_CUDA_SEPARABLE_COMPILATION ON`
- Standard: `--std=c++20`
- GPU architectures: 75, 80, 86, 90a, 100f, 120a, 120 (Turing through Blackwell)
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- DuckDB extension entry point that intercepts logical plans and converts them to GPU-executable physical plans
- Task-based pipeline execution model with materialized pipelines and stream-based data flow
- Central context ownership model (`SiriusContext`) managing all subsystems per connection
- Multi-threaded executor architecture with dedicated thread pools for GPU, scanning, task creation, and memory management
- Graceful CPU fallback mechanism when GPU constraints are reached or unsupported operations encountered
## Layers
- Purpose: Register Sirius as a DuckDB extension, expose table functions (`gpu_execution`, `gpu_buffer_init`), manage configuration
- Location: `src/sirius_extension.cpp`, `src/include/sirius_extension.hpp`
- Contains: Extension registration (`Load`, `LoadInternal`), table function bindings (`GPUExecutionBind`, `GPUExecutionFunction`), config registration (`InitialGPUConfigs`)
- Depends on: DuckDB extension API, `sirius_interface`, `sirius_prepared_statement_data`
- Used by: DuckDB runtime when queries call `CALL gpu_execution(...)` or `LOAD` the extension
- Purpose: Mediate between DuckDB's query execution context and Sirius's GPU engine; manage active query context lifecycle
- Location: `src/sirius_interface.cpp`, `src/include/sirius_interface.hpp`
- Contains: `sirius_interface` class managing query execution flow, `sirius_prepared_statement_data` wrapping logical plan + physical plan, `sirius_active_query_context` holding engine and prepared data
- Depends on: `sirius_engine`, DuckDB client context, `sirius_prepared_statement_data`
- Used by: `sirius_extension` for query invocation, handles query lifecycle (`begin_query_internal`, `fetch_result_internal`, `cleanup_internal`)
- Purpose: Convert DuckDB logical plans to Sirius physical operator trees; dispatch to operator-specific builders
- Location: `src/planner/sirius_physical_plan_generator.cpp` (dispatcher), `src/planner/sirius_plan_*.cpp` (operator-specific builders)
- Contains: `sirius_physical_plan_generator::create_plan()` (entry point with type dispatcher), operator builders for each DuckDB logical operator (filter, aggregate, join, order, scan, etc.)
- Depends on: DuckDB logical operator types, `sirius_physical_operator` hierarchy
- Used by: `sirius_interface` during bind phase to generate physical plan from logical plan
- Purpose: Define GPU-executable operator implementations; each operator knows how to execute on GPU via cuDF or custom CUDA kernels
- Location: `src/op/sirius_physical_*.cpp` implementations, `src/include/op/sirius_physical_operator.hpp` base class, `src/cuda/` for GPU kernels
- Contains: Base class `sirius_physical_operator` with virtual `execute()` method, derived operators (FILTER, PROJECTION, HASH_JOIN, UNGROUPED_AGGREGATE, TABLE_SCAN, etc.), each with CPU-side orchestration and GPU kernel dispatch
- Depends on: cuDF API, RMM, custom CUDA kernels, DuckDB expressions, data batch structures
- Used by: `sirius_engine` during execution; called via `execute()` in pipeline tasks
- Purpose: Construct materialized pipelines from physical plan, manage pipeline build state, execute operators on CUDA streams
- Location: `src/include/pipeline/sirius_pipeline.hpp` (pipeline definition), `src/pipeline/sirius_pipeline_converter.cpp` (physical plan to pipeline conversion), `src/include/pipeline/pipeline_executor.hpp` (executor), `src/include/pipeline/gpu_pipeline_executor.hpp` (GPU-specific executor)
- Contains: `sirius_pipeline` (ordered operator list with source/sink/dependencies), `sirius_meta_pipeline` (groups related pipelines), `sirius_pipeline_build_state` (controlled access during construction), `pipeline_executor` (top-level orchestrator), `gpu_pipeline_executor` (per-GPU task execution)
- Depends on: `sirius_physical_operator`, CUDA streams, memory reservations
- Used by: `sirius_engine` during initialization and execution phases
- Purpose: Create and dispatch tasks (scan tasks, GPU pipeline tasks) based on data availability; schedule work across GPU and CPU scan executors
- Location: `src/include/creator/task_creator.hpp` (task creation), `src/op/scan/duckdb_scan_executor.hpp` (DuckDB table scans), `src/op/scan/parquet_scan_task.cpp` (Parquet file scans)
- Contains: `task_creator` polls for ready operators and creates tasks, scan executors pull data from storage/sources and publish to data repositories, GPU executors consume tasks
- Depends on: `sirius_physical_operator`, data repositories, scan operators
- Used by: `pipeline_executor` and `gpu_pipeline_executor` to schedule work
- Purpose: Manage GPU, host, and disk memory via cuCascade; track reservations; handle spilling during memory pressure
- Location: `src/include/memory/sirius_memory_reservation_manager.hpp`, `src/include/downgrade/downgrade_executor.hpp`
- Contains: `sirius_memory_reservation_manager` (wrapper around cuCascade's data repository manager and memory pools), `downgrade_executor` (spills GPU data to host when memory pressure detected)
- Depends on: cuCascade library, RMM
- Used by: GPU executor (reserves memory before task execution), downgrade executor (monitors pressure and spills data)
- Purpose: Manage batched data flowing between operators; convert between DuckDB and GPU formats; cache intermediate results
- Location: `src/data/` (converters), `src/include/data/` (headers), data repositories via cuCascade
- Contains: `convertible_data_batch` (data batch wrapper), converters for different data sources (DuckDB columns, Parquet, etc.), registry of converters
- Depends on: cuCascade data repositories, DuckDB column structures, Parquet reader
- Used by: Operators during execution to transform data between formats
- Purpose: Central ownership of all subsystems within a DuckDB connection; lifecycle management
- Location: `src/include/sirius_context.hpp`, `src/sirius_context.cpp`
- Contains: `SiriusContext` owns `sirius_config`, `sirius_memory_reservation_manager`, `pipeline_executor`, `task_creator`, `downgrade_executor`, data repository manager
- Depends on: All subsystems above
- Used by: DuckDB connection lifecycle callbacks; all subsystems access context for shared resources
## Data Flow
- Task executor acquires memory reservation before dispatching operator tasks
- Each operator's `execute()` receives CUDA stream for any data movement
- Batches locked via `prepare_for_processing()` before operator execution
- After operator finishes, batches remain locked until next pipeline stage
- Downgrade executor monitors memory pressure periodically
- When threshold exceeded, moves GPU batches to host via data repository downgrades
- Task scheduler respects memory availability — reschedules tasks if locks fail
## Key Abstractions
- Purpose: Represents an executable GPU operation; defines how a logical operation maps to GPU computation
- Examples: `sirius_physical_filter`, `sirius_physical_hash_join`, `sirius_physical_ungrouped_aggregate`, `sirius_physical_table_scan`
- Pattern: Hierarchy with virtual `execute(operator_data, stream)` method; some operators are blocking (need child side done first), others streaming; each has optional source and sink logic
- Entry point: Defined via `create_plan()` methods in `src/planner/sirius_plan_*.cpp` files
- Purpose: Ordered sequence of operators executed together on CUDA streams; represents a single logical execution unit
- Pattern: Operators added during construction, then finalized when pipeline becomes ready; source/sink tracked separately until finalization, then merged into single operators list
- Lifecycle: Created during `initialize()`, executed when dependencies met, finalized when all tasks complete
- Key fields: `operators` (all ops source to sink), `source`, `sink`, `dependencies` (blocking pipelines), `tasks_created/completed` (progress tracking)
- Purpose: GPU-resident or host-resident data container; supports tiered memory model (GPU → host → disk)
- Pattern: Immutable during operator execution; locked via `prepare_for_processing()` to ensure memory space availability; moved between spaces by downgrade executor
- Used by: Operators consume/produce data_batch objects; repositories store and retrieve them
- Purpose: RAII guard ensuring GPU/host memory available for task execution
- Pattern: Acquired before task dispatch, released when task complete or batch spilled to disk
- Used by: GPU executor acquires before calling operator tasks; downgrade executor monitors available reservations
- Purpose: Named input/output port connecting pipelines; stores intermediate batches with flow control
- Pattern: Each inter-pipeline boundary has repository identified by port_id; task creator polls repositories for ready data, creates downstream tasks
- Used by: Scan operators publish results, pipeline operators consume from upstream repositories
## Entry Points
- Location: `src/sirius_extension.cpp` (`SiriusExtension::GPUExecutionBind`, `SiriusExtension::GPUExecutionFunction`)
- Triggers: `CALL gpu_execution('SELECT ...')`
- Responsibilities: Parse query string, bind DuckDB logical plan, generate physical plan, execute via `sirius_interface`
- Location: `src/sirius_extension.cpp` (`SiriusExtension::Load`, `LoadInternal`)
- Triggers: `LOAD` command or automatic on connection creation (if pre-loaded)
- Responsibilities: Register table functions, initialize config options, register optimizer extension for transparent execution
- Location: `src/transparent/sirius_optimizer_extension.cpp` (`sirius_optimizer_hook`)
- Triggers: Automatically when `gpu_execution` config is true (after optimizer runs)
- Responsibilities: Check if logical plan is supported, if so convert to physical plan and execute, else fallback to DuckDB
- Location: `src/sirius_interface.cpp` (`sirius_execute_query`)
- Triggers: From `sirius_extension::GPUExecutionFunction` or transparent execution
- Responsibilities: Create engine, initialize with physical plan, execute, return results
## Error Handling
- **Plan Generation Failure:** If `create_plan()` throws `NotImplementedException` (unsupported operator), caught in `GPUExecutionBind` with `Config::ENABLE_DUCKDB_FALLBACK` check; query retried on CPU via `run_internal_cpu_fallback_query()`
- **Execution Failure:** If operator `execute()` throws or data batch fails to lock (out of GPU memory), task is rescheduled; if consistent failure, fallback mechanism triggered
- **Type Unsupported:** When DuckDB type not mappable to GPU type (e.g., nested types), filter operator created post-scan to filter in GPU (if table function doesn't support pushdown)
- **Memory Pressure:** When GPU memory exceeds threshold, downgrade executor moves batches to host; if host exhausted, spills to disk via cuCascade
## Cross-Cutting Concerns
- Centralized via `src/include/log/logging.hpp` using spdlog
- Environment variables: `SIRIUS_LOG_DIR`, `SIRIUS_LOG_LEVEL` (trace/debug/info/warn/error/critical/off)
- Config options: `sirius_log_level`, `sirius_log_dir`, `sirius_log_flush_seconds`
- Used throughout codebase via `SIRIUS_LOG_DEBUG()`, `SIRIUS_LOG_INFO()`, etc.
- Operator type checks via `D_ASSERT` macros in debug builds
- Expression validity checked during `create_plan()` with fallback if unsupported
- Data type conversion validation in converters with type mismatch detection
- Memory reservation checks before task execution ensure GPU memory available
- NVIDIA NVTX markers for profiler integration (`nvtx3::scoped_range`)
- Per-operator NVTX ranges during `execute()` calls
- Pipeline completion tracking via atomic counters (`tasks_created`, `tasks_completed`)
- Performance metrics logged at query completion (time, throughput)
- Profiler control functions (`profiler_start`, `profiler_stop`) for nsys capture ranges
- Global `Config` class in `src/config.cpp` holds flags and parameters
- Per-connection `sirius_config` in `SiriusContext` holds operator parameters
- `SET` commands can modify config at runtime (e.g., `SET use_pin_memory = true`)
- Config options registered in `InitialGPUConfigs()` with type and setter callbacks
- `sirius_engine` finalization of pipeline operators not thread-safe (runs on DuckDB query thread)
- Pipeline executor uses task queues for thread-safe work distribution
- Atomic counters for pipeline progress (`tasks_created`, `tasks_completed`)
- Mutex + condition variable for query completion signaling (`query_finish_mutex`, `query_finish_cv`)
- Data repositories use internal synchronization for concurrent reads
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

| Skill | Description | Path |
|-------|-------------|------|
| benchmark | > Run TPC-H or TPC-DS benchmarks on Super Sirius or DuckDB CPU baseline — generate data, execute queries, validate results, and compare timings. Trigger when the user mentions benchmarking, TPC-DS, TPC-H, performance testing, query runtimes, or wants to compare Sirius vs DuckDB speed. | `.claude/skills/benchmark/SKILL.md` |
| bisect | > Use this skill to find which commit introduced a bug or regression. Uses git bisect with automated build and test. Trigger when a bug appeared recently, a query started failing, performance regressed, or the user wants to compare behavior between two commits. | `.claude/skills/bisect/SKILL.md` |
| build-errors | > Use this skill when the build fails, compilation errors occur, or you see undefined references, linker errors, CUDA compilation issues, missing headers, or template instantiation failures. Analyzes errors, suggests fixes, and iteratively rebuilds until success. | `.claude/skills/build-errors/SKILL.md` |
| config-optimizer | > Use this skill to find the optimal Sirius configuration for TPC-H workloads at any scale factor. Trigger when the user wants to tune performance, optimize config parameters, find the best thread count, batch size, or cache mode, or benchmark different Sirius configurations against each other. Also use when the user mentions "config tuning", "parameter sweep", or "optimal settings". | `.claude/skills/config-optimizer/SKILL.md` |
| dataset-manager | > Use this skill to generate benchmark datasets (TPC-H, TPC-DS, etc.). Trigger when the user needs test data at a specific scale factor for benchmarking or testing. Supports parquet and duckdb output formats. | `.claude/skills/dataset-manager/SKILL.md` |
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
