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

**inspectable_mpsc**

A new thread-safe queue class (`inspectable_mpsc<T>`) for the Sirius GPU SQL engine that supports multiple producers and a single consumer, with the ability to inspect and selectively remove elements by predicate. It lives alongside the existing `interruptible_mpmc` in `sirius::exec` and uses `std::unique_ptr<T>` ownership semantics.

**Core Value:** Thread-safe queue with predicate-based element inspection and selective removal (`pop_if`/`get_if`), enabling consumers to find specific items without draining the queue.

### Constraints

- **Tech stack**: C++20, CUDA-compatible, must compile within Sirius build system
- **Pattern**: Header-only template, same style as `interruptible_mpmc.hpp`
- **Location**: `src/include/exec/inspectable_mpsc.hpp`
- **Namespace**: `sirius::exec`
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++ (C++20 standard) - Core extension implementation, physical operators, expression execution, memory management
- CUDA (20 standard) - GPU kernels for cuDF operations, data movement, expression evaluation
- Python (3.12+) - Test harness, performance benchmarking, dataset generation, utilities
- Bash - Build scripts, environment initialization, CI/CD automation
- SQL - Test cases, TPC-H/TPC-DS benchmark queries
- CMake - Build system configuration
- YAML - Configuration files for runtime settings
## Runtime
- Linux (x86_64 and aarch64 architectures)
- CUDA 12 and 13 (configurable via pixi features)
- GPU support: NVIDIA CUDA-enabled GPUs (Turing through Blackwell architectures: 75, 80, 86, 90a, 100f, 120a, 120)
- Pixi (monorepo package and environment management)
- CMake 3.30.4+ (build orchestration)
- vcpkg (alternate C++ dependency management path)
- pip (Python dependencies in duckdb-python environment)
- Ninja (primary build generator)
- GCC or Clang (C++ compiler, configurable)
- NVCC (CUDA compiler)
- sccache (C++ and CUDA build cache)
## Frameworks
- DuckDB 1.4.4 (submodule: `duckdb/`) - SQL execution engine that Sirius extends
- RAPIDS cuDF 26.04.x - GPU DataFrame library for data manipulation
- RMM (RAPIDS Memory Manager) - GPU memory allocation and pooling
- CUDA Runtime (libcudart) - Low-level GPU execution and profiling
- libcurand-dev - CUDA random number generation
- cuCascade (submodule: `cucascade/`) - GPU memory reservation and tiered memory (GPU/host/disk)
- libnuma - NUMA-aware memory operations for CPU pinning
- yaml-cpp - YAML configuration file parsing
- spdlog 1.8.x - Structured logging with file and console outputs
- Abseil (libabseil 20260107.0+) - C++ library utilities (any_invocable, container views)
- fmt (embedded in DuckDB) - String formatting (NOTE: namespace conflict with spdlog requires careful include ordering)
## Key Dependencies
- `cudf::cudf` - libcudf and libcudf-cuda integration for GPU data structures and algorithms
- `rmm::rmm` - Memory allocation, stream management, device memory pools
- `cuCascade::cucascade` - Tiered memory management (GPU/host/disk with deferred transfers)
- `spdlog::spdlog` - Async logging with per-file rotation (config: `src/sirius_context.cpp`)
- `yaml-cpp::yaml-cpp` - Configuration parsing from `SIRIUS_CONFIG_FILE`
- `PkgConfig::NUMA` - libnuma for CPU memory pinning (required for pinned memory buffer manager)
- `absl::any_invocable` - Type-erased callable wrapper for task scheduling
- DuckDB extension infrastructure (via `extension-ci-tools` submodule)
- DuckDB built-in extensions: JSON, TPC-DS, TPC-H (for testing), Parquet, ICU
## Configuration
- Set via pixi features: `pixi shell` (default cuda13), `pixi shell -e cuda12`, `pixi shell -e vcpkg`, `pixi shell -e duckdb-python`
- GPU architectures configured in `pixi.toml` via `CUDAARCHS` env var (feature: cuda13, cuda12, vcpkg)
- Build parallelism: `CMAKE_BUILD_PARALLEL_LEVEL` (default: all cores, reduce if memory-limited)
- Compiler cache: `SCCACHE_GHA_ENABLED` (CI/CD), local sccache for dev builds
- `SIRIUS_CONFIG_FILE` - Path to YAML configuration (default: `~/.sirius.yaml` or `/etc/sirius/config.yaml`)
- `SIRIUS_LOG_DIR` - Log output directory (default: `${CMAKE_BINARY_DIR}/log`)
- `SIRIUS_LOG_LEVEL` - Logging level: trace, debug, info, warn, error (default: info)
- `SIRIUS_DISABLE` - Disable Sirius extension (set to "1" to fall back to DuckDB)
- `SIRIUS_STREAM_CHECK_LIB` - Path to stream check library for debugging (optional, requires build with `-DENABLE_STREAM_CHECK=ON`)
- `CMakeLists.txt` - Main build configuration
- `cmake/CMakePresets.json` - Build presets (debug, release, relwithdebinfo, clang variants, vcpkg variants)
- `extension_config.cmake` - DuckDB extension registration
- `pixi.toml` - Conda/Pixi environment specification with feature flags
- `.pre-commit-config.yaml` - Code formatting and linting hooks (clang-format, black, codespell, cmake-format)
- `.clang-format` - C++/CUDA formatting rules
- `.clang-tidy` - C++ linting rules
- `.codespell_words` - Custom spell-check word list
## Platform Requirements
- Linux x86_64 or aarch64
- NVIDIA GPU with CUDA 12 or 13 capability
- Pixi 0.59+
- Sufficient GPU memory (typically 4GB+ for TPC-H scale factor 1)
- Host RAM adequate for build parallelism (>16GB recommended)
- Static extension: `build/release/extension/sirius/sirius.duckdb_extension`
- Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
- Unit test binary: `build/release/extension/sirius/test/cpp/sirius_unittest`
- Debug build available: `build/debug/extension/sirius/sirius.duckdb_extension`
- Linux x86_64 host with NVIDIA GPU
- DuckDB version 1.4.4 (must match compiled extension)
- CUDA 12 or 13 runtime libraries on system
- No internet access required post-deployment (all deps statically linked or bundled)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- C++ source/header: `snake_case.cpp` / `snake_case.hpp`
- CUDA kernels: `snake_case.cu`
- Python: `snake_case.py`
- Configuration: `*.yaml`, `*.config.*` (e.g., `integration.yaml`, `memory.yaml`)
- Classes use `snake_case`: `sirius_engine`, `sirius_physical_plan_generator`, `sirius_interface`
- Struct types use `snake_case` with suffix `_config` or similar: `task_executor_config`, `operator_params`
- Functions and methods use `snake_case`: `initialize()`, `execute()`, `get_memory_space()`, `copy_null_mask()`
- Private methods: prefixed with underscores only when truly internal
- CUDA kernel functions use `snake_case` with trailing comments: `convert_uint64_to_int32<>()` with template specifiers
- Local variables: `snake_case`: `gpu_result`, `config_path`, `expected_nulls`, `actual_valid`
- Member variables: `snake_case`: `context`, `sirius_iface`, `root_pipeline_idx`, `query_finished`
- Static variables and constants: `SCREAMING_SNAKE_CASE`: `USE_PIN_MEM_FOR_CPU_PROCESSING`, `LOG_DIR`, `MAX_SORT_PARTITION_BYTES`
- Loop counters: simple `i`, `j`, `k` or descriptive names like `row_id`, `col`
- Type-generic naming in templates: `T`, `I`, `B` (block threads, items per thread)
- Enum values: `SCREAMING_SNAKE_CASE`: `AggregationType::COUNT_STAR`, `OrderByType::ASC`, `KernelColType::INT_64`
- Type aliases: `snake_case` when lowercase, `CamelCase` for complex types: `cudf::column_view`, `duckdb::shared_ptr<>`
- Primary namespace: `sirius` (new Super Sirius code)
- Legacy/DuckDB integration: `duckdb` (some shared components like `Config`)
- Nested namespaces use `snake_case`: `sirius::op::scan`, `sirius::pipeline`, `sirius::test`, `cucascade::memory`, `sirius::scan_test_utils`
## Code Style
- Tool: `clang-format` (config in `.clang-format`)
- Column limit: 100 characters
- Indent width: 2 spaces
- No tabs (UseTab: Never)
- Brace style: WebKit (opening braces on same line for most constructs)
- Tool: `clang-tidy` (config in `.clang-tidy`)
- Checks: `modernize-*` (minus selected excluded checks for stylistic/known-broken rules)
- Performance checks: `performance-for-range-copy`, `performance-unnecessary-copy-initialization`, `performance-unnecessary-value-param`
- Static analysis: `clang-analyzer-*` (minus known broken checks)
- WarningsAsErrors: `*` (all warnings are errors)
- Header filter: `.*cudf/cpp/(src|include).*` (primarily filters for cuDF code analysis)
- C++/CUDA: `clang-format` (runs via pre-commit)
- Python: `black` (rev 25.1.0, runs via pre-commit)
- CMake: `cmake-format` (via pre-commit, line-width 220, suppress decorations)
- Spell check: `codespell` (custom words in `.codespell_words`)
## Import Organization
#include "config.hpp"
#include "duckdb/main/database.hpp"
#include "data/sirius_converter_registry.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"
#include <duckdb.hpp>
#include <iostream>
## Error Handling
- **DuckDB exceptions:** Use `duckdb::InvalidInputException`, `duckdb::NotImplementedException`, `duckdb::InternalException` when integrating with DuckDB API
- **Standard C++ exceptions:** Use `std::runtime_error` for configuration/runtime errors, `std::exception` for catch-all
- **Try-catch pattern:** Wrap DuckDB query execution and external library calls
## Logging
- Environment variables: `SIRIUS_LOG_DIR`, `SIRIUS_LOG_LEVEL`
- Defaults in `src/config.cpp`: `LOG_LEVEL = "info"`, `LOG_DIR = "log"`, `LOG_FLUSH_SECONDS = 3`
- Log file pattern: `[YYYY-MM-DD HH:MM:SS.ms] [level] [file:line] message`
- Flush: Every 3 seconds (configurable)
- `SIRIUS_LOG_TRACE(...)` (no-op in CUDA code)
- `SIRIUS_LOG_DEBUG(...)`
- `SIRIUS_LOG_INFO(...)`
- `SIRIUS_LOG_WARN(...)`
- `SIRIUS_LOG_ERROR(...)`
- `SIRIUS_LOG_FATAL(...)` (maps to SPDLOG_CRITICAL)
## Comments
- Explain WHY, not WHAT (code is self-documenting)
- Clarify non-obvious algorithmic choices or GPU-specific quirks
- Mark TODO/FIXME items for future work
- Document assumptions and constraints
## Function Design
- Example: `copy_null_mask()` = 8 lines, `verify_validity_mask()` = 20 lines
- Longer functions acceptable in operator implementations when necessary
- Pass const references for large objects: `const cudf::column_view& col`
- Pass pointers for ownership transfer: `sirius_physical_operator* op`
- Use `std::optional<T>` for nullable values: `std::optional<float> float_tolerance`
- In CUDA kernels, use template parameters for block/item configuration: `template <int B, int I>`
- Use `std::unique_ptr<T>` for exclusive ownership: `duckdb::unique_ptr<QueryResult>`
- Use `std::shared_ptr<T>` for shared ownership: `duckdb::shared_ptr<SiriusContext>`
- Return bool for success/failure checks
- Return void for fire-and-forget operations
## Module Design
- No barrel files (no `index.hpp` pattern)
- Each header is self-contained with necessary includes
- Library entry point via DuckDB extension (`src/sirius_extension.cpp`)
- `src/include/` mirrors `src/` structure
- Public headers in `src/include/`, implementation in `src/`
- CUDA headers in `src/include/cuda/`, kernels in `src/cuda/`
- Header: `src/include/op/sirius_physical_<operator>.hpp` (declares class inheriting `sirius_physical_operator`)
- Implementation: `src/op/sirius_physical_<operator>.cpp` (DuckDB integration)
- Tests: `test/cpp/operator/test_<operator>.cpp`
- Use `private:` / `protected:` / `public:` sections in class declarations
- Friend classes for tight coupling: `friend class pipeline::sirius_pipeline_build_state;`
## C++ Standards and Features
- Range-based for loops: `for (auto& tag : info.tags)`
- Auto type deduction: `auto log_file = log_dir + "/sirius.log";`
- std::optional: `if (needs == env_need::SHARED && !sirius::test::g_shared_env)`
- Move semantics: `auto result_key = std::move(result.keys);`
- RAII: Resources managed via constructors/destructors
- `duckdb::idx_t` for indices
- `duckdb::shared_ptr<T>`, `duckdb::unique_ptr<T>` (DuckDB's wrapper types)
- `duckdb::vector<T>` (DuckDB's vector type)
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- DuckDB extension architecture — loads as `sirius.duckdb_extension`
- Logical-to-physical plan conversion with automatic pipeline splitting
- Multi-threaded task scheduling with dedicated thread pools (GPU executors, scan executors, task creator, downgrade monitors)
- Hierarchical operator pipeline model where operators appear as both sinks and sources across pipeline boundaries
- Unified data repository system with configurable memory barrier semantics (FULL, PARTIAL, PIPELINE)
- Graceful CPU fallback for unsupported operators/data types via `src/fallback.cpp`
- GPU memory spilling via cuCascade tiered memory (GPU → Host → Disk on pressure)
## Layers
- Location: `src/sirius_extension.cpp`
- Purpose: DuckDB extension registration, table function binding/execution, SQL parsing
- Contains: `GPUExecutionBind()` (parses SQL, generates Sirius physical plan), `GPUExecutionFunction()` (delegates to sirius_interface)
- Depends on: DuckDB parser, optimizer, sirius_interface
- Used by: DuckDB query executor (as a table function)
- Location: `src/sirius_interface.cpp`, `src/include/sirius_interface.hpp`
- Purpose: Query lifecycle management, error handling, prepared statement execution
- Contains: `sirius_interface` class with methods: `sirius_execute_query()`, `sirius_pending_statement_internal()`, `fetch_result_internal()`, `cleanup_internal()`
- Depends on: sirius_engine, DuckDB prepared statements
- Used by: Extension layer; returns `MaterializedQueryResult` to DuckDB
- Location: `src/sirius_engine.cpp`, `src/include/sirius_engine.hpp`
- Purpose: Pipeline construction, operator tree traversal, data repository wiring
- Contains: `sirius_engine` class with methods: `initialize()` (builds pipelines), `execute()` (runs query), `insert_repository()` (wires ports with barrier types)
- Pipeline construction (initialize_internal):
- Depends on: Physical plan generator, sirius_context, operators (all in src/op/)
- Used by: sirius_interface; calls pipeline_executor.start_query()
- Location: `src/planner/sirius_physical_plan_generator.cpp`, `src/planner/sirius_plan_*.cpp`
- Purpose: Convert DuckDB logical operators to Sirius physical operators
- Contains: Mapping from DuckDB LogicalOperator types to sirius_physical_operator subclasses
- Key files:
- Depends on: DuckDB logical plan classes
- Used by: Extension layer (binds) and sirius_engine (initialize)
- Location: `src/op/`, `src/include/op/`
- Purpose: Physical operator implementations with GPU and CPU fallback logic
- Categories:
- Base class: `sirius_physical_operator` (`src/include/op/sirius_physical_operator.hpp`, `src/op/sirius_physical_operator.cpp`)
- Depends on: Expression executor (GPU kernels), cuDF, RMM, cuCascade
- Used by: sirius_engine (during initialize), gpu_pipeline_executor (during execute)
- Location: `src/pipeline/`, `src/include/pipeline/`
- Purpose: Task scheduling, GPU executor management, execution orchestration
- Key files:
- Thread Model:
- Depends on: Operators, data repositories, memory manager, task creator, cuCascade
- Used by: sirius_engine (calls start_query), query completion handler
- Location: `src/creator/task_creator.cpp`, `src/include/creator/task_creator.hpp`
- Purpose: Convert ready operators into executable tasks, follow data availability chain
- Contains: `task_creator` class with method `schedule(operator*)` — pops from task creation queue, calls `get_operator_for_next_task()` to follow hint chain, creates GPU pipeline or scan task
- Task decision logic: checks `operator->get_next_task_hint()` (READY vs WAITING_FOR_INPUT_DATA), recursively follows producer chain
- Depends on: Operators, pipeline executor, data repositories
- Used by: Scan executors and GPU executors (call `schedule()` after task completion)
- Location: `src/memory/`, `src/include/memory/`
- Purpose: GPU/Host/Disk tiered memory management via cuCascade, reservation tracking
- Key files:
- Integration: GPU executor reserves memory before executing; downgrade executor monitors pressure and moves batches; RMM pools allocated at startup
- Depends on: cuCascade, RMM, DuckDB memory allocators
- Used by: GPU executors (reserve), downgrade executors (spill), scan executors (host memory for parquet I/O)
- Location: `src/downgrade/`, `src/include/downgrade/`, `src/fallback.cpp`
- Purpose: GPU memory spilling (GPU → Host) and CPU fallback for unsupported operations
- Downgrade: `src/include/downgrade/downgrade_executor.hpp` — monitors GPU memory pressure, dispatches spill tasks
- Fallback: `src/fallback.cpp` — if operator throws `NotImplementedException` or data type not supported, reverts to DuckDB CPU execution
- Depends on: Pipeline executor, data repositories, memory manager
- Used by: GPU executors (catch OOM), extension layer (graceful fallback)
- Location: `src/expression_executor/`, `src/cuda/expression_executor/`, `src/include/expression_executor/`
- Purpose: GPU-accelerated expression evaluation via cuDF AST
- Key files:
- Methods: `select()` (filter), `project()` (projection), aggregate functions
- Depends on: cuDF, expression AST (DuckDB), CUDA
- Used by: Operators during execute() phase
- Location: `src/data/`, `src/include/data/`
- Purpose: Data batch lifecycle, repositories, port routing, format conversion
- Key components:
- Sirius Converter Registry: Converts between DuckDB chunks and GPU-compatible batch formats
- Depends on: cuCascade, DuckDB chunk format
- Used by: All executors (push/pop batches), operators (input/output)
- Location: `src/include/sirius_context.hpp`, `src/sirius_engine.cpp`, `src/include/sirius_config.hpp`
- Purpose: Ownership hierarchy and lifetime management of all subsystems
- `SiriusContext` (ClientContextState subclass) owns:
- Lifecycle: `initialize()` on first query, `QueryBegin()`/`QueryEnd()` per query, `terminate()` on connection close
- Used by: Extension (registers on connection), interface (retrieves per query)
## Data Flow
- If operator throws exception: GPU executor catches, calls `completion_handler->report_error()`
- `drain_after_error()` stops task creator, drains queues, waits for executors
- Error propagates through future to main thread
- Operators maintain state via ports and repositories
- Scan operators track `exhausted` (DuckDB) or `has_more_partitions` (Parquet) atomics
- Blocking operators accumulate via `sink()`, emit via `source` + `execute()` in child pipeline
- Pipeline completion determined by `pipeline_finished` atomic + source depletion check
- Data batches transition through state machine: idle → task_created → processing → idle (or in_transit → idle for downgrades)
## Key Abstractions
- Purpose: Abstract physical operation with GPU acceleration and CPU fallback
- Base: `sirius_physical_operator` (src/include/op/sirius_physical_operator.hpp)
- Categories: Scan (produce data), Streaming (pass-through), Blocking (accumulate), Control (route/collect)
- Examples:
- Purpose: Ordered sequence of operators executing in a single batch of work
- Definition: `source` (first), `operators` list (all), `sink` (last)
- Blocking operators appear as both sink of one pipeline and source of another
- Execution: one task iterates all operators calling `execute()`, then calls sink's `sink()`
- Completion: source drained + ports empty + tasks done
- Purpose: Thread-safe queue of data batches between pipelines
- Keyed by: (operator_id, port_id)
- Registered centrally in `shared_data_repository_manager`
- Supports partitioned storage and batch state machine
- Barrier types: FULL (synchronize), PARTIAL (incremental), PIPELINE (no sync)
- Purpose: Control data flow across pipeline boundaries
- **FULL**: Downstream waits for upstream completion (hash join build)
- **PARTIAL**: Downstream can consume as data arrives (CONCAT after PARTITION in streaming joins)
- **PIPELINE**: No synchronization (within pipeline)
- Purpose: Schedulable unit of work (scan or GPU pipeline execution)
- Types: `scan_task` (DuckDB/Parquet I/O), `gpu_pipeline_task` (operator chain on CUDA stream)
- Contains: input data batch references, pipeline reference, sink operation
- Lifecycle: created by task creator, routed to executor, executed atomically
- Purpose: Atomic GPU memory allocation before task execution
- Supports: Multiple memory spaces (GPU, Host, Disk via cuCascade)
- Interface: `reserve()` (acquire), `release()` (free) on SiriusContext memory manager
- OOM handling: Task retry with exponential backoff (up to 10 retries)
## Entry Points
- Location: `src/sirius_extension.cpp`, `LoadInternal()` function
- Triggers: `LOAD 'sirius.duckdb_extension'` from SQL
- Responsibilities: Register `gpu_execution` table function, set up extension callbacks
- Location: `src/sirius_extension.cpp`, `GPUExecutionFunction()` (table function execute)
- Triggers: `CALL gpu_execution('SELECT ...')`
- Responsibilities: Parse, generate physical plan, invoke sirius_interface
- Location: `src/sirius_interface.cpp`, `sirius_execute_query()`
- Triggers: Called from extension table function
- Responsibilities: Lifecycle setup, delegate to sirius_engine, extract result
- Location: `src/sirius_engine.cpp`, `initialize()` and `execute()`
- Triggers: Called from sirius_interface
- Responsibilities: Build pipelines, start execution, collect result
- Location: `src/pipeline/pipeline_executor.cpp`, `start_query()`
- Triggers: Called from sirius_engine.execute()
- Responsibilities: Create completion handler, schedule initial tasks, route to sub-executors
## Error Handling
## Cross-Cutting Concerns
- Framework: spdlog (configured in `src/include/log/logging.hpp`)
- Levels: trace, debug, info, warn, error
- Environment: `SIRIUS_LOG_LEVEL=debug`, `SIRIUS_LOG_DIR=/path` (default: build/log)
- Macro: `SIRIUS_LOG_*()` throughout codebase
- Files log to `build/log/sirius_*.log`
- Type checking: Operators validate input/output types match operator requirements
- Expression validation: Filter/projection expressions validated during planning
- Port validation: Ports verified connected during pipeline finalization
- Batch validation: Data batches validated on push/pop with size/count checks
- Not applicable (in-process GPU execution, no network)
- Memory safety: Uses smart pointers (shared_ptr, unique_ptr) throughout
- Thread safety: Mutexes on shared state (repositories, memory manager), atomics for counters
- NVTX ranges: nvtx3 markers for profiler integration (`src/sirius_engine.cpp`, operators)
- Task counting: `pipeline->mark_task_created()`, `mark_task_completed()` track per-pipeline stats
- Performance hooks: `completion_handler` tracks overall query timing
- Metrics: Row counts per operator can be parsed from logs via `tools/parse_pipeline_log.py`
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
