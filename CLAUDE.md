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
source setup_sirius.sh        # Set SIRIUS_HOME_PATH and LDFLAGS
```

**Manual Setup:**
```bash
source setup_sirius.sh
export LIBCUDF_ENV_PREFIX=/path/to/miniconda3/envs/libcudf-env  # If using conda
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
cd duckdb-python
pip install .
cd ..
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

### Two Code Paths (Legacy vs New Sirius)

Sirius has two parallel execution modes, both coexisting in `src/`:

**Legacy Sirius** (`gpu_processing`):
- Uses `namespace duckdb`
- Entry point: `CALL gpu_processing('SELECT ...')`
- Physical plan generator: `GPUPhysicalPlanGenerator` (`src/gpu_physical_plan_generator.cpp`)
- Operators: `GPUPhysicalOperator` subclasses in `src/operator/` (e.g., `gpu_physical_hash_join.cpp`)
- Plan builders: `src/plan/` (e.g., `gpu_plan_filter.cpp`, `gpu_plan_aggregate.cpp`)
- Executor: `src/gpu_executor.cpp`
- Memory: requires `gpu_buffer_init()` before use; uses `GPUBufferManager` and `GPUContext`

**New Sirius / Super Sirius** (`gpu_execution`):
- Uses `namespace sirius`
- Entry point: `CALL gpu_execution('SELECT ...')`
- Physical plan generator: `sirius_physical_plan_generator` (`src/planner/sirius_physical_plan_generator.cpp`)
- Operators: `sirius_physical_operator` subclasses in `src/op/` (e.g., `sirius_physical_hash_join.cpp`)
- Plan builders: `src/planner/` (e.g., `sirius_plan_filter.cpp`, `sirius_plan_aggregate.cpp`)
- Engine: `src/sirius_engine.cpp`, pipelines in `src/pipeline/`
- Interface: `src/sirius_interface.cpp` (uses `sirius_interface` class)
- Includes task-based execution: `src/creator/`, `src/downgrade/`, `src/op/scan/`

**Shared code** (used by both, in `namespace duckdb`):
- `src/sirius_extension.cpp`: Extension entry point, registers both `gpu_processing` and `gpu_execution` table functions
- `src/expression_executor/`: GPU expression evaluation
- `src/config.cpp` / `src/include/config.hpp`: Runtime configuration
- `src/cuda/`: CUDA kernels (cuDF wrappers, expression dispatch)

New development should target the **new Sirius** (`namespace sirius` / `gpu_execution`) code path.

### Super Sirius Documentation

Comprehensive documentation for the new Sirius code path lives in `docs/super-sirius/`. **Read these docs before modifying Super Sirius code** — they cover the execution model, pipeline splitting rules, operator behavior, and configuration in detail.

| Document | Covers |
|----------|--------|
| [README](docs/super-sirius/README.md) | Index and reading order |
| [Architecture Overview](docs/super-sirius/architecture-overview.md) | Component diagram, ownership hierarchy, thread model |
| [Execution Flow](docs/super-sirius/execution-flow.md) | End-to-end query trace from SQL to GPU results |
| [Physical Plan Generation](docs/super-sirius/physical-plan-generation.md) | DuckDB→Sirius operator mapping, pipeline construction, **pipeline splitting rules with barrier types** |
| [Operators](docs/super-sirius/operators.md) | All physical operators — scan, streaming, blocking, pipeline breakers |
| [Pipeline Execution](docs/super-sirius/pipeline-execution.md) | Task execution, GPU executor, OOM handling, completion |
| [Task Creator](docs/super-sirius/task-creator.md) | Hint chain, per-operator overrides, scan scheduling |
| [Scan](docs/super-sirius/scan.md) | Scan subsystem — DuckDB scan, Parquet scan, caching modes |
| [Memory Management](docs/super-sirius/memory-management.md) | GPU/host/disk tiers, cuCascade, reservations, downgrade |
| [Data Management](docs/super-sirius/data-management.md) | Data batches, repositories, ports |
| [Configuration](docs/super-sirius/configuration.md) | Config file format, operator params, thread pools |
| [Optimizations](docs/super-sirius/optimizations.md) | Filter pushdown, projection elision, BUILD_PROBE mode |
| [Expression Executor](docs/super-sirius/expression-executor.md) | GPU expression evaluation via cuDF |

Key concepts from the docs:
- **After pipeline finalization**, `operators` contains ALL operators (source to sink inclusive); `source` and `sink` are just aliases for `operators[0]` and `operators.back()`
- **Pipeline splitting** in `initialize_internal()` inserts PARTITION, CONCAT, SORT_SAMPLE, MERGE_* operators with data repositories between pipelines
- **Barrier types**: `FULL` (wait for all upstream), `PARTIAL` (only PARTITION→CONCAT), `PIPELINE` (streaming — scans, ORDER_BY→SORT_SAMPLE)
- **Repositories** are always between pipelines, never in the middle of one

### Execution Flow

Sirius implements a custom execution engine that processes DuckDB's physical plans:

1. **Thread Coordinator**: Main thread receives logical plan from DuckDB, populates Pipeline Metadata Hash Map
2. **Task Creator**: Creates Scan Tasks and Pipeline Tasks based on data availability in Data Repository
3. **Scan Executor**: Uses DuckDB to scan data from storage, converts to GPU format, stores in Data Repository
4. **Pipeline Executor**: GPU thread pool executing operators via cuDF, stores results in Data Repository
5. **Downgrade Executor**: Moves data from GPU to CPU when GPU memory is constrained

### Key Components

**Data Flow:**
- `Data Batch`: Wrapper for pipeline input/output (cudf::table or spilling::allocation)
- `Data Repository`: Container for Data Batches, manages movement across memory tiers (GPU/CPU/disk via cuCascade)
- `Pipeline Task`: Operators chain + Data Batch to be executed on GPU
- `Scan Task`: DuckDB-based data scan that produces Data Batches

**Execution:**
- `sirius_engine`: Top-level orchestrator, owns pipelines and physical plan
- `sirius_pipeline`: Collection of operators that can be executed together
- `sirius_meta_pipeline`: Manages pipeline dependencies and scheduling
- `GPU Thread Pool`: Stream-per-thread model for parallel GPU execution
- `Memory Reservation Manager`: Prevents GPU OOM by enforcing memory limits

**Operators** (`src/include/operator/`):
See [Supported Features](#supported-features) for the full list of implemented operators.

### Directory Structure

**Core source code:**
- `src/include/`: Header files organized by module
  - `operator/`: GPU physical operators (filter, join, aggregate, etc.)
  - `pipeline/`: Pipeline execution framework (tasks, executors, queues)
  - `memory/`: Memory management interfaces (integrates with cuCascade)
  - `op/`: Sirius-specific physical operator wrappers
  - `planner/`: Physical plan generation and optimization
  - `data/`: Data structures (columns, batches)
  - `cudf/`: cuDF integration utilities
  - `expression_executor/`: Expression evaluation on GPU

**Important files:**
- `src/sirius_extension.cpp`: Extension entry point, registers functions with DuckDB
- `src/sirius_interface.cpp`: API for `gpu_buffer_init` and `gpu_processing`
- `src/gpu_executor.cpp`: Main GPU execution coordinator
- `src/gpu_buffer_manager.cpp`: GPU memory allocation and caching

**Third-party dependencies:**
- `cucascade/`: GPU memory management library (built as subdirectory)
- `duckdb/`: DuckDB core (git submodule)
- `third_party/`: spdlog (logging), other dependencies via CMake

**Build configuration:**
- `CMakeLists.txt`: Main build configuration
- `extension_config.cmake`: Extension-specific DuckDB config
- `third_party/*.cmake`: External dependency fetching (spdlog, cucascade)
- `pixi.toml`: Pixi environment specification (CUDA versions, dependencies)

### Memory Management

Sirius uses cuCascade for sophisticated GPU memory management:

- **GPU Caching Region**: Stores raw input data on GPU
- **GPU Processing Region**: Holds intermediate results (hash tables, join results)
- **Pinned Host Memory**: Fast CPU-GPU transfers
- **Memory Reservations**: Pre-allocation strategy to avoid OOM during execution

Initialization via `gpu_buffer_init("1 GB", "2 GB", pinned_memory_size = "4 GB")`

### Logging

Sirius uses spdlog for structured logging:

```bash
export SIRIUS_LOG_DIR=/path/to/logs      # Default: ${CMAKE_BINARY_DIR}/log
export SIRIUS_LOG_LEVEL=debug            # Levels: trace, debug, info, warn, error
```

Logs are essential for debugging GPU execution, memory allocation, and pipeline scheduling.

## Development Guidelines

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

## Common Issues

**Build Issues:**

If you see undefined reference errors related to GLIBCXX or CXXABI:
```bash
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib $LDFLAGS"
rm -rf build
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
```

**Memory Issues:**

If build consumes too much RAM, reduce parallel jobs:
```bash
CMAKE_BUILD_PARALLEL_LEVEL=4 make
```

**Test Datasets:**

TPC-H and ClickBench datasets must be generated before running tests. See `test_datasets/` and run `setup_test_datasets.sh` (automatically run in pixi activation).

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
CALL gpu_buffer_init('1 GB', '2 GB');
-- Legacy mode:
CALL gpu_processing('SELECT ...');
-- New mode (preferred):
CALL gpu_execution('SELECT ...');
```

Python:
```python
con = duckdb.connect('db.duckdb', config={"allow_unsigned_extensions": "true"})
con.execute("LOAD '/path/to/sirius.duckdb_extension'")
con.execute("CALL gpu_buffer_init('1 GB', '2 GB')")
# Legacy mode:
con.execute("CALL gpu_processing('SELECT ...')").fetchall()
# New mode (preferred):
con.execute("CALL gpu_execution('SELECT ...')").fetchall()
```

## Performance Characteristics

- **Cold runs are slow**: First query loads data from storage and converts DuckDB format to GPU format
- **Warm runs benefit from GPU caching**: Subsequent queries use cached GPU data
- **Best for**: Interactive analytics, financial workloads, ETL jobs, large aggregations/joins
- **Benchmark**: ~8x speedup on TPC-H SF=100 vs CPU at equivalent hardware cost

## Glossary Terms

Key terminology used throughout the codebase (see `docs/glossary.md` for complete definitions):

- **Pipeline**: Chain of operators executed together as a unit
- **Data Batch**: Input/output wrapper for pipeline execution
- **Data Repository**: Central storage for Data Batches with tier management
- **GPU Scheduling Thread**: Stream-associated thread that pulls tasks from queue
- **Memory Reservation**: Lease on memory to prevent oversubscription
- **Task Creator**: Thread that polls completions and creates new tasks
- **Thread Coordinator**: Main thread orchestrating Sirius execution

## Claude Code Skills

Sirius includes Claude Code skills for performance analysis and dataset management. Invoke them via slash commands:

| Skill | Command | Description |
|-------|---------|-------------|
| Profile Analyzer | `/profile-analyzer` | Analyzes GPU performance from nsys profiles — kernel occupancy, memory bandwidth, operator attribution, and regression detection. |
| Dataset Manager | `/dataset-manager` | Manages TPC-H parquet datasets — generate at any scale factor, consolidate files, inspect layout, optimize row groups. |
| Optimization Advisor | `/optimization-advisor` | Maps GPU hotspots from nsys profiles to source functions, detects efficiency bottlenecks, sync overhead, and parallelism opportunities. |
| TPC-DS Benchmark | `/tpcds-benchmark` | Runs TPC-DS benchmarks on Legacy Sirius, Super Sirius, or DuckDB CPU baseline — generate data, execute queries, and compare results. |

**Useful debugging tools:**
- `tools/parse_pipeline_log.py`: Parses Sirius pipeline logs to show per-operator row counts for debugging incorrect query results.

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Sirius-Doris aarch64 Support**

Porting the Doris integration layer (`doris/`) of the Sirius GPU SQL engine to work on aarch64 (ARM) platforms. The core Sirius C++ engine already supports aarch64, but the Doris build system, Rust build scripts, Docker deployment, and documentation have hardcoded x86_64 assumptions that prevent building and running on ARM.

**Core Value:** The Doris integration layer builds and deploys on aarch64 NVIDIA platforms (Grace Hopper, Grace Blackwell, Vera Rubin) using the same build guide as x86_64.

### Constraints

- **Submodule**: nixl changes must be minimal (one-line fix) since it's an external NVIDIA repo
- **Backwards compatible**: All changes must preserve x86_64 functionality — runtime detection, not replacement
- **Build guide**: The same 4-step build process from `BUILD_DEPLOY_TEST_GUIDE.md` must work on both architectures
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++ 20 - Core GPU acceleration engine (`src/`)
- CUDA 13+ - GPU kernels and cuDF integration (`src/cuda/`)
- Rust 1.85+ - Doris GPU Backend integration (`doris/crates/`)
- Java 17 - Apache Doris FE (optional, not built with Sirius)
- Python 3.8+ - Testing and development utilities (`test/tpch_performance/`)
## Runtime
- Linux x86_64 and aarch64 (via Pixi)
- CUDA 13.1.* (or CUDA 12.* variant)
- GPU support: Turing (75) through Blackwell (120a) architectures
- Pixi 0.59+ - Cross-platform Python/Conda environment management
- Conda channels: rapidsai, conda-forge
- Lockfile: `pixi.lock` (environment pinning)
## Frameworks
- DuckDB 1.4.4 - SQL query planning and CPU fallback engine (via git submodule)
- RAPIDS cuDF 26.02.* - GPU DataFrame operations (vector operations, joins, aggregates)
- RMM (RAPIDS Memory Manager) - GPU memory allocation and pool management
- cuCascade (NVIDIA) - GPU/CPU/disk tiered memory management with reservations
- Substrait extension (`substrait/` submodule) - Query plan IR format (protobuf-based)
- Protobuf - Substrait plan serialization
- CMake 4.1.* - C++ build configuration
- Ninja - Fast C++ build backend (recommended)
- sccache - Distributed C++ compilation caching
- clang 21+ - C++ compiler (LLVM/Clang toolchain)
- Catch2 - C++ unit testing framework (in `duckdb/third_party/catch`)
- SQLLogicTest - SQL query correctness tests (`test/sql/`)
- clang-format - C++/CUDA code formatting (`.clang-format`)
- clang-tidy - C++ linting (`.clang-tidy`)
- black - Python code formatting
- cmake-format - CMake file formatting
- codespell - Spell checker (`.codespell_words`)
- pre-commit - Git hook framework
## Key Dependencies
- `libcudf` 26.02.* - GPU DataFrame library (hash joins, aggregations, sorting)
- `librmm` - RAPIDS memory resource abstraction
- `spdlog` 1.8.* - Structured logging with file rotation
- `libconfig++` - Configuration file parsing (`.cfg` format)
- `libabseil` 20260107.0+ - Google Abseil utilities (any_invocable)
- `numa` (PkgConfig) - NUMA support for multi-socket systems
- `duckdb` (crate) 1.10501.0 - Rust DuckDB FFI bindings
- `arrow` 54 - Apache Arrow data format (IPC, serialization)
- `arrow-flight` 54 - Arrow Flight RPC protocol
- `tonic` 0.13 - gRPC framework (Rust)
- `tokio` 1.* - Async runtime (full features)
- `substrait` 0.52 - Substrait protobuf bindings (Rust)
- `prost` 0.13 - Protocol buffer code generation
- `cudarc` 0.19.4 - CUDA runtime bindings (Rust)
- `mysql_async` 0.34 - MySQL connection pooling (Doris FE queries)
- `libcurand-dev` - CUDA random number generation (GPU utilities)
- `thrift-compiler` 0.22+ - Apache Thrift RPC framework (Doris protocol)
- `protobuf` 5+ - Protocol buffers (Doris + Substrait)
- `mold` - Fast linker for build optimization
- `sqlite` 3.52.0+ - Lightweight SQL tests
## Configuration
- Set via `pixi.toml` [dependencies] and [feature.*.activation.env]
- Runtime: `.sirius/sirius.cfg` (INI format, libconfig++)
- GPU selection: CUDA architectures specified in `pixi.toml` features
- `CMakeLists.txt` - Main build orchestration
- `extension_config.cmake` - DuckDB extension loading (sirius, json, tpcds, tpch, parquet, icu, substrait)
- `.clang-format` - C++ formatting rules
- `.clang-tidy` - Clang static analysis configuration
## Platform Requirements
- Linux system with NVIDIA GPU (Turing T4 or newer recommended)
- 64GB+ RAM (C++ compilation is memory-intensive at 8+ parallel jobs)
- Pixi installed (`>=0.59`)
- CUDA 12 or 13 SDK
- Static extension: `build/release/extension/sirius/sirius.duckdb_extension`
- Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
- Unit tests: `build/release/extension/sirius/test/cpp/sirius_unittest`
- Doris GPU BE: `doris/target/release/sirius-doris-be` (Rust binary)
- NVIDIA GPU with sufficient memory (cache region 1GB+, processing region 2GB+, pinned 4GB+)
- libcudf runtime libraries available
- DuckDB extension loader support
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Snake case for all files: `sirius_physical_filter.cpp`, `sirius_physical_filter.hpp`
- Headers use `.hpp` extension
- Implementation files use `.cpp` extension
- CUDA kernels use `.cu` extension
- Organized by functional module: `src/op/`, `src/pipeline/`, `src/planner/`
- Snake case for function names: `initialize_memory_manager()`, `execute()`, `reset()`
- Public methods in classes follow lowercase_snake_case
- Private helper functions use snake_case with leading underscore rarely used
- Constructors follow class naming convention
- Local variables: lowercase with underscores: `filter_vals`, `input_batch`, `gpu_space`
- Member variables (class): no prefix convention, just lowercase_snake_case: `expression`, `sirius_pipeline`
- Static/constant members: UPPERCASE for macros only
- Loop counters: single letter `i`, `j` acceptable for short loops
- Pointers and references: `*` and `&` attach to type, not variable: `int* ptr`, `int& ref`
- Class names: lowercase_snake_case: `sirius_physical_filter`, `data_batch`, `memory_reservation_manager`
- Struct names: lowercase_snake_case when used as templates or data holders
- Enum values: UPPERCASE_SNAKE_CASE when applicable (varies by enum)
- Type aliases: lowercase_snake_case: `gpu_table_representation`, `shared_data_repository`
- Namespaces: lowercase: `sirius`, `sirius::op`, `sirius::pipeline`
- Global config variables: UPPERCASE: `Config::USE_CUDF_EXPR`, `Config::LOG_LEVEL`
- Constexpr values: lowercase_snake_case: `TEST_BUFFER_MANAGER_MEMORY_BYTES`
## Code Style
- Tool: `clang-format` (configured in `.clang-format`)
- Column limit: 100 characters
- Indentation: 2 spaces
- No tabs (`UseTab: Never`)
- Brace style: WebKit (opening braces stay on line)
- Tool: `clang-tidy` (configured in `.clang-tidy`)
- Pre-commit hook enforces formatting via `clang-format`
- Code style verification: `pre-commit run -a`
- Tool: `codespell` (configured in `.codespell_words`)
- Custom allowed words in `.codespell_words`
## Import Organization
- Project includes are quoted and relative to include directories
- External library includes are angle-bracketed
- Local project includes include hierarchy: `"op/sirius_physical_filter.hpp"`, `"log/logging.hpp"`
## Error Handling
- DuckDB assertions: `D_ASSERT()` for internal preconditions
- Catch2 test assertions: `REQUIRE()` for test failures
- Runtime errors: `throw std::runtime_error("message")`
- CUDA error checking: `verify_cuda_errors("context")`
- Throw `std::runtime_error` for runtime failures: `throw std::runtime_error("Cannot concatenate empty batch list")`
- Throw `std::invalid_argument` for parameter validation
- Catch exceptions at boundaries (DuckDB integration)
- No exception specifications on functions
- Input validation at public function entry points
- Precondition checks with `D_ASSERT()` in internal functions
- Error messages should be descriptive and include context
## Logging
- Configure logging level via `Config::LOG_LEVEL` (default: "info")
- Log directory via `Config::LOG_DIR`
- Flush interval via `Config::LOG_FLUSH_SECONDS`
- Initialize in main: `InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS)`
- No logging in CUDA device code (wrapped in `#ifdef __CUDACC__` guards)
- Entry/exit of major functions: pipeline execution, operator execution
- Configuration changes and settings applied
- Memory allocation/deallocation at significant milestones
- Task scheduling and completion
- Error conditions (before throwing or handling)
- Performance-critical sections (sparingly at debug level)
## Comments
- Complex algorithms or non-obvious logic
- Rationale for unusual design decisions
- Workarounds or known limitations
- Per-function documentation in headers
- Use `//!` for Doxygen documentation in headers
- Document public functions and classes
- Include parameter descriptions and return values
- Example from `sirius_physical_filter.hpp`:
- Use `//` for single-line explanations
- No trailing comments on same line unless very brief
- Comments stay above the code they describe when practical
## Function Design
- Keep functions under 100 lines when reasonable
- Single responsibility: one job per function
- Extract complex logic into helper functions
- Pass non-copyable types by reference: `const operator_data& input_data`
- Pass objects by const reference when not modified: `const std::vector<int>& values`
- Pass small copyable types by value: `int count`, `bool flag`
- Avoid passing raw pointers; use references or smart pointers instead
- Use `std::unique_ptr<T>` for heap-allocated single ownership: `std::unique_ptr<operator_data> execute(...)`
- Use `std::shared_ptr<T>` for shared ownership across threads: `std::shared_ptr<data_batch>`
- Return status via exceptions or result types; use `bool` only for simple queries
- Use `std::optional<T>` for optional values: `std::optional<float> float_tolerance`
- C++20 standard
- Use auto for type inference where type is obvious: `auto space = memory_manager->get_memory_space(...)`
- Structured bindings for tuples: `auto [db_owner, connection] = make_test_db_and_connection()`
- std::unique_ptr and std::shared_ptr for memory management
- No raw `new`/`delete` outside memory managers
- Range-based for loops: `for (const auto& batch : input_batches) { ... }`
## Module Design
- Header files in `src/include/` define public interfaces
- Implementation in `src/` (mirrors include structure)
- Use anonymous namespace or `static` for file-local symbols
- Namespace organization: `sirius::op::`, `sirius::pipeline::`, `duckdb::`
- Not commonly used; prefer explicit imports
- Some headers in `src/include/operator/` group related types
- Example: operator type traits included via specific header
- Use `#pragma once` at top of all headers (not `#ifndef` guards)
- Placed immediately after license comment
## Specific Conventions
- `.cu` files in `src/cuda/` and subdirectories
- Host-side logic in `.cpp`, device code in `.cu`
- Use `rmm::cuda_stream_view` for stream management
- Kernels wrapped via cuDF/cuCascade abstractions
- Use `cudf::data_type` for cuDF type representation
- DuckDB types via `duckdb::LogicalType` and `duckdb::LogicalTypeId`
- Template metaprogramming for type traits (e.g., `gpu_type_traits<TestType>`)
- Use double quotes for C++ strings: `"config setter test"`
- Raw string literals for complex patterns: `R"(regex_pattern)"`
- Top-level: `duckdb::` (legacy), `sirius::` (new)
- Sub-namespaces follow module structure: `sirius::op::`, `sirius::pipeline::`, `sirius::memory::`
- Anonymous namespaces for file-local helpers
- Using declarations in header files for convenience (rarely)
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- DuckDB extension architecture with pluggable GPU execution
- Logical plan → physical operator tree → pipeline graph transformation
- Distributed task execution across dedicated thread pools (GPU, scan, task creation, downgrade)
- Data-flow driven scheduling with per-operator memory barriers
- Graceful fallback to DuckDB CPU execution for unsupported operations
## Layers
- Purpose: DuckDB integration and API exposure
- Location: `src/sirius_extension.cpp`
- Contains: Table function registration (`gpu_execution`), extension initialization, buffer management
- Depends on: DuckDB C++ API, sirius_interface, gpu_buffer_manager
- Used by: DuckDB query execution engine
- Purpose: Query lifecycle management and result handling
- Location: `src/sirius_interface.cpp` and `src/include/sirius_interface.hpp`
- Contains: Active query context, prepared statements, pending query results, error handling
- Depends on: sirius_engine, sirius_context
- Used by: Extension layer to coordinate query execution
- Purpose: Physical plan construction and pipeline orchestration
- Location: `src/sirius_engine.cpp` and `src/include/sirius_engine.hpp`
- Contains: Physical plan ownership, pipeline graph construction, repository insertion, operator initialization
- Depends on: Physical plan generator, pipeline builders, data repository manager
- Used by: sirius_interface to execute queries
- Purpose: Logical-to-physical operator translation
- Location: `src/planner/sirius_physical_plan_generator.cpp` and `src/planner/sirius_plan_*.cpp`
- Contains: Operator mapping (TABLE_SCAN, JOIN, AGGREGATE, ORDER, etc.), plan builders for each operator type
- Depends on: DuckDB logical operators
- Used by: sirius_engine during initialization
- Purpose: Task scheduling and execution on GPU and CPU
- Location: `src/pipeline/`, `src/creator/`, `src/downgrade/`
- Contains: Pipeline executor, GPU executors, task creator, scan executor, downgrade executor
- Depends on: Sirius operators, data repositories, memory managers, thread pools
- Used by: Engine to run queries
- Purpose: Physical operator implementations (streaming and blocking)
- Location: `src/op/` (new Sirius), `src/legacy/operator/` (legacy), `src/include/op/` (headers)
- Contains: Individual operators (filter, projection, join, aggregate, scan, merge, partition, order, etc.)
- Depends on: Operator base class, expression executor, data batches
- Used by: Execution layer to transform data
- Purpose: GPU-accelerated scalar expression evaluation
- Location: `src/expression_executor/`, `src/cuda/expression_executor/`
- Contains: Expression translation to cuDF operations, specializations for each expression type (cast, comparison, conjunction, function, case, between, etc.)
- Depends on: cuDF API
- Used by: Filter and projection operators
- Purpose: Per-connection Sirius state and configuration
- Location: `src/sirius_context.cpp`, `src/sirius_config.cpp`, `src/include/sirius_context.hpp`
- Contains: Subsystem ownership (memory manager, executor pool, repositories), config file parsing, lifecycle hooks
- Depends on: cuCascade, spdlog, DuckDB ClientContext
- Used by: All layers (registered in ClientContextState)
- Purpose: GPU/host/disk memory tier management and pressure relief
- Location: `src/memory/`, `src/include/memory/`
- Contains: Memory reservation manager, allocation accessors
- Depends on: cuCascade shared_data_repository_manager
- Used by: GPU executor (reserves), downgrade executor (monitors)
- Purpose: Data batch representation and conversion between formats
- Location: `src/data/`, `src/include/data/`
- Contains: Host parquet representation, cached data representation, converter registry
- Depends on: cuDF, Arrow
- Used by: Scan operators and result collection
## Data Flow
- **Query-level state:** Per-query pipelines, operator initialization, task counters
- **Execution-level state:** Task local state (input data, operator index for resumption), memory reservations, CUDA streams
- **Global state:** Operator global state (scan cursors, hash table state for stateful operators)
## Key Abstractions
- Purpose: Base class for all GPU-executable operators
- Examples: `sirius_physical_filter.hpp`, `sirius_physical_hash_join.hpp`, `sirius_physical_grouped_aggregate.hpp`
- Pattern: Virtual `execute()` → `operator_data`, optional `sink()` for pipeline breakers
- Key fields: `operator_id`, `type`, `children`, `source_order`, ports for inter-pipeline data
- Purpose: Ordered chain of operators executing as an atomic unit
- Pattern: Container with `source`, `operators`, `sink`; tracks task count via atomic counters
- Lifecycle: Created during finalization, marked ready when dependencies complete, tasks scheduled
- Key: After finalization, `operators` includes source and sink; `source` and `sink` are aliases to first/last
- Purpose: Wrapper for column data flowing between operators
- Pattern: `cucascade::data_batch` from cuDF table, moved through `shared_data_repository` (cuCascade-managed)
- Lifecycle: Created by operator `execute()`, published to repository by `sink()`, consumed by downstream `execute()`
- Tier management: GPU → Host → Disk via cuCascade (memory pressure triggers downgrade)
- Purpose: Schedulable unit of work
- Examples: `duckdb_scan_task`, `parquet_scan_task`, `gpu_pipeline_task`
- Pattern: Carries input data, operator references, memory reservation, execution state
- Lifecycle: Created by task_creator, executed by appropriate executor, completion triggers downstream tasks
- Purpose: Forces pipeline split due to data dependencies or memory constraints
- Examples: PARTITION (distributes to multiple branches), CONCAT (merges multiple inputs), SORT_SAMPLE (samples for merge sort)
- Pattern: Created with downstream pipeline, connected via data repository with barrier type
- Barrier types: FULL (wait all upstream), PARTIAL (PARTITION→CONCAT only), PIPELINE (streaming scans)
- Purpose: GPU evaluation of scalar expressions
- Pattern: `gpu_expression_executor` translates DuckDB expressions to cuDF operations
- Examples: Specializations for CAST, COMPARISON, CONJUNCTION, FUNCTION, CASE, BETWEEN
- Lifecycle: Instantiated per operator, called during `execute()` for filtering/projection
## Entry Points
- Location: `src/sirius_extension.cpp` (registered in LoadInternal)
- Triggers: `CALL gpu_execution('SELECT ...')`
- Responsibilities: Parse SQL, prepare statement data, bind parameters, invoke sirius_interface
- Location: `src/sirius_interface.cpp`
- Triggered by: Extension table function
- Responsibilities: Receive query string, initialize sirius_engine, begin query lifecycle
- Location: `src/sirius_engine.cpp`
- Triggered by: sirius_interface after DuckDB plan generation
- Responsibilities: Take physical plan, build pipeline graph, finalize pipelines, initialize operators
- Location: `src/pipeline/pipeline_executor.cpp`
- Triggered by: sirius_interface after engine initialization
- Responsibilities: Create task_creator thread, start GPU/scan executors, queue initial tasks
- Location: `src/creator/task_creator.cpp`
- Triggered by: Pipeline executor event loop
- Responsibilities: Poll repositories for ready operators, create scan or GPU pipeline tasks
## Error Handling
## Cross-Cutting Concerns
- Framework: spdlog
- Location: `src/include/log/logging.hpp`
- Usage: Key decision points (fallback, barrier creation), task lifecycle, memory pressure
- Control: Environment variable `SIRIUS_LOG_LEVEL` (default: info), file in `$SIRIUS_LOG_DIR/log`
- Input validation: DuckDB handles SQL parsing, Sirius validates operator support during planning
- Data validation: Each operator validates column count/type on `execute()` via DuckDB data_chunk assertions
- Barrier validation: `initialize_internal()` verifies barrier types match pipeline dependencies
- Location: Inherited from DuckDB connection (no GPU-specific auth)
- Config: GPU buffer sizes via `gpu_buffer_init()` parameters, per-extension security model
- Data repositories: Protected by cuCascade's atomic operations
- Pipeline state: Atomic counters (`tasks_created`, `tasks_completed`)
- Global operator state: Operator-specific locks (e.g., hash join build table lock)
- Scan global state: Per-operator global state with mutex in `scan_task_global_state`
- NVTX regions: Pipeline task creation/completion, operator execution
- Metrics: Task count, memory reservation size, operator wall clock time
- Profiler integration: DuckDB QueryProfiler called for metrics collection
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

| Skill | Description | Path |
|-------|-------------|------|
| bisect | Find which commit introduced a bug by comparing behavior across a range of commits. Uses git bisect with automated build and test. Use when a bug appeared recently and you need to identify the culprit commit. | `.claude/skills/bisect/SKILL.md` |
| build-errors | Analyze C++/CUDA build errors, suggest fixes, and iteratively rebuild until success. Use when compilation fails. | `.claude/skills/build-errors/SKILL.md` |
| config-optimizer | find the optimal configuration for sirius running TPCH at different scale factors. | `.claude/skills/config-optimizer/SKILL.md` |
| dataset-manager | Manage TPC-H parquet datasets — generate data at any scale factor, consolidate small parquet files into fewer larger files, inspect dataset layout, and optimize row group sizes. Uses rewrite_parquet.py which auto-selects cudf (GPU) or pyarrow (CPU) with OOM fallback. | `.claude/skills/dataset-manager/SKILL.md` |
| doris-query-testing | Iterate on testing SQL queries against the Sirius-Doris cluster — start/stop the cluster, warm up BEs, run single-BE and multi-BE exchange queries, inspect logs, and diagnose issues. | `.claude/skills/doris-query-testing/SKILL.md` |
| optimization-advisor | Identify code optimization targets from nsys profiles — maps GPU hotspots to source functions, detects efficiency bottlenecks, sync overhead, memory issues, and parallelism opportunities. | `.claude/skills/optimization-advisor/SKILL.md` |
| profile-analyzer | Analyze Sirius GPU performance from nsys profiles — runs benchmarks, generates reports with kernel occupancy, memory bandwidth, operator attribution, and compares runs for regression detection. | `.claude/skills/profile-analyzer/SKILL.md` |
| race-check | Detect race conditions using ThreadSanitizer and NVIDIA Compute Sanitizer memcheck. Use when you suspect data races, deadlocks, or non-deterministic behavior in Sirius. | `.claude/skills/race-check/SKILL.md` |
| runtime-errors | Diagnose runtime errors using Sirius log files, cuda-gdb, and Compute Sanitizer. Use when a query crashes (including segfaults), throws exceptions, hangs, or triggers unexpected fallback to CPU. | `.claude/skills/runtime-errors/SKILL.md` |
| tpcds-benchmark | Run TPC-DS benchmarks on Legacy Sirius, Super Sirius, or DuckDB CPU baseline — generate data, execute queries, and compare results across engines. | `.claude/skills/tpcds-benchmark/SKILL.md` |
| update-docs | Incrementally updates Super Sirius documentation by inspecting merged PRs since the last update. Use when user says "update docs", "refresh documentation", or "sync docs with code changes". Reads commit marker from docs/super-sirius/README.md, inspects PR diffs, and updates affected doc files. | `.claude/skills/update-docs/SKILL.md` |
| validate | Diagnose incorrect query results by comparing against DuckDB CPU, analyzing per-operator row counts and data checksums to pinpoint the faulty operator. Use when a query returns wrong results. | `.claude/skills/validate/SKILL.md` |
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
