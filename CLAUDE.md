# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

**New Sirius** (`gpu_execution`):
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
