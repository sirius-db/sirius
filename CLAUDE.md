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
