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

Unsupported operations (data types, operators, row counts exceeding cuDF limits) fall back to DuckDB CPU execution via `src/fallback.cpp`.

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
```

Python:
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

**Useful debugging tools:**
- `tools/parse_pipeline_log.py`: Parses Sirius pipeline logs to show per-operator row counts for debugging incorrect query results.
