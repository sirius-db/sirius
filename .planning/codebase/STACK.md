# Technology Stack

**Analysis Date:** 2025-04-02

## Languages

**Primary:**
- C++ 20 - GPU processing engine, physical operators, expression execution, memory management
- CUDA 20 - GPU kernels, data operations, join/aggregate implementations (`src/cuda/`)
- Python 3.12+ - Optional: DuckDB Python bindings for testing and client applications

**Secondary:**
- Bash - Build scripts and environment setup (`scripts/`, `setup_test_datasets.sh`)
- CMake - Cross-platform build system
- SQL - Test queries and performance benchmarks (`test/sql/`, `scripts/`)

## Runtime

**Environment:**
- Linux (x86_64, aarch64)
- NVIDIA CUDA 12.x or 13.x (configurable via pixi features)
- NVIDIA GPU drivers >= 570 (per CLAUDE.md)

**Package Manager:**
- Pixi (recommended) - Conda-based environment with CUDA, compilers, dependencies
- Manual conda/conda-forge setup also supported
- Lockfile: Not explicit (uses `pixi.lock` for pixi environments)

## Frameworks

**Core:**
- DuckDB 1.4.4 - SQL parser, optimizer, execution framework, extension host
- cuDF (RAPIDS) 26.02.* - GPU DataFrame operations, joins, aggregates, sorting
- RMM (RAPIDS Memory Manager) - GPU memory allocation and management

**Testing:**
- Catch2 - C++ unit test framework (bundled in `duckdb/third_party/catch`)
  - Tests located in `test/cpp/`
  - Custom test runner in `test/cpp/unittest.cpp`

**Build/Dev:**
- CMake 4.1.* - Build configuration
- Ninja - Build backend
- clang 21-22 - C++ compiler (standard, also supports clang-release configs)
- clang-format - C++ code formatting
- clang-tidy - C++ linting
- pre-commit 2.x - Git hooks for code quality

**Logging:**
- spdlog 1.8.* - Structured logging framework with daily file rotation

## Key Dependencies

**Critical:**
- libcudf 26.02.* - GPU accelerated DataFrame library (RAPIDS)
  - Used for joins, aggregations, sorting, filtering, expression evaluation
  - Wraps CUDA kernels with C++ API
- librmm * - RAPIDS Memory Manager for GPU memory allocation
- cuCascade (local submodule) - GPU memory reservation and tiered memory management
  - Manages GPU caching regions, pinned host memory, disk spillover
  - Built as static library and linked into Sirius extension

**Infrastructure:**
- libconfig 3.1.* - Configuration file parsing (`.cfg` file support)
- libnuma * - NUMA support for pinned memory management
- abseil-cpp 20260107.0+ - Provides `absl::any_invocable` for GPU task dispatch
- libcurand-dev * - CUDA random number generation (via CUDA toolkit)
- cuda-nvcc * - NVIDIA CUDA compiler
- cuda-nvml-dev * - CUDA profiler API access for `cudaProfilerStart/Stop()`
- spdlog 1.8.* - Structured logging

**Build Support:**
- cmake-format 0.6.13 - CMake code formatting
- codespell 2.4.1 - Spell checker for code (custom words in `.codespell_words`)
- black 25.1.0 - Python code formatter

## Configuration

**Environment:**
- `pixi.toml` - Project dependencies and environment specification
  - Defines CUDA versions (12.x or 13.x via features)
  - GPU architectures: Turing(75), Ampere(80,86), Hopper(90a), Ada(100f), Blackwell(120a,120)
  - Default environment: cuda13

**Build:**
- `CMakeLists.txt` - Main build configuration
  - Extension static library: `build/release/extension/sirius/sirius.duckdb_extension`
  - Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
  - Unit test binary: `build/release/extension/sirius/test/cpp/sirius_unittest`
- `extension_config.cmake` - DuckDB extension loading configuration
  - Also loads: json, tpcds, tpch, parquet, icu extensions for testing
- `cmake/CMakePresets.json` - Build presets (release, debug, relwithdebinfo, clang variants)
- `.clang-format` - C++/CUDA formatting rules
- `.pre-commit-config.yaml` - Git hooks (clang-format, clang-tidy, black, codespell, cmake-format)
- Makefile - Thin wrapper around CMake for convenience (release, debug, test targets)

**Runtime:**
- `.cfg` file support via libconfig++ for operator parameters and memory configuration
- Environment variables for logging: `SIRIUS_LOG_LEVEL` (trace/debug/info/warn/error), `SIRIUS_LOG_DIR`
- DuckDB settings injected at runtime for Sirius configuration

## Platform Requirements

**Development:**
- Linux system (x86_64 or aarch64)
- NVIDIA CUDA 12.x or 13.x toolkit
- NVIDIA GPU with Turing+ architecture (CC 7.5+)
- C++ compiler: clang 21+ (via pixi)
- CMake 4.1+
- At least 16 GB GPU memory for TPC-H testing (configurable)
- At least 32 GB system RAM for full builds (parallel compilation consumes ~4GB per core)

**Production:**
- Linux with NVIDIA GPU drivers >= 570
- CUDA runtime libraries (bundled in extension or provided by runtime)
- DuckDB database engine (v1.4.4 or later for stability)
- GPU memory: 2-4 GB minimum for basic queries, 8+ GB recommended for analytics

---

*Stack analysis: 2025-04-02*
