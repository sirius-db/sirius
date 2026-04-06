# Technology Stack

**Analysis Date:** 2026-04-06

## Languages

**Primary:**
- C++ 20 - GPU extension implementation, query planning, operators, pipeline execution
- CUDA 20 - GPU kernel implementations for data operations, expression execution, joins
- Python 3.12+ - Test utilities, dataset generation, performance testing

**Build & Configuration:**
- CMake 4.1+ - Project configuration and build system
- Make - Build orchestration
- Ninja - Build execution backend

## Runtime

**Environment:**
- Linux (x86-64 and ARM64 support via pixi platforms)
- CUDA Toolkit 12.x or 13.x (feature-based selection in pixi)
- GPU architectures: Turing (75), Ampere (80, 86), Hopper (90a), Blackwell (100f, 120a, 120)

**Package Manager:**
- Pixi (conda-based) - Primary environment management
- Pip (within duckdb-python environment) - Python package installation
- Conda channels: `rapidsai`, `conda-forge`

## Frameworks

**Core SQL Engine:**
- DuckDB 1.4.4 - Integrated SQL engine (runs in-process as extension)
- DuckDB Extension API - Extension registration and integration point

**GPU Computation:**
- libcuDF 26.02.* - RAPIDS GPU DataFrame library for vectorized operations
- libRMM - RAPIDS Memory Manager for GPU memory allocation

**Testing:**
- Catch2 - C++ unit test framework (headers in `duckdb/third_party/catch`)
- SQL logic tests - DuckDB's test harness for SQL correctness

**Build/Dev:**
- clang 21.x - C++ compiler (alternative to gcc)
- clang-format 21.1.8+ - Code formatting
- clang-tidy - Static analysis
- sccache - Compilation caching
- Ninja - Build system backend

## Key Dependencies

**Critical Infrastructure:**
- cuCascade - GPU memory tiered memory management (CPU/GPU/disk spilling) - Git submodule in `cucascade/`
- libconfig++ - Configuration file parsing (SIRIUS_CONFIG_FILE support)
- libabseil 20260107.0+ - Google abseil C++ library (specifically `absl::any_invocable`)
- spdlog 1.8.* - Structured logging framework
- libnuma - NUMA-aware memory allocation

**CUDA Runtime:**
- libcurand-dev - CUDA random number generation
- cuda-nvcc - NVIDIA CUDA compiler
- cuda-nvml-dev - NVIDIA Management Library for device monitoring

**System Libraries:**
- sqlite 3.52+ - Test data storage
- pkg-config - Library discovery

## Configuration

**Environment Variables:**
- `SIRIUS_LOG_DIR` - Directory for operation logs (default: `${CMAKE_BINARY_DIR}/log`)
- `SIRIUS_LOG_LEVEL` - Logging verbosity: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)
- `SIRIUS_CONFIG_FILE` - Path to sirius.cfg configuration file (default: `~/.sirius/sirius.cfg`)
- `SIRIUS_STREAM_CHECK_LIB` - Path to stream check library for CUDA stream debugging
- `DUCKDB_SOURCE_PATH` - DuckDB source directory for Python extension build
- `CMAKE_BUILD_PARALLEL_LEVEL` - Build parallelism (recommended: `$(nproc)` or reduced if memory constrained)

**Runtime Configuration:**
Configuration file (`~/.sirius/sirius.cfg`) supports:
- Memory settings (pinned memory usage, buffer sizes)
- Scan caching levels
- Expression execution backend selection
- GPU operation optimization flags

**Build Configuration:**
Configuration files:
- `CMakeLists.txt` - Main CMake build configuration
- `pixi.toml` - Pixi environment specification with CUDA version features
- `extension_config.cmake` - DuckDB extension loader configuration
- `.clang-format` - C++ formatting rules
- `.clang-tidy` - C++ linting rules
- `.pre-commit-config.yaml` - Code quality hooks (clang-format, black, cmake-format, codespell)

## Platform Requirements

**Development:**
- Pixi package manager (>=0.59)
- NVIDIA GPU with CUDA compute capability 7.5+ (Turing era or newer)
- 8GB+ GPU memory (larger for integration tests)
- Linux OS (x86-64 or ARM64)
- ~4GB RAM for single-threaded builds; more for parallel builds

**Build Outputs:**
- Static extension: `build/release/extension/sirius/sirius.duckdb_extension`
- Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
- C++ unit tests: `build/release/extension/sirius/test/cpp/sirius_unittest`

**Python Integration:**
- Built against duckdb-python in `duckdb-python/` submodule
- Loaded via: `con.execute("LOAD 'path/to/sirius.duckdb_extension'")`
- Requires unsigned extension config: `allow_unsigned_extensions=true`

---

*Stack analysis: 2026-04-06*
