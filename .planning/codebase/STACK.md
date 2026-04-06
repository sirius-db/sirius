# Technology Stack

**Analysis Date:** 2026-04-06

## Languages

**Primary:**
- C++ 20 - GPU operator implementations, extension logic, planner, pipeline executor
- CUDA 20 - GPU kernels, cuDF wrappers, expression execution (`src/cuda/`)
- Python 3.12+ - DuckDB Python bindings, performance testing, dataset generation

**Secondary:**
- CMake - Build system configuration
- Shell - Build scripts and tooling

## Runtime

**Environment:**
- Linux (64-bit x86_64 and aarch64)
- NVIDIA GPU with CUDA 12 or 13 support
- GLIBC/standard C runtime

**Package Manager:**
- Pixi - Conda-based environment management with multiple feature profiles
  - Lockfile: `pixi.lock` (generated)
  - Main environment: `default` (cuda13)
  - Alternative: `cuda12` for systems with CUDA 12.x
  - Special: `duckdb-python` environment with pip, pybind11, scikit-build-core

## Frameworks

**Core:**
- DuckDB 1.4.4 - SQL engine, extension host, physical plan generation, execution framework
  - Submodule: `duckdb/`
  - Extension API: DuckDB extension architecture
  - Loaded via: `LOAD 'sirius.duckdb_extension'`

**GPU Compute:**
- RAPIDS cuDF 26.02.x - GPU DataFrame operations (joins, aggregations, sorting, projections)
  - Provides: `cudf::dataframe`, `cudf::join`, `cudf::groupby`, `cudf::stable_sort_keys`
  - Deployed via: libcudf CUDA library

- RMM (RAPIDS Memory Manager) - GPU memory management
  - Memory allocation, deallocation, device pool management
  - Stream-aware resource management

- cuCascade - GPU memory reservation system
  - Submodule: `cucascade/` (third-party library integrated as static library)
  - Purpose: Tiered memory management across GPU/host/disk
  - Provides: `data_repository`, `memory_reservation`, `fixed_size_host_memory_resource`

**Testing:**
- Catch2 - C++ unit testing framework
  - Config: Included via `duckdb/third_party/catch`
  - Executable: `build/release/extension/sirius/test/cpp/sirius_unittest`
  - Test files: `test/cpp/` with Catch2 tags for filtering

- SQLLogicTest - DuckDB's SQL logic testing framework
  - Test files: `test/sql/` (note: legacy gpu_processing tests skipped by default)

**Build/Dev:**
- CMake 4.1.x - Build configuration and compilation
- Ninja - Fast build backend
- Clang 21+ - C++ compiler (LLVM toolchain)
- CUDA NVCC - NVIDIA CUDA compiler

## Key Dependencies

**Critical:**
- libcudf (26.02.x) - Accelerated dataframe operations on GPU, enables query execution
- librmm - GPU memory management, essential for CUDA allocations
- libcurand-dev - CUDA random number generation library
- DuckDB (1.4.4) - SQL parsing, optimization, CPU fallback execution path
- spdlog (1.8.x) - Structured logging framework for diagnostics
- libconfig++ - Configuration file parsing (libconfig C++ bindings)
- libabseil (>=20260107.0) - Google Abseil C++ utilities (any_invocable, containers)

**Infrastructure:**
- libnuma - NUMA-aware memory allocation for multi-socket systems
- pkg-config - Dependency discovery and compilation flags
- pre-commit - Git hooks for code quality checks
- mold - High-performance linker (faster builds than ld)
- sqlite (3.52.0+) - Potential embedded database support

## Configuration

**Environment:**
- Pixi managed via `pixi.toml` with channels: rapidsai, conda-forge
- CUDA version selectable: cuda13 (default) or cuda12
- GPU architectures: Turing (75), Ampere (80, 86), Ada (90a), Hopper (100f), Blackwell (120a, 120)
  - CUDA 13: All architectures (75-real, 80-real, 86-real, 90a-real, 100f-real, 120a-real, 120)
  - CUDA 12: Turing through Ada (75-real, 80-real, 86-real, 90a-real)

**Build:**
- CMakeLists.txt: Main project configuration
- extension_config.cmake: DuckDB extension loader configuration
- cmake/CMakePresets.json: CMake build presets (release, debug, relwithdebinfo, clang variants)
- Makefile: Thin wrapper for DuckDB extension CI tools
- .clang-format: C++/CUDA formatting rules
- .clang-tidy: C++ linting rules
- .pre-commit-config.yaml: Git hooks for formatting and style

## Compiler Settings

**C++ Standards:**
- Target: C++20 (CXX_STANDARD 20)
- CUDA Standard: 20 (CUDA_STANDARD 20)
- CUDA Separable Compilation: ON
- CUDA Resolve Device Symbols: ON
- Precompiled Headers: Enabled for main extension and unittest targets

**Optimization:**
- Build types: Release, Debug, RelWithDebInfo
- Link-time optimization: Optional via clang variants
- Parallel compilation: Controlled via CMAKE_BUILD_PARALLEL_LEVEL (recommended: nproc or 8)
- Linker: mold for faster linking

## Platform Requirements

**Development:**
- Linux x86_64 or aarch64
- NVIDIA CUDA Toolkit 12 or 13
- 8+ GB RAM recommended (parallel builds can be memory-intensive)
- Git with submodule support

**Runtime:**
- Linux kernel 5.x+
- NVIDIA GPU with Turing architecture or newer (compute capability 7.5+)
- CUDA Runtime (cudart) 12.x or 13.x
- ~2GB GPU memory minimum, scales with data size

**Python:**
- Python 3.12+ (for DuckDB Python bindings)
- pip, pybind11, scikit-build-core (special duckdb-python environment)

---

*Stack analysis: 2026-04-06*
