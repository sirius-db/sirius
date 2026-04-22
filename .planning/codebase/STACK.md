# Technology Stack

**Analysis Date:** 2026-04-21

## Languages

**Primary:**
- C++ 20 - Core GPU-native SQL engine implementation
- CUDA 20 - GPU kernels and NVIDIA CUDA-X library integration
- Python 3 - Performance testing, tooling, and data generation

**Secondary:**
- CMake - Build configuration and compilation orchestration
- YAML - Configuration files for runtime settings

## Runtime

**Environment:**
- Linux (x86_64, aarch64) - Primary deployment platform
- CUDA 12.x or 13.x - GPU execution runtime (feature-selectable)
- NVIDIA GPU - Turing through Blackwell architectures (compute capability 75-120)

**Package Manager:**
- Pixi (Recommended) - Conda-based environment and dependency management
- pip - Python package installation for DuckDB Python bindings
- CMake - Cross-platform build system and dependency orchestration

**Lockfiles:**
- `pixi.lock` - Comprehensive pinned environment definition for reproducible builds

## Frameworks

**Core SQL Engine:**
- DuckDB 1.5.2 - Modular SQL database engine (submodule: `duckdb/`)
- Sirius Extension - Custom DuckDB extension implementing GPU acceleration

**GPU Libraries:**
- RAPIDS cuDF 26.04 - GPU DataFrame operations (joins, aggregations, sorting)
- RMM (RAPIDS Memory Manager) - GPU memory allocation and pool management
- cuCascade - GPU memory reservation and tiered memory management (submodule: `cucascade/`)

**Build & Compilation:**
- Ninja - Fast parallel build system
- CMake 4.x - Build configuration with version 3.30.4 minimum
- sccache - Compiler result caching for faster rebuilds
- Clang 21 / GCC - Compiler toolchain with unified standard configuration

**Testing:**
- Catch2 - C++ unit testing framework for component tests (via DuckDB's bundled headers)
- Custom SQL logic test runner - Integrated with DuckDB test infrastructure

**Development Tools:**
- pre-commit - Git hooks for code quality checks
- clang-format 20.1.4 - C++ code formatting
- clang-tidy - C++ static analysis and linting
- black - Python code formatting
- codespell - Spell checking with custom word list
- cmake-format - CMake code formatting

## Key Dependencies

**Critical:**
- `libcudf::cudf` (26.04) - Accelerated GPU DataFrame operations (joins, groupby, aggregations, sorting)
- `rmm::rmm` - GPU memory management with fallback support
- `cuCascade::cucascade` - Tiered memory (GPU/host/disk) with reservation semantics
- `duckdb::duckdb` (1.5.2) - SQL parsing, planning, CPU fallback execution

**Infrastructure:**
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

**Environment Configuration:**

Configuration is resolved in order via:
1. `SIRIUS_CONFIG_FILE` environment variable - explicit config path
2. `./sirius.yaml` - current working directory
3. `~/.sirius/sirius.yaml` - user home directory

Config file format: YAML (parsed by `yaml-cpp`)

Fallback configuration in `src/config.cpp` provides defaults:
- Expression executor strategy (AST interpretation)
- GPU memory region usage (pinned memory for CPU processing/caching)
- Table scan optimization (8 CUDA streams, 64 MB memcpy threshold)
- Logging level (info) and flush interval (3 seconds)
- Scan task batch size (512 MB)

**Build Configuration:**

CMake-based with preset profiles in `cmake/CMakePresets.json`:
- `debug` - GCC debug build with `-g -O0`
- `release` - GCC optimized release build
- `relwithdebinfo` - GCC release with debug symbols
- `clang-*` variants - Using Clang 21 as host compiler
- `legacy-release` - Legacy code path support (optional)
- `vcpkg-*` - Static linking via vcpkg package manager

Cache variables:
- `CMAKE_CUDA_COMPILER_LAUNCHER=sccache` - Compiler caching
- `CMAKE_CXX_COMPILER_LAUNCHER=sccache`
- `CMAKE_C_COMPILER_LAUNCHER=sccache`
- `CMAKE_LINKER_TYPE=MOLD` - Fast linker
- `DUCKDB_EXTENSION_CONFIGS` - Points to `extension_config.cmake`
- `EXTENSION_STATIC_BUILD=ON` - Static extension linking

**Runtime Configuration via Environment:**
- `SIRIUS_LOG_DIR` - Log output directory (default: `${CMAKE_BINARY_DIR}/log`)
- `SIRIUS_LOG_LEVEL` - Log verbosity (trace, debug, info, warn, error)
- `CUDAARCHS` - GPU compute capabilities (feature-selected via pixi)
- `VCPKG_CUDA_VERSION` - CUDA version selection for vcpkg build

**Language Standards:**
- C++ standard: 20 (CXX_STANDARD_REQUIRED ON)
- CUDA standard: 20 (CUDA_STANDARD_REQUIRED ON)
- CUDA separable compilation: ON
- CUDA device symbol resolution: ON

## Platform Requirements

**Development:**
- Linux system (x86_64 or aarch64)
- Pixi >=0.59 (for environment management)
- CMake >=3.30.4
- CUDA toolkit (12.x or 13.x feature-selected)
- NVIDIA GPU (Turing 75 or newer)
- C++20 compatible compiler (GCC or Clang 21)
- NUMA-aware system libraries (libnuma)

Optional:
- Python 3.x (for performance testing tools)
- pybind11 + scikit-build-core (for Python API building)

**Production:**
- Deployment target: Linux systems with NVIDIA GPUs
- DuckDB application environment with extension loading capability
- Sufficient GPU memory for data processing (tiered fallback to host/disk)
- Write access to log directory (default or custom via `SIRIUS_LOG_DIR`)

**GPU Architecture Support:**
- Via CUDA 13: Turing (75), Ampere (80, 86), Ada (90a), Hopper (100f, 120a), Blackwell (120)
- Via CUDA 12: Turing (75), Ampere (80, 86), Ada (90a) - Hopper/Blackwell require CUDA 13+

---

*Stack analysis: 2026-04-21*
