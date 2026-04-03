# Technology Stack

**Analysis Date:** 2026-04-03

## Languages

**Primary:**
- C++ 20 - Core GPU-accelerated SQL engine, all operators and expression evaluation
- CUDA 20 - GPU kernels for cuDF operations, expression execution, join/aggregate implementations
- Python 3.12+ (optional) - Performance testing, dataset generation, Python API bindings

**Secondary:**
- CMake - Build system and project configuration
- Bash - Build scripts and pixi activation

## Runtime

**Environment:**
- Pixi 0.59+ - Environment and dependency management
- Linux 64-bit (primary), Linux ARM64 (aarch64) support
- GPU: NVIDIA CUDA 12.x or 13.x (feature-gated)

**Package Manager:**
- Pixi - Conda-based environment from rapidsai and conda-forge channels
- Lockfile: Generated via pixi.lock

**GPU Architectures Supported:**
- Turing (75), Ampere (80, 86), Ada (90a), Hopper (100f), Blackwell (120a, 120)
- CUDA architecture selection: `CUDAARCHS` environment variable (set by pixi feature)

## Frameworks

**Core:**
- DuckDB 1.4.4 - SQL query engine, physical planner integration, extension API
- RAPIDS cuDF 26.02.* - GPU DataFrame library for joins, aggregations, ordering, filtering
- RAPIDS RMM - GPU memory management, device memory resources
- cuCascade (submodule) - GPU memory reservation and tiered memory management (GPU/host/disk)

**Testing:**
- Catch2 (DuckDB bundled) - C++ unit testing framework
- DuckDB SQL Logic Tests - End-to-end query validation

**Build/Dev:**
- CMake 4.1.* - Primary build system
- Ninja - Build execution
- CUDA nvcc compiler - CUDA code compilation
- Clang 21.x - C++ compiler with CUDA support
- Mold - Fast linker for reduced build time
- Sccache - C++ compiler cache
- pre-commit - Git hooks for code quality

## Key Dependencies

**Critical:**
- libcudf 26.02.* - Core GPU DataFrame operations (joins, aggregations, column selection)
- librmm - RAPIDS Memory Manager for GPU allocation/deallocation
- spdlog 1.8.* - Structured logging with daily file rotation and configurable levels
- libconfig 1.8.* - Configuration file parsing for runtime tuning
- libabseil 20260107.0+ - Standard library extensions (absl::any_invocable for task executors)
- NUMA (system package) - NUMA-aware memory management for host memory pools
- cuda-nvcc - NVIDIA CUDA compiler
- cuda-nvml-dev - CUDA device management API for GPU introspection

**Infrastructure:**
- cucascade - Custom GPU memory management with overflow to host memory and disk
- libcurand-dev - CUDA random number generation (for RMM initialization)
- SQLite 3.52+ - Internal storage (SQL logic test data, metadata)

**Development Only:**
- pybind11 2.6.0+ - Python binding generation for duckdb-python
- scikit-build-core 0.11.4+ - Python wheel building integration
- setuptools-scm 8.0+ - Semantic versioning from git

## Configuration

**Environment:**
- `PIXI_PROJECT_ROOT` - Sirius project root directory (set by pixi)
- `CUDAARCHS` - GPU architecture targets (75-real through 120a-real)
- `DUCKDB_SOURCE_PATH` - Path to DuckDB source for Python build (`duckdb-python` feature)
- `SIRIUS_LOG_DIR` - Log output directory (default: build/log)
- `SIRIUS_LOG_LEVEL` - Log severity threshold: trace, debug, info, warn, error (default: info)
- Runtime config: `src/include/config.hpp` static variables with defaults in `src/config.cpp`

**Build Configs:**
- `cmake/CMakePresets.json` - Build presets (release, debug, relwithdebinfo, clang variants)
- `CMakeLists.txt` - Main build definition with CUDA/C++20 requirements, link targets
- `extension_config.cmake` - DuckDB extension registration and versioning
- `.clang-format` - C++/CUDA code formatting rules
- `.clang-tidy` - C++ static analysis configuration
- `.pre-commit-config.yaml` - Auto-formatting hooks (clang-format, black, cmake-format, codespell)

**Feature Gating:**
- `[feature.cuda13]` / `[feature.cuda12]` in pixi.toml - CUDA version selection
- `[feature.duckdb-python]` in pixi.toml - Python binding build environment
- `ENABLE_STREAM_CHECK` CMake option - Debug utility for CUDA stream tracking
- `SIRIUS_ENABLE_LEGACY` CMake define - Legacy Sirius code path (optional, can be removed)

## Platform Requirements

**Development:**
- Linux x86_64 or aarch64
- NVIDIA GPU with CUDA compute capability 7.5+ (Turing)
- CUDA Toolkit 12.x or 13.x
- C++ compiler: Clang 21.x (from pixi)
- 32+ GB RAM recommended for parallel builds (CMAKE_BUILD_PARALLEL_LEVEL controls parallelism)

**Production:**
- NVIDIA GPU (same capability requirements)
- CUDA Runtime (distributed as libcudart)
- Host memory for CPU fallback and data staging
- Linux runtime libraries (glibc, libstdc++)

## Extension Loading

**Static Extension:**
- Path: `build/release/extension/sirius/sirius.duckdb_extension`
- Linked directly into DuckDB binary at build time
- Used by CLI: `duckdb db.duckdb`

**Loadable Extension:**
- Path: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
- Dynamically loaded at runtime via `LOAD 'path/to/sirius_loadable.duckdb_extension'`
- CLI usage: `SELECT * FROM duckdb_functions()` to list Sirius functions

---

*Stack analysis: 2026-04-03*
