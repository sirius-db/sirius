# Technology Stack

**Analysis Date:** 2026-04-13

## Languages

**Primary:**
- C++ (C++20 standard) - Core extension implementation, physical operators, expression execution, memory management
- CUDA (20 standard) - GPU kernels for cuDF operations, data movement, expression evaluation
- Python (3.12+) - Test harness, performance benchmarking, dataset generation, utilities

**Secondary:**
- Bash - Build scripts, environment initialization, CI/CD automation
- SQL - Test cases, TPC-H/TPC-DS benchmark queries
- CMake - Build system configuration
- YAML - Configuration files for runtime settings

## Runtime

**Environment:**
- Linux (x86_64 and aarch64 architectures)
- CUDA 12 and 13 (configurable via pixi features)
- GPU support: NVIDIA CUDA-enabled GPUs (Turing through Blackwell architectures: 75, 80, 86, 90a, 100f, 120a, 120)

**Package Manager:**
- Pixi (monorepo package and environment management)
- CMake 3.30.4+ (build orchestration)
- vcpkg (alternate C++ dependency management path)
- pip (Python dependencies in duckdb-python environment)

**Build Tool:**
- Ninja (primary build generator)
- GCC or Clang (C++ compiler, configurable)
- NVCC (CUDA compiler)
- sccache (C++ and CUDA build cache)

## Frameworks

**Core SQL Engine:**
- DuckDB 1.4.4 (submodule: `duckdb/`) - SQL execution engine that Sirius extends

**GPU Computing:**
- RAPIDS cuDF 26.04.x - GPU DataFrame library for data manipulation
- RMM (RAPIDS Memory Manager) - GPU memory allocation and pooling
- CUDA Runtime (libcudart) - Low-level GPU execution and profiling
- libcurand-dev - CUDA random number generation

**Memory Management (GPU-native):**
- cuCascade (submodule: `cucascade/`) - GPU memory reservation and tiered memory (GPU/host/disk)
- libnuma - NUMA-aware memory operations for CPU pinning

**Configuration & Serialization:**
- yaml-cpp - YAML configuration file parsing
- spdlog 1.8.x - Structured logging with file and console outputs

**Utilities:**
- Abseil (libabseil 20260107.0+) - C++ library utilities (any_invocable, container views)
- fmt (embedded in DuckDB) - String formatting (NOTE: namespace conflict with spdlog requires careful include ordering)

## Key Dependencies

**Critical (GPU Execution Path):**
- `cudf::cudf` - libcudf and libcudf-cuda integration for GPU data structures and algorithms
- `rmm::rmm` - Memory allocation, stream management, device memory pools
- `cuCascade::cucascade` - Tiered memory management (GPU/host/disk with deferred transfers)

**Infrastructure:**
- `spdlog::spdlog` - Async logging with per-file rotation (config: `src/sirius_context.cpp`)
- `yaml-cpp::yaml-cpp` - Configuration parsing from `SIRIUS_CONFIG_FILE`
- `PkgConfig::NUMA` - libnuma for CPU memory pinning (required for pinned memory buffer manager)
- `absl::any_invocable` - Type-erased callable wrapper for task scheduling

**Extension Integration:**
- DuckDB extension infrastructure (via `extension-ci-tools` submodule)
- DuckDB built-in extensions: JSON, TPC-DS, TPC-H (for testing), Parquet, ICU

## Configuration

**Environment:**
- Set via pixi features: `pixi shell` (default cuda13), `pixi shell -e cuda12`, `pixi shell -e vcpkg`, `pixi shell -e duckdb-python`
- GPU architectures configured in `pixi.toml` via `CUDAARCHS` env var (feature: cuda13, cuda12, vcpkg)
- Build parallelism: `CMAKE_BUILD_PARALLEL_LEVEL` (default: all cores, reduce if memory-limited)
- Compiler cache: `SCCACHE_GHA_ENABLED` (CI/CD), local sccache for dev builds

**Runtime:**
- `SIRIUS_CONFIG_FILE` - Path to YAML configuration (default: `~/.sirius.yaml` or `/etc/sirius/config.yaml`)
- `SIRIUS_LOG_DIR` - Log output directory (default: `${CMAKE_BINARY_DIR}/log`)
- `SIRIUS_LOG_LEVEL` - Logging level: trace, debug, info, warn, error (default: info)
- `SIRIUS_DISABLE` - Disable Sirius extension (set to "1" to fall back to DuckDB)
- `SIRIUS_STREAM_CHECK_LIB` - Path to stream check library for debugging (optional, requires build with `-DENABLE_STREAM_CHECK=ON`)

**Build:**
- `CMakeLists.txt` - Main build configuration
- `cmake/CMakePresets.json` - Build presets (debug, release, relwithdebinfo, clang variants, vcpkg variants)
- `extension_config.cmake` - DuckDB extension registration
- `pixi.toml` - Conda/Pixi environment specification with feature flags
- `.pre-commit-config.yaml` - Code formatting and linting hooks (clang-format, black, codespell, cmake-format)
- `.clang-format` - C++/CUDA formatting rules
- `.clang-tidy` - C++ linting rules
- `.codespell_words` - Custom spell-check word list

## Platform Requirements

**Development:**
- Linux x86_64 or aarch64
- NVIDIA GPU with CUDA 12 or 13 capability
- Pixi 0.59+
- Sufficient GPU memory (typically 4GB+ for TPC-H scale factor 1)
- Host RAM adequate for build parallelism (>16GB recommended)

**Build Output:**
- Static extension: `build/release/extension/sirius/sirius.duckdb_extension`
- Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
- Unit test binary: `build/release/extension/sirius/test/cpp/sirius_unittest`
- Debug build available: `build/debug/extension/sirius/sirius.duckdb_extension`

**Production/Deployment:**
- Linux x86_64 host with NVIDIA GPU
- DuckDB version 1.4.4 (must match compiled extension)
- CUDA 12 or 13 runtime libraries on system
- No internet access required post-deployment (all deps statically linked or bundled)

---

*Stack analysis: 2026-04-13*
