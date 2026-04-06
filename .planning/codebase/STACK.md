# Technology Stack

**Analysis Date:** 2026-04-06

## Languages

**Primary:**
- C++ 20 - Core GPU acceleration engine (`src/`)
- CUDA 13+ - GPU kernels and cuDF integration (`src/cuda/`)

**Secondary:**
- Rust 1.85+ - Doris GPU Backend integration (`doris/crates/`)
- Java 17 - Apache Doris FE (optional, not built with Sirius)
- Python 3.8+ - Testing and development utilities (`test/tpch_performance/`)

## Runtime

**Environment:**
- Linux x86_64 and aarch64 (via Pixi)
- CUDA 13.1.* (or CUDA 12.* variant)
- GPU support: Turing (75) through Blackwell (120a) architectures

**Package Manager:**
- Pixi 0.59+ - Cross-platform Python/Conda environment management
- Conda channels: rapidsai, conda-forge
- Lockfile: `pixi.lock` (environment pinning)

## Frameworks

**Core:**
- DuckDB 1.4.4 - SQL query planning and CPU fallback engine (via git submodule)
- RAPIDS cuDF 26.02.* - GPU DataFrame operations (vector operations, joins, aggregates)
- RMM (RAPIDS Memory Manager) - GPU memory allocation and pool management
- cuCascade (NVIDIA) - GPU/CPU/disk tiered memory management with reservations

**Substrait:**
- Substrait extension (`substrait/` submodule) - Query plan IR format (protobuf-based)
- Protobuf - Substrait plan serialization

**Build/Dev:**
- CMake 4.1.* - C++ build configuration
- Ninja - Fast C++ build backend (recommended)
- sccache - Distributed C++ compilation caching
- clang 21+ - C++ compiler (LLVM/Clang toolchain)

**Testing:**
- Catch2 - C++ unit testing framework (in `duckdb/third_party/catch`)
- SQLLogicTest - SQL query correctness tests (`test/sql/`)

**Code Quality:**
- clang-format - C++/CUDA code formatting (`.clang-format`)
- clang-tidy - C++ linting (`.clang-tidy`)
- black - Python code formatting
- cmake-format - CMake file formatting
- codespell - Spell checker (`.codespell_words`)
- pre-commit - Git hook framework

## Key Dependencies

**Critical:**
- `libcudf` 26.02.* - GPU DataFrame library (hash joins, aggregations, sorting)
- `librmm` - RAPIDS memory resource abstraction
- `spdlog` 1.8.* - Structured logging with file rotation
- `libconfig++` - Configuration file parsing (`.cfg` format)
- `libabseil` 20260107.0+ - Google Abseil utilities (any_invocable)
- `numa` (PkgConfig) - NUMA support for multi-socket systems

**Doris Integration:**
- `duckdb` (crate) 1.10501.0 - Rust DuckDB FFI bindings
- `arrow` 54 - Apache Arrow data format (IPC, serialization)
- `arrow-flight` 54 - Arrow Flight RPC protocol
- `tonic` 0.13 - gRPC framework (Rust)
- `tokio` 1.* - Async runtime (full features)
- `substrait` 0.52 - Substrait protobuf bindings (Rust)
- `prost` 0.13 - Protocol buffer code generation
- `cudarc` 0.19.4 - CUDA runtime bindings (Rust)
- `mysql_async` 0.34 - MySQL connection pooling (Doris FE queries)

**Infrastructure:**
- `libcurand-dev` - CUDA random number generation (GPU utilities)
- `thrift-compiler` 0.22+ - Apache Thrift RPC framework (Doris protocol)
- `protobuf` 5+ - Protocol buffers (Doris + Substrait)
- `mold` - Fast linker for build optimization
- `sqlite` 3.52.0+ - Lightweight SQL tests

## Configuration

**Environment:**
- Set via `pixi.toml` [dependencies] and [feature.*.activation.env]
- Runtime: `.sirius/sirius.cfg` (INI format, libconfig++)
- GPU selection: CUDA architectures specified in `pixi.toml` features
  - CUDA 13: `75-real;80-real;86-real;90a-real;100f-real;120a-real;120` (all modern GPUs)
  - CUDA 12: `75-real;80-real;86-real;90a-real` (up to Hopper)

**Build:**
- `CMakeLists.txt` - Main build orchestration
- `extension_config.cmake` - DuckDB extension loading (sirius, json, tpcds, tpch, parquet, icu, substrait)
- `.clang-format` - C++ formatting rules
- `.clang-tidy` - Clang static analysis configuration

## Platform Requirements

**Development:**
- Linux system with NVIDIA GPU (Turing T4 or newer recommended)
- 64GB+ RAM (C++ compilation is memory-intensive at 8+ parallel jobs)
- Pixi installed (`>=0.59`)
- CUDA 12 or 13 SDK

**Build Outputs:**
- Static extension: `build/release/extension/sirius/sirius.duckdb_extension`
- Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
- Unit tests: `build/release/extension/sirius/test/cpp/sirius_unittest`
- Doris GPU BE: `doris/target/release/sirius-doris-be` (Rust binary)

**Production:**
- NVIDIA GPU with sufficient memory (cache region 1GB+, processing region 2GB+, pinned 4GB+)
- libcudf runtime libraries available
- DuckDB extension loader support

---

*Stack analysis: 2026-04-06*
