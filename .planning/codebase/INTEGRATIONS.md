# External Integrations

**Analysis Date:** 2026-04-13

## APIs & External Services

**DuckDB Core Integration:**
- DuckDB 1.4.4 via submodule `duckdb/`
  - Query planner integration: `src/planner/sirius_physical_plan_generator.cpp`
  - Physical operator interface: `src/op/sirius_physical_operator.cpp`
  - Expression evaluation: `src/expression_executor/gpu_expression_executor.cpp`
  - Extension entry point: `src/sirius_extension.cpp`

**RAPIDS GPU Libraries:**
- libcudf (GPU DataFrame library)
  - CUDA data structures and algorithms
  - Implementation: `src/cuda/cudf/` (join, groupby, orderby, aggregate, duplicate elimination)
- RMM (RAPIDS Memory Manager)
  - GPU memory allocation and pooling
  - Stream management for async operations
  - Configuration: `src/memory/sirius_memory_reservation_manager.cpp`

**GPU Vendor APIs:**
- NVIDIA CUDA Runtime (libcudart)
  - Low-level GPU kernel execution and profiling
  - Forward-declared in `src/sirius_extension.cpp`: `cudaProfilerStart()`, `cudaProfilerStop()`
- NVIDIA libcurand-dev
  - Random number generation (linked but minimal direct usage)

## Data Storage

**Databases:**
- DuckDB (embedded)
  - In-memory or file-based SQL engine
  - Configuration: DuckDB connection parameters in `src/sirius_interface.cpp`
  - Client: Direct C++ API via DuckDB headers (`duckdb/main/database.hpp`, etc.)

**File Storage:**
- Parquet files (primary supported format)
  - Reader: cuDF's hybrid scan Parquet reader (experimental)
  - Implementation: `src/op/scan/parquet_scan_task.cpp`
  - Data representation: `src/include/data/host_parquet_representation.hpp`
  - Metadata scanning: `src/op/scan/sirius_parquet_metadata_scan_operator.cpp`
  - Converters: `src/data/host_parquet_representation_converters.cpp`

- Iceberg tables (metadata + Parquet)
  - Metadata reader: `src/op/scan/iceberg_metadata_reader.cpp`
  - Avro delete files: `src/op/scan/iceberg_avro_reader.cpp`
  - Scan task: `src/op/scan/iceberg_scan_task.cpp`
  - Delete filters: `src/op/scan/equality_delete_filter.cpp`, `src/op/scan/positional_delete_filter.cpp`
  - Equality delete mask: `src/cuda/iceberg/equality_delete_mask.cu`
  - Pipeline: `src/op/scan/iceberg_delete_pipeline.cpp`

- DuckDB catalog tables (secondary fallback)
  - Scan executor: `src/op/scan/duckdb_scan_executor.cpp`
  - Scan task: `src/op/scan/duckdb_scan_task.cpp`

**Caching:**
- CPU-side cache (host memory)
  - Implementation: `src/cpu_cache.cpp`
  - Memory representation: `src/data/host_parquet_representation.hpp`
  - Pinned memory allocation (DMA-friendly host buffers)

- GPU-side cache (GPU VRAM)
  - cuCascade tiered memory representation

## Authentication & Identity

**Auth Provider:**
- Custom (none)
- No authentication layer in Sirius core
- Extension loads into existing DuckDB database connection (inherits DuckDB's auth model if any)

## Monitoring & Observability

**Error Tracking:**
- None (no external error tracking service)
- Fallback to DuckDB CPU execution on GPU errors: `src/fallback.cpp`

**Logs:**
- spdlog-based structured logging
  - Output directory: `SIRIUS_LOG_DIR` env var (default: `${CMAKE_BINARY_DIR}/log`)
  - Log level: `SIRIUS_LOG_LEVEL` env var (trace, debug, info, warn, error; default: info)
  - Flush interval: configurable via `Config::LOG_FLUSH_SECONDS` (`src/config.cpp`)
  - File rotation: per-file in `${SIRIUS_LOG_DIR}/`
  - Async logging enabled (spdlog thread pool)

**Debug Utilities:**
- Stream check library (optional, requires build flag `ENABLE_STREAM_CHECK`)
  - Detects CUDA default stream usage
  - Path: `SIRIUS_STREAM_CHECK_LIB` env var
  - Implementation: `src/util/stream_check_wrapper.cpp`, `utils/stream_check/`

- Segfault backtrace handler
  - Captures stack traces on segfaults
  - Output: `${SIRIUS_LOG_DIR}/backtrace.log`
  - Implementation: `src/util/segfault_backtrace_handler.cpp`

- GPU memory profiling (via NVIDIA's nsys profiler)
  - CUDA Profiler API integration: `cudaProfilerStart()`/`cudaProfilerStop()` in `src/sirius_extension.cpp`

## CI/CD & Deployment

**Hosting:**
- GitHub Actions (self-hosted runners for GPU testing)
  - Build runners: ubuntu-24.04 (CPU), ubuntu-22.04 (GPU T4)
  - ARM runners: ubuntu-24.04-arm (aarch64)

**CI Pipeline:**
- `.github/workflows/check.yml` - Linting and formatting (pre-commit hooks)
- `.github/workflows/test.yml` - Build, unit tests, TPC-H performance snapshot
  - Pre-commit hooks: `.pre-commit-config.yaml`
    - clang-format (C++/CUDA code style)
    - black (Python formatting)
    - codespell (spell checking with `.codespell_words`)
    - cmake-format (CMake code formatting)
    - json/yaml/toml validation
  - Build: `make release` (CMAKE_BUILD_PARALLEL_LEVEL, sccache)
  - Unit tests: `./build/release/extension/sirius/test/cpp/sirius_unittest`
  - Integration tests: TPC-H benchmark with result validation

**Build System:**
- DuckDB extension template via `extension-ci-tools` submodule
- CMake presets in `cmake/CMakePresets.json` (linked to `duckdb/CMakePresets.json`)
- Makefile wrapper: `Makefile` delegates to CMake presets

## Environment Configuration

**Required env vars for runtime:**
- `SIRIUS_CONFIG_FILE` - Path to YAML configuration file (optional; defaults to `~/.sirius.yaml` or system path)
- `SIRIUS_LOG_DIR` - Directory for log output (optional; default: `${CMAKE_BINARY_DIR}/log`)
- `SIRIUS_LOG_LEVEL` - Logging verbosity (optional; default: info)

**Optional env vars:**
- `SIRIUS_DISABLE` - Set to "1" to disable Sirius and fall back to DuckDB
- `SIRIUS_STREAM_CHECK_LIB` - Path to stream check debug library (requires build flag)

**Required env vars for build:**
- `CMAKE_BUILD_PARALLEL_LEVEL` - Number of parallel build jobs (default: auto-detect cores)
- `CUDAARCHS` - Target GPU architectures (set by pixi features: cuda13, cuda12, vcpkg)
- `VCPKG_CUDA_VERSION` - CUDA version for vcpkg builds (12 or 13)

**Secrets/Credentials:**
- None required by Sirius core
- GitHub Actions: Uses `SCCACHE_GHA_ENABLED` for distributed caching (no token needed, built-in)

## Webhooks & Callbacks

**Incoming:**
- None (Sirius does not expose HTTP endpoints)

**Outgoing:**
- None (Sirius does not make external HTTP calls)

## Performance Tools Integration

**Benchmarking:**
- TPC-H dataset generation and execution
  - Script: `test/tpch_performance/generate_test_data.py`
  - Queries: `scripts/tpch-queries.sql`
  - Results validation: `test/tpch_performance/benchmark_and_validate.sh`
  - Comparison: `tools/compare_runs.py`

- TPC-DS support (via DuckDB's built-in extension)

**Profiling:**
- NVIDIA Nsys integration
  - Manual CUDA profiler API calls: `cudaProfilerStart()`, `cudaProfilerStop()`
  - Profile analysis tools: `tools/parse_duckdb_log.py`, `tools/parse_pipeline_log.py`
  - Operator attribution via pipeline logs: `tools/parse_pipeline_log.py`

## Submodule Dependencies

**Core Submodules:**
- `duckdb/` - DuckDB 1.4.4 base engine (branch: main)
- `cucascade/` - NVIDIA cuCascade for GPU tiered memory (branch: main)
- `duckdb-python/` - Python bindings for DuckDB (branch: main)
- `extension-ci-tools/` - DuckDB extension build infrastructure (branch: main)

**Optional/Third-party Submodules:**
- `substrait/` - Sirius Substrait extension for query interchange (branch: main)
- `vcpkg/` - Microsoft vcpkg for C++ dependency management (alternate build path)
- `.claude/claude-tools/` - Claude AI code generation tools (private sirius-db repo)

---

*Integration audit: 2026-04-13*
