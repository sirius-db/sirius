# External Integrations

**Analysis Date:** 2026-04-06

## APIs & External Services

**GPU Compute APIs:**
- CUDA Runtime API - Direct GPU kernel execution and memory management
- NVIDIA CUDA Profiler API (`cudaProfilerStart/Stop`) - Performance profiling hooks in `src/sirius_extension.cpp`
- NVIDIA Management Library (NVML) - GPU device monitoring and telemetry

**DuckDB Extension Interface:**
- DuckDB Extension API - Sirius loads as unsigned DuckDB extension via `LOAD` command
- DuckDB table functions - Custom SQL interface via `gpu_execution()` and `gpu_buffer_init()` procedures
- DuckDB physical plan interface - Query optimization and execution planning integration

## Data Storage

**Databases:**
- DuckDB 1.4.4 - In-process SQL database (extension targets)
  - Connection: In-process via `duckdb.connect()`
  - Client: DuckDB C++ API (headers in `duckdb/` submodule)
  - Configuration: `allow_unsigned_extensions=true` required for Sirius loading

**File Storage:**
- Parquet files - Primary columnar data format
  - Reading: via `src/op/scan/parquet_scan_task.cpp` with DuckDB's parquet reader
  - Format: DuckDB-compatible Parquet files (scanned into GPU memory)
  - Conversion: `src/data/host_parquet_representation.cpp` manages CPU<->GPU transfer
- Iceberg tables - Delta lake format support
  - Reading: `src/op/scan/iceberg_scan_task.cpp` with Avro metadata parsing
  - Delete filters: `src/op/scan/equality_delete_filter.cpp`, `src/op/scan/positional_delete_filter.cpp`
  - Metadata: `src/op/scan/iceberg_metadata_reader.cpp`
  - Avro format: `src/op/scan/iceberg_avro_reader.cpp` for Iceberg metadata rows
- Local filesystem - Direct file I/O for data sources and test datasets

**Caching:**
- CPU-side caching: Scan result caching via `src/op/scan/cached_ranges.cpp`
- GPU tiered memory: cuCascade manages GPU/host/disk spilling (`cucascade/` submodule)
- Configuration: `SIRIUS_CACHE_LEVEL` environment option controls CPU cache behavior

## Memory Management

**GPU Memory:**
- RMM (RAPIDS Memory Manager) - GPU memory allocation and pooling
  - Device allocators: `src/cuda/allocator.cu`
  - Stream pooling: `cucascade/memory/stream_pool.hpp`
- cuCascade - Tiered memory (GPU/host/disk) for data exceeding GPU capacity
  - Data repository: `cucascade/data/data_repository.hpp`
  - Memory reservation: `cucascade/memory/memory_reservation.hpp`
  - OOM handling: `src/memory/defragmenter_oom_policy.cpp`

**CPU Memory:**
- Pinned host memory - DuckDB-allocated pinned memory via `cudf::pinned_memory`
- Configuration: `USE_PIN_MEM_FOR_CPU_PROCESSING`, `USE_PIN_MEM_FOR_CACHING` in `src/config.cpp`

## Authentication & Identity

**Auth Provider:**
- None - Sirius uses DuckDB's authentication context
- Database access: Inherits DuckDB connection credentials and permissions
- Extension security: Requires unsigned extension flag (`allow_unsigned_extensions=true`)

## Monitoring & Observability

**Logging:**
- spdlog (1.8.*) - Structured logging framework
  - Configuration: `src/log/logging.hpp`
  - Log directory: `$SIRIUS_LOG_DIR` (default: `${CMAKE_BINARY_DIR}/log`)
  - Log level: `$SIRIUS_LOG_LEVEL` (default: `info`)
  - Flush interval: 3 seconds (configurable via `Config::LOG_FLUSH_SECONDS`)
- Query logging: Query begin/end events in `src/sirius_context.cpp`
- Pipeline logging: Per-operator row counts and execution state

**Error Tracking:**
- Structured error handling via C++ exceptions
- Backtrace support: `src/util/segfault_backtrace_handler.cpp` for crash analysis
- Stream check library: Optional CUDA stream validation (loaded from `$SIRIUS_STREAM_CHECK_LIB`)

**Performance Profiling:**
- NVIDIA nsys integration - CUDA profiler hooks via `cudaProfilerStart/Stop` in extension initialization
- Performance test harness: `test/tpch_performance/` for TPC-H benchmarking

## CI/CD & Deployment

**Hosting:**
- GitHub-based repository (sirius)
- Build artifacts: `build/release/extension/sirius/` directory

**Build System:**
- CMake 4.1+ with DuckDB extension template
- Parallel compilation: `CMAKE_BUILD_PARALLEL_LEVEL` variable for tuning
- Presets: `cmake/CMakePresets.json` defines build configurations

**Testing:**
- C++ unit tests: `test/cpp/` with Catch2 framework
- SQL logic tests: DuckDB test runner in `duckdb/test/`
- Integration tests: TPC-H and TPC-DS benchmarking

## Environment Configuration

**Required env vars for build:**
- `PIXI_PROJECT_ROOT` - Project root (set by pixi automatically)
- `CMAKE_CUDA_ARCHITECTURES` - GPU target architectures (auto-set by pixi CUDAARCHS feature)
- `DUCKDB_SOURCE_PATH` - DuckDB source for Python builds (set in duckdb-python task)

**Required env vars for runtime:**
- None mandatory; all have sensible defaults
- Optional: `SIRIUS_LOG_DIR`, `SIRIUS_LOG_LEVEL`, `SIRIUS_CONFIG_FILE` for customization

**Secrets location:**
- Configuration file: `~/.sirius/sirius.cfg` (user home directory)
- Environment variables: No sensitive data should be passed via env vars
- DuckDB config: `allow_unsigned_extensions=true` must be set in connection config

## Webhooks & Callbacks

**Incoming:**
- None - Sirius is a query processing library, not a network service

**Outgoing:**
- CUDA device callbacks: Stream callbacks registered with CUDA runtime for async event handling
- No external API calls or webhooks

## Test Data & Benchmarks

**TPC-H Benchmarking:**
- Dataset generation: `test/tpch_performance/generate_test_data.py`
- Performance testing: `test/tpch_performance/performance_test.py`
- Scale factors: Configurable (1, 10, 100+ supported)
- Dataset format: Parquet files in `test_datasets/tpch_parquet_sf{N}/`

**TPC-DS Benchmarking:**
- Dataset generation: `test_datasets/generate_tpch/` and `test_datasets/tpcds_parquet_sf1/`
- Supported scale factors: SF1, SF10
- Integration tests: `test/cpp/integration/test_gpu_execution_tpch.cpp`

**Iceberg Test Data:**
- Metadata reading: `test_datasets/gen_iceberg_test_data.py`
- Format: Apache Iceberg with Avro metadata

## Code Quality & Linting

**External Tools:**
- pre-commit hooks: `.pre-commit-config.yaml` with:
  - clang-format (v20.1.4+) for C++/CUDA formatting
  - black (v25.1.0) for Python formatting
  - cmake-format (v0.6.13) for CMake files
  - codespell (v2.4.1) for spell checking with custom words in `.codespell_words`
  - Standard hooks (trailing whitespace, JSON formatting, TOML validation)

---

*Integration audit: 2026-04-06*
