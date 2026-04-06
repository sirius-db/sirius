# External Integrations

**Analysis Date:** 2026-04-06

## APIs & External Services

**DuckDB Core:**
- DuckDB 1.4.4 (submodule `duckdb/`)
  - Integration: Sirius is a DuckDB extension loaded via `LOAD 'sirius.duckdb_extension'`
  - API: DuckDB C++ extension API for plan generation, operator execution, result collection
  - Usage: `gpu_execution('SELECT ...')` table function entry point
  - Type hints: Uses DuckDB's physical plan AST, expression types, vector batches

**RAPIDS GPU Libraries:**
- libcudf 26.02.x - GPU dataframe operations
  - Headers: `<cudf/io/parquet.hpp>`, `<cudf/io/experimental/hybrid_scan.hpp>`, `<cudf/io/datasource.hpp>`, `<cudf/io/parquet_schema.hpp>`
  - Operators: `cudf::left_join()`, `cudf::inner_join()`, `cudf::stable_sort_keys()`, `cudf::groupby::aggregate()`
  - Usage: Join operators (`src/cuda/operator/hash_join_*.cu`), aggregation (`src/cuda/cudf/cudf_groupby.cu`), ordering (`src/cuda/cudf/cudf_orderby.cu`)
  - Client: `libcudf.so` (linked via CMake find_package)

- RMM (RAPIDS Memory Manager)
  - Headers: `<rmm/cuda_stream_view.hpp>`
  - Purpose: GPU memory allocation, stream management, resource pools
  - Usage: Memory reservations in `src/memory/sirius_memory_reservation_manager.cpp`

**GPU Acceleration:**
- NVIDIA CUDA Runtime (cudart)
  - Profiler API: `cudaProfilerStart()`, `cudaProfilerStop()` (linked via libcudart)
  - Usage: Performance profiling in `src/sirius_extension.cpp`
  - Streams: CUDA stream management for concurrent kernel execution

- CUDA Libraries:
  - libcurand-dev: Random number generation on GPU
  - curand: CUDA random number generation kernels

## Data Storage

**Databases:**
- DuckDB - In-process SQL database
  - Connection: Via DuckDB C++ API (`duckdb::Connection`, `duckdb::ClientContext`)
  - Primary use: Query parsing, optimization, CPU fallback execution

**File Storage:**
- Local Parquet Files
  - Scanned via cuDF and DuckDB parquet readers
  - Implementation: `src/op/scan/parquet_scan_task.cpp`
  - Format support: Parquet 1.0+
  - Schema discovery: Via parquet metadata (footer reads)

- Iceberg Table Format
  - Metadata: Iceberg manifests and delete files
  - Implementation: `src/op/scan/iceberg_scan_task.cpp`, `src/op/scan/iceberg_metadata_reader.cpp`
  - Delete handling: Positional and equality delete filters (`src/op/scan/iceberg_delete_filter.cpp`)
  - Avro support: `src/op/scan/iceberg_avro_reader.cpp` (for manifest files)

**GPU Memory Storage:**
- GPU Device Memory
  - Allocated via: RMM memory pools
  - Tiered management: cuCascade handles spilling to host/disk

- Host Memory (Pinned)
  - GPU-accessible pinned memory for zero-copy transfers
  - Controlled via: `use_pin_memory` config option

**Caching:**
- CPU Cache
  - Implementation: `src/cpu_cache.cpp`
  - Purpose: Cache frequently accessed host data to avoid repeated GPU transfers
  - Controlled via: `use_pin_memory_for_caching` config option

## Authentication & Identity

**Auth Provider:**
- Not applicable - Sirius is a query execution engine, auth delegated to DuckDB/host

## Monitoring & Observability

**Error Tracking:**
- None detected - Error handling via C++ exceptions

**Logs:**
- spdlog (1.8.x) - Structured logging framework
  - Configuration: `SIRIUS_LOG_LEVEL` (trace, debug, info, warn, error), `SIRIUS_LOG_DIR`
  - Usage: Logging in `src/include/log/logging.hpp`
  - Sink: File-based logging to `$SIRIUS_LOG_DIR` or `${CMAKE_BINARY_DIR}/log`
  - Per-component: Log files in `build/release/extension/sirius/test/cpp/log` for unit tests

**Performance Profiling:**
- NVIDIA Nsys Integration
  - Entry point: CUDA Profiler API via libcudart
  - Used by: `/profile-analyzer` Claude Code skill for kernel analysis
  - Output: nsys profile files (`.qdrep`)

## CI/CD & Deployment

**Hosting:**
- GitHub (source control)
  - Submodules: duckdb, cucascade, duckdb-python
  - Integration: Sirius delivered as loadable DuckDB extension

**CI Pipeline:**
- Not explicitly configured in source (relies on DuckDB extension CI tools)
- Build integration: `extension-ci-tools` included via Makefile

**Distribution:**
- Static extension: `build/release/extension/sirius/sirius.duckdb_extension`
- Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
- Python package: `duckdb-python/` (links against Sirius extension)

## Environment Configuration

**Required env vars:**
- `CUDA_VISIBLE_DEVICES` - GPU device selection (inherited from CUDA runtime)
- `SIRIUS_LOG_LEVEL` - Logging verbosity (optional, default: info)
- `SIRIUS_LOG_DIR` - Log output directory (optional, default: `${CMAKE_BINARY_DIR}/log`)

**Runtime config (via DuckDB CALL commands):**
- `gpu_buffer_init(gpu_mem, pinned_mem)` - Initialize GPU buffer pools (legacy mode)
- `gpu_execution('SELECT ...')` - Execute query on GPU (Super Sirius)
- `gpu_processing('SELECT ...')` - Execute query on GPU (legacy mode)

**Build-time configuration:**
- `CMAKE_CUDA_ARCHITECTURES` - Target GPU architectures (auto-detected if not set)
- `ENABLE_STREAM_CHECK` - Build stream-check debugging library (default: OFF)

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected (Sirius is a query execution engine, no external API calls)

## Integration Patterns

**DuckDB Extension Model:**
- Sirius intercepts DuckDB's physical plan execution
- Physical plan transformation: `sirius_physical_plan_generator` converts DuckDB logical plans to GPU-executable plans
- Fallback: Unsupported operations automatically downgrade to DuckDB CPU execution via `src/fallback.cpp`
- Result collection: GPU results marshaled back to DuckDB's vector format via `src/op/result/host_table_chunk_reader.cpp`

**GPU Computation Pipeline:**
1. DuckDB sends physical plan to Sirius
2. Plan generator creates GPU operators (`src/op/*.cpp`)
3. Pipeline executor schedules operators as tasks (`src/parallel/task_executor.cpp`)
4. CUDA kernels execute on GPU (via cuDF and custom kernels in `src/cuda/`)
5. Results spilled to GPU memory via cuCascade tiered memory management
6. Results transferred to host and collected by result collector

**Data Format Conversions:**
- Parquet → GPU Dataframe: `src/data/host_parquet_representation_converters.cpp`
- Arrow → GPU Dataframe: Via cuDF's Arrow C Data Interface
- DuckDB Vector → GPU Memory: Task-based copying with stream management

---

*Integration audit: 2026-04-06*
