# External Integrations

**Analysis Date:** 2026-04-03

## APIs & External Services

**GPU Compute:**
- NVIDIA CUDA C API - GPU kernel execution, memory transfers, stream management
  - SDK/Client: cuda-nvcc, libcudart (CUDA runtime)
  - Headers: Direct inclusion of `<cuda.h>`, `<cuda_runtime.h>`

**DuckDB Planner & Executor:**
- DuckDB Extension API (1.4.4) - Physical plan generation, operator execution framework
  - SDK: libduckdb (linked as duckdb_static)
  - Integration: `src/sirius_extension.cpp` registers extension via LoadInternal function
  - Entry point: `CALL gpu_execution('SELECT ...')` table function

**RAPIDS Libraries:**
- cuDF - GPU DataFrame operations (joins, groupby, orderby, filtering)
  - SDK: libcudf::cudf CMake target
  - Used in: `src/cuda/cudf/*.cu` CUDA implementations
- RMM - RAPIDS Memory Manager for GPU memory allocation
  - SDK: rmm::rmm CMake target
  - Configuration: `src/memory/sirius_memory_reservation_manager.hpp` wraps RMM device memory resources

## Data Storage

**Databases:**
- DuckDB 1.4.4 - Primary query execution engine
  - Connection: Embedded in-process via DuckDB C++ API
  - Client: Native C++ API (no ORM, direct connection management)
  - Storage: Parquet (via DuckDB's scanner)

**File Storage:**
- Parquet Format - Primary external data source
  - Reader: DuckDB's built-in Parquet scanner
  - Handler: `src/op/sirius_physical_parquet_scan.cpp` and task-based scanning
  - Scanning: Task-based parallel reads via `src/op/scan/parquet_scan_task.cpp`

- Iceberg Format - Data format support with metadata
  - Reader: Custom `src/op/scan/iceberg_avro_reader.cpp` for Avro metadata
  - Handler: `src/op/sirius_physical_iceberg_scan.cpp`
  - Delete handling: `src/op/scan/iceberg_delete_pipeline.cpp`, `equality_delete_filter.cpp`, `positional_delete_filter.cpp`
  - Metadata: `src/op/scan/iceberg_metadata_reader.cpp` for manifest parsing

**Caching:**
- Host/GPU Cache - Internal tiered memory management
  - Disk overflow: cuCascade library manages spilling to disk when GPU memory exhausted
  - CPU cache: `src/cpu_cache.cpp` with configurable sizing
  - Reservation: `src/memory/sirius_memory_reservation_manager.hpp` coordinates RMM + cuCascade

## Authentication & Identity

**Auth Provider:**
- None - Sirius is embedded in DuckDB and inherits its authentication model
- DuckDB connection authentication applies (if configured)

## Monitoring & Observability

**Error Tracking:**
- None - Errors logged to files

**Logs:**
- Approach: spdlog daily file rotation
- Location: Default `${CMAKE_BINARY_DIR}/log/sirius.log`
- Configuration: Environment variables `SIRIUS_LOG_DIR`, `SIRIUS_LOG_LEVEL`
- Levels: trace, debug, info, warn, error (default: info)
- Macro: `SIRIUS_LOG_DEBUG()`, `SIRIUS_LOG_INFO()`, etc. in `src/include/log/logging.hpp`
- Sink: Daily file sink (rotates daily at midnight)
- Pattern: `[YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] [file:line] message`
- Flush interval: 3 seconds (configurable via `LOG_FLUSH_SECONDS` in `src/config.cpp`)

**Profiling:**
- CUDA Profiler API - Optional CUDA profiling integration
  - Functions: `cudaProfilerStart()`, `cudaProfilerStop()` (forward-declared in `src/sirius_extension.cpp`)
  - Usage: Hook into query execution for nsys-style profiling

## CI/CD & Deployment

**Hosting:**
- No cloud/SaaS integration - Sirius is an on-premises library
- Deployment: Statically or dynamically linked into DuckDB binary

**CI Pipeline:**
- None detected - No CI/CD integration points in codebase

## Environment Configuration

**Required env vars for runtime:**
- `SIRIUS_LOG_DIR` - Log directory (default: log)
- `SIRIUS_LOG_LEVEL` - Log level: trace|debug|info|warn|error (default: info)

**Build-time env vars:**
- `CMAKE_BUILD_PARALLEL_LEVEL` - Parallel build jobs (recommended: nproc)
- `CUDAARCHS` - GPU architectures to compile for (set by pixi feature)
- `DUCKDB_SOURCE_PATH` - DuckDB source root for Python builds (set by pixi duckdb-python feature)

**Secrets location:**
- No secrets storage - Sirius is a stateless compute engine

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Configuration Parameters

**Query Execution:**
- `use_cudf_expr` (default: true) - Use cuDF for expression evaluation
- `use_custom_top_n` (default: true) - Use custom GPU TOP-N implementation
- `use_pin_memory` (default: true) - Use pinned memory for CPU processing
- `use_pin_memory_for_caching` (default: false) - Use pinned memory for GPU caching
- `use_opt_table_scan` (default: true) - Use optimized table scan with async I/O
- `opt_table_scan_num_streams` (default: 8) - CUDA streams for parallel memcpy
- `opt_table_scan_memcpy_size` (default: 64 MB) - Chunk size for host-to-GPU transfers

**Memory Management:**
- `enable_fallback_check` (default: false) - Check if operation should fallback to CPU
- `enable_duckdb_fallback` (default: false) - Actually fallback to DuckDB on error
- `max_sort_partition_bytes` (default: 0 = auto) - Max memory per sort partition (33% GPU)

**Logging:**
- `log_level` (default: info) - Severity threshold
- `log_dir` (default: log) - Output directory
- `log_flush_seconds` (default: 3) - Flush interval

**Other:**
- `print_gpu_table_max_rows` (default: 1000) - Max rows for GPU table display
- `enable_regex_jit_impl` (default: true) - JIT regex evaluation
- `default_scan_task_batch_size` (default: 512 MB) - Scan task batch size
- `default_scan_task_varchar_size` (default: 256 bytes) - VARCHAR size for row estimation

## Data Flow & Integration Architecture

**Query Execution Pipeline:**

1. **Entry Point:** `CALL gpu_execution('SELECT ...')` in `src/sirius_extension.cpp`
2. **Plan Generation:** DuckDB planner produces physical plan → `sirius_physical_plan_generator.cpp` transforms to GPU operators
3. **Operator Execution:** `src/op/` operators execute in task-based pipeline
4. **GPU Operations:** cuDF wrappers in `src/cuda/cudf/` execute on GPU via RMM allocator
5. **Memory Management:** cuCascade reservation manager oversees GPU/host/disk tiering
6. **Fallback:** If unsupported → `src/fallback.cpp` routes to DuckDB CPU execution
7. **Results:** `sirius_physical_result_collector.cpp` collects and returns results

**Data Representation:**
- GPU: `cudf::table` (columnar format with device memory)
- Host: `HostTable` intermediate format for CPU staging/cache
- Parquet: Task-based streaming readers in `src/op/scan/parquet_scan_task.cpp`
- Iceberg: Metadata + Parquet with delete vectors managed by delete filter operators

---

*Integration audit: 2026-04-03*
