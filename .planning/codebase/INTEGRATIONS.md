# External Integrations

**Analysis Date:** 2026-04-06

## APIs & External Services

**DuckDB Extension Interface:**
- DuckDB 1.4.4 (SQL engine) - `src/sirius_extension.cpp`
  - Client: C++ DuckDB API (libduckdb)
  - Functions: `gpu_buffer_init()`, `gpu_processing()`, `gpu_execution()`, `gpu_execution_substrait()`
  - Protocol: Table function interface with result collector

**Substrait Query Plan IR:**
- Substrait extension (submodule) - `substrait/` directory
  - Format: Protocol buffers (binary or JSON)
  - Used by: Doris FE for query plan serialization, GPU execution planner
  - Client: `src/sirius_extension.cpp` uses `SubstraitToDuckDB` for deserialization

**Apache Doris Integration:**
- Doris FE (Java) - RPC via gRPC/Thrift
  - Port: 9030 (FE leader), 19050+ (BE heartbeat), 19060+ (BE thrift), 18060+ (BE brpc)
  - Protocol: gRPC (VERSION_3 fragment execution), Thrift (legacy metadata), bRPC (data sink)
  - Entry: `doris/crates/sirius-doris-be/` Rust backend
  - Fragment execution: `doris/crates/doris-rpc/` (7k LOC RPC handlers)

**Arrow Flight RPC:**
- Arrow Flight 54 - Result streaming from GPU BE to Doris FE
  - Port: 18071 (default GPU BE Arrow Flight server)
  - Serialization: Apache Arrow IPC format
  - Implementation: `doris/crates/result-formatter/` (Arrow Flight server)

## Data Storage

**Databases:**
- DuckDB 1.4.4 (in-process)
  - Connection: Via C++ API (`src/sirius_interface.cpp`)
  - Client: `duckdb::Connection`
  - Purpose: Query planner, CPU fallback, scan executor

**File Storage:**
- Parquet files - Primary input format
  - Reader: DuckDB Parquet extension (built-in)
  - GPU path: `src/op/scan/parquet_scan_task.cpp` (hybrid scan via libcudf)
  - CPU path: `src/op/scan/duckdb_scan_executor.cpp` (DuckDB reader)

**Iceberg Tables:**
- Iceberg v2 metadata reader - `src/op/scan/iceberg_metadata_reader.cpp`
  - Avro deserialization: `src/op/scan/iceberg_avro_reader.cpp`
  - Delete filters: `src/op/scan/equality_delete_filter.cpp`, `src/op/scan/positional_delete_filter.cpp`

**GPU Memory Tiers:**
- cuCascade (NVIDIA) - Tiered memory management
  - Submodule: `cucascade/` (built as static library)
  - Interfaces: GPU caching region, GPU processing region, pinned host memory, disk spill
  - Management: `src/memory/sirius_memory_reservation_manager.cpp`

**Caching:**
- GPU pinned memory cache - CPU↔GPU fast transfers
  - Manager: `src/gpu_buffer_manager.cpp`
  - Reservation system: `src/memory/sirius_memory_reservation_manager.cpp`
- CPU cache - Multi-level data representation
  - Implementation: `src/cpu_cache.cpp`, `src/include/cpu_cache.hpp`

## Authentication & Identity

**Auth Provider:**
- None (in-process DuckDB extension)
- Doris FE authentication handled separately (MySQL protocol)
- No external identity service

## Monitoring & Observability

**Error Tracking:**
- None (stderr/file logging only)

**Logs:**
- spdlog 1.8.* - Structured logging to files
  - Directory: Environment variable `SIRIUS_LOG_DIR` (default: `${CMAKE_BINARY_DIR}/log`)
  - Level: `SIRIUS_LOG_LEVEL` (trace, debug, info, warn, error)
  - Flush: `SIRIUS_LOG_FLUSH_SECONDS` (auto-flush interval)
  - Usage: `src/include/log/logging.hpp` (SIRIUS_LOG_* macros)
  - Test logs: `build/release/extension/sirius/test/cpp/log/`

**Performance Profiling:**
- NVIDIA CUDA Profiler API - cudaProfilerStart/Stop
  - Reference: `src/sirius_extension.cpp` (extern "C" declarations)
  - Typical usage: `nsys profile` for kernel attribution

## CI/CD & Deployment

**Hosting:**
- No external hosting dependency (single-process GPU extension)
- Doris FE: Optional Java deployment (external)
- Doris GPU BE: Rust binary deployment (`sirius-doris-be`)

**CI Pipeline:**
- GitHub Actions workflow (`.github/workflows/`)
- Build matrix: CUDA 12, 13; Linux x86_64, aarch64
- Test: SQL Logic Tests, C++ unit tests
- Extension versioning: `dev` (source build)

## Environment Configuration

**Required env vars (for DuckDB extension):**
- `SIRIUS_LOG_DIR` - Log output directory
- `SIRIUS_LOG_LEVEL` - Log verbosity
- `SIRIUS_LOG_FLUSH_SECONDS` - Log flush interval

**Required env vars (for Doris GPU BE):**
- `NIXL_PREFIX` - nixl C++ library path (if using GPU-direct exchange)
- `LD_LIBRARY_PATH` - Include conda libdir for cudf/rmm/ucx
- `CONDA_PREFIX` - Pixi environment root

**Build env vars:**
- `CMAKE_BUILD_PARALLEL_LEVEL` - Parallel compilation jobs (default: nproc)
- `CUDAARCHS` - GPU architectures (set by pixi feature)
- `CUDA_HOME` / `CONDA_PREFIX` - CUDA toolkit location

**Secrets location:**
- None stored in codebase
- Runtime config via `~/.sirius/sirius.cfg`

## Webhooks & Callbacks

**Incoming:**
- None (in-process extension)

**Outgoing:**
- DuckDB table function results - Arrow IPC batches
- Exchange sink - bRPC or nixl to Doris FE/other BE
- Arrow Flight responses - Arrow IPC to clients

## GPU-Direct Exchange (nixl)

**Service:**
- nixl (GPU-direct collective library) - GPU-GPU data transfer
  - Submodule: `doris/thirdparty/nixl/` (UCX plugin)
  - Build: Meson build system (compiled into conda prefix libdir)
  - Rust FFI: `doris/crates/doris-rpc/` (nvlink/P2P exchange)

**Data Format:**
- RAPIDS packed tables (cudf::table serialized format)
- Registration: `register_packed_table()` for GPU execution
- Transfer: Point-to-point GPU memory via UCX/NVLink

## Substrait Plan Compilation

**Service:**
- Substrait extension (DuckDB) - Plan IR to DuckDB logical plans
  - Library: `substrait/` (git submodule, duckdb-substrait-extension fork)
  - Deserialization: `SubstraitToDuckDB` class
  - Path: `src/expression_executor/specializations/gpu_execute_function.cpp` (comment references)

**Fallback Path:**
- `from_substrait` table function - CPU execution of Substrait
  - Trigger: GPU plan failure, unsupported types, memory limits
  - Implementation: Built-in DuckDB extension

## Data Exchange Protocols

**Between Sirius GPU BE and Doris FE:**
- gRPC - Fragment execution, metadata
  - Port: 19060 (default BE thrift), 18060 (default BE brpc)
  - Protocol: VERSION_3 TPipelineFragmentParamsList

- Arrow Flight - Result streaming
  - Port: 18071 (default GPU BE)
  - Format: Apache Arrow IPC

- bRPC - Result block sink
  - Port: 28060+ (for multi-BE setup)
  - Format: Doris PBlock (optional StreamVByte/LZ4 compression)

**Between Sirius GPU BEs (hash partitioned exchange):**
- nixl (GPU-direct) - GPU table transfer
  - Protocol: UCX-based (NVLink, InfiniBand, Ethernet fallback)
  - Format: RAPIDS packed cudf::table

- bRPC (fallback) - CPU fallback exchange
  - Protocol: CRC32/CRC32C hash partitioning in `hash_partitioner.rs`
  - Format: Doris PBlock

---

*Integration audit: 2026-04-06*
