# External Integrations

**Analysis Date:** 2025-04-02

## APIs & External Services

**Database Engines:**
- DuckDB 1.4.4 - Primary SQL query engine host
  - Sirius loads as DuckDB extension (`sirius.duckdb_extension`)
  - Uses DuckDB's parser, optimizer, planner, and fallback CPU execution
  - Extension entry point: `src/sirius_extension.cpp` (LoadInternal function)
  - API: Table functions `gpu_execution()` and legacy `gpu_processing()`

**GPU Compute Framework:**
- NVIDIA cuDF (libcudf 26.02.*) - GPU DataFrame operations
  - SDK/Client: `libcudf` C++ library, included via `find_package(cudf REQUIRED CONFIG)`
  - Purpose: Accelerated joins (hash, nested loop), aggregations, sorting, filtering
  - Data model: cudf::table, cudf::column, cudf::column_view
  - Include paths: `<cudf/...>` throughout `src/` for operations
  - No authentication required (local library)

**GPU Memory Management:**
- NVIDIA RMM (librmm) - RAPIDS Memory Manager
  - SDK/Client: `librmm` C++ library via `find_package(rmm REQUIRED CONFIG)` (implicit via cudf)
  - Purpose: GPU memory allocation, deallocation, and resource management
  - Used in: `src/data/host_parquet_representation_converters.cpp`, data batch allocation

**Data Format Support:**
- Apache Parquet (via cuDF) - Primary data input format
  - Reader: cuDF's `cudf::io::parquet::read()` and `cudf::io::experimental::hybrid_scan()`
  - Files: `src/op/scan/parquet_scan_task.cpp` implements Parquet scanning on GPU
  - Feature: Handles multi-file parquet datasets, row group selection, column projection
  - Status: Full GPU integration via cuDF's parquet plugin

- Apache Arrow (implicitly via cuDF) - In-memory data representation
  - Format used for GPU <-> CPU data interchange
  - DuckDB's table chunks converted to Arrow format for GPU processing

- Iceberg Table Format (partial support)
  - Metadata reader: `src/op/scan/iceberg_metadata_reader.cpp`
  - Avro schema reader: `src/op/scan/iceberg_avro_reader.cpp`
  - Delete handling: Equality and positional delete filters in `src/op/scan/`
  - Purpose: Support Iceberg catalog queries on GPU

- DuckDB Native Format - Fallback for unsupported data types
  - Scanned via DuckDB's internal scan operators
  - Converted to GPU format (cudf::table) for processing
  - File: `src/op/scan/duckdb_scan_task.cpp`

## Data Storage

**Databases:**
- DuckDB in-process database (default) - Query source
  - Connection: In-process (no remote connection)
  - File: `build/release/extension/sirius/*.duckdb` or in-memory
  - Client: Native DuckDB C++ API via ClientContext and Connection

**File Storage:**
- Local filesystem only for data input
  - Parquet files: Read via DuckDB's virtual filesystem or direct cuDF reader
  - No cloud storage integration (S3, GCS, etc.) detected
  - Iceberg metadata/data: Assumes local filesystem or DuckDB's catalog

**Caching:**
- cuCascade (GPU memory cache) - Tiered memory management
  - Caching regions: GPU memory caching raw input data
  - Host memory caching: Pinned host memory for fast CPU-GPU transfers
  - Disk spillover: Out-of-core processing when data exceeds GPU+host capacity
  - Configuration: `src/sirius_config.cpp` configures memory space topology
  - Cache levels: NONE, GPU, CPU (configurable per scan operator)

**Temporary Storage:**
- GPU memory regions (configurable via `gpu_buffer_init`)
  - GPU caching region: Stores input data
  - GPU processing region: Holds intermediate results (hash tables, joins)
  - Managed by cuCascade's memory reservation manager

## Authentication & Identity

**Auth Provider:**
- None - Sirius is an in-process GPU extension
- DuckDB handles authentication (if database file is protected)
- No user/role management in Sirius itself

## Monitoring & Observability

**Error Tracking:**
- None - No external error tracking service
- Errors logged to `sirius.log` via spdlog
- DuckDB's error propagation mechanism used

**Logs:**
- Structured logging via spdlog (1.8.*)
  - Log file: `${SIRIUS_LOG_DIR}/sirius.log` (default: build directory)
  - Levels: trace, debug, info, warn, error, critical
  - Daily file rotation enabled
  - Format: `[YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] [file:line] message`
  - Configuration: `SIRIUS_LOG_LEVEL` env var (default: info)
  - Flush interval: `SIRIUS_LOG_FLUSH_SECONDS` (configurable, default in config.cpp)
  - Pattern: Set in `src/include/log/logging.hpp` InitGlobalLogger function

**Performance Profiling:**
- CUDA profiler API integration (via cudaProfilerStart/Stop in `src/sirius_extension.cpp`)
  - Allows nsys profiling with synchronized regions
  - No telemetry sent externally

**GPU Monitoring:**
- CUDA NVML API - GPU health/status (via `cuda-nvml-dev` package)
  - Used for topology discovery and device management
  - No external reporting

## CI/CD & Deployment

**Hosting:**
- Self-hosted: Compiled as DuckDB extension (static or loadable)
- Container: Can be deployed in Docker (user-provided, not in repo)

**CI Pipeline:**
- None detected in codebase - uses GitHub Actions (external CI/CD)
- Pre-commit hooks managed locally via `.pre-commit-config.yaml`
- Build tested via `pixi run make` locally

**Deployment:**
- Static extension: `build/release/extension/sirius/sirius.duckdb_extension`
  - Linked directly into DuckDB at build time
- Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
  - Loaded at runtime via `LOAD 'path/to/sirius_loadable.duckdb_extension'`

## Environment Configuration

**Required env vars:**
- `SIRIUS_LOG_LEVEL` - Logging level (trace, debug, info, warn, error)
- `SIRIUS_LOG_DIR` - Log file directory (default: CMAKE_BINARY_DIR/log)
- `SIRIUS_LOG_FLUSH_SECONDS` - Log flush interval in seconds

**Optional:**
- `CUDAARCHS` - GPU architectures for compilation (set by pixi features: cuda12/cuda13)
- `LIBCUDF_ENV_PREFIX` - Path to libcudf conda environment (if using conda)

**Secrets location:**
- No credentials stored in codebase
- DuckDB auth handled by database file permissions
- GPU access via driver installed on host

## Webhooks & Callbacks

**Incoming:**
- None - Sirius is synchronous batch SQL engine

**Outgoing:**
- None - Results returned directly to DuckDB query client

## External Library Dependencies

**Direct Linkage (CMakeLists.txt):**
```
target_link_libraries(sirius_extension 
  cudf::cudf                          # GPU DataFrame operations
  rmm::rmm                            # GPU memory management
  spdlog::spdlog                      # Structured logging
  cuCascade::cucascade                # Tiered memory management
  PkgConfig::NUMA                     # NUMA-aware memory allocation
  ${LIBCONFIG++_LIBRARIES}            # Config file parsing
  absl::any_invocable                 # Type-erased function storage
)
```

**Optional/Conditional:**
- DuckDB bundled submodule: `duckdb/` (git submodule)
  - Provides: Parser, planner, optimizer, execution framework
  - Version: Tracked as submodule (currently v1.4.4)
  - Built as `duckdb_static` library and linked to Sirius

**Test Framework:**
- Catch2 (bundled, third-party in DuckDB)
  - Location: `duckdb/third_party/catch`
  - No external dependency, compiled into test binary

## Data Flow with External Systems

```
DuckDB Parser → DuckDB Optimizer → DuckDB Planner
                                        ↓
                        Sirius Physical Plan Generator
                              (src/planner/)
                                  ↓
                    [GPU Execution Path]
                    Task Creator → Scan Tasks (parquet/iceberg/duckdb)
                                        ↓
                        Data Repository (cuCascade-managed)
                        GPU/CPU/Disk memory tiers
                                        ↓
                    GPU Pipeline Executor (cuDF operations)
                        [hash joins, aggregates, filters, etc.]
                                        ↓
                        Downgrade Executor (when GPU OOM)
                        [fallback to CPU via DuckDB]
                                        ↓
                        Result Collector → DuckDB QueryResult
```

## Integration Points with DuckDB

1. **Extension Registration** (`src/sirius_extension.cpp`)
   - LoadInternal: Registers `gpu_execution()` and `gpu_processing()` table functions
   - Sets DuckDB configuration options for Sirius parameters

2. **Query Interception**
   - Physical plan passed to Sirius at execution time
   - Sirius intercepts supported operators, delegates to GPU
   - Unsupported operators fall back to DuckDB CPU execution (`src/fallback.cpp`)

3. **Memory Integration**
   - GPU buffer initialization: `CALL gpu_buffer_init('1 GB', '2 GB')`
   - Reserves GPU memory regions before query execution

4. **Expression Evaluation**
   - `src/expression_executor/gpu_expression_translator.cpp` converts DuckDB expressions to cuDF AST
   - Filters, projections, and aggregations evaluated on GPU

5. **Scan Integration**
   - DuckDB scan operators converted to Sirius scan tasks
   - Parquet/Iceberg readers use cuDF's native readers where possible
   - Falls back to DuckDB scan for unsupported formats

---

*Integration audit: 2025-04-02*
