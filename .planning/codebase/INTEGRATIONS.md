# External Integrations

**Analysis Date:** 2026-04-21

## APIs & External Services

**DuckDB Extension System:**
- DuckDB main extension API - Sirius loads as a DuckDB extension
  - SDK/Client: `duckdb` submodule (1.5.2 pinned)
  - Entry point: `CALL gpu_execution('SELECT ...')` in SQL
  - Extension registration: `src/sirius_extension.cpp`

**NVIDIA CUDA Runtime:**
- CUDA Profiler API - Optional GPU profiling integration
  - Functions: `cudaProfilerStart()`, `cudaProfilerStop()` (linked via libcudart)
  - Used in: `src/sirius_extension.cpp`
  - Usage: CUDA profiling enablement for performance analysis

**NVIDIA RAPIDS Libraries:**
- RAPIDS cuDF 26.04 - GPU DataFrame library
  - SDK/Client: `libcudf::cudf` (conda package)
  - Usage: All GPU-accelerated operations (joins, aggregations, sorting, filters)
  - Headers: `cudf/table/table.hpp`, `cudf/join/*.hpp`, `cudf/aggregation.hpp`, `cudf/ast/expressions.hpp`
  - Version handling: Version-specific APIs (e.g., `cudf/join/distinct_hash_join.hpp` for CUDF 25.04+)

- RAPIDS RMM (Memory Manager)
  - SDK/Client: `rmm::rmm` (conda package)
  - Usage: GPU memory allocation, pooling, stream management
  - Headers: `rmm/cuda_stream_view.hpp`, `cudf/utilities/pinned_memory.hpp`
  - Integration: Core memory management for all GPU operations

**cuCascade - GPU Memory Reservation Library:**
- NVIDIA cuCascade (submodule: `cucascade/`)
- Purpose: Tiered memory management across GPU/host/disk
- Headers: `cucascade/memory/fixed_size_host_memory_resource.hpp`, `cucascade/memory/small_pinned_host_memory_resource.hpp`
- Build integration: Added as subdirectory in `CMakeLists.txt` (lines 85-97)
- Static linking: `CUCASCADE_BUILD_STATIC_LIBS=ON`
- Linked target: `cuCascade::cucascade`
- NUMA support: Links against `libnuma` for system memory awareness

## Data Storage

**Databases:**
- DuckDB (in-process)
  - Type: Embedded SQL database engine
  - Connection: In-process via DuckDB connection manager
  - Client: DuckDB C++ API
  - Usage: Core query execution engine, fallback for unsupported operations
  - Submodule path: `duckdb/`
  - Version: 1.5.2 (pinned in `pixi.toml`)

**Data Formats Supported:**
- Parquet - GPU-accelerated scanning via cuDF hybrid scan
  - Reader: `cudf/io/experimental/hybrid_scan.hpp`
  - Metadata scanning: `sirius_parquet_metadata_scan_operator` (`src/op/scan/sirius_parquet_metadata_scan_operator.cpp`)
  - Task-based execution: `parquet_scan_task.cpp`

- Iceberg Format - Table metadata and data scanning
  - Scanning operators: `sirius_physical_iceberg_scan.cpp` (`src/op/sirius_physical_iceberg_scan.cpp`)
  - Metadata reader: `iceberg_metadata_reader.cpp`
  - Delete filters: Equality and positional delete handling
  - AVRO parsing: `iceberg_avro_reader.cpp`

- Arrow Format - Intermediate representation
  - Integration: Via cuDF table format and DuckDB serialization

**File Storage:**
- Local filesystem only
  - Parquet file scanning (GPU-accelerated via cuDF)
  - Spill-to-disk for memory overflow (via cuCascade tiered memory)
  - Configuration files (YAML) from local paths or home directory

**Caching:**
- CPU Cache - Host-side data caching
  - Implementation: `src/cpu_cache.cpp`, `src/include/cpu_cache.hpp`
  - Purpose: Cache hot data on host before GPU transfer
  - Test coverage: `test/cpp/memory_management/test_cpu_cache.cpp`

- GPU Memory Hierarchy:
  - GPU VRAM - Primary execution memory
  - Host pinned memory - Zero-copy GPU access
  - Host pageable memory - System RAM fallback
  - Disk spill - cuCascade-managed overflow

## Authentication & Identity

**Auth Provider:**
- Custom / None - Sirius is an in-process DuckDB extension
- DuckDB manages client authentication (if enabled)
- No external identity service integration

**Database Access Control:**
- DuckDB native access control (if configured)
- Sirius inherits DuckDB catalog permissions

## Monitoring & Observability

**Error Tracking:**
- None (no external service)
- Internal exception handling via `sirius::exception`
- Fallback to DuckDB error reporting

**Logging:**
- Framework: spdlog 1.8
- Output: File-based logging to directory specified by:
  - Environment variable: `SIRIUS_LOG_DIR` (default: `${CMAKE_BINARY_DIR}/log`)
  - Config setting: `LOG_DIR` in `src/config.cpp`
- Log levels: trace, debug, info, warn, error
  - Configurable via: `SIRIUS_LOG_LEVEL` environment variable
  - Default: "info" (set in `src/config.cpp`)
- Flush interval: 3 seconds (configurable via `LOG_FLUSH_SECONDS`)
- Sinks: `spdlog/sinks/basic_file_sink.h` for file output
- Setup: `sirius_context.cpp` initializes logger with spdlog

**Performance Instrumentation:**
- NVIDIA CUDA Profiler API - Optional runtime profiling
  - Hooks: `cudaProfilerStart()`, `cudaProfilerStop()`
  - Used for: GPU kernel profiling and performance analysis

**Diagnostics:**
- Pipeline execution logging - Per-operator row counts and timing
  - Tools: `tools/parse_pipeline_log.py` (included in repo)
- Segfault backtrace handler - Debug crash diagnostics
  - Implementation: `src/util/segfault_backtrace_handler.cpp`

## CI/CD & Deployment

**Hosting:**
- Self-hosted or on-premise deployment model
- DuckDB extension distribution (static + loadable variants)
- Build outputs:
  - Static extension: `build/release/extension/sirius/sirius.duckdb_extension`
  - Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`

**CI Pipeline:**
- Build system: CMake + Ninja with sccache caching
- Presets defined in `cmake/CMakePresets.json`
- Pre-commit hooks: `.pre-commit-config.yaml` (v6.0.0)
  - Hooks: clang-format, clang-tidy, black, codespell, cmake-format
  - Runs on: C++, Python, CMake, spell check
- DuckDB extension-ci-tools integration
  - Submodule: `extension-ci-tools` (sirius branch)
  - Makefile wrapper: Thin CMake invocation via `Makefile`

**Build Configuration:**
- Configuration file: `cmake/CMakePresets.json`
- Extension config: `extension_config.cmake` (specifies extension loading)
- Test discovery: SQLLogicTests in `test/sql/` and C++ unit tests in `test/cpp/`

## Environment Configuration

**Required Environment Variables:**

Optional runtime configuration:
- `SIRIUS_LOG_DIR` - Log output directory (default: `${CMAKE_BINARY_DIR}/log`)
- `SIRIUS_LOG_LEVEL` - Log verbosity level (default: "info")
- `SIRIUS_CONFIG_FILE` - Explicit config file path
- `CUDAARCHS` - GPU architectures (set by pixi feature)
- `VCPKG_CUDA_VERSION` - CUDA version for vcpkg builds

**Config File Format:**
- YAML configuration file: `sirius.yaml` or `sirius.cfg` (legacy)
- Search locations:
  1. Path specified in `SIRIUS_CONFIG_FILE` env var
  2. `./sirius.yaml` in current working directory
  3. `~/.sirius/sirius.yaml` in home directory
- Parser: `yaml-cpp` library
- Config loading: `sirius_context.cpp`

**Secrets Location:**
- No external secrets management
- All configuration via environment variables or YAML files
- Credentials (if any) managed by DuckDB connection settings

## Webhooks & Callbacks

**Incoming:**
- None - Sirius is an in-process SQL execution engine

**Outgoing:**
- None - No external callbacks or webhook integrations

## Language Bindings

**Python API:**
- DuckDB Python binding with Sirius extension loading
- Build: `pixi run -e duckdb-python build-duckdb-python`
- Location: `duckdb-python/` submodule
- Build tools: pybind11, scikit-build-core
- Usage example:
  ```python
  import duckdb
  con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
  con.execute("LOAD 'build/release/extension/sirius/sirius.duckdb_extension'")
  result = con.execute("CALL gpu_execution('SELECT ...')").fetchall()
  ```

## Submodule Dependencies

**Core Submodules:**
- `duckdb/` - Main SQL engine (v1.5.2 main branch)
- `cucascade/` - GPU memory tiering library (NVIDIA, main branch)
- `duckdb-python/` - Python bindings (DuckDB main)
- `extension-ci-tools/` - Build infrastructure (sirius branch)

**Optional/Experimental:**
- `substrait/` - DuckDB Substrait extension (sirius branch, experimental)

**Vendor:**
- `vcpkg/` - Package manager for static builds

**Developer Tools:**
- `.claude/claude-tools/` - Claude Code integration scripts

## Build Dependencies Resolution

**CMake Find-Module Integration:**
- `find_package(cudf REQUIRED CONFIG)`
- `find_package(spdlog REQUIRED CONFIG)`
- `find_package(yaml-cpp REQUIRED CONFIG)`
- `find_package(absl REQUIRED CONFIG)`
- `find_package(PkgConfig REQUIRED)` - For NUMA library resolution

**Link Targets:**
- `cudf::cudf` - GPU operations
- `rmm::rmm` - Memory management
- `spdlog::spdlog` - Logging
- `cuCascade::cucascade` - Memory tiering
- `yaml-cpp::yaml-cpp` - Configuration
- `PkgConfig::NUMA` - NUMA operations
- `absl::any_invocable` - Utility (static extension only)
- `duckdb_static` - Core SQL engine

---

*Integration audit: 2026-04-21*
