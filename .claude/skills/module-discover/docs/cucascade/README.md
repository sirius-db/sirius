# cuCascade — Module Reference

**Version**: git submodule (see `cucascade/` in repo root)
**Location**: `./cucascade/`
**Namespace**: `cucascade` (data types), `cucascade::memory` (memory management)

## Module Map

| Module | Status | Description | Key APIs Used |
|--------|--------|-------------|---------------|
| data | USED | Data representation, batching, and repository management | `data_batch`, `gpu_table_representation`, `shared_data_repository`, `data_repository_manager` |
| memory | USED | GPU/Host/Disk memory spaces, reservations, and resource adaptors | `memory_reservation_manager`, `reservation_aware_resource_adaptor`, `memory_space`, `fixed_size_host_memory_resource` |
| utils | UNUSED | Internal atomic helpers and metaprogramming utilities | — |

## Our Usage Summary

We use **2 of 3** modules. Primary integration points:

- **Data representation**: All pipeline data flows through `cucascade::data_batch` wrapping either `gpu_table_representation` (GPU cudf::table) or `host_data_representation` (CPU). Repositories (`shared_data_repository`) store batches between pipeline stages.
- **Memory management**: `memory_reservation_manager` orchestrates tiered memory (GPU→Host→Disk). Each GPU pipeline task gets a `reservation` that wraps a `reservation_aware_resource_adaptor` for OOM-safe allocation. Host memory uses `fixed_size_host_memory_resource` with block-based allocation for pinned memory.
- **Configuration**: `reservation_manager_configurator` (builder pattern) creates memory space configs. `topology_discovery` detects GPU/NUMA topology at startup.
- **Stream management**: `exclusive_stream_pool` provides CUDA streams for parallel GPU execution.

## Files That Reference This Library

| Source File | Modules Used | Key APIs |
|-------------|-------------|----------|
| `src/sirius_engine.cpp` | data | `data_repository_manager` |
| `src/sirius_context.cpp` | memory | `fixed_size_host_memory_resource`, `small_pinned_host_memory_resource` |
| `src/sirius_config.cpp` | memory | `config`, `reservation_manager_configurator` |
| `src/pipeline/gpu_pipeline_task.cpp` | data, memory | `data_batch`, `gpu_table_representation`, `cpu_data_representation`, `memory_space`, `reservation_aware_resource_adaptor` |
| `src/pipeline/task_scheduler.cpp` | memory | `memory_reservation`, `memory_space`, `common` |
| `src/pipeline/gpu_pipeline_executor.cpp` | memory | `stream_pool`, `memory_reservation`, `memory_space` |
| `src/memory/sirius_memory_reservation_manager.cpp` | memory | `memory_reservation_manager`, `Tier` |
| `src/op/scan/duckdb_scan_task.cpp` | data, memory | `cpu_data_representation`, `data_batch`, `memory_space`, `memory_reservation` |
| `src/op/scan/parquet_scan_task.cpp` | data, memory | `data_batch`, `cpu/gpu_data_representation`, `fixed_size_host_memory_resource`, `memory_reservation_manager` |
| `src/op/scan/duckdb_scan_executor.cpp` | data, memory | `cpu/gpu_data_representation`, `data_batch`, `common` |
| `src/expression_executor/gpu_expression_executor.cpp` | data | `data_batch`, `gpu_table_representation` |
| `src/downgrade/downgrade_task.cpp` | data, memory | `cpu_data_representation`, `common` |
| `src/data/host_parquet_representation_converters.cpp` | data, memory | `gpu_data_representation`, `fixed_size_host_memory_resource`, `memory_space` |
| `src/op/sirius_physical_result_collector.cpp` | data, memory | `cpu_data_representation`, `data_batch`, `memory_reservation_manager` |
| `src/op/sirius_physical_ungrouped_aggregate.cpp` | data | `data_batch`, `gpu_table_representation` |

## CMake Integration

```cmake
# Built as subdirectory
add_subdirectory(cucascade "${CMAKE_BINARY_DIR}/cucascade" EXCLUDE_FROM_ALL)

# Linked as static library
target_link_libraries(sirius_extension ... cuCascade::cucascade)
```
