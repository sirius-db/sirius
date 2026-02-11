# File Index

Quick reference guide to important files in the Sirius codebase, organized by category.

---

## Entry Points and Extension

### DuckDB Integration
| File | Description |
|------|-------------|
| `src/sirius_extension.cpp` | Main extension registration and table functions |
| `src/sirius_interface.hpp` | Query execution interface (New Mode) |
| `src/sirius_interface.cpp` | Query execution implementation |
| `src/sirius_context.hpp` | Per-connection context |
| `src/sirius_context.cpp` | Context management |
| `src/sirius_config.hpp` | Global configuration |
| `src/sirius_config.cpp` | Configuration implementation |

### Table Functions
- **Legacy**: `src/sirius_extension.cpp:240-339` - `gpu_processing()`
- **New**: `src/sirius_extension.cpp:353-452` - `gpu_execution()`

---

## Physical Planning

### Plan Generation
| File | Description |
|------|-------------|
| `src/planner/sirius_physical_plan_generator.hpp` | Main planner interface |
| `src/planner/sirius_physical_plan_generator.cpp` | Physical plan generation |
| `src/planner/sirius_plan_aggregate.cpp` | Aggregate planning |
| `src/planner/sirius_plan_filter.cpp` | Filter planning |
| `src/planner/sirius_plan_get.cpp` | Scan planning |
| `src/planner/sirius_plan_join.cpp` | Join planning |
| `src/planner/sirius_plan_projection.cpp` | Projection planning |
| `src/planner/sirius_plan_order.cpp` | Sort planning |
| `src/planner/sirius_plan_limit.cpp` | Limit planning |

### Legacy Planner
| File | Description |
|------|-------------|
| `src/gpu_physical_plan_generator.hpp` | Legacy planner interface |
| `src/gpu_physical_plan_generator.cpp` | Legacy physical planning |

---

## Operators

### New Mode Operators (sirius_physical_operator)

#### Base Classes
| File | Description |
|------|-------------|
| `src/include/op/sirius_physical_operator.hpp` | Base operator class |
| `src/include/op/sirius_physical_operator_type.hpp` | Operator type enum |
| `src/op/sirius_physical_operator.cpp` | Base implementation |
| `src/op/sirius_physical_operator_type.cpp` | Type utilities |

#### Scans
| File | Description |
|------|-------------|
| `src/include/op/sirius_physical_table_scan.hpp` | Table scan operator |
| `src/op/sirius_physical_table_scan.cpp` | Table scan implementation |
| `src/include/op/sirius_physical_duckdb_scan.hpp` | DuckDB scan operator |
| `src/op/sirius_physical_duckdb_scan.cpp` | DuckDB scan implementation |
| `src/include/op/sirius_physical_dummy_scan.hpp` | Dummy scan (testing) |
| `src/op/sirius_physical_dummy_scan.cpp` | Dummy scan implementation |

#### Filters and Projections
| File | Description |
|------|-------------|
| `src/include/op/sirius_physical_filter.hpp` | Filter operator |
| `src/op/sirius_physical_filter.cpp` | Filter implementation |
| `src/include/op/sirius_physical_projection.hpp` | Projection operator |
| `src/op/sirius_physical_projection.cpp` | Projection implementation |
| `src/include/op/sirius_physical_limit.hpp` | Limit operator |
| `src/op/sirius_physical_limit.cpp` | Limit implementation |

#### Aggregates
| File | Description |
|------|-------------|
| `src/include/op/sirius_physical_ungrouped_aggregate.hpp` | Ungrouped aggregate |
| `src/op/sirius_physical_ungrouped_aggregate.cpp` | Ungrouped aggregate impl |
| `src/include/op/sirius_physical_hash_group_by.hpp` | Hash group by |
| `src/op/sirius_physical_hash_group_by.cpp` | Hash group by impl |

#### Joins
| File | Description |
|------|-------------|
| `src/include/op/sirius_physical_hash_join.hpp` | Hash join operator |
| `src/op/sirius_physical_hash_join.cpp` | Hash join implementation |
| `src/include/op/sirius_physical_nested_loop_join.hpp` | Nested loop join |
| `src/op/sirius_physical_nested_loop_join.cpp` | Nested loop join impl |

#### Sorting
| File | Description |
|------|-------------|
| `src/include/op/sirius_physical_order_by.hpp` | Order by operator |
| `src/op/sirius_physical_order_by.cpp` | Order by implementation |
| `src/include/op/sirius_physical_top_n.hpp` | Top N operator |
| `src/op/sirius_physical_top_n.cpp` | Top N implementation |
| `src/include/op/sirius_physical_merge_sort.hpp` | Merge sort operator |
| `src/op/sirius_physical_merge_sort.cpp` | Merge sort implementation |

#### Partitioning
| File | Description |
|------|-------------|
| `src/include/op/sirius_physical_partition.hpp` | Partition operator |
| `src/op/sirius_physical_partition.cpp` | Partition implementation |
| `src/include/op/sirius_physical_sort_partition.hpp` | Sort partition operator |
| `src/op/sirius_physical_sort_partition.cpp` | Sort partition implementation |

#### Output
| File | Description |
|------|-------------|
| `src/include/op/sirius_physical_result_collector.hpp` | Result collector |
| `src/op/sirius_physical_result_collector.cpp` | Result collection impl |

### Legacy Mode Operators (GPUPhysicalOperator)

#### Base Class
| File | Description |
|------|-------------|
| `src/include/gpu_physical_operator.hpp` | Legacy operator base |
| `src/gpu_physical_operator.cpp` | Legacy operator impl |

#### Legacy Operators
| File | Description |
|------|-------------|
| `src/operator/gpu_physical_filter.cpp` | Legacy filter |
| `src/operator/gpu_physical_projection.cpp` | Legacy projection |
| `src/operator/gpu_physical_hash_aggregate.cpp` | Legacy aggregate |
| `src/operator/gpu_physical_hash_join.cpp` | Legacy hash join |
| `src/operator/gpu_physical_table_scan.cpp` | Legacy table scan |
| `src/operator/gpu_physical_result_collector.cpp` | Legacy result collector |

---

## Pipeline Infrastructure

### New Mode Pipelines
| File | Description |
|------|-------------|
| `src/include/pipeline/sirius_pipeline.hpp` | Pipeline structure |
| `src/include/pipeline/sirius_pipeline.cpp` | Pipeline implementation |
| `src/include/pipeline/sirius_meta_pipeline.hpp` | Meta pipeline (DAG) |
| `src/include/pipeline/sirius_meta_pipeline.cpp` | Meta pipeline impl |
| `src/include/pipeline/sirius_pipeline_itask.hpp` | Task interface |
| `src/include/pipeline/sirius_pipeline_itask.cpp` | Task implementation |
| `src/include/pipeline/sirius_pipeline_build_state.hpp` | Pipeline builder |
| `src/include/pipeline/sirius_pipeline_build_state.cpp` | Builder impl |

### Legacy Pipelines
| File | Description |
|------|-------------|
| `src/include/gpu_pipeline.hpp` | Legacy pipeline structure |
| `src/gpu_pipeline.cpp` | Legacy pipeline impl |
| `src/include/gpu_meta_pipeline.hpp` | Legacy meta pipeline |
| `src/gpu_meta_pipeline.cpp` | Legacy meta pipeline impl |

---

## Execution Engines

### New Mode Engine
| File | Description |
|------|-------------|
| `src/include/sirius_engine.hpp` | Engine interface |
| `src/sirius_engine.cpp` | Engine implementation |

### Legacy Executor
| File | Description |
|------|-------------|
| `src/include/gpu_executor.hpp` | Legacy executor interface |
| `src/gpu_executor.cpp` | Legacy executor implementation |

---

## Task Execution

### Task Executors
| File | Description |
|------|-------------|
| `src/include/parallel/task_executor.hpp` | Task executor base |
| `src/include/parallel/itask.hpp` | Task interface |
| `src/parallel/pipeline_executor.hpp` | GPU pipeline executor |
| `src/parallel/pipeline_executor.cpp` | Pipeline executor impl |
| `src/parallel/task_creator.hpp` | Task creator |
| `src/parallel/task_creator.cpp` | Task creator impl |
| `src/parallel/downgrade_executor.hpp` | Memory downgrade executor |
| `src/parallel/downgrade_executor.cpp` | Downgrade executor impl |
| `src/parallel/duckdb_scan_executor.hpp` | DuckDB scan executor |
| `src/parallel/duckdb_scan_executor.cpp` | DuckDB scan impl |

### Legacy Executor
| File | Description |
|------|-------------|
| `src/gpu_pipeline_executor.cpp` | Legacy pipeline executor |

---

## Memory Management

### New Mode Memory
| File | Description |
|------|-------------|
| `src/include/memory/sirius_memory_reservation_manager.hpp` | Memory reservations |
| `src/include/memory/sirius_memory_reservation_manager.cpp` | Reservation impl |
| `src/include/memory/sirius_memory_manager.hpp` | Memory manager |
| `src/include/memory/sirius_memory_manager.cpp` | Memory manager impl |

### Legacy Memory
| File | Description |
|------|-------------|
| `src/include/gpu_buffer_manager.hpp` | Legacy buffer manager |
| `src/gpu_buffer_manager.cpp` | Legacy buffer manager impl |

### Cucascade (New Mode)
| File | Description |
|------|-------------|
| `cucascade/include/cucascade/data/data_batch.hpp` | Data batch structure |
| `cucascade/include/cucascade/data/data_repository.hpp` | Data repository |
| `cucascade/include/cucascade/data/data_repository_manager.hpp` | Repository manager |
| `cucascade/include/cucascade/memory/memory_reservation_manager.hpp` | Memory reservations |

---

## Data Structures

### New Mode Data
| File | Description |
|------|-------------|
| `src/include/data/sirius_column.hpp` | Column structure |
| `src/include/data/sirius_table.hpp` | Table structure |
| `src/data/sirius_converter_registry.hpp` | Type converters |
| `src/data/sirius_converter_registry.cpp` | Converter impl |

### Legacy Data
| File | Description |
|------|-------------|
| `src/include/gpu_columns.hpp` | GPUColumn, GPUIntermediateRelation |
| `src/gpu_columns.cpp` | Legacy data structures impl |

---

## Expression Evaluation

| File | Description |
|------|-------------|
| `src/include/expression/gpu_expression_executor.hpp` | Expression executor |
| `src/expression/gpu_expression_executor.cpp` | Expression execution |
| `src/include/expression/gpu_expression.hpp` | Expression types |

---

## Configuration and Context

| File | Description |
|------|-------------|
| `src/sirius_config.hpp` | Global configuration |
| `src/sirius_config.cpp` | Config implementation |
| `src/sirius_context.hpp` | Per-connection context |
| `src/sirius_context.cpp` | Context implementation |
| `src/config.hpp` | Legacy config |
| `src/config.cpp` | Legacy config impl |

---

## Utilities

### Logging
| File | Description |
|------|-------------|
| `src/include/log/logging.hpp` | Logging macros |
| `src/log/logging.cpp` | Logging implementation |

### Types
| File | Description |
|------|-------------|
| `src/include/helper/types.hpp` | Type utilities |
| `src/include/helper/cuda_helper.hpp` | CUDA helper functions |

---

## Tests

### C++ Unit Tests
| Directory | Description |
|-----------|-------------|
| `test/cpp/operator/` | Operator unit tests |
| `test/cpp/pipeline/` | Pipeline unit tests |
| `test/cpp/memory/` | Memory management tests |
| `test/cpp/planner/` | Planner tests |
| `test/cpp/expression/` | Expression evaluation tests |

### SQL Integration Tests
| Directory | Description |
|-----------|-------------|
| `test/sql/operators/` | Operator SQL tests |
| `test/sql/tpch/` | TPC-H benchmark queries |
| `test/sql/correctness/` | Correctness tests |

---

## Build System

| File | Description |
|------|-------------|
| `CMakeLists.txt` | Root CMake configuration |
| `src/CMakeLists.txt` | Source CMake configuration |
| `test/CMakeLists.txt` | Test CMake configuration |
| `pixi.toml` | Pixi environment configuration |
| `.github/workflows/` | CI/CD workflows |

---

## Quick Navigation by Task

### Adding a New Operator
1. Create header: `src/include/op/sirius_physical_<name>.hpp`
2. Create impl: `src/op/sirius_physical_<name>.cpp`
3. Add to type enum: `src/include/op/sirius_physical_operator_type.hpp`
4. Add planner: `src/planner/sirius_plan_<name>.cpp`
5. Add tests: `test/cpp/operator/test_<name>.cpp`

### Debugging Memory Issues
1. Check: `src/include/memory/sirius_memory_reservation_manager.hpp`
2. Check: `cucascade/include/cucascade/data/data_repository.hpp`
3. Enable logging in: `src/include/log/logging.hpp`

### Understanding Query Flow
1. Start: `src/sirius_extension.cpp` (entry points)
2. Planning: `src/planner/sirius_physical_plan_generator.cpp`
3. Execution: `src/sirius_engine.cpp`
4. Operators: `src/op/sirius_physical_*.cpp`
5. Results: `src/op/sirius_physical_result_collector.cpp`

---

## File Organization Principles

### Naming Conventions
- **Headers**: `.hpp` (C++ header)
- **Implementation**: `.cpp` (C++ source)
- **New Mode**: `sirius_*` prefix
- **Legacy Mode**: `gpu_*` or `GPU*` prefix
- **Operators**: `*_physical_*` in name

### Directory Structure
- `src/include/` - Public headers
- `src/` - Implementation files
- `src/op/` - New mode operators
- `src/operator/` - Legacy operators
- `src/planner/` - Physical planning
- `src/parallel/` - Task execution
- `src/memory/` - Memory management
- `src/data/` - Data structures
- `src/expression/` - Expression evaluation
- `cucascade/` - Submodule for data management

---

## See Also

- [API Reference](api-reference.md) - Class and method documentation
- [Glossary](glossary.md) - Term definitions
- [Code Organization](../07-development/code-organization.md) - Directory structure guide
