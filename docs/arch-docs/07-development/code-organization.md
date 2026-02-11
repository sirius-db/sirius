# Code Organization

This document provides a comprehensive guide to the Sirius codebase structure, helping developers navigate and understand where different components live.

## Table of Contents

1. [Overview](#overview)
2. [Top-Level Directory Structure](#top-level-directory-structure)
3. [Source Code Organization](#source-code-organization)
4. [Key Files and Directories](#key-files-and-directories)
5. [Naming Conventions](#naming-conventions)
6. [Include Paths](#include-paths)
7. [Build System](#build-system)
8. [Next Steps](#next-steps)

---

## Overview

The Sirius codebase is organized into several major directories:

```
sirius/
├── src/                 # Source code (.cpp)
├── src/include/         # Header files (.hpp)
├── test/                # Tests (unit, SQL, integration)
├── benchmark/           # Performance benchmarks
├── scripts/             # Build and utility scripts
├── third_party/         # External dependencies
├── examples/            # Example queries and usage
├── docs/                # Additional documentation
└── CMakeLists.txt       # Build configuration
```

---

## Top-Level Directory Structure

### `src/`

Main source code directory containing all `.cpp` implementation files.

**Subdirectories:**

- `src/op/` - New Mode operators (`sirius_physical_*`)
- `src/operator/` - Legacy Mode operators (`gpu_physical_*`)
- `src/planner/` - Physical plan generation
- `src/pipeline/` - Pipeline and task management
- `src/parallel/` - Task executors and threading
- `src/expression_executor/` - Expression evaluation
- `src/cuda/` - CUDA kernels and GPU utilities
- `src/memory/` - Memory management
- `src/data/` - Data structures and converters
- `src/downgrade/` - Memory downgrade/spilling
- `src/util/` - Utility functions
- `src/creator/` - Task creation logic

### `src/include/`

Header files (`.hpp`) matching the structure of `src/`.

**Subdirectories:**

- `src/include/op/` - New Mode operator headers
- `src/include/operator/` - Legacy Mode operator headers
- `src/include/planner/` - Planner headers
- `src/include/pipeline/` - Pipeline headers
- `src/include/parallel/` - Threading headers
- `src/include/memory/` - Memory management headers
- `src/include/log/` - Logging system headers
- `src/include/helper/` - Helper utilities

### `test/`

All test code and test data.

**Subdirectories:**

- `test/cpp/` - C++ unit tests (Google Test)
- `test/sql/` - SQL logic tests
- `test/integration/` - Integration tests
- `test/data/` - Test data files (TPC-H, etc.)

### `benchmark/`

Performance benchmarks using Google Benchmark.

### `third_party/`

External dependencies (cuDF, RMM, DuckDB, etc.).

---

## Source Code Organization

### New Mode (Recommended)

**Operators**: `src/op/sirius_physical_*.cpp`

| File | Operator | Type |
|------|----------|------|
| `sirius_physical_filter.cpp` | Filter rows by predicate | Intermediate |
| `sirius_physical_projection.cpp` | Compute new columns | Intermediate |
| `sirius_physical_table_scan.cpp` | Scan DuckDB tables | Source |
| `sirius_physical_hash_join.cpp` | Hash join (build/probe) | Sink + Source |
| `sirius_physical_grouped_aggregate.cpp` | Grouped aggregation | Sink + Source |
| `sirius_physical_ungrouped_aggregate.cpp` | Global aggregation | Sink + Source |
| `sirius_physical_order.cpp` | Sort by columns | Sink + Source |
| `sirius_physical_limit.cpp` | Limit rows | Intermediate |
| `sirius_physical_top_n.cpp` | Optimized top-k | Sink + Source |
| `sirius_physical_result_collector.cpp` | Collect final results | Sink |

**Planners**: `src/planner/sirius_plan_*.cpp`

| File | Logical Operator |
|------|------------------|
| `sirius_plan_filter.cpp` | LogicalFilter → sirius_physical_filter |
| `sirius_plan_projection.cpp` | LogicalProjection → sirius_physical_projection |
| `sirius_plan_get.cpp` | LogicalGet → sirius_physical_table_scan |
| `sirius_plan_comparison_join.cpp` | LogicalComparisonJoin → sirius_physical_hash_join |
| `sirius_plan_aggregate.cpp` | LogicalAggregate → sirius_physical_*_aggregate |
| `sirius_plan_order.cpp` | LogicalOrder → sirius_physical_order |
| `sirius_plan_limit.cpp` | LogicalLimit → sirius_physical_limit |

**Pipeline Components**: `src/pipeline/`

| File | Component |
|------|-----------|
| `sirius_pipeline.cpp` | Pipeline structure |
| `sirius_meta_pipeline.cpp` | Meta pipeline (grouping) |
| `sirius_pipeline_itask.cpp` | Task interface |
| `sirius_pipeline_build_state.cpp` | Pipeline building state |

**Task Executors**: `src/parallel/`

| File | Executor |
|------|----------|
| `pipeline_executor.cpp` | GPU pipeline execution |
| `task_creator.cpp` | Plan → tasks conversion |
| `downgrade_executor.cpp` | Memory spilling |
| `duckdb_scan_executor.cpp` | CPU scan execution |

**Expression Evaluation**: `src/expression_executor/`

| File | Component |
|------|-----------|
| `gpu_expression_executor.cpp` | Main expression executor |
| `gpu_dispatcher.cpp` | Dispatch to type-specific handlers |
| `specializations/` | Type-specific implementations |
| `regex/gpu_regex.cpp` | Regex pattern matching |

### Legacy Mode

**Operators**: `src/operator/gpu_physical_*.cpp`

| File | Operator |
|------|----------|
| `gpu_physical_filter.cpp` | Filter rows |
| `gpu_physical_projection.cpp` | Compute columns |
| `gpu_physical_table_scan.cpp` | Scan tables |
| `gpu_physical_hash_join.cpp` | Hash join |
| `gpu_physical_grouped_aggregate.cpp` | Grouped aggregate |
| `gpu_physical_order.cpp` | Sort |
| `gpu_physical_result_collector.cpp` | Result collection |

**Execution**: `src/`

| File | Component |
|------|-----------|
| `gpu_executor.cpp` | Legacy mode executor |
| `gpu_pipeline.cpp` | Legacy pipeline |
| `gpu_meta_pipeline.cpp` | Legacy meta pipeline |
| `gpu_physical_plan_generator.cpp` | Legacy planner |

**Memory Management**: `src/`

| File | Component |
|------|-----------|
| `gpu_buffer_manager.cpp` | Singleton memory manager |
| `gpu_columns.cpp` | GPUColumn/GPUIntermediateRelation |

### Core Infrastructure

**Configuration**: `src/`

| File | Component |
|------|-----------|
| `sirius_config.cpp` | Global configuration |
| `sirius_context.cpp` | Per-connection context |

**Memory Management**: `src/memory/`

| File | Component |
|------|-----------|
| `sirius_memory_reservation_manager.cpp` | Multi-tier memory |
| `sirius_memory_pool.cpp` | Memory pools |

**Extension Entry Point**: `src/`

| File | Component |
|------|-----------|
| `sirius_extension.cpp` | DuckDB extension registration |
|  | `gpu_processing()` - Legacy Mode entry |
|  | `gpu_execution()` - New Mode entry |

**Logging**: `src/log/`

| File | Component |
|------|-----------|
| `logging.cpp` | Logging system (spdlog-based) |

**Utilities**: `src/util/`

| File | Component |
|------|-----------|
| `types.cpp` | Type conversions |
| `string_utils.cpp` | String utilities |
| `cudf_utils.cpp` | cuDF helpers |

### CUDA Kernels

**Location**: `src/cuda/`

| File | Component |
|------|-----------|
| `cuda/operator/*.cu` | Operator-specific kernels |
| `cuda/expression_executor/*.cu` | Expression evaluation kernels |
| `cuda/cudf/*.cu` | cuDF integration helpers |

---

## Key Files and Directories

### Entry Points

**DuckDB Extension Registration:**

```
src/sirius_extension.cpp
  ├─ gpu_processing() table function (Legacy Mode)
  ├─ gpu_execution() table function (New Mode)
  ├─ sirius_engine() table function (New Mode alternative)
  └─ Extension initialization
```

**New Mode Execution:**

```
src/sirius_interface.cpp
  └─ Main coordinator for New Mode
src/sirius_engine.cpp
  └─ Execution engine
src/op/*.cpp
  └─ Operator implementations
```

**Legacy Mode Execution:**

```
src/gpu_executor.cpp
  └─ Legacy mode executor
src/operator/*.cpp
  └─ Legacy operator implementations
```

### Planning

**New Mode Planner:**

```
src/planner/sirius_physical_plan_generator.cpp
  ├─ Logical → Physical conversion
  └─ create_plan() methods for each logical operator type

src/planner/sirius_plan_*.cpp
  └─ Operator-specific planning logic
```

**Legacy Mode Planner:**

```
src/gpu_physical_plan_generator.cpp
  └─ Legacy logical → physical conversion
```

### Pipeline Execution

**New Mode:**

```
src/pipeline/sirius_pipeline.cpp
  ├─ Pipeline structure
  └─ Task generation

src/pipeline/sirius_meta_pipeline.cpp
  └─ Pipeline grouping and dependency management

src/pipeline/sirius_pipeline_itask.cpp
  └─ Task interface
```

**Legacy Mode:**

```
src/gpu_pipeline.cpp
  └─ Legacy pipeline structure

src/gpu_meta_pipeline.cpp
  └─ Legacy pipeline grouping
```

### Memory Management

**New Mode:**

```
src/memory/sirius_memory_reservation_manager.cpp
  └─ Multi-tier memory (GPU/HOST/DISK)

src/downgrade/downgrade_manager.cpp
  └─ Automatic memory spilling
```

**Legacy Mode:**

```
src/gpu_buffer_manager.cpp
  └─ Singleton memory manager
src/gpu_columns.cpp
  └─ GPUColumn data structures
```

### Threading and Task Execution

```
src/parallel/pipeline_executor.cpp
  └─ GPU pipeline task execution

src/parallel/task_creator.cpp
  └─ Plan → tasks conversion

src/parallel/downgrade_executor.cpp
  └─ Memory spilling tasks

src/parallel/duckdb_scan_executor.cpp
  └─ CPU scan tasks
```

### Expression Evaluation

```
src/expression_executor/gpu_expression_executor.cpp
  ├─ Main expression executor
  └─ select() for filters

src/expression_executor/gpu_dispatcher.cpp
  └─ Type-specific dispatch

src/expression_executor/specializations/
  └─ Type-specific implementations
    ├─ int32_specialization.cpp
    ├─ int64_specialization.cpp
    ├─ float_specialization.cpp
    ├─ string_specialization.cpp
    └─ ...
```

---

## Naming Conventions

### Files

**New Mode:**

- **Operators**: `sirius_physical_<name>.cpp/hpp`
- **Planners**: `sirius_plan_<name>.cpp/hpp`
- **Infrastructure**: `sirius_<component>.cpp/hpp`

**Legacy Mode:**

- **Operators**: `gpu_physical_<name>.cpp/hpp`
- **Planners**: `gpu_physical_plan_generator.cpp`
- **Infrastructure**: `gpu_<component>.cpp/hpp`

### Classes

**New Mode:**

- **Operators**: `sirius_physical_<name>` (e.g., `sirius_physical_filter`)
- **Infrastructure**: `sirius_<component>` (e.g., `sirius_pipeline`)

**Legacy Mode:**

- **Operators**: `GPUPhysical<Name>` (e.g., `GPUPhysicalFilter`)
- **Infrastructure**: `GPU<Component>` (e.g., `GPUPipeline`)

### Namespaces

```cpp
// New Mode
namespace sirius {
namespace op {
    class sirius_physical_filter { ... };
}
namespace pipeline {
    class sirius_pipeline { ... };
}
namespace planner {
    class sirius_physical_plan_generator { ... };
}
}

// Legacy Mode
namespace duckdb {
    class GPUPhysicalFilter { ... };
    class GPUPipeline { ... };
}
```

### Variables

**Conventions:**

- `snake_case` for variables and functions
- `camelCase` for member variables (cuDF style in some areas)
- `UPPER_CASE` for constants

**Examples:**

```cpp
// Good
size_t row_count = batch->get_row_count();
auto filter_op = create_filter_operator();

// Member variables
class SiriusOperator {
    size_t row_count_;           // Trailing underscore
    std::vector<LogicalType> types;
};
```

---

## Include Paths

### Include Style

**Use angle brackets for system/third-party headers:**

```cpp
#include <cucascade/data/data_batch.hpp>
#include <cudf/table/table.hpp>
#include <duckdb/common/types.hpp>
```

**Use quotes for project headers:**

```cpp
#include "op/sirius_physical_filter.hpp"
#include "log/logging.hpp"
#include "helper/types.hpp"
```

### Include Order

**Recommended order:**

1. Corresponding header (for .cpp files)
2. Project headers
3. DuckDB headers
4. cuDF/cucascade headers
5. System headers

**Example:**

```cpp
// src/op/sirius_physical_filter.cpp

#include "op/sirius_physical_filter.hpp"       // 1. Corresponding header

#include "log/logging.hpp"                     // 2. Project headers
#include "expression_executor/gpu_expression_executor.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"  // 3. DuckDB

#include <cucascade/data/data_batch.hpp>       // 4. cuDF/cucascade
#include <cudf/table/table.hpp>

#include <chrono>                              // 5. System headers
#include <stdexcept>
```

---

## Build System

### CMakeLists.txt Structure

**Top-level**: `CMakeLists.txt`

```cmake
project(Sirius)

# Options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ENABLE_CUDA "Enable CUDA support" ON)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(cudf REQUIRED)
find_package(duckdb REQUIRED)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmark)

# Define library
add_library(sirius SHARED ${SIRIUS_SOURCES})
target_link_libraries(sirius cudf::cudf duckdb ...)
```

**Source directory**: `src/CMakeLists.txt`

```cmake
set(SIRIUS_SOURCES
    # New Mode operators
    op/sirius_physical_filter.cpp
    op/sirius_physical_projection.cpp
    op/sirius_physical_hash_join.cpp
    # ... more operators ...

    # Planners
    planner/sirius_physical_plan_generator.cpp
    planner/sirius_plan_filter.cpp
    # ... more planners ...

    # Infrastructure
    sirius_engine.cpp
    sirius_interface.cpp
    pipeline/sirius_pipeline.cpp
    # ... more files ...
)

add_library(sirius ${SIRIUS_SOURCES})
```

### Adding New Files

**To add a new operator:**

1. Create `src/op/sirius_physical_<name>.cpp`
2. Create `src/include/op/sirius_physical_<name>.hpp`
3. Create `src/planner/sirius_plan_<name>.cpp`
4. Add to `src/CMakeLists.txt`:

```cmake
set(SIRIUS_SOURCES
    # ... existing files ...
    op/sirius_physical_<name>.cpp
    planner/sirius_plan_<name>.cpp
)
```

5. Create test file `test/cpp/operator/test_<name>.cpp`

---

## Next Steps

**Related Documentation:**

- **[Building and Testing](building-and-testing.md)**: Setup development environment
- **[Adding Operators](adding-operators.md)**: Implement new operators
- **[Debugging](debugging.md)**: Debug the codebase

**Architecture:**

- **[System Overview](../02-architecture/system-overview.md)**: High-level architecture
- **[New Mode Overview](../04-new-mode/overview.md)**: New Mode architecture
- **[Legacy Mode Overview](../03-legacy-mode/overview.md)**: Legacy Mode architecture

**Reference:**

- **[File Index](../08-reference/file-index.md)**: Complete file listing
- **[API Reference](../08-reference/api-reference.md)**: Key classes and methods
