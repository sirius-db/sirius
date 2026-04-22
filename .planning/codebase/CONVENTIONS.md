# Coding Conventions

**Analysis Date:** 2026-04-21

## Naming Patterns

**Files:**
- All lowercase with underscores: `sirius_engine.cpp`, `sirius_physical_operator.hpp`
- CUDA kernels: `.cu` extension in `src/cuda/` directories
- Headers mirror source structure in `src/include/` using same naming

**Functions:**
- snake_case: `initialize_test_buffer_manager()`, `calculate_test_cpu_cache_size()`
- Member functions use snake_case: `get_data_batches()`, `prepare_for_processing()`

**Variables:**
- Local variables: snake_case: `num_records`, `gpu_column`, `cpu_cache_bytes`
- Member variables: prefixed with underscore: `_data_batches`, `_use_custom_hint`, `_custom_hint`
- Static members: PascalCase: `Config::LOG_DIR`, `Config::USE_PIN_MEM_FOR_CPU_PROCESSING`
- Constants: ALL_CAPS: `CPU_CACHE_TEST_MEM_SF`, `SIRIUS_UNITTEST_LOG_DIR`

**Types:**
- Enum classes: PascalCase: `TaskCreationHint`, `MemoryBarrierType`, `OrderByType`
- Struct/Class names: snake_case: `sirius_engine`, `sirius_physical_operator`, `shared_env_listener`
- DuckDB types use full namespace: `duckdb::shared_ptr<>`, `duckdb::unique_ptr<>`

## Namespace Organization

**Primary namespaces:**
- `namespace sirius {}` - Super Sirius (new code path): Contains `sirius_engine`, operators, pipeline infrastructure, memory management, data types
- `namespace duckdb {}` - Legacy/integration layer: DuckDB extension integration, logging, configuration
- Nested namespaces follow domain:
  - `sirius::op` - Operators
  - `sirius::pipeline` - Pipeline execution infrastructure
  - `sirius::memory` - Memory management
  - `sirius::creator` - Task creation
  - `sirius::parallel` - Parallel execution utilities
  - `sirius::data` - Data representations and conversion
  - `sirius::exec` - Execution context
  - `sirius::test` - Test utilities and fixtures
  - `sirius::utils` - Utility functions

**Using declarations in test files:**
```cpp
using namespace sirius::creator;
using namespace sirius::exec;
using namespace sirius::parallel;
using namespace sirius::op::scan;
using namespace sirius::op;
```

## Code Style

**Formatting Tool:**
- Tool: clang-format 20.1.4
- Configuration: `.clang-format` defines all style rules
- Applied automatically via pre-commit hooks (see `.pre-commit-config.yaml`)
- Run formatting: `clang-format -fallback-style=none -style=file -i <file>`

**Key formatting settings:**
- Indentation: 2 spaces (not tabs)
- Column limit: 100 characters
- Brace style: WebKit (opening brace on same line, no space before)
- Pointer alignment: Left (e.g., `int* ptr` not `int *ptr`)
- No space after C-style casts: `(int)value` not `(int) value`
- Constructor init lists: Break before colon
- Always break template declarations

**Example formatting:**
```cpp
class sirius_interface {
 public:
  sirius_interface(duckdb::ClientContext& client_context);
  duckdb::ClientContext& client_context;
  duckdb::unique_ptr<sirius_active_query_context> sirius_active_query;

 private:
  duckdb::BaseQueryResult* open_result = nullptr;
};
```

**Linting:**
- Tool: clang-tidy with custom configuration (`.clang-tidy`)
- Enabled checks: modernize, performance, clang-analyzer
- Warnings treated as errors: `WarningsAsErrors: '*'`
- Disabled checks: use-equals-default, concat-nested-namespaces, use-trailing-return-type, use-bool-literals, use-designated-initializers (all stylistic or C++20 specific)
- Run linting: `pre-commit run clang-tidy -a`

## Import Organization

**Order (enforced by clang-format):**
1. Quoted local includes: `"sirius_engine.hpp"` (Priority 1)
2. Benchmark/test includes: `<benchmarks/>`, `<tests/>` (Priority 2)
3. cuDF test includes: `<cudf_test/>` (Priority 3)
4. cuDF includes: `<cudf/>` (Priority 4)
5. Other RAPIDS: `<nvtext>`, `<cudf_kafka>` (Priority 5)
6. RAPIDS libraries: `<cugraph>`, `<cuml>`, `<raft>`, `<kvikio>` (Priority 6)
7. RMM includes: `<rmm/>` (Priority 7)
8. CCCL: `<thrust/>`, `<cub/>`, `<cuda/>` (Priority 8)
9. CUDA: `<cooperative_groups>`, `<cuco>`, `<cuda_runtime>`, etc. (Priority 8)
10. System includes with dots: `<yaml-cpp/yaml.h>` (Priority 9)
11. STL includes (no dots): `<vector>`, `<string>`, `<memory>` (Priority 10)

**In source files (`src/sirius_engine.cpp` example):**
```cpp
#include "sirius_engine.hpp"                      // local header
#include "log/logging.hpp"                        // local log header
#include "op/sirius_physical_table_scan.hpp"      // local operator header
#include "pipeline/sirius_pipeline_converter.hpp" // local pipeline header
#include "sirius/exception.hpp"                   // sirius namespace header

#include <vector>
#include <memory>
#include <string>
```

## Error Handling

**Strategy:** Exception-based with custom exception types

**Custom exceptions (in `src/include/sirius/exception.hpp`):**
- `sirius::internal_exception` - Invariant violations, internal logic errors
- `sirius::not_implemented_exception` - Feature not yet implemented
- `sirius::invalid_input_exception` - Invalid input parameters, precondition failures

**Throwing:**
```cpp
throw internal_exception("can_create_more_tasks not implemented for operator " + get_name());
throw invalid_input_exception("input_batches is empty");
throw not_implemented_exception("Unsupported feature: {}", feature_name);
```

**Catch blocks in tests:**
- Use Catch2 assertions: `REQUIRE_THROWS_AS(expression, exception_type)`
- Example: `REQUIRE_THROWS_AS(r.required("name", name), std::runtime_error);`

**DuckDB errors:**
- Integration code uses DuckDB's exception system
- Fallback throws `duckdb::InvalidInputException`

## Logging

**Framework:** spdlog (configured in `src/include/log/logging.hpp`)

**Macros (CPU code only):**
- `SIRIUS_LOG_TRACE(...)` - Trace level
- `SIRIUS_LOG_DEBUG(...)` - Debug level
- `SIRIUS_LOG_INFO(...)` - Info level (default)
- `SIRIUS_LOG_WARN(...)` - Warning level
- `SIRIUS_LOG_ERROR(...)` - Error level
- `SIRIUS_LOG_FATAL(...)` - Critical/fatal level

**CUDA kernels:**
- CUDA compilation cannot include spdlog headers
- All logging macros are no-ops in `.cu` files (see `#ifdef __CUDACC__`)

**Initialization:**
- Called in `test/cpp/unittest.cpp`: `InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, flush_seconds)`
- Log output: `${SIRIUS_LOG_DIR}/sirius.log`
- Pattern: `[%Y-%m-%d %T.%e] [%l] [%s:%#] %v` (timestamp, level, file:line, message)

**Runtime control:**
- Set log level via env var: `SIRIUS_LOG_LEVEL=debug` (before initialization)
- Change at runtime: `SetGlobalLogLevel("warn")`
- Flush interval: `SetGlobalLogFlush(flush_seconds)`

**Usage patterns:**
```cpp
SIRIUS_LOG_DEBUG("Initializing sirius_engine");
SIRIUS_LOG_ERROR("Error executing query: {}", e.what());
SIRIUS_LOG_INFO("Query Plan:\n{}", plan_printer.render());
SIRIUS_LOG_WARN("[sirius_engine] SiriusContext not available");
```

## Comments

**When to comment:**
- Document public APIs with intent/usage
- Explain non-obvious logic, especially in GPU code
- Mark deprecated code or workarounds
- High-level overview comments at start of complex functions

**JSDoc/Doxygen comments:**
- Use `/** ... */` for public APIs
- Common tags:
  - `@brief` - One-line description
  - `@param` - Parameter description
  - `@return` - Return value description
  - `@throws` - Exceptions thrown
  - `@note` - Important notes
  - `@example` - Usage examples

**Example (from `src/include/op/sirius_physical_operator.hpp`):**
```cpp
/**
 * @brief Lock all data batches for processing in the requested memory space.
 *
 * Iterates over all batches and locks (or converts then locks) each one into the
 * requested memory space. Returns the processing handles that keep the batches locked
 * until they go out of scope.
 *
 * Returns std::nullopt if any batch fails to lock (triggers a retry/reschedule).
 * Propagates rmm::out_of_memory so the caller can record metrics and reschedule.
 *
 * @param requested_memory_space  Target memory space; may be nullptr to use each batch's
 *                                current space.
 * @param stream                  CUDA stream used for any data-movement kernels.
 * @return Processing handles for all batches, or std::nullopt on lock failure.
 */
virtual std::optional<std::vector<::cucascade::data_batch_processing_handle>>
prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                       rmm::cuda_stream_view stream);
```

## Function Design

**Size:** Keep functions focused and under 200 lines where practical. Complex operators can be longer.

**Parameters:**
- Pass by const reference for input: `const std::string& query`
- Pass by reference for output/modification: `duckdb::ClientContext& context`
- DuckDB uses smart pointers: `duckdb::shared_ptr<>`, `duckdb::unique_ptr<>`
- Move semantics for heavy objects: `std::vector<...> data_batches` or `std::move(...)`

**Return values:**
- Use `std::optional<T>` for nullable returns: `std::optional<std::vector<...>> prepare_for_processing(...)`
- Return by value for small types (enums, small structs)
- Return const reference for large read-only data
- Use `[[nodiscard]]` attribute for important return values

**Example from `src/include/op/sirius_physical_operator.hpp`:**
```cpp
[[nodiscard]] const std::vector<std::shared_ptr<::cucascade::data_batch>>& get_data_batches()
  const
{
  return _data_batches;
}

std::vector<std::shared_ptr<::cucascade::data_batch>> release_data_batches()
{
  return std::move(_data_batches);
}
```

## Module Design

**Headers in include/ directories:**
- One primary class/interface per header file
- Name matches file name
- Include guard: `#pragma once` (not `#ifndef`)
- All includes at top, organized by priority (see Import Organization)

**Exports/Public API:**
- Public interfaces via namespace + class name
- Private implementation in `.cpp` files
- Use `namespace {}` anonymous blocks for internal helpers in `.cpp`

**Example structure (`src/include/sirius_interface.hpp`):**
```cpp
#pragma once

namespace sirius {

class sirius_interface {
 public:
  // Public API
  sirius_interface(duckdb::ClientContext& client_context);
  void check_executable_internal(...);
  duckdb::unique_ptr<duckdb::QueryResult> fetch_result_internal(...);

 private:
  // Private data
  duckdb::ClientContext& client_context;
  duckdb::unique_ptr<sirius_active_query_context> sirius_active_query;
};

}  // namespace sirius
```

**Barrel Files (optional):**
- Not commonly used; most code includes specific headers
- When used, typically in `include/` for convenience exports

## Header Organization

**Header structure order:**
1. Copyright notice (Apache 2.0)
2. Include guard: `#pragma once`
3. Conditional CUDA handling: `#ifdef __CUDACC__` for kernel-only headers
4. Standard library includes
5. External library includes (cudf, rmm, duckdb, etc.)
6. Local includes from same project
7. Namespace declarations
8. Forward declarations for reduced coupling
9. Type definitions (enums, structs, classes)
10. Namespace closing

**Example (simplified `src/include/sirius_engine.hpp`):**
```cpp
/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <vector>
#include <memory>

#include "duckdb/main/connection.hpp"
#include "duckdb/planner/physical_operator.hpp"

#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"

namespace sirius {

class sirius_interface;

class sirius_engine {
 public:
  sirius_engine(duckdb::ClientContext& context);
  void execute();

 private:
  duckdb::unique_ptr<op::sirius_physical_operator> sirius_physical_plan;
};

}  // namespace sirius
```

## CUDA Kernel Conventions

**Kernel naming:**
- Kernel functions: `__global__ void kernel_name(...)`
- Device functions: `__device__ void device_function(...)`
- Device types: Use cuDF types (`cudf::bitmask_type`, `cudf::mutable_column_view`)

**Logging in kernels:**
- NO logging allowed - spdlog/fmt incompatible with nvcc
- Use assertions for debugging: `assert(condition)` in debug builds only
- Coordinate with CPU-side logging in wrapper functions

**Memory management:**
- Use RMM for GPU memory allocation: `rmm::cuda_stream_view stream`
- cuCascade handles CPU↔GPU transfers via data batches
- No manual `cudaMalloc`/`cudaFree` - managed via RMM

**Compilation:**
- CMake: `CMAKE_CUDA_SEPARABLE_COMPILATION ON`
- Standard: `--std=c++20`
- GPU architectures: 75, 80, 86, 90a, 100f, 120a, 120 (Turing through Blackwell)

---

*Convention analysis: 2026-04-21*
