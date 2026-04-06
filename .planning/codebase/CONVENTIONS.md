# Coding Conventions

**Analysis Date:** 2026-04-06

## Naming Patterns

**Files:**
- Source files: `snake_case.cpp` (e.g., `sirius_physical_filter.cpp`, `gpu_expression_executor.cpp`)
- Header files: `snake_case.hpp` (e.g., `sirius_interface.hpp`, `gpu_buffer_manager.hpp`)
- Test files: `test_snake_case.cpp` (e.g., `test_physical_filter.cpp`, `test_cpu_cache.cpp`)
- Utility/helper headers in tests: `name_utils.hpp` or `name_test_utils.hpp` (e.g., `operator_test_utils.hpp`, `test_utils.hpp`)
- Legacy code: placed in `legacy/` directory with same naming pattern

**Functions:**
- Snake_case for all functions: `sirius_physical_filter()`, `initialize_memory_manager()`, `copy_column_to_host()`
- Private/helper functions: also snake_case, no special prefix
- Constructors follow class naming (see Classes below)

**Classes:**
- Snake_case with no prefix: `sirius_physical_filter`, `sirius_interface`, `gpu_type_traits<T>` (template)
- Exception: legacy classes may use `GPU` prefix: `GPUBufferManager`, `GPUColumn`, `GPUIntermediateRelation`
- Trait classes: `gpu_type_traits<T>` for template specializations providing type information

**Variables:**
- Local and member variables: snake_case: `gpu_column`, `output_batches`, `filter_vals`
- Static/global config: SCREAMING_SNAKE_CASE: `USE_PIN_MEM_FOR_CPU_PROCESSING`, `ENABLE_FALLBACK_CHECK`
- Namespaces: lowercase: `sirius`, `sirius::op`, `sirius::test`
- Const variables: snake_case (not screaming): `limit_ratio = 0.75`, `gpu_capacity = 512ull << 20`

**Types/Templates:**
- Custom types: snake_case: `data_batch`, `operator_data`, `gpu_data_representation`
- Enum classes: PascalCase values: `SiriusPhysicalOperatorType::FILTER`, `LogicalTypeId::BIGINT`
- Type aliases: snake_case: `using data_repository_mgr = ...`

**Constants in Code:**
- Inline literals with descriptive names for magic numbers:
  ```cpp
  const size_t gpu_capacity = 512ull << 20;  // 512MB
  const double limit_ratio = 0.75;
  constexpr size_t CPU_CACHE_TEST_MEM_SF = 8;
  ```

## Code Style

**Formatting:**
- Tool: clang-format (enforced via pre-commit hooks)
- Config file: `.clang-format`
- Key settings:
  - Column limit: 100 characters
  - Indentation: 2 spaces (no tabs)
  - Brace style: WebKit (opening brace on same line for functions/classes)
  - Pointer alignment: Left (`int* ptr` not `int *ptr`)
  - Constructor initializer list: One per line, break before colon

**Linting:**
- Tool: clang-tidy
- Config file: `.clang-tidy`
- Enforced checks: `modernize-*`, `performance-for-range-copy`, `performance-unnecessary-copy-initialization`, `clang-analyzer-*`
- Warnings treated as errors: `WarningsAsErrors: '*'`
- Notable disabled checks:
  - `modernize-use-equals-default` (auto-fix broken)
  - `modernize-use-trailing-return-type` (stylistic, no benefit)
  - `clang-analyzer-cplusplus.NewDeleteLeaks` (has bugs in llvm)

**Spell Check:**
- Tool: codespell (via pre-commit)
- Custom words file: `.codespell_words`
- Current custom words: `aktion`, `ans`, `foto`

**C++ Standard:**
- C++20 (specified in `.clang-format` as `Standard: c++20`)
- CUDA standard 20 (specified in CMakeLists.txt)

## Import Organization

**Order (from `.clang-format` IncludeCategories):**
1. Quoted includes (local project headers): `"relative/path.hpp"`
2. Benchmark/test includes: `<benchmarks/...>`, `<tests/...>`
3. cuDF test includes: `<cudf_test/...>`
4. cuDF includes: `<cudf/...>`
5. Other libcudf: `<nvtext/>`, `<cudf_kafka>`
6. Other RAPIDS: `<cugraph/>`, `<cuml/>`, `<raft/>`, `<kvikio>`
7. RMM includes: `<rmm/...>`
8. CCCL (thrust, cub, cuda): `<thrust/>`, `<cub/>`, `<cuda/>`
9. Cooperative groups and CUDA utilities: `<cooperative_groups/>`, `<cuda.h>`, `<cuda_runtime>`
10. System includes with dots: `<sys/...>`, `<iostream>`
11. STL includes (no dots): `<algorithm>`, `<vector>`, `<string>`

**Path Aliases:**
- Project includes use relative path: `#include "op/sirius_physical_filter.hpp"`
- From source: paths are relative to `src/include/` via CMake include directories
- No CMake-style path aliases observed; paths are relative within include directories

**Sorting:**
- `SortIncludes: true` in `.clang-format` — includes automatically sorted per category
- `SortUsingDeclarations: true` — using declarations sorted alphabetically

## Error Handling

**Assertion Style:**
- `D_ASSERT(condition)` - Used throughout codebase for developer assertions
- `REQUIRE(condition)` - Used in Catch2 tests for test assertions
- Example: `D_ASSERT(!sirius_active_query);` in `sirius_interface.cpp`
- Example: `REQUIRE(outputs->get_data_batches().size() == 1);` in `test_physical_filter.cpp`

**Exception Throwing:**
- DuckDB exceptions used: `duckdb::InvalidInputException("message")`
- Example: `throw duckdb::InvalidInputException("Attempting to execute a closed pending query result")`
- Constructed with descriptive messages, not bare throws

**Error Functions:**
- Template function for error results: `sirius_error_result<T>(ErrorData error, query_string)`
- Error processing: `sirius_process_error(error, query)` adds location info and JSON conversion if needed

**CUDA Error Handling:**
- Helper function: `verify_cuda_errors(message)` in test code
- Called after CUDA operations to check for kernel/memory errors
- Example: `verify_cuda_errors("CUDA Errors in CPU Caching Test")`

**Fallback Strategy:**
- Graceful fallback to DuckDB CPU when GPU operations not supported
- Controlled by config: `Config::ENABLE_DUCKDB_FALLBACK`
- Implementation in `src/fallback.cpp`

## Logging

**Framework:** spdlog

**Macros (in `src/include/log/logging.hpp`):**
- `SIRIUS_LOG_TRACE(...)` - Trace level
- `SIRIUS_LOG_DEBUG(...)` - Debug level
- `SIRIUS_LOG_INFO(...)` - Info level
- `SIRIUS_LOG_WARN(...)` - Warning level
- `SIRIUS_LOG_ERROR(...)` - Error level
- `SIRIUS_LOG_FATAL(...)` - Critical/Fatal level

**Configuration:**
- Set via: `InitGlobalLogger(log_level_str, log_dir, flush_seconds)`
- Environment variables:
  - `SIRIUS_LOG_LEVEL` - log level (trace, debug, info, warn, error, critical, off)
  - `SIRIUS_LOG_DIR` - output directory (default: `log/`)
  - `SIRIUS_LOG_FLUSH_SECONDS` - flush interval (default: 3 seconds)
- Log file format: `[YYYY-MM-DD HH:MM:SS.ms] [LEVEL] [file:line] message`
- Default output: `${CMAKE_BINARY_DIR}/log/sirius.log`

**CUDA Compilation Note:**
- Logging macros are no-ops in CUDA `.cu` files (nvcc cannot compile spdlog headers)
- Defined as empty macros: `#define SIRIUS_LOG_DEBUG(...)`

**Usage Patterns:**
```cpp
SIRIUS_LOG_DEBUG("Fetching result from GPU executor");
SIRIUS_LOG_INFO("Operation completed");
```

## Comments

**When to Comment:**
- Complex algorithms or GPU kernel logic: describe intent and non-obvious steps
- Business logic constraints: explain why a particular approach is needed
- Workarounds and temporary fixes: explain issue and reference bug number if available
- Public API (headers): Doxygen-style comments for classes and methods

**JSDoc/Doxygen Style:**
- Used in header files (`.hpp`) for public APIs
- Format: `/** @brief Description */` or multi-line `/** ... **/`
- Parameter documentation: `@param name Description`
- Return documentation: `@return Description`
- Example from `sirius_test_env.hpp`:
  ```cpp
  /**
   * @brief Create a new connection to the shared DuckDB instance.
   *
   * The extension callback's OnConnectionOpened automatically registers
   * the shared SiriusContext into the new connection's registered_state.
   */
  duckdb::Connection make_connection();
  ```

**Comment Style:**
- Single-line: `// Comment`
- Multi-line: `/* ... */` or multiple `//`
- Avoid redundant comments that restate obvious code

**Inline Comments:**
- Use sparingly; code should be self-documenting
- When used: `// Why we do this` not `// What we do`

## Function Design

**Size:**
- Guideline: Functions should fit on a screen (< ~40 lines preferred)
- Longer functions acceptable for: GPU kernels, complex multi-step operations with clear phases
- Example of reasonable length: `sirius_physical_filter::execute()` (~20 lines)

**Parameters:**
- Pass by const reference for large objects: `const operator_data& input_data`
- Pass by value for small types (primitives)
- Move semantics for ownership transfer: `std::move(types)`, `std::move(select_list)`
- Use unique_ptr for exclusive ownership: `duckdb::unique_ptr<Expression> expression`
- Use shared_ptr for shared ownership: `duckdb::shared_ptr<GPUColumn>`

**Return Values:**
- Prefer returning by unique_ptr for objects: `std::unique_ptr<operator_data> execute(...)`
- Return small types by value: `int`, `bool`, `size_t`
- Const references for non-owning returns: `const operator_data&`
- void for in-out operations (modify parameters)

**Constructor Initializer Lists:**
- One member per line, break before colon
- Sort by member declaration order in class
- Example:
  ```cpp
  sirius_physical_filter::sirius_physical_filter(
    duckdb::vector<duckdb::LogicalType> types,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list,
    duckdb::idx_t estimated_cardinality)
    : sirius_physical_operator(
        SiriusPhysicalOperatorType::FILTER, std::move(types), estimated_cardinality)
  {
  ```

## Module Design

**Exports:**
- Header files (`*.hpp`) contain public API
- Implementation details in `.cpp` (not exposed in headers)
- Namespaces group related functionality: `sirius::op::`, `sirius::test::`, `sirius::memory::`
- Files generally export one main class/component

**Barrel Files:**
- Limited use; most modules are single-file
- Include dependencies explicitly rather than via barrel files
- Test utilities exported via typed headers: `operator_test_utils.hpp`, `test_utils.hpp`

**Namespace Organization:**
- `sirius` - Main namespace for active (Super Sirius) implementation
- `sirius::op` - Physical operators
- `sirius::test` - Test utilities and fixtures
- `duckdb` - DuckDB integration (some config in `duckdb::Config`)
- `duckdb::sirius` - Expression executors (GPU versions)
- No inline anonymous namespaces; use explicit `namespace { ... }` when needed

## Header Guards

**Format:**
- `#pragma once` at top of file (modern C++, works everywhere)
- Followed by copyright header and includes

**Example:**
```cpp
#pragma once

#include "op/sirius_physical_operator.hpp"

namespace sirius {
namespace op {
// ...
```

## Copyright Headers

**Format:**
- All source files include Apache 2.0 license header
- Consistent format across all files
- Example:
  ```cpp
  /*
   * Copyright 2025, Sirius Contributors.
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * ...
   * limitations under the License.
   */
  ```

## Memory Management Patterns

**Smart Pointers:**
- DuckDB types: `duckdb::unique_ptr<T>` and `duckdb::shared_ptr<T>`
- Standard library: `std::unique_ptr<T>`, `std::shared_ptr<T>`
- GPU memory: `rmm::device_buffer`, `rmm::device_uvector<T>`
- CUDA streams: `rmm::cuda_stream_view` (non-owning), `rmm::cuda_stream` (owning)

**Memory Spaces:**
- Managed via `sirius::memory::sirius_memory_reservation_manager`
- Multiple tiers: GPU, HOST, DISK (via cuCascade)
- Passed as: `cucascade::memory::memory_space*`

**Move Semantics:**
- Actively used for ownership transfer of collections and unique_ptrs
- Example: `make_uniq<FilterOp>(std::move(types), std::move(expressions), ...)`

---

*Convention analysis: 2026-04-06*
