# Coding Conventions

**Analysis Date:** 2026-04-03

## Naming Patterns

**Files:**
- C++ source files: `snake_case.cpp` (e.g., `sirius_interface.cpp`)
- C++ header files: `snake_case.hpp` (e.g., `sirius_interface.hpp`)
- Test files: `test_<component>.cpp` (e.g., `test_cpu_cache.cpp`, `test_config.cpp`)
- SQL test files: `<feature>-sirius.test` or `<category>.test` (e.g., `tpch-sirius.test`, `bugfix.test`)
- Python scripts: `snake_case.py` (e.g., `performance_test.py`)

**Functions:**
- Function names: `snake_case` for most functions (e.g., `collect_bound_ref_indices()`)
- Member functions: `snake_case` (e.g., `are_conditions_supported()`)
- Private methods: Prefix with underscore not used; rely on access modifiers instead
- Example: `sirius_interface::sirius_process_error()`, `::collect_bound_ref_indices()`

**Variables:**
- Local variables: `snake_case` (e.g., `cpu_cache_bytes`, `gpu_column`, `num_records`)
- Member variables: `snake_case` with underscore suffix for private members (e.g., `config_path_`, `db_`, `original_config_env_`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `CPU_CACHE_TEST_MEM_SF`, `PINNED_MEMORY_PARAM_KEY`)
- Template parameters: `PascalCase` (e.g., `T`, `SRC`)

**Types & Classes:**
- Class names: `snake_case` (e.g., `shared_test_env`, `bounded_thread_pool`, `sirius_interface`)
- Enum names: `snake_case` (e.g., `env_need`, `HASH_JOIN_MODE`)
- Struct names: `snake_case` (e.g., `sirius_active_query_context`, `SiriusTableFunctionData`)
- Type aliases: `snake_case` (e.g., `sirius_prepared_statement_data`)

**Namespaces:**
- Primary namespace: `sirius` for new code (active development)
- Legacy namespace: `duckdb` for older code (gpu_processing, gpu_context)
- Sub-namespaces: `sirius::op`, `sirius::pipeline`, `sirius::exec`, `sirius::test`, `sirius::memory`
- Nested namespaces flatten in function names: `collect_bound_ref_indices()` in file scope, not as method on anonymous namespace types

## Code Style

**Formatting:**
- Tool: `clang-format` (style defined in `.clang-format`)
- Indent width: 2 spaces
- Tab width: 2 spaces
- Line length limit: 100 characters (ColumnLimit: 100)
- Pointer alignment: Left (e.g., `T* var`, not `T *var`)
- No space in empty parentheses: `func()` not `func( )`

**Brace style:**
- Control statement braces: WebKit style (opening brace on same line)
  ```cpp
  if (condition) {
    // body
  }
  ```
- Function braces: Opening brace on same line
  ```cpp
  void func() {
    // body
  }
  ```
- Class/struct/namespace braces: Opening brace on same line
- No split empty functions, records, or namespaces

**Linting:**
- Tool: `clang-tidy` with modernize checks enabled (see `.clang-tidy`)
- Warnings as errors: Yes
- Header filter regex: Sirius extensions should match the cuDF convention
- Common disabled checks:
  - `modernize-use-equals-default` (auto-fix broken)
  - `modernize-use-trailing-return-type` (stylistic preference)
  - `modernize-return-braced-init-list` (prefer explicit return type at return site)

## Import Organization

**C++ Include Order** (enforced by clang-format with IncludeBlocks: Regroup):

1. Quoted includes (project local headers): `#include "..."`
2. Benchmark/test includes: `#include <benchmarks/...>`, `#include <tests/...>`
3. cuDF test includes: `#include <cudf_test/...>`
4. cuDF includes: `#include <cudf/...>`
5. Other RAPIDS includes: `#include <nvtext/>`, `#include <cudf_kafka/>`, etc.
6. More RAPIDS: `#include <cugraph/>`, `#include <cuml/>`, `#include <raft/>`, `#include <kvikio/>`
7. RMM includes: `#include <rmm/...>`
8. CCCL includes: `#include <thrust/>`, `#include <cub/>`, `#include <cuda/>`
9. CUDA/cooperative groups: `#include <cooperative_groups/>`, `#include <cuco/>`, `#include <cuda_runtime>`
10. System includes with dot: `#include <sys/...>`
11. STL includes (no dot): `#include <vector>`, `#include <string>`

**Include guards:**
- Use `#pragma once` at top of header files (preferred over traditional guards)
- Example: `#pragma once` (no include guard macros needed)

**Using declarations:**
- Sort using declarations (SortUsingDeclarations: true)
- Example: `using namespace sirius;` then `using namespace sirius::op;`

**Path aliases:**
- Not heavily used; imports tend to be fully qualified
- Relative imports from project root are preferred

## Error Handling

**Strategy:** Mix of exception-based and explicit error checking.

**Patterns:**

**1. DuckDB exceptions for user-facing errors:**
```cpp
throw duckdb::BinderException("gpu_execution cannot be called with a NULL parameter");
throw duckdb::InvalidInputException("Invalid format");
throw std::runtime_error("Error in SiriusGeneratePhysicalPlan: " + error.RawMessage());
```

**2. Try-catch for recovery scenarios:**
```cpp
try {
  // operation
} catch (std::exception& e) {
  // log and handle
  SIRIUS_LOG_ERROR("Error: {}", e.what());
}
```

**3. Assertions for internal invariants:**
```cpp
D_ASSERT(!sirius_active_query);  // DuckDB assertion macro
D_ASSERT(engine.has_result_collector());
```

**4. CUDA error checking:**
```cpp
verify_cuda_errors("CUDA Errors in CPU Caching Test");  // Custom helper
```

**5. Optional/result handling:**
```cpp
std::optional<int> value = ...;
if (value.has_value()) { /* use value.value() */ }
```

**Not used:**
- C-style error codes
- Custom exception types (use DuckDB/std exceptions)

## Logging

**Framework:** spdlog via `SIRIUS_LOG_*` macros (defined in `src/include/log/logging.hpp`)

**Macros available:**
- `SIRIUS_LOG_TRACE(...)` - Detailed diagnostic info
- `SIRIUS_LOG_DEBUG(...)` - Debug-level information
- `SIRIUS_LOG_INFO(...)` - Informational messages
- `SIRIUS_LOG_WARN(...)` - Warning-level issues
- `SIRIUS_LOG_ERROR(...)` - Error messages
- `SIRIUS_LOG_FATAL(...)` - Critical/fatal errors

**In CUDA code (.cu files):**
- Macros are defined as no-ops (nvcc cannot compile spdlog chrono headers)
- Log critical info in CPU-side wrapper code instead

**Configuration:**
- Environment variables:
  - `SIRIUS_LOG_LEVEL` (trace, debug, info, warn, error, critical, off) — defaults to "info"
  - `SIRIUS_LOG_DIR` (path to log directory) — defaults to "${CMAKE_BINARY_DIR}/log"
  - `SIRIUS_LOG_FLUSH_SECONDS` (flush interval) — defaults to 3
- Initialized in `test/cpp/unittest.cpp` via `InitGlobalLogger()`

**Pattern:**
```cpp
#include "log/logging.hpp"

SIRIUS_LOG_DEBUG("Fetching result from GPU executor");
SIRIUS_LOG_INFO("Total meta pipelines {}", to_schedule.size());
SIRIUS_LOG_ERROR("Error executing query: {}", e.what());
```

## Comments

**When to comment:**
- Explain WHY, not WHAT (code shows what it does)
- Non-obvious design decisions or algorithm choices
- Workarounds and known limitations
- License headers required on all source files (Apache 2.0)

**License header format:**
```cpp
/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```

**JSDoc/TSDoc:**
- C++ doc comments use `///` for doxygen (less common in this codebase)
- Inline comments use `//` for single line, `/* */` for multi-line
- Example from test_utils.hpp:
  ```cpp
  /**
   * @brief Initialize the memory reservation manager for tests.
   *
   * Sets up GPU, HOST, and DISK memory tiers with test-appropriate sizes.
   */
  ```

## Function Design

**Size:** No strict line limit; prefer single responsibility

**Parameters:**
- Pass by reference for mutable objects: `GPUBufferManager& manager`
- Pass by const reference for read-only large objects: `const std::vector<Column>&`
- Pass by value for small types (int, bool, enum): `bool invalidated`
- Use smart pointers for ownership: `duckdb::unique_ptr<T>`, `duckdb::shared_ptr<T>`
- Example:
  ```cpp
  void check_executable_internal(duckdb::PendingQueryResult& pending);
  duckdb::unique_ptr<duckdb::QueryResult> fetch_result_internal(
    duckdb::PendingQueryResult& pending);
  ```

**Return values:**
- Use `duckdb::unique_ptr<T>` for new heap allocations with exclusive ownership
- Use `duckdb::shared_ptr<T>` when multiple owners are needed
- Use raw pointers for non-owning references (short-lived): `BaseQueryResult* open_result`
- Return by value for small types
- Return by const reference for large read-only objects (rare)
- Example:
  ```cpp
  duckdb::unique_ptr<T> sirius_error_result(duckdb::ErrorData error, ...);
  bool is_active() const { return db_ != nullptr; }
  ```

## Module Design

**Exports:**
- Header files expose public API; implementation in .cpp
- All public symbols in namespace `sirius` or its sub-namespaces
- Internal symbols use anonymous namespace or `sirius::internal` (less common)

**Barrel files:**
- Not heavily used in this codebase
- Each module has its own interface file (e.g., `sirius_interface.hpp`)

**Class design patterns:**
- Non-copyable (delete copy constructor/assignment): Used for resource-owning classes
  ```cpp
  shared_test_env(const shared_test_env&) = delete;
  shared_test_env& operator=(const shared_test_env&) = delete;
  ```
- Move-only (deleted copy, defaulted move): Used for unique_ptr wrappers
  ```cpp
  shared_test_env(shared_test_env&&) = delete;  // Sometimes deleted for safety
  ```
- RAII for resource management (memory, GPU buffer, DuckDB connection)

## Special Patterns

**DuckDB Conventions:**
- Use DuckDB's smart pointers: `duckdb::unique_ptr`, `duckdb::shared_ptr` (not std::)
- Use DuckDB string type: `duckdb::string` (not std::string in API boundaries)
- Use DuckDB assert: `D_ASSERT()` (not standard assert)
- Use DuckDB cast: `expr.Cast<Type>()` (not C++ cast operators)

**CUDA/GPU conventions:**
- CUDA kernel code goes in `src/cuda/*.cu` files
- CPU-side wrapper code in `src/op/*.cpp` or `src/*.cpp`
- Use cuDF APIs for GPU data manipulation (no direct cuDF kernel calls from CPU)
- Memory allocation via RMM: `rmm::cuda_stream`, `rmm::device_memory_resource`

**Test conventions:**
- Catch2 TEST_CASE naming: `"description", "[tag1][tag2]"`
- Test namespaces: `sirius::test`, `sirius::scan_test_utils`
- Avoid test-specific exports in production headers; use inline helpers in test headers

---

*Convention analysis: 2026-04-03*
