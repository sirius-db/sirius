# Coding Conventions

**Analysis Date:** 2026-04-13

## Naming Patterns

**Files:**
- C++ source/header: `snake_case.cpp` / `snake_case.hpp`
  - Example: `sirius_engine.cpp`, `cpu_cache.hpp`, `sirius_physical_plan_generator.cpp`
- CUDA kernels: `snake_case.cu`
  - Example: `cudf_aggregate.cu`, `allocator.cu`, `cudf_orderby.cu`
- Python: `snake_case.py`
  - Example: `performance_test.py`, `generate_test_data.py`
- Configuration: `*.yaml`, `*.config.*` (e.g., `integration.yaml`, `memory.yaml`)

**Classes and Structs:**
- Classes use `snake_case`: `sirius_engine`, `sirius_physical_plan_generator`, `sirius_interface`
  - Legacy pattern also uses `snake_case`: `gpu_physical_ungrouped_aggregate`
  - Exceptions: Some class names use `CamelCase` when inherited from DuckDB API (e.g., `SiriusExtension`, `SiriusContext`)
- Struct types use `snake_case` with suffix `_config` or similar: `task_executor_config`, `operator_params`

**Functions and Methods:**
- Functions and methods use `snake_case`: `initialize()`, `execute()`, `get_memory_space()`, `copy_null_mask()`
- Private methods: prefixed with underscores only when truly internal
- CUDA kernel functions use `snake_case` with trailing comments: `convert_uint64_to_int32<>()` with template specifiers

**Variables:**
- Local variables: `snake_case`: `gpu_result`, `config_path`, `expected_nulls`, `actual_valid`
- Member variables: `snake_case`: `context`, `sirius_iface`, `root_pipeline_idx`, `query_finished`
- Static variables and constants: `SCREAMING_SNAKE_CASE`: `USE_PIN_MEM_FOR_CPU_PROCESSING`, `LOG_DIR`, `MAX_SORT_PARTITION_BYTES`
- Loop counters: simple `i`, `j`, `k` or descriptive names like `row_id`, `col`
- Type-generic naming in templates: `T`, `I`, `B` (block threads, items per thread)

**Enums and Type Aliases:**
- Enum values: `SCREAMING_SNAKE_CASE`: `AggregationType::COUNT_STAR`, `OrderByType::ASC`, `KernelColType::INT_64`
- Type aliases: `snake_case` when lowercase, `CamelCase` for complex types: `cudf::column_view`, `duckdb::shared_ptr<>`

**Namespaces:**
- Primary namespace: `sirius` (new Super Sirius code)
- Legacy/DuckDB integration: `duckdb` (some shared components like `Config`)
- Nested namespaces use `snake_case`: `sirius::op::scan`, `sirius::pipeline`, `sirius::test`, `cucascade::memory`, `sirius::scan_test_utils`

## Code Style

**Formatting:**
- Tool: `clang-format` (config in `.clang-format`)
- Column limit: 100 characters
- Indent width: 2 spaces
- No tabs (UseTab: Never)
- Brace style: WebKit (opening braces on same line for most constructs)
  - Example:
    ```cpp
    if (condition) {
      // code
    }
    ```

**Linting:**
- Tool: `clang-tidy` (config in `.clang-tidy`)
- Checks: `modernize-*` (minus selected excluded checks for stylistic/known-broken rules)
- Performance checks: `performance-for-range-copy`, `performance-unnecessary-copy-initialization`, `performance-unnecessary-value-param`
- Static analysis: `clang-analyzer-*` (minus known broken checks)
- WarningsAsErrors: `*` (all warnings are errors)
- Header filter: `.*cudf/cpp/(src|include).*` (primarily filters for cuDF code analysis)

**Code style tools:**
- C++/CUDA: `clang-format` (runs via pre-commit)
- Python: `black` (rev 25.1.0, runs via pre-commit)
- CMake: `cmake-format` (via pre-commit, line-width 220, suppress decorations)
- Spell check: `codespell` (custom words in `.codespell_words`)

Run formatting/linting:
```bash
pre-commit run -a              # Run all hooks on all files
pre-commit install             # Install git hooks (runs on every commit)
```

## Import Organization

**Order (per `.clang-format` IncludeCategories):**

1. Quoted includes (local project headers first): `"config.hpp"`
2. Benchmark/test includes: `<benchmarks/`, `<tests/`
3. cuDF test includes: `<cudf_test/`
4. cuDF includes: `<cudf/`
5. Other libcudf: `<nvtext/`, `<cudf_kafka>`
6. Other RAPIDS: `<cugraph/`, `<cuml/`, `<raft/`, `<kvikio>`
7. RMM includes: `<rmm/`
8. CCCL (CUDA collective communications): `<thrust/`, `<cub/`, `<cuda/`
9. CUDA includes: `<cooperative_groups/`, `<cuco/`, `<cuda.h/`, etc.
10. Other system includes (with dot): `<chrono>`, `<iostream>`
11. STL includes (no dot): `<vector>`, `<string>`, `<memory>`

**Example from `src/sirius_extension.cpp`:**
```cpp
#include "config.hpp"
#include "duckdb/main/database.hpp"
#include "data/sirius_converter_registry.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"
#include <duckdb.hpp>
#include <iostream>
```

**SortIncludes:** true (enforce alphabetical within each category)

## Error Handling

**Patterns:**
- **DuckDB exceptions:** Use `duckdb::InvalidInputException`, `duckdb::NotImplementedException`, `duckdb::InternalException` when integrating with DuckDB API
  - Example: `throw duckdb::InvalidInputException("Attempting to execute a closed pending query result")`
- **Standard C++ exceptions:** Use `std::runtime_error` for configuration/runtime errors, `std::exception` for catch-all
  - Example: `throw std::runtime_error("Failed to load config from " + config_path.string())`
- **Try-catch pattern:** Wrap DuckDB query execution and external library calls
  - All top-level GPU execution paths have try-catch in `sirius_engine.cpp` and `sirius_interface.cpp`
  - On exception, log error details, trigger cleanup (e.g., `clear_all_repositories()`), and re-throw

**Error propagation:** Errors surface through `QueryResult::HasError()` check (DuckDB API pattern)
```cpp
auto result = con->Query(query);
REQUIRE(result);
if (result->HasError()) {
  UNSCOPED_INFO("error: " << result->GetError());
}
REQUIRE_FALSE(result->HasError());
```

**CUDA/cuDF errors:** Checked via `gpuErrchk()` macro and `CHECK_ERROR()` (see `cuda_helper.cuh`)

## Logging

**Framework:** `spdlog` (custom-configured in `src/include/log/logging.hpp`)

**Log levels:** trace, debug, info, warn, error, critical (via SPDLOG)

**Initialization:** Happens in unit test main (`test/cpp/unittest.cpp`):
```cpp
auto log_dir = SIRIUS_UNITTEST_LOG_DIR;
Config::LOG_DIR = log_dir;
InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS);
```

**Runtime configuration:**
- Environment variables: `SIRIUS_LOG_DIR`, `SIRIUS_LOG_LEVEL`
- Defaults in `src/config.cpp`: `LOG_LEVEL = "info"`, `LOG_DIR = "log"`, `LOG_FLUSH_SECONDS = 3`
- Log file pattern: `[YYYY-MM-DD HH:MM:SS.ms] [level] [file:line] message`
- Flush: Every 3 seconds (configurable)

**Logging macros (CUDA-safe):**
- `SIRIUS_LOG_TRACE(...)` (no-op in CUDA code)
- `SIRIUS_LOG_DEBUG(...)`
- `SIRIUS_LOG_INFO(...)`
- `SIRIUS_LOG_WARN(...)`
- `SIRIUS_LOG_ERROR(...)`
- `SIRIUS_LOG_FATAL(...)` (maps to SPDLOG_CRITICAL)

In CUDA code (`__CUDACC__`), all macros are no-ops to avoid nvcc compilation errors.

**Usage example:**
```cpp
SIRIUS_LOG_DEBUG("CUDF Aggregate");
SIRIUS_LOG_DEBUG("Input size: {}", column[0]->column_length);
SIRIUS_LOG_ERROR("CUDA initialization error for gpu {}: {}", gpu, cudaGetErrorString(err));
```

## Comments

**When to Comment:**
- Explain WHY, not WHAT (code is self-documenting)
- Clarify non-obvious algorithmic choices or GPU-specific quirks
- Mark TODO/FIXME items for future work
- Document assumptions and constraints

**JSDoc/Doxygen style:**
Used selectively in header files for public APIs:
```cpp
/**
 * @brief Copy null mask from GPU column to host vector.
 * @param col cuDF column with potential null mask
 * @return Vector of bitmask_type, empty if no nulls
 */
std::vector<cudf::bitmask_type> copy_null_mask(const cudf::column_view& col);
```

**Block comments:** Use `/* ... */` for multi-line explanations
```cpp
// Pause environments that should not be active for this test
if (needs != env_need::SHARED && sirius::test::g_shared_env &&
    sirius::test::g_shared_env->is_active()) {
  sirius::test::g_shared_env->pause();
}
```

**TODO/FIXME:** Inline format
```cpp
// TODO: probably want to use sirius config for these two values
```

## Function Design

**Size:** Keep functions focused and reasonably short (prefer < 100 lines for clarity)
- Example: `copy_null_mask()` = 8 lines, `verify_validity_mask()` = 20 lines
- Longer functions acceptable in operator implementations when necessary

**Parameters:**
- Pass const references for large objects: `const cudf::column_view& col`
- Pass pointers for ownership transfer: `sirius_physical_operator* op`
- Use `std::optional<T>` for nullable values: `std::optional<float> float_tolerance`
- In CUDA kernels, use template parameters for block/item configuration: `template <int B, int I>`

**Return Values:**
- Use `std::unique_ptr<T>` for exclusive ownership: `duckdb::unique_ptr<QueryResult>`
- Use `std::shared_ptr<T>` for shared ownership: `duckdb::shared_ptr<SiriusContext>`
- Return bool for success/failure checks
- Return void for fire-and-forget operations

**Template functions:** Specialize for common types (see `allocator.cu` pattern for cudaMalloc templates)

## Module Design

**Exports:**
- No barrel files (no `index.hpp` pattern)
- Each header is self-contained with necessary includes
- Library entry point via DuckDB extension (`src/sirius_extension.cpp`)

**Header organization:**
- `src/include/` mirrors `src/` structure
- Public headers in `src/include/`, implementation in `src/`
- CUDA headers in `src/include/cuda/`, kernels in `src/cuda/`

**Operator structure (Super Sirius):**
- Header: `src/include/op/sirius_physical_<operator>.hpp` (declares class inheriting `sirius_physical_operator`)
- Implementation: `src/op/sirius_physical_<operator>.cpp` (DuckDB integration)
- Tests: `test/cpp/operator/test_<operator>.cpp`

**Visibility:**
- Use `private:` / `protected:` / `public:` sections in class declarations
- Friend classes for tight coupling: `friend class pipeline::sirius_pipeline_build_state;`

## C++ Standards and Features

**Standard:** C++20 (specified in `.clang-format` and `CMakeLists.txt`)

**Modern C++ idioms:**
- Range-based for loops: `for (auto& tag : info.tags)`
- Auto type deduction: `auto log_file = log_dir + "/sirius.log";`
- std::optional: `if (needs == env_need::SHARED && !sirius::test::g_shared_env)`
- Move semantics: `auto result_key = std::move(result.keys);`
- RAII: Resources managed via constructors/destructors

**DuckDB type aliases (for consistency with DuckDB codebase):**
- `duckdb::idx_t` for indices
- `duckdb::shared_ptr<T>`, `duckdb::unique_ptr<T>` (DuckDB's wrapper types)
- `duckdb::vector<T>` (DuckDB's vector type)

---

*Convention analysis: 2026-04-13*
