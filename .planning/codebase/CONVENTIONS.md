# Coding Conventions

**Analysis Date:** 2025-04-02

## Naming Patterns

**Files:**
- C++ source: `snake_case.cpp` (e.g., `sirius_physical_hash_join.cpp`, `gpu_expression_translator.cpp`)
- C++ headers: `snake_case.hpp` (e.g., `sirius_interface.hpp`, `fallback.hpp`)
- CUDA kernels: `snake_case.cu` (e.g., `gpu_hash_join.cu`)
- Test files: `test_*.cpp` (e.g., `test_config.cpp`, `test_gpu_execution_tpch.cpp`)
- SQL logic tests: `*.test` (e.g., `tpch-sirius.test`, `clickbench-sirius.test`)

**Functions:**
- Snake case: `bind_prepared_statement_parameters()`, `collect_bound_ref_indices()`, `moveDataToCPU()` (for DuckDB C++ interop functions that match DuckDB style)
- Mixed when matching DuckDB API: `sirius_process_error()` combines snake_case with prefix
- Static/utility functions in files: snake_case with descriptive names
- Class methods: snake_case (e.g., `are_conditions_supported()`, `execute()`, `get_types()`)
- Getters and setters: `get_*()`, `set_*()` pattern (e.g., `get_result()`, `get_types()`)

**Variables:**
- Local variables: `snake_case` (e.g., `cpu_result`, `gpu_sql`, `chunk_id`)
- Member variables: `snake_case_` with trailing underscore for private members (e.g., `is_initialized_`)
- Loop variables: Single letter or abbreviated snake case (e.g., `i`, `c`, `r`, `const auto& cond`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `CPU_CACHE_TEST_MEM_SF`, `CATCH_CONFIG_RUNNER`)

**Types:**
- Classes: `PascalCase` (e.g., `GPUBufferManager`, `SiriusContext`, `GPUExecutionFixtureBase`)
- Enums: `PascalCase` (e.g., `MemoryBarrierType`, `TaskCreationHint`)
- Struct names: `snake_case` or `PascalCase` depending on context (e.g., `task_creation_hint`, `sirius_config_env_guard`)
- Template parameters: `PascalCase` (e.g., `TARGET`)
- Type aliases and using declarations: `snake_case` (e.g., `using TestEventListenerBase = ...`)

**Namespaces:**
- Primary: `sirius` (new GPU execution path) or `duckdb` (shared/legacy code)
- Sub-namespaces: `sirius::op`, `sirius::test`, `sirius::planner`, `duckdb::*` (following DuckDB conventions)
- Namespace closing: `}  // namespace name` with comment

## Code Style

**Formatting:**
- Tool: `clang-format` configured via `.clang-format`
- Line width: 100 characters (ColumnLimit: 100)
- Indentation: 2 spaces, no tabs (TabWidth: 2, UseTab: Never)
- Brace style: WebKit (opening brace on same line as control statement)
- Pointer alignment: Left (`int* ptr` not `int *ptr`)

**Critical clang-format settings:**
- `AlignAfterOpenBracket: Align` — Align function parameters/arguments
- `BreakConstructorInitializers: BeforeColon` — Colon on new line for constructor init lists
- `ConstructorInitializerAllOnOneLineOrOnePerLine: true` — No mixed init list formatting
- `AllowShortFunctionsOnASingleLine: All` — Single-line OK for short functions
- `BinPackArguments: false` — Parameters on separate lines when wrapping
- `BinPackParameters: false` — Same for function declarations
- `AlwaysBreakTemplateDeclarations: Yes` — Template declaration keywords on new line

**Linting:**
- Tool: `clang-tidy` configured via `.clang-tidy`
- Enabled checks: `modernize-*`, `performance-*`, `clang-analyzer-*` (with specific exclusions)
- `WarningsAsErrors: '*'` — Treat warnings as errors
- Key disabled checks: `modernize-use-equals-default`, `modernize-use-trailing-return-type` (stylistic), `clang-analyzer-cplusplus.NewDeleteLeaks` (has bugs)

**Pre-commit hooks** (`.pre-commit-config.yaml`):
- clang-format: C++/CUDA code formatting (auto-fix: `-i`)
- black: Python formatting
- codespell: Spell checking with custom words in `.codespell_words`
- cmake-format: CMake file formatting
- Standard hooks: trailing whitespace, YAML/JSON checks, large files, mixed line endings

Run all hooks:
```bash
pre-commit run -a
```

## Import Organization

**Order:**
1. Project-local headers in quotes: `"sirius_interface.hpp"` (Priority 1)
2. Benchmark/test headers: `<benchmarks/...>`, `<tests/...>` (Priority 2)
3. cuDF test headers: `<cudf_test/...>` (Priority 3)
4. cuDF headers: `<cudf/...>` (Priority 4)
5. Other RAPIDS: `<nvtext/...>`, `<cugraph/...>`, `<raft/...>` (Priority 5-6)
6. RMM headers: `<rmm/...>` (Priority 7)
7. CUDA/CCCL: `<cuda/...>`, `<thrust/...>`, `<cub/...>` (Priority 8)
8. System headers with dots: `<sys/types.h>` (Priority 9)
9. STL headers (no dots): `<vector>`, `<memory>`, `<string>` (Priority 10)

**Settings:**
- `IncludeBlocks: Regroup` — Regroup includes by priority
- `SortIncludes: true` — Sort within each group
- `SortUsingDeclarations: true` — Sort `using` statements

**Path aliases:**
- Sirius includes use relative paths from `src/include`: `#include "op/sirius_physical_hash_join.hpp"`
- DuckDB includes use angle brackets with full path: `#include <duckdb/main/client_context.hpp>`
- cuDF/RAPIDS includes: angle brackets `#include <cudf/...>`

## Error Handling

**Patterns:**
- DuckDB exceptions: `throw duckdb::InternalException(...)`, `throw duckdb::InvalidInputException(...)`
- Sirius context: `throw std::runtime_error(...)` for initialization/config errors
- Assertions: `D_ASSERT(condition)` from DuckDB for debug-only checks
- Error data: `duckdb::ErrorData` struct with `FinalizeError()`, `ConvertErrorToJSON()` methods

**Example error handling** (`src/sirius_interface.cpp`):
```cpp
void sirius_interface::sirius_process_error(duckdb::ErrorData& error,
                                            const duckdb::string& query) const
{
  error.FinalizeError();
  if (duckdb::Settings::Get<duckdb::ErrorsAsJSONSetting>(client_context)) {
    error.ConvertErrorToJSON();
  } else {
    error.AddErrorLocation(query);
  }
}
```

**Example validation throws** (`src/gpu_buffer_manager.cpp`):
```cpp
if (ptr == nullptr) throw InvalidInputException("Pointer is nullptr");
if (ptr already exists) throw InvalidInputException("Pointer already exists in allocation table");
```

**Fallback checker pattern** (`src/fallback.cpp`):
- Visitor pattern for checking unsupported operations
- Switch on expression type/operator type with explicit case handlers
- Recursively check child expressions
- Throw with formatted message on unsupported feature

## Logging

**Framework:** spdlog

**Macros** (defined in `src/include/log/logging.hpp`):
- `SIRIUS_LOG_TRACE(...)` — Trace level
- `SIRIUS_LOG_DEBUG(...)` — Debug level
- `SIRIUS_LOG_INFO(...)` — Info level
- `SIRIUS_LOG_WARN(...)` — Warning level
- `SIRIUS_LOG_ERROR(...)` — Error level
- `SIRIUS_LOG_FATAL(...)` — Fatal (CRITICAL) level

**CUDA files:**
- In `.cu` files: macros expand to no-ops (spdlog cannot be compiled by nvcc)
- Use macros in `.cpp` files only

**Initialization** (`test/cpp/unittest.cpp`):
```cpp
std::string log_dir = SIRIUS_UNITTEST_LOG_DIR;
Config::LOG_DIR     = log_dir;
InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS);
```

**Environment variables:**
- `SIRIUS_LOG_LEVEL`: debug, info, warn, error (defaults to Config::LOG_LEVEL)
- `SIRIUS_LOG_DIR`: Directory for log files (defaults to CMAKE_BINARY_DIR/log)

## Comments

**When to comment:**
- Algorithm explanation: Complex join logic, memory management decisions
- Performance notes: Why specific optimizations were chosen
- Warnings about gotchas: "Mixed join: cuDF requires equality and conditional columns to be disjoint"
- References to external docs: Link to DuckDB or cuDF docs for non-obvious patterns
- Not for obvious code: Don't comment `++i` or `value += 1`

**JSDoc/TSDoc-style documentation:**
- Use `/// @brief` for function descriptions (copied from RAPIDS conventions)
- Use `/// @param` for parameters
- Use `/// @return` for return values
- Example (`test/cpp/unittest.cpp`):
```cpp
/**
 * @brief Catch2 listener that activates/deactivates shared test environments
 * based on test tags.
 *
 * Only one shared environment can be active at a time (each owns the extension
 * lock).  The listener uses a transition-based design: it pauses the wrong
 * environment and resumes the right one in testCaseStarting, so consecutive
 * tests of the same type share a single DuckDB/SiriusContext instance without
 * any intermediate teardown.
 *
 *   [shared_context]  → g_shared_env      (scan/operator unit tests)
 *   [integration]     → g_integration_env (GPU execution integration tests)
 *   anything else     → no env active     (isolated / standalone tests)
 */
struct shared_env_listener : Catch::TestEventListenerBase { ... };
```

## Function Design

**Size:**
- Prefer small, focused functions (max ~50 lines for critical paths)
- Static helper functions for complex logic (e.g., `collect_bound_ref_indices()`)
- Use early returns to reduce nesting

**Parameters:**
- Use `const auto&` for loop variables when iterating containers: `for (auto const& cond : conditions)`
- Use `auto&` for mutable references
- Use `duckdb::unique_ptr<T>` for owned resources
- Use `duckdb::shared_ptr<T>` for shared ownership (GPU memory wrappers)
- DuckDB-style parameters: pass by const reference, return smart pointers

**Return values:**
- Errors: throw exceptions (DuckDB pattern)
- Success: return value via `duckdb::unique_ptr<T>` or by-reference parameter
- Optional: use `std::optional<T>` (e.g., float_tolerance parameter in tests)

**Example function signature** (`src/op/sirius_physical_hash_join.cpp`):
```cpp
static void collect_bound_ref_indices(duckdb::Expression& expr,
                                      std::unordered_set<duckdb::idx_t>& indices)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    indices.insert(expr.Cast<duckdb::BoundReferenceExpression>().index);
    return;
  }
  duckdb::ExpressionIterator::EnumerateChildren(
    expr, [&](duckdb::Expression& child) { collect_bound_ref_indices(child, indices); });
}
```

## Module Design

**Exports:**
- Header files contain declarations; implementations in `.cpp`
- Public APIs in headers under `src/include/`; implementation details in `src/`
- Use `inline` for small utility functions in headers

**Barrel files:**
- Not used; direct includes by path (e.g., `#include "op/sirius_physical_hash_join.hpp"`)

**Class organization:**
- Public: Constructors, main methods, public getters
- Protected: Virtual methods, protected data for derived classes
- Private: Implementation details, private members with trailing `_`

**Example** (`src/include/operator/cpu_cache.hpp`):
```cpp
class CPUCache {
 public:
  virtual uint32_t moveDataToCPU(shared_ptr<GPUIntermediateRelation> relationship) = 0;
  virtual shared_ptr<GPUIntermediateRelation> moveDataToGPU(uint32_t chunk_id,
                                                            bool evict_from_cpu) = 0;
};
```

## Type Safety

**Smart pointers:**
- `duckdb::unique_ptr<T>` — Sole ownership (RAII)
- `duckdb::shared_ptr<T>` — Shared ownership
- `duckdb::optional_ptr<T>` — Nullable non-owning pointer (replaces `T*` in DuckDB)

**Type casting:**
- `expr.Cast<TargetType>()` — DuckDB-safe cast for expressions
- `dynamic_cast<T*>` — General C++ polymorphism (avoid in hot paths)
- `reinterpret_cast<T*>` — Only for GPU memory pointers

**Const correctness:**
- Mark member functions `const` if they don't modify state
- Use `const auto&` in loops: `for (auto const& item : items)`
- Use `const` on parameters that shouldn't be modified

---

*Convention analysis: 2025-04-02*
