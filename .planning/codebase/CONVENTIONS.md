# Coding Conventions

**Analysis Date:** 2026-04-06

## Naming Patterns

**Files:**
- C++/CUDA files: `snake_case.cpp`, `snake_case.hpp` or `snake_case.cu`
- Examples: `sirius_interface.cpp`, `gpu_expression_executor.hpp`, `gpu_order_impl.cu`
- Operator classes: `gpu_<operation>_impl.cpp` (e.g., `gpu_aggregate_impl.cpp`, `gpu_partition_impl.cpp`)

**Functions & Methods:**
- PascalCase for public methods (following DuckDB conventions as Sirius integrates with DuckDB)
- snake_case for helper/internal functions
- Examples: `Execute()`, `GetOperatorState()`, `AddExpression()`, `reset()`, `insert_repository()`
- Getter methods: no `get_` prefix; use direct name like `database()`, `get_memory_space()`

**Variables:**
- snake_case for local and member variables
- Prefix/suffix for clarity: `*_idx` for indices, `*_size` for sizes, `*_count` for counters
- Examples: `root_pipeline_idx`, `num_groups`, `estimated_cardinality`, `expected_data`
- Member variables: trailing underscore for private: `config_path_`, `db_`, `had_original_config_env_`

**Classes & Types:**
- PascalCase with `sirius_` or `gpu_` prefix where appropriate
- Examples: `sirius_physical_filter`, `sirius_engine`, `GpuExpressionExecutor`, `gpu_type_traits<TestType>`
- Enum classes: PascalCase, e.g., `SiriusPhysicalOperatorType`, `MemoryBarrierType`

**Constants & Config:**
- UPPER_SNAKE_CASE for compile-time constants and configuration values
- Examples: `DEFAULT_SCAN_TASK_BATCH_SIZE`, `MAX_SORT_PARTITION_BYTES`, `LOG_LEVEL`
- Static member constants in namespace `duckdb::Config`

## Code Style

**Formatting:**
- Tool: clang-format (strict enforcement via pre-commit hooks)
- Config file: `.clang-format`
- Key settings:
  - Column limit: 100
  - Indent width: 2 spaces
  - Pointer alignment: Left (`Type* var`)
  - Brace style: WebKit (opening braces on same line as declaration)
  - Break template declarations: Yes

**Linting:**
- Tool: clang-tidy (integrated via pre-commit)
- Config file: `.clang-tidy`
- Checks enabled: modernize-*, performance-*, clang-analyzer-*
- Enforcement: WarningsAsErrors enabled (violations block commits)
- Notable disabled checks: modernize-use-equals-default, modernize-use-trailing-return-type (stylistic reasons)

**Code Quality Tools:**
- Python formatting: black (via pre-commit)
- CMake formatting: cmake-format with cmake-lint
- Spell check: codespell with custom words in `.codespell_words`

## Import Organization

**Order (enforced by clang-format with IncludeBlocks: Regroup):**
1. Quoted includes (local project files): `"sirius_interface.hpp"`
2. Benchmark/test includes: `<benchmarks/...>`, `<tests/...>`
3. cuDF includes: `<cudf/...>`
4. RAPIDS includes: `<cuml/>`, `<nvtext/>`, etc.
5. RMM includes: `<rmm/...>`
6. CCCL/CUDA includes: `<thrust/>`, `<cub/>`, `<cuda/>`
7. System includes with dots: `<iostream>`, etc.
8. STL includes: `<vector>`, `<string>`, etc.

**Path Aliases:**
- No explicit using aliases found, but `namespace sirius` and `namespace duckdb` are primary namespaces
- Common imports use fully qualified paths: `duckdb::`, `sirius::`, `cucascade::`

## Error Handling

**Patterns:**
- DuckDB integration: use `D_ASSERT()` for assertions (`src/sirius_interface.cpp` line 63-76)
- Exceptions: throw DuckDB exception types: `duckdb::InvalidInputException()`, `duckdb::ErrorData()`
- CUDA/GPU errors: checked via `CUDA_CHECK` macros and cuDF error handling
- Fallback strategy: exceptions trigger graceful fallback to DuckDB CPU execution via `src/fallback.cpp`
- Error context: error messages include query information and location details via `AddErrorLocation()`

**Example from `src/sirius_interface.cpp`:**
```cpp
if (invalidated) {
  if (pending.HasError()) {
    throw duckdb::InvalidInputException(
      "Attempting to execute an unsuccessful pending query result\n");
  }
  throw duckdb::InvalidInputException("Attempting to execute a closed pending query result");
}
```

## Logging

**Framework:** spdlog (via `src/include/log/logging.hpp`)

**Levels:** trace, debug, info, warn, error, critical (mapped to SIRIUS_LOG_*)

**Initialization Pattern:**
```cpp
#include "log/logging.hpp"
// In main or initialization:
InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS);
```

**Usage:**
- Macros: `SIRIUS_LOG_DEBUG("message")`, `SIRIUS_LOG_INFO("format {}", value)`
- Format: spdlog fmt-style with file:line in pattern `[%s:%#]`
- Environment variables:
  - `SIRIUS_LOG_LEVEL`: trace, debug, info, warn, error (default: info)
  - `SIRIUS_LOG_DIR`: directory for logs (default: log/)
  - `SIRIUS_LOG_FLUSH_SECONDS`: flush interval (default: 3)

**Special Case (CUDA):**
- CUDA compilation units (.cu files) define logging macros as no-ops
- Line 19-26 in `logging.hpp`: `#ifdef __CUDACC__` guards prevent spdlog inclusion in CUDA code

## Comments

**When to Comment:**
- Before class/struct definitions: brief description (single line)
- Complex algorithm sections: explain intent, not what the code does
- Non-obvious design choices (e.g., "GPU memory layout optimized for coalesced access")
- License headers: Apache 2.0 on all source files (Copyright 2025, Sirius Contributors)

**JSDoc/TSDoc (Doxygen-compatible):**
- Used sparingly in headers for public APIs
- Example from `logging.hpp`: `@brief` for single-line descriptions
- Function signatures show parameter types clearly

## Function Design

**Size:**
- Range: 20-150 lines typical
- Shorter for utility functions, longer acceptable for complex operators
- Example: `sirius_engine::insert_repository()` at ~40 lines; `GpuExpressionExecutor::Execute()` handling multiple cases

**Parameters:**
- Pass vectors/large objects by reference or unique_ptr
- Small types (int, bool, enum) by value
- Pattern: const-reference for inputs, move semantics for ownership transfer
- Example from test: `make_two_column_batch<int64_t, typename Traits::type>(*space, filter_vals, data_vals, ...)`

**Return Values:**
- Return by value for small types and POD
- Return `duckdb::unique_ptr<T>` for allocated objects (DuckDB convention)
- Return `std::shared_ptr<T>` for shared lifetime (GPU buffers, data_batch)
- Return `std::optional<T>` for optional results

## Module Design

**Exports:**
- Header-only utilities in `src/include/` with `.hpp` extension
- Implementation in `src/` with `.cpp` or `.cu` extension
- Class methods in `src/include/` declared, implemented in `src/`

**Barrel Files:**
- Not extensively used
- Main entry points: `src/sirius_interface.hpp`, `src/include/config.hpp`
- Test utilities: `test/cpp/operator/operator_test_utils.hpp` aggregates utility functions

## Namespacing

**Primary Namespaces:**
- `namespace sirius` - Main engine code, operators, expression executor
- `namespace sirius::op` - Physical operators (`sirius_physical_filter`, etc.)
- `namespace sirius::pipeline` - Pipeline execution infrastructure
- `namespace sirius::memory` - Memory management (reservation manager, cache)
- `namespace sirius::test` - Test utilities and fixtures
- `namespace duckdb` - DuckDB integration layer (config, legacy code)
- `namespace cucascade` - GPU cascade memory library (used for data representations)

**Anonymous Namespaces:**
- Used in .cpp files for private helper functions
- Pattern: unnamed namespace `{}` at file scope (e.g., `src/expression_executor/gpu_expression_executor.cpp` line 44-52)

## Type Conversions & Casts

**Pattern:**
- Avoid C-style casts `(Type)`
- Use `static_cast<>` for safe conversions
- Use `Cast<>()` method on operator base classes for type-safe downcasts
- Example: `next_op->Cast<op::sirius_physical_right_delim_join>()`

## Memory & Ownership

**DuckDB Conventions:**
- Use `duckdb::make_uniq<T>()` instead of `std::make_unique<T>()`
- Use `duckdb::unique_ptr<T>` type alias
- Use `duckdb::shared_ptr<T>` for reference-counted resources

**CUDA/GPU Memory:**
- Use `rmm::device_buffer`, `rmm::device_uvector` for GPU allocations
- Allocators passed via `rmm::device_async_resource_ref`
- Example: `auto mr = get_resource_ref(*space)` in tests

---

*Convention analysis: 2026-04-06*
