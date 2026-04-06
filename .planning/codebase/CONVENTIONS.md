# Coding Conventions

**Analysis Date:** 2026-04-06

## Naming Patterns

**Files:**
- Snake case for all files: `sirius_physical_filter.cpp`, `sirius_physical_filter.hpp`
- Headers use `.hpp` extension
- Implementation files use `.cpp` extension
- CUDA kernels use `.cu` extension
- Organized by functional module: `src/op/`, `src/pipeline/`, `src/planner/`

**Functions and Methods:**
- Snake case for function names: `initialize_memory_manager()`, `execute()`, `reset()`
- Public methods in classes follow lowercase_snake_case
- Private helper functions use snake_case with leading underscore rarely used
- Constructors follow class naming convention

**Variables:**
- Local variables: lowercase with underscores: `filter_vals`, `input_batch`, `gpu_space`
- Member variables (class): no prefix convention, just lowercase_snake_case: `expression`, `sirius_pipeline`
- Static/constant members: UPPERCASE for macros only
- Loop counters: single letter `i`, `j` acceptable for short loops
- Pointers and references: `*` and `&` attach to type, not variable: `int* ptr`, `int& ref`

**Types and Classes:**
- Class names: lowercase_snake_case: `sirius_physical_filter`, `data_batch`, `memory_reservation_manager`
- Struct names: lowercase_snake_case when used as templates or data holders
- Enum values: UPPERCASE_SNAKE_CASE when applicable (varies by enum)
- Type aliases: lowercase_snake_case: `gpu_table_representation`, `shared_data_repository`
- Namespaces: lowercase: `sirius`, `sirius::op`, `sirius::pipeline`

**Configuration and Constants:**
- Global config variables: UPPERCASE: `Config::USE_CUDF_EXPR`, `Config::LOG_LEVEL`
- Constexpr values: lowercase_snake_case: `TEST_BUFFER_MANAGER_MEMORY_BYTES`

## Code Style

**Formatting:**
- Tool: `clang-format` (configured in `.clang-format`)
- Column limit: 100 characters
- Indentation: 2 spaces
- No tabs (`UseTab: Never`)
- Brace style: WebKit (opening braces stay on line)

**Key Formatting Rules (from .clang-format):**
```cpp
// Breaking style for function declarations
BreakConstructorInitializers: BeforeColon
BreakInheritanceList: BeforeColon

// Alignment
AlignAfterOpenBracket: Align
AlignTrailingComments: true
AlignConsecutiveAssignments: true
AlignConsecutiveMacros: true

// Constructor initialization
ConstructorInitializerAllOnOneLineOrOnePerLine: true

// Pointer alignment
PointerAlignment: Left  // int* ptr, not int *ptr

// Template spacing
SpaceAfterTemplateKeyword: true
```

**Linting:**
- Tool: `clang-tidy` (configured in `.clang-tidy`)
- Pre-commit hook enforces formatting via `clang-format`
- Code style verification: `pre-commit run -a`

**Spell Check:**
- Tool: `codespell` (configured in `.codespell_words`)
- Custom allowed words in `.codespell_words`

## Import Organization

**Order (from .clang-format IncludeCategories):**
1. Quoted includes (project local): `"config.hpp"`
2. Benchmark/test includes: `<benchmarks/...>`, `<tests/...>`
3. cuDF includes: `<cudf/...>`, `<cudf_test/...>`
4. Other RAPIDS: `<cugraph/...>`, `<cuml/...>`, `<raft/...>`
5. RMM includes: `<rmm/...>`
6. CCCL includes: `<thrust/...>`, `<cub/...>`, `<cuda/...>`
7. CUDA includes: `<cooperative_groups>`, `<cuda.h>`, `<device_types>`
8. System includes with dots: `<sys/types.h>`
9. STL includes: `<vector>`, `<string>`, `<memory>`

**Path Aliases:**
- Project includes are quoted and relative to include directories
- External library includes are angle-bracketed
- Local project includes include hierarchy: `"op/sirius_physical_filter.hpp"`, `"log/logging.hpp"`

## Error Handling

**Assertion Patterns:**
- DuckDB assertions: `D_ASSERT()` for internal preconditions
- Catch2 test assertions: `REQUIRE()` for test failures
- Runtime errors: `throw std::runtime_error("message")`
- CUDA error checking: `verify_cuda_errors("context")`

**Exception Usage:**
- Throw `std::runtime_error` for runtime failures: `throw std::runtime_error("Cannot concatenate empty batch list")`
- Throw `std::invalid_argument` for parameter validation
- Catch exceptions at boundaries (DuckDB integration)
- No exception specifications on functions

**Validation:**
- Input validation at public function entry points
- Precondition checks with `D_ASSERT()` in internal functions
- Error messages should be descriptive and include context

## Logging

**Framework:** spdlog (via macros in `src/include/log/logging.hpp`)

**Macros:**
```cpp
SIRIUS_LOG_TRACE(...)   // Trace-level logging
SIRIUS_LOG_DEBUG(...)   // Debug-level logging
SIRIUS_LOG_INFO(...)    // Info-level logging
SIRIUS_LOG_WARN(...)    // Warning-level logging
SIRIUS_LOG_ERROR(...)   // Error-level logging
SIRIUS_LOG_FATAL(...)   // Critical/fatal errors
```

**Usage Patterns:**
- Configure logging level via `Config::LOG_LEVEL` (default: "info")
- Log directory via `Config::LOG_DIR`
- Flush interval via `Config::LOG_FLUSH_SECONDS`
- Initialize in main: `InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS)`
- No logging in CUDA device code (wrapped in `#ifdef __CUDACC__` guards)

**When to Log:**
- Entry/exit of major functions: pipeline execution, operator execution
- Configuration changes and settings applied
- Memory allocation/deallocation at significant milestones
- Task scheduling and completion
- Error conditions (before throwing or handling)
- Performance-critical sections (sparingly at debug level)

## Comments

**When to Comment:**
- Complex algorithms or non-obvious logic
- Rationale for unusual design decisions
- Workarounds or known limitations
- Per-function documentation in headers

**JSDoc/Doxygen Style:**
- Use `//!` for Doxygen documentation in headers
- Document public functions and classes
- Include parameter descriptions and return values
- Example from `sirius_physical_filter.hpp`:
```cpp
//! sirius_physical_filter represents a filter operator. It removes non-matching tuples
//! from the result. Note that it does not physically change the data, it only
//! adds a selection vector to the chunk.
```

**Inline Comments:**
- Use `//` for single-line explanations
- No trailing comments on same line unless very brief
- Comments stay above the code they describe when practical

## Function Design

**Size:**
- Keep functions under 100 lines when reasonable
- Single responsibility: one job per function
- Extract complex logic into helper functions

**Parameters:**
- Pass non-copyable types by reference: `const operator_data& input_data`
- Pass objects by const reference when not modified: `const std::vector<int>& values`
- Pass small copyable types by value: `int count`, `bool flag`
- Avoid passing raw pointers; use references or smart pointers instead

**Return Values:**
- Use `std::unique_ptr<T>` for heap-allocated single ownership: `std::unique_ptr<operator_data> execute(...)`
- Use `std::shared_ptr<T>` for shared ownership across threads: `std::shared_ptr<data_batch>`
- Return status via exceptions or result types; use `bool` only for simple queries
- Use `std::optional<T>` for optional values: `std::optional<float> float_tolerance`

**Modern C++ Features:**
- C++20 standard
- Use auto for type inference where type is obvious: `auto space = memory_manager->get_memory_space(...)`
- Structured bindings for tuples: `auto [db_owner, connection] = make_test_db_and_connection()`
- std::unique_ptr and std::shared_ptr for memory management
- No raw `new`/`delete` outside memory managers
- Range-based for loops: `for (const auto& batch : input_batches) { ... }`

## Module Design

**Exports:**
- Header files in `src/include/` define public interfaces
- Implementation in `src/` (mirrors include structure)
- Use anonymous namespace or `static` for file-local symbols
- Namespace organization: `sirius::op::`, `sirius::pipeline::`, `duckdb::`

**Barrel Files:**
- Not commonly used; prefer explicit imports
- Some headers in `src/include/operator/` group related types
- Example: operator type traits included via specific header

**Header Guards:**
- Use `#pragma once` at top of all headers (not `#ifndef` guards)
- Placed immediately after license comment

## Specific Conventions

**CUDA Code:**
- `.cu` files in `src/cuda/` and subdirectories
- Host-side logic in `.cpp`, device code in `.cu`
- Use `rmm::cuda_stream_view` for stream management
- Kernels wrapped via cuDF/cuCascade abstractions

**GPU Type System:**
- Use `cudf::data_type` for cuDF type representation
- DuckDB types via `duckdb::LogicalType` and `duckdb::LogicalTypeId`
- Template metaprogramming for type traits (e.g., `gpu_type_traits<TestType>`)

**String Literals:**
- Use double quotes for C++ strings: `"config setter test"`
- Raw string literals for complex patterns: `R"(regex_pattern)"`

**Namespace Conventions:**
- Top-level: `duckdb::` (legacy), `sirius::` (new)
- Sub-namespaces follow module structure: `sirius::op::`, `sirius::pipeline::`, `sirius::memory::`
- Anonymous namespaces for file-local helpers
- Using declarations in header files for convenience (rarely)

---

*Convention analysis: 2026-04-06*
