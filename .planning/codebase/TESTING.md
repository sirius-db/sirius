# Testing Patterns

**Analysis Date:** 2026-04-06

## Test Framework

**Runner:**
- Catch2 (header-only framework)
- Config: `#define CATCH_CONFIG_RUNNER` in `test/cpp/unittest.cpp`
- Location: `test/cpp/sirius_unittest` (binary produced by CMake)

**Assertion Library:**
- Catch2 macros: `REQUIRE()`, `REQUIRE_NOTHROW()`, `REQUIRE_THROWS()`, `CHECK()`
- DuckDB assertions: `D_ASSERT()` (only for developer assertions, not tests)

**Run Commands:**
```bash
make test                                      # Run all SQLLogicTests
make test_debug                                # Debug build tests

# C++ unit tests (after build)
build/release/extension/sirius/test/cpp/sirius_unittest
build/release/extension/sirius/test/cpp/sirius_unittest "[filter]"              # Run tests tagged [filter]
build/release/extension/sirius/test/cpp/sirius_unittest "test_name"             # Run specific test

# SQL logic tests
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

## Test File Organization

**Location:**
- C++ unit tests: `test/cpp/` organized by component
- SQL logic tests: `test/sql/` (end-to-end)

**Naming:**
- Test files: `test_component_name.cpp` (e.g., `test_physical_filter.cpp`, `test_cpu_cache.cpp`)
- Utility headers: `component_test_utils.hpp` or `component_utils.hpp`
- Example structure:
  - `test/cpp/operator/test_physical_filter.cpp` - Tests for filter operator
  - `test/cpp/operator/operator_test_utils.hpp` - Shared operator test utilities
  - `test/cpp/scan/test_utils.hpp` - Scan-specific test utilities

**Directory Structure:**
```
test/cpp/
├── config/                  # Configuration tests
├── data/                    # Data representation tests
├── downgrade/               # Downgrade executor tests
├── exec/                    # Execution infrastructure tests
├── expression_executor/     # Expression evaluation tests
├── integration/             # End-to-end GPU execution tests
├── memory/                  # Memory management tests
├── memory_management/       # CPU cache and buffer tests
├── operator/                # Physical operator tests
│   ├── aggregate/           # Aggregation operator tests
│   └── *.cpp                # Individual operator tests
├── parallel/                # Parallel execution tests
├── pipeline/                # Pipeline execution tests
├── scan/                    # Data scan tests
├── utils/                   # Test utilities and environment
├── unittest.cpp             # Test runner with listener
└── *.cpp                    # Other unit tests

test/sql/
├── bugfix.test              # Bugfix regression tests
├── tpch-sirius.test         # TPC-H benchmark tests
└── clickbench-sirius.test   # ClickBench tests
```

## Test Structure

**Suite Organization:**
```cpp
// Example from test_physical_filter.cpp

#include "memory/sirius_memory_reservation_manager.hpp"
#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"
#include <catch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <op/sirius_physical_filter.hpp>

using namespace duckdb;
using namespace sirius::op;

namespace {
  using namespace sirius::test::operator_utils;
}

TEMPLATE_TEST_CASE("sirius_physical_filter executes on data_batch for multiple numeric types",
                   "[physical_filter]",
                   int32_t,
                   int64_t,
                   float,
                   double)
{
  // Test body
}
```

**Key Patterns:**
- `TEMPLATE_TEST_CASE()` - Parameterized tests across multiple types
- `TEST_CASE()` - Single test case
- Tag syntax: `"[tag1][tag2]"` - Multiple tags for grouping
- Special tags:
  - `[shared_context]` - Tests using shared DuckDB environment
  - `[integration]` - End-to-end integration tests
  - `[cpu_cache]` - CPU caching tests
  - `[.][cpu_cache]` - Skipped by default, use explicit tag to run
  - `.` prefix - Skipped test (hidden/slow tests)

**Shared Test Environment:**
- Global environments: `sirius::test::g_shared_env` (operator tests), `sirius::test::g_integration_env` (integration tests)
- Listener `shared_env_listener` in `unittest.cpp` manages activation/deactivation
- Tests tagged `[shared_context]` share a single DuckDB instance and SiriusContext
- Tests tagged `[integration]` share separate instance with different config
- Untagged tests get isolated context (slower but independent)

## Test Structure Details

**Setup/Teardown:**
- No explicit setup/teardown macros; inline initialization in test body
- Fixture-style using helper functions: `initialize_memory_manager()`, `initialize_test_buffer_manager()`
- Config environment guards for isolated tests:
  ```cpp
  struct sirius_config_env_guard {
    sirius_config_env_guard(const std::string& config_path) {
      setenv("SIRIUS_CONFIG_FILE", config_path.c_str(), 1);
    }
    ~sirius_config_env_guard() { unsetenv("SIRIUS_CONFIG_FILE"); }
  };
  ```

**Assertion Patterns:**
- Basic assertions: `REQUIRE(condition)` - Test fails if false
- Exception assertions: `REQUIRE_THROWS(statement)` - Verify exception thrown
- No-throw assertions: `REQUIRE_NOTHROW(statement)` - Verify no exception
- Equality: `REQUIRE(actual == expected)` - Can print both sides
- Example:
  ```cpp
  REQUIRE(outputs->get_data_batches().size() == 1);
  REQUIRE(reloaded_column->segment_id == -1);
  REQUIRE_NOTHROW(cpu_cache.moveDataToGPU(copy_chunk_id, true));
  REQUIRE_THROWS(cpu_cache.moveDataToGPU(chunk_id, true));
  ```

## Mocking

**Framework:**
- No explicit mocking library (Google Mock, Catch2 Matchers not used extensively)
- Mocking done via: test doubles (fake implementations), helper classes

**Test Doubles:**
- Memory manager: `sirius::memory::sirius_memory_reservation_manager` with test sizes
  ```cpp
  const size_t gpu_capacity = 512ull << 20;  // 512MB (small for tests)
  const size_t host_capacity = 1ull << 30;   // 1GB
  ```
- Data builders: `make_two_column_batch<T1, T2>()` creates test data
- Buffer manager: `initialize_test_buffer_manager()` with test-appropriate sizes

**What to Mock:**
- GPU memory: Use test-sized memory managers instead of production sizes
- Test data: Generate via type traits and builder functions
- DuckDB connections: Create via shared test env (already isolated)

**What NOT to Mock:**
- cuDF operations: Use real GPU calls (tests run on actual GPUs)
- RMM memory allocator: Use real GPU memory (tests allocate/deallocate)
- cuCascade: Use real reservation manager (tests validate memory handling)
- DuckDB planner/executor: Use real DuckDB (only mock at extension boundary)

## Fixtures and Factories

**Test Data:**
- Type traits: `gpu_type_traits<T>` provides:
  - `logical_type()` - DuckDB type
  - `cudf_type` - cuDF type ID
  - `sample_values()` - Vector of test data
  - `threshold()` - Comparison threshold
  - `is_decimal`, `is_string`, `is_ts` - Type classification

**Factories:**
- `make_two_column_batch<T1, T2>()` - Create 2-column data batches
- `initialize_memory_manager(n_gpus)` - Setup memory space with test sizes
- `create_column_with_random_data(type, num_records, chars_per_record)` - Legacy column creation
- `copy_column_to_host<T>()` - Copy GPU column to host for verification

**Location:**
- Test utilities: `test/cpp/utils/sirius_test_env.hpp`, `test/cpp/operator/operator_test_utils.hpp`
- Scan utilities: `test/cpp/scan/test_utils.hpp`
- Type traits: `test/cpp/operator/operator_type_traits.hpp`
- Helper classes: `test/cpp/utils/utils.hpp`, `test/cpp/operator/aggregate/aggregate_test_utils.hpp`

## Coverage

**Requirements:**
- No explicit coverage target enforced
- Coverage measurement: Optional (not required by build)

**View Coverage:**
```bash
# Not currently configured in project
# Could be added via lcov or similar if needed
```

**Coverage Gaps:**
- Legacy code paths (GPU processing, gpu_executor.cpp, etc.) have lower priority
- Focus on Super Sirius (namespace sirius) code coverage
- GPU-specific paths harder to cover (need GPU hardware)

## Test Types

**Unit Tests:**
- **Scope:** Single component (operator, expression executor, memory manager)
- **Approach:**
  - Isolated: Create minimal test data for component
  - Fast: Run on small datasets (1KB-1MB)
  - Deterministic: Same input always produces same output
  - Examples:
    - `test/cpp/operator/test_physical_filter.cpp` - Filter operator in isolation
    - `test/cpp/config/test_config.cpp` - Configuration parsing
    - `test/cpp/expression_executor/test_gpu_expression_executor.cpp` - Expression evaluation

**Integration Tests:**
- **Scope:** Multiple components working together; full query execution
- **Approach:**
  - Full pipeline: DuckDB planner -> Sirius -> GPU execution -> result
  - Test data: TPC-H or multi-format datasets
  - Tag: `[integration]`
  - Slower but validates end-to-end correctness
  - Examples:
    - `test/cpp/integration/test_gpu_execution_tpch.cpp` - TPC-H queries via GPU
    - `test/cpp/integration/test_gpu_execution_multi_format.cpp` - Multiple data formats

**E2E Tests (SQL Logic Tests):**
- **Framework:** SQLLogicTest format (`.test` files)
- **Scope:** Entire query execution via SQL
- **Files:**
  - `test/sql/bugfix.test` - Regressions for fixed bugs
  - `test/sql/tpch-sirius.test` - TPC-H query suite
  - `test/sql/clickbench-sirius.test` - ClickBench query suite
- **Format:**
  ```
  # Load extensions
  require sirius

  # Setup
  statement ok
  call gpu_buffer_init("1 GB", "2 GB");

  # Test queries
  query I
  call gpu_processing("SELECT rowID FROM table WHERE condition;");
  ----
  expected_result_rows
  ```
- **Run:** `build/release/test/unittest --test-dir . test/sql/file.test`

## Common Patterns

**Type-Parameterized Tests:**
```cpp
TEMPLATE_TEST_CASE("Filter executes for multiple types",
                   "[physical_filter]",
                   int32_t, int64_t, float, double,
                   int16_t, bool, string_tag, date32_tag)
{
  using Traits = gpu_type_traits<TestType>;
  auto data_vals = Traits::sample_values();
  // ... test using Traits::logical_type(), Traits::cudf_type, etc.
}
```

**Async Testing:**
```cpp
// GPU operations are synchronous in tests (streams wait before return)
auto outputs = filter.execute(operator_data(inputs), cudf::get_default_stream());
// Stream is synchronized by CUDA stream semantics (operations complete before next statement)

// For async patterns in pipelines, tasks are submitted to executor:
// task_executor.submit(task);
// task_executor.wait_all();
```

**Error Testing:**
```cpp
// Exception assertions
REQUIRE_THROWS(cpu_cache.moveDataToGPU(evicted_chunk_id, true));

// CUDA error checking
verify_cuda_errors("CUDA Errors in test");

// Assertion on conditions
D_ASSERT(sirius_active_query->is_open_result(pending));
```

**Memory Verification:**
```cpp
// Verify GPU column data after operation
auto host_vals = copy_column_to_host<int32_t>(gpu_column);
REQUIRE(host_vals == expected_values);

// Verify column structure
REQUIRE(reloaded_column->segment_id == -1);
```

## Test Logging

**Output:**
- Logs saved to: `build/release/extension/sirius/test/cpp/log/`
- Controlled by: `Config::LOG_DIR` and `Config::LOG_LEVEL`
- Set in `unittest.cpp`:
  ```cpp
  std::string log_dir = SIRIUS_UNITTEST_LOG_DIR;
  Config::LOG_DIR = log_dir;
  InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS);
  ```

**Debugging:**
- Use `SIRIUS_LOG_DEBUG(...)` within source code to trace execution
- Examine logs in `log/sirius.log` after test run
- For pipeline debugging: `tools/parse_pipeline_log.py` parses task execution logs

## Performance Testing

**Framework:**
- Manual performance tests in `test/tpch_performance/`
- TPC-H data generation and execution timing

**Run Performance Tests:**
```bash
python3 test/tpch_performance/generate_test_data.py {SCALE_FACTOR}
python3 test/tpch_performance/performance_test.py {SCALE_FACTOR}
# Requires: pixi run -e duckdb-python build-duckdb-python
```

## Test Configuration

**Config Files:**
- `test/cpp/integration/integration.cfg` - Integration test GPU config
- `test/cpp/scan/memory.cfg` - Scan test memory config
- Format: libconfig++ (C++ configuration file library)

**Environment Variables:**
- `SIRIUS_CONFIG_FILE` - Path to config file (set by test guard)
- `SIRIUS_INTEGRATION_TEST_DB_PATH` - Path to TPC-H database
- `SIRIUS_LOG_LEVEL` - Log level for tests
- `SIRIUS_LOG_DIR` - Log output directory

---

*Testing analysis: 2026-04-06*
