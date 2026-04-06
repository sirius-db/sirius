# Testing Patterns

**Analysis Date:** 2026-04-06

## Test Framework

**C++ Unit Tests:**
- Runner: Catch2 (header-only test framework)
- Config: `src/include/catch.hpp` (Catch2 single header)
- Entry point: `test/cpp/unittest.cpp` with custom test listener

**Python Integration Tests:**
- Framework: DuckDB Python API (duckdb module)
- Test file: `test/test_python.py` (loads extension and executes via Python)
- Build requirement: `pixi run -e duckdb-python build-duckdb-python`

**SQL Logic Tests (End-to-End):**
- Format: DuckDB SQLLogicTest (.test files)
- Test files: `test/sql/tpch-sirius.test`, `test/sql/bugfix.test`
- Runner: `build/release/test/unittest --test-dir . test/sql/*.test`

**Run Commands:**
```bash
# All C++ unit tests
make test                              # Release mode
make test_debug                        # Debug mode

# Specific C++ test by tag/name
build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"

# SQL logic tests
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test

# Performance tests
python3 test/tpch_performance/generate_test_data.py {SCALE_FACTOR}
python3 test/tpch_performance/performance_test.py {SCALE_FACTOR}
```

**Test Output Location:**
- C++ test logs: `build/release/extension/sirius/test/cpp/log/`
- Integration test database: `test/cpp/integration/integration.duckdb`

## Test File Organization

**Location:**
- Unit tests: co-located with source in `test/cpp/` mirroring `src/` structure
- Operator tests: `test/cpp/operator/` (e.g., `test_physical_filter.cpp`)
- Expression tests: `test/cpp/expression_executor/`
- Memory tests: `test/cpp/memory/`, `test/cpp/memory_management/`
- Integration tests: `test/cpp/integration/`
- Test utilities: `test/cpp/utils/` and `test/cpp/operator/` (shared headers)

**Naming:**
- Test files: `test_<component>.cpp` (e.g., `test_physical_filter.cpp`)
- Test utilities: `<component>_test_utils.hpp` (e.g., `operator_test_utils.hpp`)
- Shared test env: `sirius_test_env.cpp/hpp`

**Structure:**
```
test/cpp/
├── unittest.cpp                    # Main entry, Catch2 listener registration
├── operator/
│   ├── test_physical_filter.cpp
│   ├── operator_test_utils.hpp    # Shared utilities for operator tests
│   └── aggregate/
│       └── test_physical_grouped_aggregate.cpp
├── utils/
│   ├── sirius_test_env.cpp        # Shared test environment (pause/resume)
│   └── utils.cpp
└── integration/
    └── test_gpu_execution_tpch.cpp
```

## Test Structure

**Suite Organization (Catch2):**
```cpp
TEMPLATE_TEST_CASE("test description",
                   "[tag_name]",
                   int32_t,
                   int64_t,
                   float,
                   double)
{
  using Traits = gpu_type_traits<TestType>;

  // Setup
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);  // Assert setup succeeded

  // Action
  auto result = operator_under_test.execute(input_data, stream);

  // Verify
  REQUIRE(result->get_data_batches().size() == 1);
  // Detailed assertions
}
```

**Test Tags (from `test/cpp/unittest.cpp`):**
- `[shared_context]`: Tests using shared DuckDB/SiriusContext (scan, operator unit tests)
- `[integration]`: GPU execution integration tests using full pipeline
- No tag: Isolated/standalone tests with independent setup

**Patterns:**

1. **Template Test Pattern** (from `test_physical_filter.cpp` lines 38-114):
   - Use `TEMPLATE_TEST_CASE` to test multiple type combinations
   - Trait classes `gpu_type_traits<T>` provide type metadata
   - Tests run once per type variant

2. **Setup Pattern**:
   ```cpp
   auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
   auto* space = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
   REQUIRE(space != nullptr);
   ```

3. **Data Creation Pattern**:
   ```cpp
   auto [input_table, expected_table] =
     sirius::test::make_test_data_for_grouped_aggregate<Traits>(
       num_groups, partition_count, stream, mr);
   std::shared_ptr<data_batch> input_batch = sirius::make_data_batch(
     std::move(input_table), *space);
   ```

4. **Operator Execution Pattern**:
   ```cpp
   auto outputs = operator_instance.execute(
     operator_data(inputs), cudf::get_default_stream());
   REQUIRE(outputs->get_data_batches().size() == 1);
   ```

5. **Assertion Pattern**:
   ```cpp
   auto& output_table = outputs->get_data_batches()[0]
                          ->get_data()
                          ->cast<gpu_table_representation>()
                          .get_table();
   auto host_vals = copy_column_to_host<typename Traits::type>(
     output_table.view().column(1));
   REQUIRE(host_vals == expected_data);
   ```

## Mocking

**Framework:** Manual/dependency injection (no mock library)

**Patterns:**
- Memory manager mocking: Create isolated `sirius_memory_reservation_manager` instances for each test
- DuckDB context mocking: Create isolated `duckdb::DuckDB` and `duckdb::Connection` per test
- Shared environment strategy: Use `shared_test_env` pause/resume to switch between environments without full teardown

**What to Mock:**
- GPU memory manager: Always initialize per test with `initialize_memory_manager()` to control capacity
- DuckDB client context: Create new instance unless test requires persistence

**What NOT to Mock:**
- cuDF operations: Use real cuDF functions with test data
- GPU execution: Tests validate actual GPU operator behavior
- Data representations: Use real `data_batch`, `gpu_table_representation`

**Test Environment Lifecycle (from `unittest.cpp` lines 45-82):**
```cpp
struct shared_env_listener : Catch::TestEventListenerBase {
  void testCaseStarting(Catch::TestCaseInfo const& info) override {
    // Pause environments for this test type
    // Resume the matching environment
  }
};
// Two global shared environments:
sirius::test::g_shared_env      // [shared_context] tests
sirius::test::g_integration_env // [integration] tests
```

## Fixtures and Factories

**Test Data:**
- Factory functions: `sirius::test::operator_utils::make_two_column_batch<T1, T2>()`
- Location: `test/cpp/operator/operator_test_utils.hpp` (lines 93-140)
- Pattern:
  ```cpp
  std::shared_ptr<data_batch> make_two_column_batch<int64_t, float>(
    *space, filter_vals, data_vals, cudf::type_id::FLOAT32, std::nullopt);
  ```

**Test Data Builders:**
- Aggregate test data: `sirius::test::make_test_data_for_grouped_aggregate<Traits>()`
- Aggregate with AVG: `sirius::test::make_test_data_for_grouped_aggregate_with_avg<Traits>()`
- Expression data: Factory pattern with sample values from `Traits::sample_values()`
- Memory fixtures: `sirius::test::operator_utils::initialize_memory_manager()` with preset capacities

**Location:**
- Fixtures in test headers alongside usage
- Shared utilities in `test/cpp/operator/operator_test_utils.hpp`
- Per-component utilities in corresponding test directories

## Coverage

**Requirements:** No enforced coverage target

**View Coverage:** No instrumentation configured

**Coverage Practice:**
- Unit tests: Aim for operator execute() path and edge cases
- Integration tests: Validate end-to-end query execution against expected results
- Performance tests: Validate correctness at scale with real TPC-H/TPC-DS data

## Test Types

**Unit Tests:**
- Scope: Single operator or component (filter, aggregate, sort, etc.)
- Approach: Direct instantiation with crafted data_batch inputs
- Data: Small synthetic datasets (5-1000 rows)
- File pattern: `test/cpp/operator/test_physical_*.cpp`
- Example: `test_physical_filter.cpp` (114 lines, multiple TEMPLATE_TEST_CASEs)

**Operator Tests:**
- Scope: Execute operator, validate output columns and cardinality
- Approach: Type-parameterized tests using traits
- Data: Type-specific sample values from `Traits::sample_values()`
- Validation: Column-by-column comparison after copying to host

**Integration Tests:**
- Scope: Full query execution via GPU execution pipeline
- Approach: Create connection, load extension, execute SQL
- File: `test/cpp/integration/test_gpu_execution_tpch.cpp`, `test_gpu_execution_multi_format.cpp`
- Data: Real TPC-H tables or minimal hand-crafted datasets
- Tag: `[integration]`

**E2E Tests:**
- Framework: SQLLogicTest format (.test files)
- Scope: Query execution with result validation
- File: `test/sql/tpch-sirius.test`, `test/sql/bugfix.test`
- Queries: Real TPC-H queries and regression tests
- Runner: `build/release/test/unittest --test-dir . test/sql/tpch-sirius.test`

**Performance Tests:**
- Framework: Python script with DuckDB Python API
- Scope: Query execution time and memory usage at scale
- Files: `test/tpch_performance/performance_test.py` (runner), `generate_test_data.py` (data gen)
- Data: TPC-H parquet files at configurable scale factor
- Invocation: `python3 test/tpch_performance/performance_test.py {SCALE_FACTOR}`

## Common Patterns

**Async Testing (CUDA streams):**
- Stream handling: `auto stream = cudf::get_default_stream()`
- Synchronization: Implicit via cuDF operations; explicit `stream.synchronize()` if needed
- Pattern from test: Operations passed `cudf::get_default_stream()` to execute methods

**Type-Parameterized Testing (cuDF type system):**
```cpp
TEMPLATE_TEST_CASE("description",
                   "[tag]",
                   int32_t, int64_t, float, double,
                   string_tag, decimal64_tag, timestamp_us_tag, date32_tag)
{
  using Traits = gpu_type_traits<TestType>;
  // Traits::cudf_type, Traits::logical_type(), Traits::sample_values()
}
```

**Error Testing:**
- Pattern: Check exception is thrown or fallback is triggered
- No dedicated error test files found; error paths tested via integration tests
- Approach: Invalid queries or unsupported operations fall through to DuckDB CPU

**Data Type Traits Pattern (from `operator_type_traits.hpp`):**
```cpp
template <typename T>
struct gpu_type_traits {
  using type = T;
  static constexpr cudf::type_id cudf_type = cudf::type_id::INT32;
  static constexpr bool is_string = false;
  static constexpr bool is_decimal = false;
  static std::vector<T> sample_values() { return {1, 2, 3, 5, 7}; }
  duckdb::LogicalType logical_type() { return duckdb::LogicalType::INTEGER; }
};
// Specializations: string_tag, decimal64_tag, timestamp_us_tag, date32_tag
```

## Test Configuration

**Memory Config Files:**
- Scan tests: `test/cpp/scan/memory.cfg`
- Integration tests: `test/cpp/integration/integration.cfg`
- Usage: Passed to `shared_test_env(config_path)` to control GPU/host memory limits

**Environment Variables:**
- `SIRIUS_CONFIG_FILE`: Path to config file (set by test environment)
- `SIRIUS_LOG_LEVEL`: Test log verbosity (default: info)
- `SIRIUS_LOG_DIR`: Log directory for test output
- `SIRIUS_PROJECT_ROOT`: Set at CMake configure time for test asset location

**CMake Integration:**
- Test binary: `build/release/extension/sirius/test/cpp/sirius_unittest`
- Build target: `unittest` (default test build target)
- Parallel build: `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make`

---

*Testing analysis: 2026-04-06*
