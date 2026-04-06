# Testing Patterns

**Analysis Date:** 2026-04-06

## Test Framework

**Runner:**
- Framework: Catch2
- Config: Compiled into single binary `build/release/extension/sirius/test/cpp/sirius_unittest`
- Main test driver: `test/cpp/unittest.cpp`

**Assertion Library:**
- Catch2 built-in assertions
- Custom helpers from test utilities (`test/cpp/utils/utils.hpp`, `test/cpp/operator/operator_test_utils.hpp`)

**Run Commands:**
```bash
# Run all C++ unit tests
build/release/extension/sirius/test/cpp/sirius_unittest

# Run tests with specific tag
build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"

# Run specific test by name
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"

# Run SQL logic tests (end-to-end)
make test

# Run with debug build
make test_debug

# Run specific SQL test
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

**Test Logs:**
- Location: `build/release/extension/sirius/test/cpp/log`
- Log level configured via `SIRIUS_LOG_LEVEL` environment variable
- Directory configured via `SIRIUS_LOG_DIR` environment variable

## Test File Organization

**Location:**
- Unit tests co-located with functionality being tested
- Test directory mirrors source structure: `test/cpp/operator/`, `test/cpp/pipeline/`, `test/cpp/config/`
- Integration tests: `test/cpp/integration/`
- Utilities: `test/cpp/utils/`, `test/cpp/operator/`, `test/cpp/scan/`

**Naming:**
- Test files: `test_<component>.cpp`
- Examples: `test_physical_filter.cpp`, `test_config.cpp`, `test_gpu_pipeline_executor.cpp`

**Structure:**
```
test/cpp/
├── config/                          # Configuration tests
├── operator/                        # Physical operator unit tests
│   ├── operator_test_utils.hpp     # Test utilities for operators
│   ├── aggregate/                  # Aggregate operator tests
│   └── test_physical_filter.cpp
├── integration/                     # End-to-end integration tests
│   ├── test_gpu_execution_tpch.cpp
│   └── test_gpu_execution_multi_format.cpp
├── pipeline/                        # Pipeline execution tests
├── memory_management/               # Memory management tests
├── scan/                           # Scan operator tests
│   ├── test_utils.hpp
│   └── test_parquet_scan_task.cpp
└── utils/                          # Shared test utilities
    ├── utils.hpp
    ├── utils.cpp
    └── data_utils.hpp
```

## Test Structure

**Basic Test Case (Catch2):**
```cpp
TEST_CASE("test_cpu_cache_basic_fixed_single_col", "[.][cpu_cache]")
{
  // Setup
  size_t num_records = 1024;
  GPUBufferManager* gpuBufferManager = initialize_test_buffer_manager();

  // Exercise
  auto gpu_column = create_column_with_random_data(GPUColumnTypeId::INT32, num_records);
  uint32_t chunk_id = cpu_cache.moveDataToCPU(relationship);

  // Verify
  REQUIRE(chunk_id == 0);
  verify_gpu_column_equality(reloaded_column, gpu_column);
}
```

**Template Test (Type-Parameterized):**
```cpp
TEMPLATE_TEST_CASE("sirius_physical_filter executes on data_batch for multiple numeric types",
                   "[physical_filter]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   decimal64_tag,
                   string_tag,
                   timestamp_us_tag,
                   date32_tag)
{
  using Traits = gpu_type_traits<TestType>;

  // Test body uses TestType via type traits
  auto data_vals = Traits::sample_values();
  auto input_batch = make_two_column_batch<int64_t, typename Traits::type>(
    *space, filter_vals, data_vals, Traits::cudf_type, std::nullopt);

  REQUIRE(host_vals == expected_data);
}
```

**Patterns:**
- **Setup:** Initialize dependencies (memory managers, test data, fixtures)
- **Exercise:** Call function under test with prepared inputs
- **Verify:** Assert outputs match expectations using `REQUIRE()`, `REQUIRE_FALSE()`, `REQUIRE_THROWS_AS()`, `REQUIRE_NOTHROW()`
- **Cleanup:** Implicit via destructors and RAII

## Test Tags and Environments

**Test Tags (in brackets after test name):**
- `[physical_filter]` - Component identifier
- `[config_opt]` - Feature area
- `[basic]`, `[optional]`, `[complex]` - Test variant/scope
- `[cpu_cache]` - Specific functionality area
- `[integration]` - Integration test requiring shared environment
- `[shared_context]` - Unit test requiring DuckDB shared context
- `[.]` - Disabled test (skipped by default)

**Shared Test Environments (from unittest.cpp):**
```cpp
struct shared_env_listener : Catch::TestEventListenerBase {
  enum class env_need { NONE, SHARED, INTEGRATION };

  // Three environment tiers:
  // [shared_context] → g_shared_env (scan/operator unit tests)
  // [integration]    → g_integration_env (GPU execution integration tests)
  // anything else    → no env active (isolated tests)
};
```

- Only one shared environment active at a time (extension lock contention)
- Listener pauses/resumes environments as needed
- Consecutive tests with same tag share single DuckDB instance without teardown
- Isolated tests (no tag) have no active environment

## Mocking

**Framework:** Minimal mocking; most tests use real objects

**Patterns:**
- **In-place memory initialization:** `initialize_memory_manager()` creates configured GPU space
- **Test data creation:** Helper functions create realistic test data
- **Converter registry reset:** `sirius::converter_registry::reset_for_testing()` prevents cross-test leakage

**Fixture Pattern:**
```cpp
class GPUExecutionFixtureBase {
 public:
  GPUExecutionFixtureBase() {
    // Use shared env if active, fallback to isolated DuckDB
    if (sirius::test::g_integration_env && g_integration_env->is_active()) {
      con = std::make_unique<duckdb::Connection>(g_integration_env->make_connection());
    } else {
      db  = std::make_unique<duckdb::DuckDB>(nullptr);
      con = std::make_unique<duckdb::Connection>(*db);
    }
  }

  void compare_gpu_vs_cpu(const std::string& query, std::optional<float> float_tolerance = std::nullopt) {
    con->Query("SET enable_duckdb_fallback = false;");
    // Compare GPU results against CPU execution
  }
};
```

**What to Mock:**
- Complex external dependencies (file I/O for determinism)
- Environment-specific behavior (file paths, configuration)
- Third-party services (not present in this codebase)

**What NOT to Mock:**
- GPU memory management (use real cuCascade reservation manager)
- cuDF operations (call actual cuDF functions)
- DuckDB execution (use real DuckDB for integration tests)
- Operator logic (test actual implementations)

## Test Utilities

**Test Data Creation (from operator_test_utils.hpp):**
```cpp
// Initialize GPU memory manager with configured spaces
std::unique_ptr<sirius_memory_reservation_manager> initialize_memory_manager(size_t n_gpus = 1);

// Get default GPU memory space (cached)
cucascade::memory::memory_space* get_default_gpu_space();

// Create test data_batch with multiple columns
std::shared_ptr<cucascade::data_batch> make_two_column_batch<T1, T2>(
  memory_space& space,
  std::vector<T1>& vals1,
  std::vector<T2>& vals2,
  cudf::data_type type2,
  std::optional<int> scale);

// Concatenate multiple data_batch objects horizontally
std::shared_ptr<cucascade::data_batch> concatenate_batches_horizontal(
  const std::vector<std::shared_ptr<data_batch>>& batches,
  memory_space& space);
```

**Legacy Test Utilities (from utils.hpp):**
```cpp
// Buffer manager
GPUBufferManager* initialize_test_buffer_manager();

// Column creation
duckdb::shared_ptr<GPUColumn> create_column_with_random_data(
  GPUColumnTypeId col_type,
  size_t num_records,
  size_t chars_per_record = 1,
  size_t num_materialize_row_ids = 0,
  bool has_null_mask = false);

// Verification
void verify_gpu_column_equality(shared_ptr<GPUColumn> col1, shared_ptr<GPUColumn> col2);
void verify_cuda_errors(const char* msg);

// Random data
GPUBufferManager* initialize_test_buffer_manager();
std::mt19937_64& global_rng();
```

**DuckDB/Connection Helper:**
```cpp
std::pair<std::unique_ptr<duckdb::DuckDB>, duckdb::Connection> make_test_db_and_connection();
```
- Returns shared DuckDB connection if shared environment active
- Creates isolated DuckDB instance otherwise
- Simplifies switching between shared and standalone test modes

## Test Types

**Unit Tests:**
- **Scope:** Single operator or component (filter, aggregate, join, etc.)
- **Location:** `test/cpp/operator/`, `test/cpp/config/`, `test/cpp/memory/`
- **Approach:** Test in isolation with mocked/initialized dependencies
- **Example:** `test_physical_filter.cpp` tests filter operator with various data types
- **Setup time:** < 1 second

**Integration Tests:**
- **Scope:** Full query execution through GPU execution pipeline
- **Location:** `test/cpp/integration/`
- **Approach:** Use real DuckDB, real cuDF, full memory management
- **Example:** `test_gpu_execution_tpch.cpp` runs TPC-H queries and compares GPU vs CPU results
- **Setup time:** 1-5 seconds per test

**End-to-End SQL Tests:**
- **Framework:** SQLLogicTest format
- **Location:** `test/sql/*.test`
- **Approach:** SQL queries with expected results
- **Run:** `make test` or `build/release/test/unittest --test-dir . test/sql/tpch-sirius.test`
- **Setup time:** Minutes (one-time dataset generation)

## Common Patterns

**Async Testing (GPU operations):**
```cpp
// Stream management
rmm::cuda_stream_view stream = cudf::get_default_stream();

// Execute with stream parameter
auto outputs = filter.execute(operator_data(inputs), stream);

// GPU operations are sync'd implicitly in test context
verify_gpu_column_equality(reloaded_column, gpu_column);  // No explicit sync needed
```

**Error Testing:**
```cpp
// Expect exception
REQUIRE_THROWS_AS(
  setter.apply(libconfig.getRoot()),
  std::runtime_error
);

// Expect no exception
REQUIRE_NOTHROW(
  setter.apply(libconfig.getRoot())
);

// Expect condition false
REQUIRE_FALSE(int_value.has_value());
```

**Type Parameterization (for multiple types):**
```cpp
TEMPLATE_TEST_CASE("test name", "[tag]", int32_t, int64_t, float, double, ...) {
  using Traits = gpu_type_traits<TestType>;

  // Use Traits to get:
  auto type_value = Traits::sample_values();      // Sample data
  auto cudf_type = Traits::cudf_type;             // cuDF type ID
  bool is_string = Traits::is_string;             // Type properties
  auto logical_type = Traits::logical_type();     // DuckDB type
}
```

**Floating Point Comparison:**
```cpp
// Approx macro for floating point equality
REQUIRE(double_value == Approx(6.28));

// Custom tolerance in GPU vs CPU comparison
void compare_gpu_vs_cpu(
  const std::string& query,
  std::optional<float> float_tolerance = std::nullopt  // e.g., 0.01f
);
```

**Memory Configuration in Tests:**
```cpp
// Test constants from utils.hpp
constexpr size_t TEST_BUFFER_MANAGER_MEMORY_BYTES = 2L * 1024L * 1024L * 1024L;  // 2 GB

// Memory setup in fixtures
const size_t gpu_capacity = 512ull << 20;  // 512 MB
const double limit_ratio = 0.75;
builder.set_gpu_usage_limit(gpu_capacity / n_gpus)
       .set_reservation_fraction_per_gpu(limit_ratio);
```

## Coverage

**Requirements:** Not enforced (no minimum coverage target)

**Coverage Gaps:**
- Most operator unit tests have good coverage of core logic
- Some error paths and edge cases not tested
- Performance/regression testing via TPC-H benchmarks (separate from unit tests)

**Adding Coverage:**
- Add test case to existing test file
- Create new `test_<component>.cpp` for new components
- Use template test case for type coverage
- Integration tests for cross-operator scenarios

## Best Practices

**Test Reliability:**
- Tests must be deterministic (no random failure)
- Avoid timing-dependent assertions
- Use `REQUIRE_NOTHROW()` for expected no-throw paths
- Reset global state between tests (converter registry, config state)

**Test Clarity:**
- Test names describe what is tested: `"test_cpu_cache_basic_fixed_single_col"`
- Use tags to group related tests: `[cpu_cache]`, `[integration]`
- Keep test bodies focused on single scenario
- Avoid complex setup logic in test body; extract to helpers

**Test Performance:**
- Mark slow tests with `[.]` to skip by default: `TEST_CASE("slow test", "[.][integration]")`
- Reuse shared environments (tagged `[shared_context]` or `[integration]`) to avoid startup overhead
- Isolated tests (untagged) pay full startup cost; use sparingly

**Test Maintenance:**
- Test utilities in `test/cpp/utils/` and `test/cpp/operator/operator_test_utils.hpp`
- Shared test data generators in utility headers
- Update tests when component contracts change
- Document non-obvious test scenarios in comments

---

*Testing analysis: 2026-04-06*
