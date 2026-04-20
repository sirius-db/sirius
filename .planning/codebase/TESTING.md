# Testing Patterns

**Analysis Date:** 2025-04-02

## Test Framework

**Runner:**
- Catch2 framework (header-only)
- Config: Defined via `#define CATCH_CONFIG_RUNNER` in test runner (`test/cpp/unittest.cpp`)
- Invoked: `build/release/extension/sirius/test/cpp/sirius_unittest`

**Assertion Library:**
- Catch2 macros: `REQUIRE()`, `REQUIRE_FALSE()`, `REQUIRE_THROWS()`, `REQUIRE_THROWS_AS()`, `CHECK()`, `UNSCOPED_INFO()`
- Approx comparisons: `REQUIRE(value == Approx(6.28))` for floating point

**Run Commands:**
```bash
# Run all tests
build/release/extension/sirius/test/cpp/sirius_unittest

# Run tests with specific tag
build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"

# Run specific test by name
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"

# Run with pattern matching
build/release/extension/sirius/test/cpp/sirius_unittest "[integration]"
```

## Test File Organization

**Location:**
- Unit tests: `test/cpp/` with subdirectories by component
- SQL logic tests: `test/sql/*.test` (DuckDB SQLLogicTest format)
- Performance tests: `test/tpch_performance/` (Python-based)

**Naming:**
- C++ unit test: `test_*.cpp` (e.g., `test_config.cpp`, `test_cpu_cache.cpp`)
- SQL tests: `*-sirius.test` or `*_sirius.test` (e.g., `tpch-sirius.test`, `clickbench-sirius.test`)

**Directory structure:**
```
test/cpp/
├── config/           # Config and context tests
├── creator/          # Task creator tests
├── data/             # Data representation tests
├── downgrade/        # Memory downgrade executor tests
├── exec/             # Execution framework tests
├── expression_executor/  # GPU expression evaluation tests
├── integration/       # End-to-end GPU execution tests (with shared DuckDB instance)
├── memory_management/ # Memory management and caching tests
├── operator/         # Individual operator tests
├── parallel/         # Thread pool and parallelism tests
├── pipeline/         # Pipeline execution tests
├── scan/             # Scan executor tests
├── utils/            # Test utilities and fixtures
└── unittest.cpp      # Main test runner with Catch2 listener
```

## Test Structure

**Catch2 test case format:**
```cpp
TEST_CASE("descriptive test name", "[tag1][tag2]")
{
  // Setup
  // Action
  REQUIRE(condition);
  // Teardown (RAII handles this)
}
```

**Tagged test execution:**
- Tag format: `[tag_name]` in test case definition
- Examples: `[cpu_cache]`, `[shared_context]`, `[integration]`
- Special tags:
  - `[shared_context]` — Uses shared DuckDB instance across multiple tests (managed by `shared_env_listener`)
  - `[integration]` — Full GPU execution tests requiring GPU memory
  - `[.][tag]` — Disabled test (run with explicit tag only)

**Sections for sub-tests:**
```cpp
TEST_CASE("parent test", "[tag]")
{
  // Setup shared across sections
  
  SECTION("sub-test 1") {
    REQUIRE(...);
  }
  
  SECTION("sub-test 2") {
    REQUIRE(...);
  }
}
```

## Shared Test Environments

**Purpose:**
- Reuse DuckDB/Sirius instances across multiple unit tests
- Avoid repeated GPU memory initialization overhead
- Manage exclusive access to extension lock (only one environment active at a time)

**Architecture** (`test/cpp/unittest.cpp`):
```cpp
struct shared_env_listener : Catch::TestEventListenerBase {
  enum class env_need { NONE, SHARED, INTEGRATION };
  
  static env_need classify(Catch::TestCaseInfo const& info);
  void testCaseStarting(Catch::TestCaseInfo const& info) override;
};

CATCH_REGISTER_LISTENER(shared_env_listener)
```

**Listener behavior:**
- Pauses environments that don't match current test tag
- Resumes environment matching test tag
- Consecutive tests with same tag share single DuckDB instance (no teardown between tests)
- Test types: `g_shared_env` (operator tests), `g_integration_env` (full GPU execution)

**Accessing shared environment** (`test/cpp/integration/test_gpu_execution_tpch.cpp`):
```cpp
class GPUExecutionFixtureBase {
 public:
  GPUExecutionFixtureBase() {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con = std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      // Fallback for standalone test run
      db  = std::make_unique<duckdb::DuckDB>(nullptr);
      con = std::make_unique<duckdb::Connection>(*db);
    }
  }
};
```

## Test Patterns

**Config test pattern** (`test/cpp/config/test_config.cpp`):
```cpp
TEST_CASE("use configuration basic setters", "[config_opt][basic]")
{
  using namespace sirius;
  config::configuration_setter setter;
  int int_value = 0;
  double double_value = 0.0;
  
  setter.add_config("int_value", int_value);
  setter.add_config("double_value", double_value);
  
  libconfig::Config libconfig;
  libconfig.readString(R"(int_value = 100; double_value = 6.28;)");
  
  try {
    setter.apply(libconfig.getRoot());
  } catch (const std::exception& e) {
    std::cerr << "Setting not found: " << e.what() << std::endl;
  }
  
  REQUIRE(int_value == 100);
  REQUIRE(double_value == Approx(6.28));
}
```

**GPU execution comparison pattern** (`test/cpp/integration/test_gpu_execution_tpch.cpp`):
```cpp
class GPUExecutionFixtureBase {
  void compare_gpu_vs_cpu(const std::string& query,
                          std::optional<float> float_tolerance = std::nullopt) {
    con->Query("SET enable_duckdb_fallback = false;");
    
    // Run GPU path
    auto gpu_sql    = "CALL gpu_execution(\"" + query + "\")";
    auto gpu_result = con->Query(gpu_sql);
    REQUIRE(gpu_result);
    REQUIRE_FALSE(gpu_result->HasError());
    
    // Run CPU baseline
    auto cpu_result = con->Query(query);
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());
    
    // Compare row counts and column counts
    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());
    
    // Sort both results for deterministic comparison
    auto ncols = gpu_result->ColumnCount();
    std::string order_clause = " ORDER BY ";
    for (duckdb::idx_t c = 0; c < ncols; c++) {
      if (c > 0) order_clause += ", ";
      order_clause += std::to_string(c + 1);
    }
    
    auto gpu_sorted = con->Query("SELECT * FROM gpu_execution(\"" + query + "\")" + order_clause);
    auto cpu_sorted = con->Query("SELECT * FROM (" + query + ") t" + order_clause);
    
    // Value-by-value comparison with floating point tolerance
    for (duckdb::idx_t r = 0; r < gpu_sorted->RowCount(); r++) {
      for (duckdb::idx_t c = 0; c < gpu_sorted->ColumnCount(); c++) {
        auto gpu_value = gpu_sorted->GetValue(c, r);
        auto cpu_value = cpu_sorted->GetValue(c, r);
        
        if (float_tolerance.has_value() && is_floating_point(gpu_value.type().id())) {
          double gpu_d = gpu_value.GetValue<double>();
          double cpu_d = cpu_value.GetValue<double>();
          double diff  = std::fabs(gpu_d - cpu_d);
          REQUIRE(diff <= static_cast<double>(float_tolerance.value()));
        } else {
          REQUIRE(gpu_value.ToString() == cpu_value.ToString());
        }
      }
    }
  }
};
```

**Fixture inheritance pattern:**
```cpp
class GPUExecutionDuckDBFixture : public GPUExecutionFixtureBase {
 public:
  GPUExecutionDuckDBFixture() {
    auto db_path = get_tpch_db_path().string();
    auto result = con->Query("ATTACH IF NOT EXISTS '" + db_path + "' AS tpch (READ_ONLY);");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }
};

TEST_CASE_METHOD(GPUExecutionDuckDBFixture, "test name", "[integration]") {
  compare_gpu_vs_cpu("SELECT * FROM tpch.region");
}
```

## Memory Caching Test Pattern

**Basic cache test** (`test/cpp/memory_management/test_cpu_cache.cpp`):
```cpp
TEST_CASE("test_cpu_cache_basic_fixed_single_col", "[.][cpu_cache]")
{
  // Initialize the buffer manager
  size_t num_records = 1024;
  GPUBufferManager* gpuBufferManager = initialize_test_buffer_manager();
  
  // Create a GPU column
  duckdb::shared_ptr<GPUColumn> gpu_column =
    create_column_with_random_data(GPUColumnTypeId::INT32, num_records);
  duckdb::shared_ptr<GPUIntermediateRelation> relationship =
    make_shared_ptr<GPUIntermediateRelation>(1);
  relationship->columns[0] = gpu_column;
  
  // Cache to CPU
  size_t cpu_cache_bytes = calculate_test_cpu_cache_size(2 * gpu_column->getTotalColumnSize());
  MallocCPUCache cpu_cache(cpu_cache_bytes, 1);
  uint32_t chunk_id = cpu_cache.moveDataToCPU(relationship);
  REQUIRE(chunk_id == 0);
  
  // Load back from cache
  duckdb::shared_ptr<GPUIntermediateRelation> loaded_relationship =
    cpu_cache.moveDataToGPU(chunk_id, true);
  REQUIRE(loaded_relationship->columns.size() == 1);
  
  // Verify data matches
  verify_gpu_column_equality(loaded_relationship->columns[0], gpu_column);
  verify_cuda_errors("CUDA Errors in CPU Caching Test");
}
```

## SQL Logic Test Format

**Structure** (`test/sql/tpch-sirius.test`):
```
# =============================================================================
# Copyright 2025, Sirius Contributors.
# ...
# =============================================================================

# name: test/sql/tpch-sirius.test
# description: test TPC-H queries with GPU processing
# group: [sirius]

# Load required extensions
require sirius

# DDL statements
statement ok
CREATE TABLE nation ( n_nationkey INTEGER NOT NULL UNIQUE PRIMARY KEY, ... );

# Query tests with expected results
query I
SELECT COUNT(*) FROM nation;
----
25

# Multi-row result test
query III
SELECT n_nationkey, n_regionkey, COUNT(*) FROM nation GROUP BY n_nationkey, n_regionkey;
----
0    0    1
1    1    1
...
```

**Result formats:**
- `query I` — Single integer column
- `query III` — Three integer columns
- `query R` — Real (float) column
- `----` — Separator between query and results
- Empty line — End of test

## Error Testing Patterns

**Exception testing:**
```cpp
// Expect specific exception type
REQUIRE_THROWS_AS(setter.apply(libconfig.getRoot()), std::runtime_error);

// Expect exception with no value change
REQUIRE_THROWS_AS(setter.apply(invalid_config), std::runtime_error);
REQUIRE(int_value == 0);  // value unchanged due to validation failure

// Expect no exception
REQUIRE_NOTHROW(cpu_cache.moveDataToGPU(copy_chunk_id, true));
REQUIRE_THROWS(cpu_cache.moveDataToGPU(chunk_id, true));  // Expect any exception
```

**DuckDB query error testing:**
```cpp
auto result = con->Query("invalid sql");
REQUIRE(result);
if (result->HasError()) {
  UNSCOPED_INFO("Error: " << result->GetError());
}
REQUIRE_FALSE(result->HasError());
```

## Mocking & Test Doubles

**Approach:**
- No external mocking framework (gmock not used)
- Manual test fixtures with controlled DuckDB instances
- Shared environments for integration tests (`shared_env_listener`)
- Isolated test environments for unit tests

**Test environment setup:**
```cpp
struct sirius_config_env_guard {
  sirius_config_env_guard(const std::string& config_path) {
    setenv("SIRIUS_CONFIG_FILE", config_path.c_str(), 1);
  }
  ~sirius_config_env_guard() { unsetenv("SIRIUS_CONFIG_FILE"); }
};
```

**Database fixtures:**
- `integration.duckdb` — Shared TPC-H database for integration tests
- `integration.cfg` — Configuration file specifying memory limits, device selection
- `scan_config_path` — Configuration for CPU cache tests

## Fixtures and Factories

**Test utilities location:** `test/cpp/utils/` and component-specific headers

**Helper functions** (`test/cpp/memory_management/test_cpu_cache.cpp`):
```cpp
GPUBufferManager* initialize_test_buffer_manager();
duckdb::shared_ptr<GPUColumn> create_column_with_random_data(GPUColumnTypeId type, 
                                                              size_t num_records);
void verify_gpu_column_equality(duckdb::shared_ptr<GPUColumn> col1,
                                duckdb::shared_ptr<GPUColumn> col2);
void verify_cuda_errors(const std::string& context);
```

**Test environment class** (`test/cpp/utils/sirius_test_env.hpp`):
```cpp
class shared_test_env {
 public:
  shared_test_env(const std::filesystem::path& config_path);
  void pause();
  void resume();
  bool is_active() const;
  duckdb::Connection make_connection();
};

// Global instances
extern shared_test_env* g_shared_env;      // for unit tests
extern shared_test_env* g_integration_env; // for GPU execution tests
```

**Fixture inheritance:**
- `GPUExecutionFixtureBase` — Base with comparison logic
- `GPUExecutionDuckDBFixture(GPUExecutionFixtureBase)` — Adds database attachment
- `TEST_CASE_METHOD(Fixture, ...)` — Catch2 pattern for using fixtures

## Coverage

**Requirements:** No formal coverage target enforced

**View coverage:**
- Not currently automated; manual review via log analysis
- Operators: Check test counts in `test/cpp/operator/` vs supported operators in CLAUDE.md

**Coverage gaps (observed):**
- Legacy GPU path (`src/gpu_physical_plan_generator.cpp`) — Limited tests vs. new path
- Error paths and edge cases — Some fallback scenarios untested
- Memory spill scenarios — Limited disk I/O testing

**Test categorization:**
- `[cpu_cache]` — CPU memory caching tests
- `[shared_context]` — Shared DuckDB instance tests
- `[integration]` — Full GPU execution tests requiring TPC-H data
- `[config_opt]` — Configuration setter tests
- `[.][tag]` — Disabled tests (use explicit tag to run)

## Test Execution Workflow

**Build tests:**
```bash
cd /home/felipe/sirius_2
pixi shell
source setup_sirius.sh
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
```

**Run unit tests:**
```bash
# All tests
build/release/extension/sirius/test/cpp/sirius_unittest

# Specific component
build/release/extension/sirius/test/cpp/sirius_unittest "[config_opt]"

# With logging
SIRIUS_LOG_LEVEL=debug SIRIUS_LOG_DIR=/tmp/logs \
  build/release/extension/sirius/test/cpp/sirius_unittest
```

**Run SQL tests:**
```bash
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

**Run performance tests:**
```bash
python3 test/tpch_performance/generate_test_data.py 1    # SF=1
python3 test/tpch_performance/performance_test.py 1
```

**Test logs:**
- C++ tests: `build/release/extension/sirius/test/cpp/log/`
- SQL tests: Output in console

---

*Testing analysis: 2025-04-02*
