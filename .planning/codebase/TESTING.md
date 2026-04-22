# Testing Patterns

**Analysis Date:** 2026-04-21

## Test Framework

**Runner:**
- Framework: Catch2
- Config: Integrated via `duckdb/third_party/catch` (single-header)
- Build target: `sirius_unittest` (CMakeLists.txt)
- Binary: `build/release/extension/sirius/test/cpp/sirius_unittest`

**Assertion Library:**
- Built into Catch2: `REQUIRE()`, `REQUIRE_FALSE()`, `REQUIRE_NOTHROW()`, `REQUIRE_THROWS_AS()`, etc.

**Run Commands:**
```bash
# Run all unit tests
build/release/extension/sirius/test/cpp/sirius_unittest

# Run tests with specific tag
build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"

# Run single named test
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"

# SQL logic tests
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test

# Run tests with debug build
make test_debug
```

**Log output:**
- Unit test logs saved to: `build/release/extension/sirius/test/cpp/log/`
- Named: `sirius.log` (daily rotating)
- Configure via: `SIRIUS_LOG_LEVEL=debug`, `SIRIUS_LOG_DIR=/path/to/logs`

## Test File Organization

**Location:**
- C++ unit tests: `test/cpp/` - organized by component
- SQL logic tests: `test/sql/` - end-to-end verification
- Performance tests: `test/tpch_performance/`, `test/tpcds_performance/`

**Directory structure:**
```
test/cpp/
├── config/               # Configuration tests
├── creator/              # Task creator tests
├── data/                 # Data conversion tests
├── downgrade/            # Memory downgrade tests
├── helper/               # Type/conversion utilities tests
├── integration/          # GPU execution integration tests (shared context)
├── memory_management/    # CPU cache, memory management tests
├── operator/             # Operator-specific tests
├── parallel/             # Task executor, parallelism tests
├── planner/              # Plan generation tests
├── scan/                 # Table scan operator tests
├── unittest.cpp          # Main test runner with shared environment listener
└── utils/                # Test fixtures and utilities
    ├── operator_test_utils.hpp
    ├── sirius_test_env.hpp
    └── transparent_execution_test_utils.hpp
```

**Naming:**
- Test files: `test_*.cpp` or `*_test.cpp` (both conventions used)
- Examples: `test_config.cpp`, `test_cpu_cache.cpp`, `test_task_creator.cpp`

## Test Structure

**Basic test case:**
```cpp
#include "catch.hpp"
#include "config.hpp"

using namespace duckdb;

TEST_CASE("yaml reader basic types", "[config_opt][basic]")
{
  auto node = YAML::Load(R"(
    int_value: 100
    double_value: 6.28
    string_value: "config setter test"
  )");

  int int_value = 0;
  double double_value = 0.0;
  std::string string_value;

  yaml::reader r(node);
  r.optional("int_value", int_value);
  r.optional("double_value", double_value);
  r.optional("string_value", string_value);
  r.reject_unknown();

  REQUIRE(int_value == 100);
  REQUIRE(double_value == Approx(6.28));
  REQUIRE(string_value == "config setter test");
}
```

**Test fixture with setup/teardown:**
```cpp
class GPUExecutionFixtureBase {
 public:
  GPUExecutionFixtureBase()
  {
    // Initialization in constructor
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con = std::make_unique<duckdb::Connection>(
        sirius::test::g_integration_env->make_connection());
    } else {
      db = std::make_unique<duckdb::DuckDB>(nullptr);
      con = std::make_unique<duckdb::Connection>(*db);
    }
  }

  ~GPUExecutionFixtureBase() = default;  // Cleanup happens here (RAII)

 protected:
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
};

TEST_CASE_METHOD(GPUExecutionDuckDBFixture, "test_q1_hash_join", "[integration][gpu_exec]")
{
  // Test body - fixture already initialized
  REQUIRE(con != nullptr);
}
```

**Shared test environment:**
- Controlled by Catch2 test event listener in `test/cpp/unittest.cpp`
- Listener class: `shared_env_listener` (extends `Catch::TestEventListenerBase`)
- Manages global shared environments: `g_shared_env`, `g_integration_env`
- Only ONE environment active at a time (they share GPU memory and extension lock)
- Tests tagged with `[shared_context]` get `g_shared_env`
- Tests tagged with `[integration]` get `g_integration_env`
- All other tests get isolated/standalone environments

**Environment registration in test tags:**
```cpp
TEST_CASE("test_cpu_cache_basic_fixed_single_col", "[.][cpu_cache]")
{
  // No shared env - standalone test (. = hidden/optional)
}

TEST_CASE("test_convertible_data_batch", "[convertible_data_batch]")
{
  // Will use shared_context env if enabled
}

TEST_CASE_METHOD(GPUExecutionDuckDBFixture, "test_q1", "[integration][gpu_exec]")
{
  // Will use integration_env - has access to GPU + DuckDB context
}
```

## Mocking

**Framework:** Hand-rolled mocks for testing

**Patterns:**
```cpp
class mock_sirius_physical_operator : public sirius_physical_operator {
 public:
  mock_sirius_physical_operator(SiriusPhysicalOperatorType op_type =
    SiriusPhysicalOperatorType::PROJECTION)
    : sirius_physical_operator(op_type, {}, 0), _use_custom_hint(false)
  {
  }

  /**
   * @brief Enable custom hint mode and set the hint to return.
   */
  void set_custom_hint(std::optional<sirius::op::task_creation_hint> hint)
  {
    _use_custom_hint = true;
    _custom_hint = std::move(hint);
  }

  void clear_custom_hint() { _use_custom_hint = false; }

  std::optional<task_creation_hint> get_next_task_hint() override
  {
    if (_use_custom_hint) { return _custom_hint; }
    return sirius_physical_operator::get_next_task_hint();
  }

 private:
  bool _use_custom_hint;
  std::optional<sirius::op::task_creation_hint> _custom_hint;
};
```

**What to Mock:**
- GPU operators when testing integration points
- Memory managers and managers when testing task scheduling
- Task hints to control scheduling paths
- Pipelines to test barriers and completion conditions

**What NOT to Mock:**
- Core data structures (data_batch, convertible_data_batch) - test with real objects
- Memory allocation paths - use real RMM/cuCascade where possible
- Error conditions - trigger real exceptions when testing error handling

## Fixtures and Factories

**Test data helpers (in `test/cpp/utils/`):**

**Example fixture (`test/cpp/memory_management/test_cpu_cache.cpp`):**
```cpp
constexpr size_t CPU_CACHE_TEST_MEM_SF = 8;  // Multiplier for cache size

size_t calculate_test_cpu_cache_size(size_t bytes_to_cache)
{
  return std::pow(2.0, std::ceil(std::log2(CPU_CACHE_TEST_MEM_SF * bytes_to_cache)));
}

TEST_CASE("test_cpu_cache_basic_fixed_single_col", "[.][cpu_cache]")
{
  // Create test data
  size_t num_records = 1024;
  GPUBufferManager* gpuBufferManager = initialize_test_buffer_manager();

  duckdb::shared_ptr<GPUColumn> gpu_column =
    create_column_with_random_data(GPUColumnTypeId::INT32, num_records);

  duckdb::shared_ptr<GPUIntermediateRelation> relationship =
    make_shared_ptr<GPUIntermediateRelation>(1);
  relationship->columns[0] = gpu_column;

  // Test the functionality
  size_t cpu_cache_bytes = calculate_test_cpu_cache_size(2 * gpu_column->getTotalColumnSize());
  MallocCPUCache cpu_cache(cpu_cache_bytes, 1);
  uint32_t chunk_id = cpu_cache.moveDataToCPU(relationship);
  REQUIRE(chunk_id == 0);
}
```

**Factory helper (`test/cpp/data/test_convertible_data_batch.cpp`):**
```cpp
namespace {

struct test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;
  rmm::cuda_stream conv_stream;

  test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0)),
      conv_stream()
  {
  }

  rmm::cuda_stream_view stream() { return conv_stream.view(); }
};

test_env& env()
{
  static test_env e;
  return e;
}

}  // anonymous namespace

TEST_CASE("convertible_data_batch converts GPU batch to HOST", "[convertible_data_batch]")
{
  auto& e = env();
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3, 4, 5}, cudf::type_id::INT32);

  // Test...
}
```

**Location:**
- Test fixtures: `test/cpp/utils/sirius_test_env.hpp`, `test/cpp/utils/operator_test_utils.hpp`
- Factories: Inline in test files or in `utils/` headers
- Not committed outside tests (test data generated dynamically)

## Coverage

**Requirements:** Not enforced with minimum threshold

**View Coverage:**
- Code coverage tracking: Uses CMake build + gcov (optional)
- No built-in coverage target currently configured
- Manual: `cmake --build . --target coverage` (if configured)

**Good coverage targets:**
- Data conversion paths (core business logic)
- Memory management (fallback conditions)
- Error handling (exception paths)
- Operator correctness (with multiple input shapes/types)

## Test Types

**Unit Tests:**
- Location: `test/cpp/` (all component directories)
- Scope: Single operator, module, or utility in isolation
- Approach: Test one behavior, mock external dependencies
- Examples: `test_cpu_cache.cpp`, `test_config.cpp`, `test_task_creator.cpp`

**Integration Tests:**
- Location: `test/cpp/integration/` with `[integration]` tag
- Scope: End-to-end GPU execution with DuckDB
- Approach: Run query, compare GPU result vs CPU baseline
- Fixtures: `GPUExecutionFixtureBase`, subclasses for DuckDB/Parquet data sources
- Shared environment: Uses `g_integration_env` (paused/resumed per test)
- Examples: `test_gpu_execution_tpch.cpp`, `test_transparent_execution.cpp`

**SQL Logic Tests (SQLLogicTest):**
- Location: `test/sql/` - `.test` format files
- Scope: Query correctness against DuckDB gold standard
- Approach: DuckDB parser, statement execution, result validation
- Test files:
  - `tpch-sirius.test` - TPC-H queries
  - `clickbench-sirius.test` - ClickBench queries
  - `bugfix.test` - Regression tests for fixed issues
  - `tpch_mod_sirius.test` - Modified TPC-H (performance tuning)

**Performance Tests:**
- Location: `test/tpch_performance/`, `test/tpcds_performance/`
- Scope: Throughput, latency under realistic loads
- Approach: Run benchmark queries, capture timings
- Commands:
  ```bash
  python3 test/tpch_performance/generate_test_data.py {SCALE_FACTOR}
  python3 test/tpch_performance/performance_test.py {SCALE_FACTOR}
  ```

## Common Patterns

**Async Testing (GPU operations):**
```cpp
TEST_CASE("convertible_data_batch converts GPU batch to HOST", "[convertible_data_batch]")
{
  auto& e = env();

  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3, 4, 5}, cudf::type_id::INT32);

  REQUIRE(batch->get_memory_space() == e.gpu_space);
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);

  sirius::convertible_data_batch wrapper(batch);
  auto result = wrapper.convert({e.host_space}, e.stream(), *e.mgr);  // Async operation

  // Results are wrapped in std::optional
  REQUIRE(result.has_value());
  REQUIRE((*result).size() == 1);
  REQUIRE((*result)[0] > 0);
  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
}
```

**Error Testing:**
```cpp
TEST_CASE("yaml reader required field throws if missing", "[config_opt][required]")
{
  auto node = YAML::Load("int_value: 100");

  int int_value = 0;
  std::string name;

  yaml::reader r(node);
  r.optional("int_value", int_value);
  REQUIRE_THROWS_AS(r.required("name", name), std::runtime_error);
}
```

**Exception safety testing:**
```cpp
TEST_CASE("operation is nothrow", "[nothrow]")
{
  REQUIRE_NOTHROW(register_parquet_converters(registry));
}
```

**CUDA memory testing:**
```cpp
TEST_CASE("verify cuda errors", "[cuda]")
{
  duckdb::shared_ptr<GPUColumn> reloaded_column = /* ... */;
  duckdb::shared_ptr<GPUColumn> gpu_column = /* ... */;

  verify_cuda_errors("CUDA Errors in CPU Caching Test");
  verify_gpu_column_equality(reloaded_column, gpu_column);
}
```

**Floating-point comparisons:**
```cpp
TEST_CASE("floating point approximation", "[float]")
{
  double result = 6.28;
  REQUIRE(result == Approx(6.28).epsilon(0.01));  // 1% tolerance
}
```

## SQL Logic Test Format

**File format (`test/sql/tpch-sirius.test`):**
```sql
# =============================================================================
# Copyright header and description
# =============================================================================

# name: test/sql/tpch-sirius.test
# description: test TPC-H queries with GPU processing
# group: [sirius]

# Load required extensions
require sirius

# Setup: create tables
statement ok
DROP TABLE IF EXISTS nation;

statement ok
CREATE TABLE nation (n_nationkey INTEGER NOT NULL, ...);

# Run query via GPU
statement ok
call gpu_buffer_init('1 GB', '2 GB');

# Test result - must match exactly
query I
call gpu_processing('SELECT rowID FROM issue_56 WHERE ...');
----
1
2
3
```

**Line syntax:**
- `# comment` - Comment line
- `require extension` - Load required extension
- `statement ok` - Statement should succeed (no result checked)
- `query <type>` - Query returning results (`I`=integer, `R`=real, `T`=text, etc.)
- `----` - Separates query from expected results
- Results: One value per line, row-by-row

**Test execution:**
```bash
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

---

*Testing analysis: 2026-04-21*
