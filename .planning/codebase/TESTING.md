# Testing Patterns

**Analysis Date:** 2026-04-13

## Test Framework

**Unit Test Runner:**
- Framework: `Catch2` (C++ testing framework)
- Entry point: `test/cpp/unittest.cpp`
- Build output: `build/release/extension/sirius/test/cpp/sirius_unittest`
- Config: CMakeLists.txt integration (no separate config file)

**Assertion Library:**
- Native Catch2 assertions: `REQUIRE`, `REQUIRE_FALSE`, `REQUIRE_NOTHROW`, `REQUIRE_THROWS_AS`, `REQUIRE_THROWS`
- Floating-point: `REQUIRE(actual == Approx(expected))` (e.g., `REQUIRE(double_value == Approx(6.28))`)
- Custom macros: `UNSCOPED_INFO` for diagnostic output in test failures

**Run Commands:**
```bash
# Build and run all unit tests
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
build/release/extension/sirius/test/cpp/sirius_unittest

# Run tests with specific tag
build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"

# Run specific test by name
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"

# Watch mode (via test framework)
# (Not directly supported; run full suite and grep for changes)

# Coverage
# (Coverage not explicitly enforced; manual instrumentation via SIRIUS_LOG_* macros)
```

**SQL Logic Tests (End-to-End):**
- Runner: `test/unittest` binary (DuckDB SQLLogicTest runner)
- Test files: `.test` format in `test/sql/`
- Example files:
  - `test/sql/tpch-sirius.test` (TPC-H queries with gpu_processing calls)
  - `test/sql/bugfix.test` (regression tests)
  - `test/sql/clickbench-sirius.test` (clickbench queries)

```bash
# Run all SQL logic tests
make test

# Run specific test file
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

**Performance Tests:**
- Framework: Python script (`test/tpch_performance/performance_test.py`)
- Data generation: `test/tpch_performance/generate_test_data.py`
- Requires: `duckdb-python` to be built via `pixi run -e duckdb-python build-duckdb-python`

```bash
python3 test/tpch_performance/generate_test_data.py {SCALE_FACTOR}
python3 test/tpch_performance/performance_test.py {SCALE_FACTOR}
```

## Test File Organization

**Location:**
- Unit tests: `test/cpp/` (co-located with DuckDB main build)
- SQL logic tests: `test/sql/` (separate from unit tests)
- Performance tests: `test/tpch_performance/` (Python-based)

**Directory structure under `test/cpp/`:**
```
test/cpp/
├── config/              # Configuration tests (YAML parsing, validation)
├── creator/             # Task creator tests
├── data/                # Data representation tests (host parquet, converters)
├── downgrade/           # Downgrade executor lifecycle tests
├── exec/                # Execution infrastructure (thread pool, MPMC queue)
├── integration/         # Integration tests (GPU execution end-to-end)
│   ├── integration.yaml # Sirius config for integration tests
│   └── data/            # Test databases (duckdb files)
├── memory_management/   # Memory cache tests (CPU cache)
├── memory/              # Host memory utilities tests
├── parallel/            # Parallel task execution tests
├── planner/             # Physical plan generation tests
├── scan/                # Scan operator tests
│   └── memory.yaml      # Sirius config for scan tests
├── unittest.cpp         # Test harness with Catch2 listener
└── (no shared base fixtures at .cpp level)
```

**Naming Convention:**
- Test files: `test_*.cpp` or `test_<feature>.cpp`
- Examples: `test_config.cpp`, `test_cpu_cache.cpp`, `test_task_executor.cpp`, `test_gpu_execution_tpch.cpp`

## Test Structure

**Catch2 Suite Organization:**
```cpp
TEST_CASE("yaml reader basic types", "[config_opt][basic]")
{
  auto node = YAML::Load(R"(
    int_value: 100
    double_value: 6.28
    string_value: "config setter test"
  )");

  int int_value       = 0;
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

**Test tags:** `[tag_name]` in square brackets for filtering
- Common tags: `[config_opt]`, `[basic]`, `[optional]`, `[required]`, `[conditional]`
- Shared environment tags: `[shared_context]` (uses shared scan environment), `[integration]` (uses integration environment)
- Example: `TEST_CASE("name", "[integration][gpu]")`

**Patterns:**
- **Setup:** Inline in test body (no separate setup methods; prefer clear variable initialization)
  - Example: `auto cfg_path = fs::path(__FILE__).parent_path() / "integration.yaml";`
- **Teardown:** Automatic (RAII destructors clean up; no explicit teardown in most tests)
  - Example: `std::unique_ptr<duckdb::DuckDB> db` automatically destroyed at scope end
- **Assertions:** Sequential `REQUIRE` statements (stop at first failure)
  - Example:
    ```cpp
    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());
    REQUIRE_FALSE(gpu_result->HasError());
    ```

## Shared Test Environments

**Problem solved:** GPU resources (memory, extension lock) can only be used by one test at a time.

**Solution:** Catch2 listener-based environment management in `test/cpp/unittest.cpp`:
```cpp
struct shared_env_listener : Catch::TestEventListenerBase {
  enum class env_need { NONE, SHARED, INTEGRATION };

  static env_need classify(Catch::TestCaseInfo const& info)
  {
    for (auto const& tag : info.tags) {
      if (tag == "shared_context") return env_need::SHARED;
      if (tag == "integration") return env_need::INTEGRATION;
    }
    return env_need::NONE;
  }

  void testCaseStarting(Catch::TestCaseInfo const& info) override
  {
    auto needs = classify(info);
    // Pause wrong environment, resume right one
  }
};

CATCH_REGISTER_LISTENER(shared_env_listener)
```

**Environments:**
- `sirius::test::g_shared_env`: Shared context for scan/operator unit tests (config: `test/cpp/scan/memory.yaml`)
  - Paused by default, activated for tests tagged `[shared_context]`
- `sirius::test::g_integration_env`: Shared context for GPU execution integration tests (config: `test/cpp/integration/integration.yaml`)
  - Paused by default, activated for tests tagged `[integration]`
- Untagged tests: Use isolated environments (standalone GPU context or no GPU)

**Test environment config files:**
- `test/cpp/scan/memory.yaml`: GPU memory config (usage_limit_fraction, reservation limits, thread pools)
- `test/cpp/integration/integration.yaml`: Full Sirius config including memory, topology, executor settings

## Mocking

**Framework:** No explicit mocking framework (not used in this codebase)

**Patterns:**
- **DuckDB integration tests:** Use real DuckDB instance (`duckdb::DuckDB` and `duckdb::Connection`)
  - Example: `db = std::make_unique<duckdb::DuckDB>(nullptr);` creates in-memory database
- **GPU operators:** Use real GPU via `sirius::test::shared_test_env` (manages GPU memory context)
- **Test doubles:** Manual stubs for simple objects
  - Example: `struct str_top_n_record_type { uint32_t row_id; uint64_t key_prefix; }` for sort key comparison

**What to Mock:**
- Not commonly mocked; real components preferred for fidelity
- If needed, stub simple utilities (e.g., mock metadata readers)

**What NOT to Mock:**
- GPU execution paths (always test on real GPU)
- DuckDB query engine (always use real engine)
- Memory managers (always test real allocation/deallocation)

## Fixtures and Factories

**Test Data:**
- **YAML files as config:** `test/cpp/*/memory.yaml`, `test/cpp/integration/integration.yaml`
- **In-memory data:** Created within test body via `YAML::Load(R"(...)") `
- **Files on disk:** Test databases at `test/cpp/integration/data/duckdb/integration.duckdb`

**Helper functions (not factory pattern):**
```cpp
// From test_host_table_utils.cpp
std::filesystem::path get_test_config_path()
{
  return std::filesystem::path(__FILE__).parent_path() / "memory.yaml";
}

memory_space* get_memory_space(duckdb::shared_ptr<duckdb::SiriusContext> sirius_ctx,
                               cucascade::memory::Tier tier,
                               int device_id)
{
  auto& manager = sirius_ctx->get_memory_manager();
  auto* space   = manager.get_memory_space(tier, device_id);
  if (space) { return space; }
  auto spaces = manager.get_memory_spaces_for_tier(tier);
  if (!spaces.empty()) { return const_cast<memory_space*>(spaces.front()); }
  return nullptr;
}

template <typename T>
void verify_numeric_column(const cudf::column_view& col,
                           const std::vector<T>& expected,
                           const std::vector<bool>& expected_valid)
{
  REQUIRE(static_cast<size_t>(col.size()) == expected.size());
  REQUIRE(expected.size() == expected_valid.size());
  // ... verification logic
}
```

**Location:** Helpers defined inline in test `.cpp` files, not in separate factory headers

## Test Types

**Unit Tests:**
- **Scope:** Single operator, function, or small module
- **Example:** `test/cpp/config/test_config.cpp` - tests YAML parsing logic
- **Approach:** Direct function calls, check output
- **Isolation:** Minimal setup, no GPU resource contention (unless tagged `[shared_context]`)

**Integration Tests:**
- **Scope:** Full GPU execution pipeline (plan → execute → result)
- **Example:** `test/cpp/integration/test_gpu_execution_tpch.cpp`
- **Approach:** Execute query via `gpu_execution()`, compare GPU vs CPU results
- **Isolation:** Uses shared integration environment, compares results deterministically

```cpp
class GPUExecutionFixtureBase {
 public:
  void compare_gpu_vs_cpu(const std::string& query,
                          std::optional<float> float_tolerance = std::nullopt)
  {
    con->Query("SET enable_duckdb_fallback = false;");

    auto gpu_sql    = "CALL gpu_execution(\"" + query + "\")";
    auto gpu_result = con->Query(gpu_sql);
    REQUIRE_FALSE(gpu_result->HasError());

    auto cpu_result = con->Query(query);
    REQUIRE_FALSE(cpu_result->HasError());

    // Compare results row-by-row after sorting by all columns
    // Handles floating-point tolerance if needed
  }
};
```

**End-to-End Tests (SQL Logic Tests):**
- **Scope:** Query correctness across different data types and operations
- **Files:** `.test` format in `test/sql/`
- **Approach:** Execute SQL statement, check result rows
- **Format:**
  ```
  statement ok
  call gpu_buffer_init("1 GB", "2 GB");

  query I
  call gpu_processing("SELECT rowID FROM issue_56 WHERE ...");
  ----
  1
  2
  3
  ```

**Performance Tests:**
- **Scope:** Benchmark GPU vs CPU execution on TPC-H/TPC-DS workloads
- **Framework:** Python script
- **Approach:** Generate data, execute queries, measure time, validate results match
- **Output:** Timing comparisons and result verification

## Common Patterns

**Async Testing (with GPU execution):**
- Not explicitly async; GPU operations block until complete
- `con->Query()` returns `std::unique_ptr<QueryResult>` after execution finishes
- Multi-threaded task execution happens internally (test just waits for result)

Example from integration test:
```cpp
auto gpu_result = con->Query(gpu_sql);
// Blocks until GPU execution completes and result is ready
REQUIRE(gpu_result);
REQUIRE_FALSE(gpu_result->HasError());
```

**Error Testing:**
- Test expected exceptions with `REQUIRE_THROWS_AS`:
  ```cpp
  REQUIRE_THROWS_AS(r.required("name", name), std::runtime_error);
  ```

- Test error conditions:
  ```cpp
  auto node = YAML::Load("value: 100");
  int value = 0;
  yaml::reader r(node);
  REQUIRE_THROWS_AS(r.optional("value", value, yaml::greater_than<int>{150}),
                    std::runtime_error);
  REQUIRE(value == 0);  // Unchanged due to validation failure
  ```

**Floating-Point Comparison:**
- Use `Approx` for tolerance-based comparison:
  ```cpp
  REQUIRE(double_value == Approx(6.28));
  ```

- In integration tests, pass optional tolerance for column comparison:
  ```cpp
  compare_gpu_vs_cpu(query, std::optional<float>{0.0001f});
  ```

**Result Validation in Integration Tests:**
- Sort both result sets by all columns for deterministic comparison:
  ```cpp
  std::string order_clause = " ORDER BY ";
  for (duckdb::idx_t c = 0; c < ncols; c++) {
    if (c > 0) order_clause += ", ";
    order_clause += std::to_string(c + 1);
  }

  auto gpu_sorted = con->Query("SELECT * FROM gpu_execution(\"" + query + "\")" + order_clause);
  auto cpu_sorted = con->Query("SELECT * FROM (" + query + ") t" + order_clause);
  ```

- Compare rows as strings to normalize type differences (HUGEINT vs BIGINT both render as "50"):
  ```cpp
  for (duckdb::idx_t r = 0; r < gpu_sorted->RowCount(); r++) {
    for (duckdb::idx_t c = 0; c < gpu_sorted->ColumnCount(); c++) {
      auto gpu_val = gpu_sorted->GetValue(c, r).ToString();
      auto cpu_val = cpu_sorted->GetValue(c, r).ToString();
      REQUIRE(gpu_val == cpu_val);
    }
  }
  ```

## Test Logs

**Location:** `build/release/extension/sirius/test/cpp/log`

**Configuration:**
- Log level set via `Config::LOG_LEVEL` (default: "info")
- Log directory set via `Config::LOG_DIR` (default: "log")
- Flush every 3 seconds (configurable via `Config::LOG_FLUSH_SECONDS`)

**Runtime control (unit tests):**
```cpp
std::string log_dir = SIRIUS_UNITTEST_LOG_DIR;
Config::LOG_DIR = log_dir;
InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS);
```

**Format:** `[YYYY-MM-DD HH:MM:SS.ms] [level] [file:line] message`

**Parsing logs:** Tool at `tools/parse_pipeline_log.py` parses Sirius pipeline logs to show per-operator row counts (useful for debugging incorrect query results).

## Coverage

**Requirements:** No enforced coverage target (not explicitly configured)

**Measurement approach:**
- Manual instrumentation via `SIRIUS_LOG_*` macros (see CONVENTIONS.md)
- Integration tests validate end-to-end correctness (imply coverage of execution paths)

**Gaps to consider:**
- Fallback paths (DuckDB execution) not heavily tested in unit tests
- Error recovery scenarios (cleanup after exception) limited coverage
- Memory spilling edge cases (cascading memory pressure) minimal tests

## Test Debugging

**Run single test:**
```bash
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"
```

**Run tests with specific tag:**
```bash
build/release/extension/sirius/test/cpp/sirius_unittest "[config_opt]"
```

**Enable debug logging:**
```bash
# Set environment before running
export SIRIUS_LOG_LEVEL=debug
export SIRIUS_LOG_DIR=./my_logs
CMAKE_BUILD_PARALLEL_LEVEL=4 make
build/release/extension/sirius/test/cpp/sirius_unittest
```

**Check test output:**
- stderr/stdout: Captured by Catch2, printed on failure
- Log files: In `SIRIUS_LOG_DIR` (default: `log/sirius.log`)
- `UNSCOPED_INFO` messages: Printed before assertion failure

Example debug output:
```
INFO: [2026-04-13 14:22:31.456] [debug] [sirius_engine.cpp:123] Initializing sirius engine
DEBUG: Input size: 1000000
DEBUG: CUDF Aggregate result count: 42
```

---

*Testing analysis: 2026-04-13*
