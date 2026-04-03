# Testing Patterns

**Analysis Date:** 2026-04-03

## Test Framework

**Runner:**
- Framework: Catch2 (header-only, version 3.x)
- Test binary: `build/release/extension/sirius/test/cpp/sirius_unittest`
- Config: Catch2 integrated into CMakeLists.txt (no separate config file needed)

**Assertion Library:**
- Catch2 built-in assertions (REQUIRE, CHECK, SECTION)
- Approx for floating-point comparisons: `REQUIRE(double_value == Approx(6.28))`

**Run Commands:**
```bash
# Run all unit tests
build/release/extension/sirius/test/cpp/sirius_unittest

# Run tests matching a tag (e.g., cpu_cache tests)
build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"

# Run specific test by name
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"

# Run SQL logic tests (end-to-end)
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

## Test File Organization

**Location:**
- C++ unit tests: `test/cpp/<component>/test_*.cpp` (e.g., `test/cpp/config/test_config.cpp`)
- SQL logic tests: `test/sql/*.test` (e.g., `test/sql/tpch-sirius.test`)
- Performance tests: `test/tpch_performance/` (Python scripts)
- Test utilities/fixtures: `test/cpp/scan/test_utils.hpp`, `test/cpp/utils/sirius_test_env.hpp`

**Naming:**
- C++ test files: `test_<component>.cpp`
- SQL test files: `<feature>-sirius.test` or `<category>.test`
- Performance test files: `*_test.py` or `*.py` with test functions

**Structure:**
```
test/
├── cpp/
│   ├── config/
│   │   ├── test_config.cpp
│   │   └── test_context.cpp
│   ├── memory_management/
│   │   └── test_cpu_cache.cpp
│   ├── scan/
│   │   ├── test_parquet_scan_task.cpp
│   │   ├── test_utils.hpp
│   │   └── memory.cfg
│   ├── integration/
│   │   ├── test_gpu_execution_tpch.cpp
│   │   └── integration.cfg
│   ├── utils/
│   │   └── sirius_test_env.hpp
│   ├── unittest.cpp        # Main entry point with Catch2 listener
│   └── [other components]/
├── sql/
│   ├── tpch-sirius.test
│   ├── bugfix.test
│   └── clickbench-sirius.test
└── tpch_performance/
    ├── generate_test_data.py
    ├── performance_test.py
    └── queries.py
```

## Test Structure

**Suite Organization:**
```cpp
TEST_CASE("use configuration basic setters", "[config_opt][basic]")
{
  using namespace sirius;
  config::configuration_setter setter;
  
  // Arrange
  int int_value = 0;
  setter.add_config("int_value", int_value);
  
  // Act
  libconfig::Config libconfig;
  libconfig.readString("int_value = 100;");
  setter.apply(libconfig.getRoot());
  
  // Assert
  REQUIRE(int_value == 100);
}
```

**Patterns:**

**1. Arrange-Act-Assert (AAA) pattern:**
- Arrange: Set up test data and fixtures
- Act: Call the function/method under test
- Assert: Verify the result with REQUIRE or CHECK

**2. Setup with shared test environment:**
```cpp
TEST_CASE("test with shared context", "[shared_context]")
{
  // This test gets a connection from g_shared_env (see unittest.cpp listener)
  auto conn = sirius::test::g_shared_env->make_connection();
  // test code
}
```

**3. Isolated tests with their own environment:**
```cpp
TEST_CASE("test with isolated context", "[isolated_context]")
{
  // This test runs with its own SiriusContext (g_shared_env paused)
  duckdb::DuckDB db(":memory:");
  duckdb::Connection conn(db);
  // test code
}
```

**4. Integration tests (GPU execution):**
```cpp
TEST_CASE("GPU execution integration test", "[integration]")
{
  // Uses g_integration_env with GPU memory setup
  auto conn = sirius::test::g_integration_env->make_connection();
  // GPU execution code
}
```

## Mocking

**Framework:** None — tests use real objects

**Patterns:**

**1. Mock GPU Physical Operator (for testing task creation):**
```cpp
// From test/cpp/creator/test_task_creator.cpp
class mock_gpu_operator : public sirius::op::sirius_physical_operator {
 public:
  virtual std::string get_name() const override { return "MockGPUOp"; }
  virtual void execute(sirius::pipeline::sirius_pipeline& pipeline) override { }
  // ... implement required pure virtual methods
};
```

**2. Memory manager initialization (for GPU tests):**
```cpp
// From test/cpp/scan/test_utils.hpp
auto manager = initialize_memory_manager();  // Allocates 2GB GPU, 4GB host
sirius::converter_registry::initialize();    // Set up data type converters
```

**3. Test configuration files:**
```
test/cpp/scan/memory.cfg        # Memory tier configuration for scan tests
test/cpp/integration/integration.cfg  # GPU buffer configuration for integration tests
```

**What to Mock:**
- GPU buffers in unit tests (use CPU cache or mock allocation)
- DuckDB connections in isolated tests (use `:memory:`)

**What NOT to Mock:**
- GPU execution functions (use real cuDF operations)
- Memory managers (use real RMM/cuCascade)
- Data representations (use actual converter registry)

## Fixtures and Factories

**Test Data:**
```cpp
// From test/cpp/memory_management/test_cpu_cache.cpp
size_t calculate_test_cpu_cache_size(size_t bytes_to_cache) {
  return std::pow(2.0, std::ceil(std::log2(CPU_CACHE_TEST_MEM_SF * bytes_to_cache)));
}

TEST_CASE("test_cpu_cache_basic_fixed_single_col", "[.][cpu_cache]") {
  size_t num_records = 1024;
  GPUBufferManager* gpuBufferManager = initialize_test_buffer_manager();
  
  duckdb::shared_ptr<GPUColumn> gpu_column =
    create_column_with_random_data(GPUColumnTypeId::INT32, num_records);
  duckdb::shared_ptr<GPUIntermediateRelation> relationship =
    make_shared_ptr<GPUIntermediateRelation>(1);
  relationship->columns[0] = gpu_column;
  // ...
}
```

**Location:**
- Inline test fixtures in test files (prefer)
- Reusable helpers in `test/cpp/scan/test_utils.hpp` for scan tests
- Shared environment in `test/cpp/utils/sirius_test_env.hpp` for all tests
- Memory initialization: `sirius::test::shared_test_env` constructor

**Shared Test Environment** (`test/cpp/utils/sirius_test_env.hpp`):
```cpp
class shared_test_env {
 public:
  explicit shared_test_env(const std::filesystem::path& config_path);
  duckdb::Connection make_connection();
  duckdb::DuckDB& database();
  bool is_active() const;  // Returns true if DuckDB instance exists
  void pause();            // Destroy for isolated tests
  void resume();           // Recreate after isolated tests
};

extern shared_test_env* g_shared_env;        // For [shared_context] tests
extern shared_test_env* g_integration_env;   // For [integration] tests
```

## Coverage

**Requirements:** No explicit coverage target enforced

**View Coverage:**
```bash
# No built-in coverage command documented
# Coverage would be measured via gcov or lcov if enabled in CMake
```

**Organization:**
- Core functionality well-covered by unit tests in `test/cpp/`
- End-to-end validation via SQL logic tests in `test/sql/`
- Performance regression detection via `test/tpch_performance/`

## Test Types

**Unit Tests:**
- **Scope:** Individual functions, operators, utilities
- **Approach:** Direct function calls, mock GPU operations
- **Location:** `test/cpp/<component>/test_*.cpp`
- **Examples:**
  - `test/cpp/config/test_config.cpp` — Config system
  - `test/cpp/memory_management/test_cpu_cache.cpp` — CPU caching
  - `test/cpp/exec/test_bounded_thread_pool.cpp` — Thread pool
- **Environment:** Isolated or shared DuckDB context (tag-driven via Catch2 listener)

**Integration Tests:**
- **Scope:** Full query execution through GPU pipeline
- **Approach:** Execute SQL via `CALL gpu_execution(...)` or `CALL gpu_processing(...)`
- **Location:** `test/cpp/integration/test_gpu_execution_*.cpp`
- **Examples:**
  - `test/cpp/integration/test_gpu_execution_tpch.cpp` — TPC-H queries
  - `test/cpp/integration/test_tpcds_plan_translation.cpp` — TPC-DS plan validation
- **Environment:** Dedicated GPU memory (g_integration_env)

**SQL Logic Tests (End-to-End):**
- **Scope:** Query correctness, result matching
- **Approach:** DuckDB SQLLogicTest format (statement ok / query / result rows)
- **Location:** `test/sql/*.test`
- **Examples:**
  - `test/sql/tpch-sirius.test` — TPC-H queries with `CALL gpu_processing(...)`
  - `test/sql/bugfix.test` — Regression tests (Issue #56 IS NOT DISTINCT, etc.)
  - `test/sql/clickbench-sirius.test` — ClickBench queries
- **Format:**
  ```
  statement ok
  DROP TABLE IF EXISTS table_name;
  
  statement ok
  CREATE TABLE table_name (...);
  
  query I
  CALL gpu_execution("SELECT ...");
  ----
  expected_row_1
  expected_row_2
  ```

**Performance Tests:**
- **Scope:** TPC-H execution on variable scale factors
- **Approach:** Python scripts measuring query time, verifying results
- **Location:** `test/tpch_performance/performance_test.py`
- **Scripts:**
  - `generate_test_data.py` — Generate parquet datasets at scale factor {1,10,30,100,300}
  - `performance_test.py` — Execute all 22 queries, time execution, verify results
- **Invocation:**
  ```bash
  python3 test/tpch_performance/generate_test_data.py 10   # Generate SF=10
  python3 test/tpch_performance/performance_test.py 10     # Run SF=10 tests
  ```

## Common Patterns

**Async Testing:**
Not heavily used (GPU operations are synchronous from CPU's perspective).

Example with thread pool (rare):
```cpp
// From test/cpp/exec/test_bounded_thread_pool.cpp
std::atomic<int> active{0};
std::atomic<int> peak{0};

for (int i = 0; i < 12; ++i) {
  auto s = pool.reserve();
  pool.dispatch(std::move(s), [&active, &peak] {
    int cur = active.fetch_add(1) + 1;
    // Atomic updates track concurrent execution
  });
}
```

**Error Testing:**
```cpp
// From test/cpp/memory_management/test_cpu_cache.cpp
REQUIRE_NOTHROW(cpu_cache.moveDataToGPU(copy_chunk_id, true));
REQUIRE_THROWS(cpu_cache.moveDataToGPU(chunk_id, true));  // After eviction
```

**Float/Double Comparisons:**
```cpp
// From test/cpp/config/test_config.cpp
REQUIRE(double_value == Approx(6.28));
```

**Resource cleanup:**
```cpp
// Destructors handle GPU memory, CUDA streams, DuckDB connections
// No explicit cleanup needed (RAII pattern)
~shared_test_env() { /* DuckDB destroyed, lock released */ }
```

## Test Execution Environment

**Catch2 Listener** (`test/cpp/unittest.cpp`):
- Registers a `shared_env_listener` that manages test environment lifecycle
- Tags determine which environment is active:
  - `[shared_context]` — Uses g_shared_env (single DuckDB/SiriusContext for multiple tests)
  - `[integration]` — Uses g_integration_env (separate GPU memory setup)
  - No tag — Isolated (test creates its own DuckDB)
- Listener pauses wrong environment, resumes correct one before each test
- Benefit: Avoids GPU memory conflicts, reduces context creation overhead

**Initialization sequence:**
```cpp
// In unittest.cpp main()
Config::LOG_DIR = log_dir;
InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS);

auto scan_config_path = ... / "test/cpp/scan/memory.cfg";
sirius::test::shared_test_env scan_env(scan_config_path);
scan_env.pause();
sirius::test::g_shared_env = &scan_env;

auto integration_config_path = ... / "test/cpp/integration/integration.cfg";
sirius::test::shared_test_env integration_env(integration_config_path);
integration_env.pause();
sirius::test::g_integration_env = &integration_env;

Catch::Session session;
session.applyCommandLine(argc, argv);
int result = session.run();
```

**Log output:**
- Logs written to: `${CMAKE_BINARY_DIR}/log/sirius.log` (configured by SIRIUS_LOG_DIR)
- Test logs in: `build/release/extension/sirius/test/cpp/log/`

## Building Tests

**CMake configuration:**
- Tests integrated into main CMakeLists.txt
- Catch2 headers included via: `#include "catch.hpp"`
- SIRIUS_UNITTEST_LOG_DIR set at compile time (project root relative)

**Build commands:**
```bash
# Build with tests
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make

# After build, run tests
build/release/extension/sirius/test/cpp/sirius_unittest
```

---

*Testing analysis: 2026-04-03*
