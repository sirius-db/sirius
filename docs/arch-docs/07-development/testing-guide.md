# Testing Guide

This comprehensive guide covers testing strategies for Sirius, including unit tests, integration tests, SQL logic tests, and performance benchmarks.

## Table of Contents

1. [Overview](#overview)
2. [Test Types](#test-types)
3. [Unit Testing (C++)](#unit-testing-c)
4. [SQL Logic Tests](#sql-logic-tests)
5. [Integration Testing](#integration-testing)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Test Data Generation](#test-data-generation)
8. [CI/CD Integration](#cicd-integration)
9. [Best Practices](#best-practices)
10. [Next Steps](#next-steps)

---

## Overview

Sirius uses a multi-layered testing approach:

1. **Unit Tests (C++)**: Test individual operators and components in isolation
2. **SQL Logic Tests**: Test query correctness against expected results
3. **Integration Tests**: Test end-to-end query execution
4. **Performance Benchmarks**: Measure and track performance
5. **Regression Tests**: Ensure fixes stay fixed

**Testing Frameworks:**

- **Google Test (gtest/gmock)**: C++ unit tests
- **SQLLogicTest**: SQL integration tests
- **Google Benchmark**: Performance benchmarks

---

## Test Types

### 1. Unit Tests (C++)

**Purpose**: Test individual operators, functions, and components in isolation.

**Location**: `test/cpp/`

**Framework**: Google Test

**Example**:

```cpp
TEST(FilterTest, BasicFilter) {
    // Arrange
    auto input = create_test_batch(...);
    sirius::op::sirius_physical_filter filter_op(...);

    // Act
    auto result = filter_op.execute({input}, stream);

    // Assert
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->get_row_count(), expected_rows);
}
```

### 2. SQL Logic Tests

**Purpose**: Test query correctness with SQL queries and expected results.

**Location**: `test/sql/`

**Framework**: SQLLogicTest

**Example**:

```sql
statement ok
CREATE TABLE t (a INTEGER, b VARCHAR);

query II
SELECT * FROM gpu_execution('SELECT * FROM t WHERE a > 10');
----
15	hello
20	world
```

### 3. Integration Tests

**Purpose**: Test complete query execution paths, including planning, execution, and result collection.

**Location**: `test/integration/`

**Example**: TPC-H queries, complex joins, window functions

### 4. Performance Benchmarks

**Purpose**: Measure performance and track regressions.

**Location**: `benchmark/`

**Framework**: Google Benchmark

**Example**:

```cpp
static void BM_Filter_1M_Rows(benchmark::State& state) {
    for (auto _ : state) {
        run_filter_query(1000000);
    }
}
BENCHMARK(BM_Filter_1M_Rows);
```

---

## Unit Testing (C++)

### Setup

**Prerequisites:**

```bash
cd build
cmake -DBUILD_TESTING=ON ..
make
```

**Run all unit tests:**

```bash
ctest
```

**Run specific test:**

```bash
./test/cpp/operator/test_filter
```

### Writing Unit Tests

**Test Structure:**

```cpp
#include <gtest/gtest.h>
#include "op/sirius_physical_filter.hpp"
#include "test_helpers.hpp"

// Test fixture (optional, for shared setup/teardown)
class FilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code (runs before each test)
        stream = rmm::cuda_stream_default;
    }

    void TearDown() override {
        // Cleanup code (runs after each test)
    }

    rmm::cuda_stream_view stream;
};

// Test case
TEST_F(FilterTest, EmptyInput) {
    // Arrange
    sirius::op::sirius_physical_filter filter_op(
        {duckdb::LogicalType::INTEGER},
        create_expression("a > 10"),
        100
    );

    // Act
    auto result = filter_op.execute({}, stream);

    // Assert
    ASSERT_TRUE(result.empty());
}

TEST_F(FilterTest, AllRowsPass) {
    auto input = create_test_batch({{"a", {15, 20, 25}}});
    sirius::op::sirius_physical_filter filter_op(...);

    auto result = filter_op.execute({input}, stream);

    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->get_row_count(), 3);  // All 3 rows pass
}

TEST_F(FilterTest, NoRowsPass) {
    auto input = create_test_batch({{"a", {1, 2, 3}}});
    sirius::op::sirius_physical_filter filter_op(...);

    auto result = filter_op.execute({input}, stream);

    ASSERT_TRUE(result.empty() || result[0]->get_row_count() == 0);
}

TEST_F(FilterTest, PartialFilter) {
    auto input = create_test_batch({{"a", {5, 15, 10, 25, 8}}});
    sirius::op::sirius_physical_filter filter_op(...);

    auto result = filter_op.execute({input}, stream);

    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->get_row_count(), 2);  // 15 and 25 pass
}

TEST_F(FilterTest, WithNulls) {
    auto input = create_test_batch_with_nulls({
        {"a", {5, std::nullopt, 15, 10, std::nullopt}}
    });

    sirius::op::sirius_physical_filter filter_op(...);
    auto result = filter_op.execute({input}, stream);

    // Nulls should be filtered out
    ASSERT_EQ(result[0]->get_row_count(), 2);  // 15 and (10 if > 10)
}
```

### Test Helpers

**Create test batches:**

```cpp
// test/cpp/test_helpers.hpp

// Create simple test batch
std::shared_ptr<cucascade::data_batch> create_test_batch(
    std::map<std::string, std::vector<int32_t>> columns)
{
    std::vector<std::unique_ptr<cudf::column>> cudf_columns;

    for (const auto& [name, values] : columns) {
        cudf_columns.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(
            values.begin(), values.end()
        ).release());
    }

    auto cudf_table = std::make_unique<cudf::table>(std::move(cudf_columns));
    return cucascade::data_batch::from_cudf_table(std::move(cudf_table), stream);
}

// Create batch with nulls
std::shared_ptr<cucascade::data_batch> create_test_batch_with_nulls(
    std::map<std::string, std::vector<std::optional<int32_t>>> columns)
{
    std::vector<std::unique_ptr<cudf::column>> cudf_columns;

    for (const auto& [name, values] : columns) {
        std::vector<int32_t> data;
        std::vector<bool> validity;

        for (const auto& val : values) {
            data.push_back(val.value_or(0));
            validity.push_back(val.has_value());
        }

        cudf_columns.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(
            data.begin(), data.end(), validity.begin()
        ).release());
    }

    auto cudf_table = std::make_unique<cudf::table>(std::move(cudf_columns));
    return cucascade::data_batch::from_cudf_table(std::move(cudf_table), stream);
}

// Compare batches
void assert_batches_equal(
    std::shared_ptr<cucascade::data_batch> actual,
    std::shared_ptr<cucascade::data_batch> expected)
{
    ASSERT_EQ(actual->get_row_count(), expected->get_row_count());
    ASSERT_EQ(actual->get_column_count(), expected->get_column_count());

    // Compare each column
    for (size_t i = 0; i < actual->get_column_count(); i++) {
        auto actual_col = actual->get_column(i);
        auto expected_col = expected->get_column(i);

        ASSERT_TRUE(cudf::test::expect_columns_equal(
            actual_col->view(),
            expected_col->view()
        ));
    }
}
```

### Google Test Assertions

```cpp
// Boolean assertions
ASSERT_TRUE(condition);
ASSERT_FALSE(condition);
EXPECT_TRUE(condition);   // Continue on failure

// Equality assertions
ASSERT_EQ(a, b);          // a == b
ASSERT_NE(a, b);          // a != b
ASSERT_LT(a, b);          // a < b
ASSERT_LE(a, b);          // a <= b
ASSERT_GT(a, b);          // a > b
ASSERT_GE(a, b);          // a >= b

// Floating point assertions
ASSERT_FLOAT_EQ(a, b);
ASSERT_DOUBLE_EQ(a, b);
ASSERT_NEAR(a, b, epsilon);

// String assertions
ASSERT_STREQ(a, b);
ASSERT_STRCASEEQ(a, b);

// Exception assertions
ASSERT_THROW(statement, exception_type);
ASSERT_NO_THROW(statement);
ASSERT_ANY_THROW(statement);
```

### Parameterized Tests

**Test multiple inputs:**

```cpp
class FilterParameterizedTest :
    public ::testing::TestWithParam<std::tuple<std::vector<int32_t>, int, size_t>>
{
};

TEST_P(FilterParameterizedTest, VariousInputs) {
    auto [input_values, threshold, expected_count] = GetParam();

    auto input = create_test_batch({{"a", input_values}});
    auto filter_op = create_filter_op(threshold);

    auto result = filter_op.execute({input}, stream);

    ASSERT_EQ(result[0]->get_row_count(), expected_count);
}

INSTANTIATE_TEST_SUITE_P(
    FilterTests,
    FilterParameterizedTest,
    ::testing::Values(
        std::make_tuple(std::vector<int32_t>{1, 2, 3}, 5, 0),     // None pass
        std::make_tuple(std::vector<int32_t>{10, 20, 30}, 5, 3),  // All pass
        std::make_tuple(std::vector<int32_t>{1, 10, 20}, 5, 2)    // Some pass
    )
);
```

---

## SQL Logic Tests

### Overview

**SQLLogicTest** is a declarative testing format for SQL queries.

**File Format**: `test/sql/<feature>.test`

### Basic Syntax

**Create table:**

```sql
statement ok
CREATE TABLE t (a INTEGER, b VARCHAR);
```

**Insert data:**

```sql
statement ok
INSERT INTO t VALUES (1, 'hello'), (2, 'world');
```

**Query with expected results:**

```sql
query II
SELECT * FROM gpu_execution('SELECT * FROM t ORDER BY a');
----
1	hello
2	world
```

**Query types:**

- `query I`: Single INTEGER column
- `query II`: Two INTEGER columns
- `query III`: Three INTEGER columns
- `query T`: VARCHAR column
- `query IT`: INTEGER and VARCHAR columns
- `query R`: REAL/FLOAT column

### Example Test File

**File**: `test/sql/filter.test`

```sql
# Test basic filtering

statement ok
CREATE TABLE users (id INTEGER, name VARCHAR, age INTEGER);

statement ok
INSERT INTO users VALUES
    (1, 'Alice', 25),
    (2, 'Bob', 30),
    (3, 'Charlie', 35);

# Test filter with age > 25
query IT
SELECT id, name FROM gpu_execution('SELECT * FROM users WHERE age > 25 ORDER BY id');
----
2	Bob
3	Charlie

# Test filter with no matches
query IT
SELECT id, name FROM gpu_execution('SELECT * FROM users WHERE age > 100');
----

# Test filter with all matches
query IT
SELECT id, name FROM gpu_execution('SELECT * FROM users WHERE age > 0 ORDER BY id');
----
1	Alice
2	Bob
3	Charlie

# Test compound filter
query I
SELECT id FROM gpu_execution('SELECT * FROM users WHERE age > 25 AND age < 35 ORDER BY id');
----
2

# Test NULL handling
statement ok
INSERT INTO users VALUES (4, 'Dave', NULL);

query IT
SELECT id, name FROM gpu_execution('SELECT * FROM users WHERE age IS NULL');
----
4	Dave

query IT
SELECT id, name FROM gpu_execution('SELECT * FROM users WHERE age IS NOT NULL ORDER BY id');
----
1	Alice
2	Bob
3	Charlie
```

### Running SQL Tests

**Run all SQL tests:**

```bash
./build/release/test/sqllogictest test/sql/*.test
```

**Run specific test:**

```bash
./build/release/test/sqllogictest test/sql/filter.test
```

**Run with verbose output:**

```bash
./build/release/test/sqllogictest -v test/sql/filter.test
```

### Advanced Features

**Loop tests:**

```sql
loop i 0 10

statement ok
INSERT INTO t VALUES (${i}, 'value_${i}');

endloop

query I
SELECT COUNT(*) FROM t;
----
10
```

**Conditional tests:**

```sql
require gpu

query I
SELECT * FROM gpu_execution('SELECT 1');
----
1
```

**Error tests:**

```sql
statement error
SELECT * FROM gpu_execution('SELECT * FROM nonexistent_table');
```

---

## Integration Testing

### TPC-H Queries

**Location**: `test/integration/tpch/`

**Run TPC-H tests:**

```bash
cd test/integration/tpch
./run_tpch.sh
```

**Example TPC-H Query 1:**

```sql
-- test/integration/tpch/q1.sql

SELECT
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM gpu_execution('
    SELECT * FROM lineitem
    WHERE l_shipdate <= DATE ''1998-09-02''
')
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;
```

### Custom Integration Tests

**Create integration test:**

```cpp
// test/integration/test_complex_join.cpp

#include <gtest/gtest.h>
#include "test_helpers.hpp"

TEST(IntegrationTest, ComplexJoin) {
    // Setup database
    auto db = create_test_database();
    db.execute("CREATE TABLE orders (order_id INT, customer_id INT, total FLOAT)");
    db.execute("CREATE TABLE customers (customer_id INT, name VARCHAR)");
    db.execute("INSERT INTO orders VALUES (1, 10, 100.0), (2, 20, 200.0), (3, 10, 150.0)");
    db.execute("INSERT INTO customers VALUES (10, 'Alice'), (20, 'Bob')");

    // Execute query
    auto result = db.query(R"(
        SELECT c.name, COUNT(*), SUM(o.total)
        FROM gpu_execution('
            SELECT * FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
        ')
        GROUP BY c.name
        ORDER BY c.name
    )");

    // Verify result
    ASSERT_EQ(result.row_count(), 2);
    ASSERT_EQ(result.get_value(0, 0), "Alice");
    ASSERT_EQ(result.get_value(0, 1), 2);      // 2 orders
    ASSERT_FLOAT_EQ(result.get_value(0, 2), 250.0);  // 100 + 150
}
```

---

## Performance Benchmarks

### Setup

**Build benchmarks:**

```bash
cmake -DBUILD_BENCHMARKS=ON ..
make benchmarks
```

**Run benchmarks:**

```bash
./benchmark/filter_benchmark
```

### Writing Benchmarks

**Basic benchmark:**

```cpp
#include <benchmark/benchmark.h>
#include "op/sirius_physical_filter.hpp"

static void BM_Filter_SimpleExpression(benchmark::State& state) {
    // Setup
    size_t row_count = state.range(0);
    auto input = create_large_test_batch(row_count);
    auto filter_op = create_filter_op("a > 10");

    // Benchmark loop
    for (auto _ : state) {
        auto result = filter_op.execute({input}, rmm::cuda_stream_default);
        cudaStreamSynchronize(rmm::cuda_stream_default);
    }

    // Report throughput
    state.SetItemsProcessed(state.iterations() * row_count);
    state.SetBytesProcessed(state.iterations() * row_count * sizeof(int32_t));
}

// Register benchmark with different row counts
BENCHMARK(BM_Filter_SimpleExpression)
    ->Args({1000})      // 1K rows
    ->Args({10000})     // 10K rows
    ->Args({100000})    // 100K rows
    ->Args({1000000})   // 1M rows
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
```

**Output:**

```
---------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations
---------------------------------------------------------------------
BM_Filter_SimpleExpression/1000     0.15 ms         0.15 ms     4500
BM_Filter_SimpleExpression/10000    0.25 ms         0.25 ms     2800
BM_Filter_SimpleExpression/100000   1.20 ms         1.20 ms      580
BM_Filter_SimpleExpression/1000000  10.5 ms         10.5 ms       67
```

### Advanced Benchmarks

**Compare GPU vs CPU:**

```cpp
static void BM_Filter_GPU(benchmark::State& state) {
    auto input = create_test_batch(state.range(0));
    auto filter_op = create_gpu_filter_op();

    for (auto _ : state) {
        auto result = filter_op.execute({input}, stream);
        cudaStreamSynchronize(stream);
    }
}

static void BM_Filter_CPU(benchmark::State& state) {
    auto input = create_test_batch(state.range(0));
    auto filter_op = create_cpu_filter_op();

    for (auto _ : state) {
        auto result = filter_op.execute(input);
    }
}

BENCHMARK(BM_Filter_GPU)->Range(1000, 1000000);
BENCHMARK(BM_Filter_CPU)->Range(1000, 1000000);
```

---

## Test Data Generation

### Generate Large Datasets

```cpp
// test/cpp/test_helpers.cpp

std::shared_ptr<cucascade::data_batch> generate_random_batch(
    size_t row_count,
    const std::vector<duckdb::LogicalType>& types,
    uint64_t seed = 0)
{
    std::mt19937 rng(seed);

    std::vector<std::unique_ptr<cudf::column>> columns;

    for (const auto& type : types) {
        switch (type.id()) {
            case duckdb::LogicalTypeId::INTEGER: {
                std::uniform_int_distribution<int32_t> dist(0, 1000000);
                std::vector<int32_t> data(row_count);
                for (auto& val : data) {
                    val = dist(rng);
                }
                columns.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(
                    data.begin(), data.end()
                ).release());
                break;
            }
            // ... other types ...
        }
    }

    auto table = std::make_unique<cudf::table>(std::move(columns));
    return cucascade::data_batch::from_cudf_table(std::move(table), stream);
}
```

### Load TPC-H Data

```bash
# Generate TPC-H data (1GB scale factor)
cd test/data
./generate_tpch.sh 1

# Load into DuckDB
./duckdb test.db
> CREATE TABLE lineitem AS SELECT * FROM read_parquet('tpch_lineitem_sf1.parquet');
```

---

## CI/CD Integration

### GitHub Actions

**File**: `.github/workflows/test.yml`

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.0-devel-ubuntu22.04

    steps:
    - uses: actions/checkout@v3

    - name: Build
      run: |
        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON .
        make -j$(nproc)

    - name: Run Unit Tests
      run: ctest --output-on-failure

    - name: Run SQL Tests
      run: ./build/release/test/sqllogictest test/sql/*.test

    - name: Run Benchmarks
      run: |
        ./benchmark/filter_benchmark --benchmark_min_time=0.1
```

---

## Best Practices

### 1. Test Naming

**Good:**

```cpp
TEST(FilterTest, EmptyInput_ReturnsEmptyOutput)
TEST(FilterTest, AllRowsPass_ReturnsAllRows)
TEST(FilterTest, NoRowsPass_ReturnsEmptyOutput)
TEST(FilterTest, WithNulls_FiltersNullValues)
```

**Bad:**

```cpp
TEST(FilterTest, Test1)
TEST(FilterTest, Filter)
TEST(FilterTest, Works)
```

### 2. Test Independence

**Each test should be independent:**

```cpp
// ❌ BAD: Tests depend on each other
TEST(FilterTest, Setup) {
    global_state = initialize();
}

TEST(FilterTest, TestFilter) {
    use(global_state);  // Depends on Setup
}

// ✅ GOOD: Each test is independent
TEST(FilterTest, TestFilter1) {
    auto state = initialize();
    use(state);
}

TEST(FilterTest, TestFilter2) {
    auto state = initialize();  // Independent setup
    use(state);
}
```

### 3. Test Coverage

**Aim for comprehensive coverage:**

- **Happy path**: Normal inputs, expected outputs
- **Edge cases**: Empty inputs, single row, large inputs
- **Error cases**: Invalid inputs, out-of-bounds, nulls
- **Performance**: Large datasets, worst-case scenarios

### 4. Assertions

**Use descriptive assertion messages:**

```cpp
// ❌ BAD
ASSERT_EQ(result->get_row_count(), 10);

// ✅ GOOD
ASSERT_EQ(result->get_row_count(), 10)
    << "Expected 10 rows after filtering, but got " << result->get_row_count();
```

---

## Next Steps

**Related Documentation:**

- **[Adding Operators](adding-operators.md)**: Implement and test new operators
- **[Debugging](debugging.md)**: Debug failing tests
- **[Building and Testing](building-and-testing.md)**: Setup test environment

**External Resources:**

- **Google Test**: https://google.github.io/googletest/
- **Google Benchmark**: https://github.com/google/benchmark
- **SQLLogicTest**: https://www.sqlite.org/sqllogictest/
