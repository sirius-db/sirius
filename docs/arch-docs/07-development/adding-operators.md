# Adding New Operators to Sirius

This guide walks through the process of adding a new operator to Sirius, covering both **New Mode** (`sirius_physical_operator`) and **Legacy Mode** (`GPUPhysicalOperator`).

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [New Mode Operator Checklist](#new-mode-operator-checklist)
4. [Example: Adding a DISTINCT Operator](#example-adding-a-distinct-operator)
5. [Legacy Mode Operator Checklist](#legacy-mode-operator-checklist)
6. [Testing Your Operator](#testing-your-operator)
7. [Common Pitfalls](#common-pitfalls)
8. [Next Steps](#next-steps)

---

## Overview

Adding a new operator to Sirius involves several steps:

1. **Define operator type**: Add to operator type enum
2. **Implement operator class**: Create header and source files
3. **Add planner support**: Convert logical → physical operator
4. **Implement execution logic**: Write GPU kernels or cuDF operations
5. **Write tests**: Unit tests (C++) and integration tests (SQL)
6. **Document**: Add to operator catalog and documentation

This guide focuses on **New Mode** operators, as they are the recommended path forward. Legacy Mode operators are covered briefly for completeness.

---

## Prerequisites

Before adding a new operator, ensure you understand:

- **New Mode Architecture**: [New Mode Overview](../04-new-mode/overview.md)
- **Operator Guide**: [New Mode Operators](../04-new-mode/operators.md)
- **cuDF Operations**: Familiarity with RAPIDS cuDF API
- **DuckDB Operators**: Understand the corresponding DuckDB logical operator

**Recommended Reading:**

- [Task Creation](../04-new-mode/task-creation.md): How operators generate tasks
- [Pipeline Execution](../04-new-mode/pipeline-execution.md): How operators fit into pipelines
- [Expression Executor](../05-core-components/expression-executor.md): How to evaluate expressions on GPU

---

## New Mode Operator Checklist

### Step 1: Add Operator Type

**File**: `src/include/op/sirius_physical_operator_type.hpp`

Add your operator to the `SiriusPhysicalOperatorType` enum:

```cpp
enum class SiriusPhysicalOperatorType : uint8_t {
    INVALID = 0,
    // ... existing operators ...
    DISTINCT,            // ← Add your new operator
    // ... more operators ...
};
```

Update the `SiriusPhysicalOperatorTypeToString()` function:

```cpp
inline std::string SiriusPhysicalOperatorTypeToString(SiriusPhysicalOperatorType type) {
    switch (type) {
        // ... existing cases ...
        case SiriusPhysicalOperatorType::DISTINCT:
            return "DISTINCT";
        // ... more cases ...
    }
}
```

### Step 2: Create Operator Header

**File**: `src/include/op/sirius_physical_distinct.hpp`

```cpp
#pragma once

#include "op/sirius_physical_operator.hpp"
#include <cucascade/data/data_batch.hpp>

namespace sirius {
namespace op {

class sirius_physical_distinct : public sirius_physical_operator {
public:
    static constexpr const SiriusPhysicalOperatorType TYPE =
        SiriusPhysicalOperatorType::DISTINCT;

    sirius_physical_distinct(duckdb::vector<duckdb::LogicalType> types,
                             duckdb::idx_t estimated_cardinality);

    // Override base class methods
    std::string get_name() const override { return "DISTINCT"; }

    // Execute method for intermediate operators
    std::vector<std::shared_ptr<cucascade::data_batch>> execute(
        const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
        rmm::cuda_stream_view stream) override;

    // OR: Sink/source methods for pipeline-breaking operators
    // void sink(std::shared_ptr<cucascade::data_batch> batch,
    //           rmm::cuda_stream_view stream) override;
    // task_creation_hint get_next_task_hint() override;
    // std::shared_ptr<cucascade::data_batch> get_next_task_input_batch(
    //     rmm::cuda_stream_view stream) override;

    // Pipeline building
    void build_pipelines(pipeline::sirius_pipeline& current,
                        pipeline::sirius_meta_pipeline& meta_pipeline) override;

private:
    // Private helper methods
    std::shared_ptr<cucascade::data_batch> remove_duplicates(
        std::shared_ptr<cucascade::data_batch> input,
        rmm::cuda_stream_view stream);
};

}  // namespace op
}  // namespace sirius
```

**Key Decisions:**

1. **Intermediate vs Sink?**
   - **Intermediate**: Operator can process batches independently (FILTER, PROJECTION)
   - **Sink**: Operator needs to see all data before producing output (HASH_JOIN, AGGREGATE)

2. **State Management:**
   - **Stateless**: No shared state between batches (FILTER, PROJECTION)
   - **Stateful**: Maintains global state (HASH_JOIN, AGGREGATE)

### Step 3: Implement Operator

**File**: `src/op/sirius_physical_distinct.cpp`

```cpp
#include "op/sirius_physical_distinct.hpp"
#include "log/logging.hpp"
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace sirius {
namespace op {

sirius_physical_distinct::sirius_physical_distinct(
    duckdb::vector<duckdb::LogicalType> types,
    duckdb::idx_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::DISTINCT,
      std::move(types),
      estimated_cardinality)
{
}

std::vector<std::shared_ptr<cucascade::data_batch>>
sirius_physical_distinct::execute(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
    rmm::cuda_stream_view stream)
{
    SIRIUS_LOG_DEBUG("DISTINCT: Processing {} input batches", input_batches.size());

    std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
    output_batches.reserve(input_batches.size());

    for (auto const& batch : input_batches) {
        if (!batch || batch->get_row_count() == 0) {
            continue;
        }

        // Remove duplicates from this batch
        auto distinct_batch = remove_duplicates(batch, stream);

        if (distinct_batch && distinct_batch->get_row_count() > 0) {
            output_batches.push_back(std::move(distinct_batch));
        }
    }

    SIRIUS_LOG_DEBUG("DISTINCT: Produced {} output batches", output_batches.size());
    return output_batches;
}

std::shared_ptr<cucascade::data_batch>
sirius_physical_distinct::remove_duplicates(
    std::shared_ptr<cucascade::data_batch> input,
    rmm::cuda_stream_view stream)
{
    // Step 1: Convert data_batch to cuDF table
    auto cudf_table = input->to_cudf_table();

    // Step 2: Use cuDF to remove duplicates
    auto distinct_table = cudf::unique(
        cudf_table->view(),
        std::vector<cudf::size_type>{},  // Consider all columns
        cudf::duplicate_keep_option::KEEP_FIRST,
        cudf::null_equality::EQUAL,
        stream
    );

    // Step 3: Convert back to data_batch
    auto output_batch = cucascade::data_batch::from_cudf_table(
        std::move(distinct_table),
        stream
    );

    SIRIUS_LOG_DEBUG("DISTINCT: {} rows → {} rows",
                    input->get_row_count(),
                    output_batch->get_row_count());

    return output_batch;
}

void sirius_physical_distinct::build_pipelines(
    pipeline::sirius_pipeline& current,
    pipeline::sirius_meta_pipeline& meta_pipeline)
{
    // For intermediate operators: add to current pipeline
    auto& state = meta_pipeline.get_state();
    state.add_pipeline_operator(current, *this);

    // Continue building with child operator
    children[0]->build_pipelines(current, meta_pipeline);
}

}  // namespace op
}  // namespace sirius
```

**Implementation Notes:**

- **Use cuDF when possible**: Leverage RAPIDS cuDF for common operations
- **Log progress**: Add `SIRIUS_LOG_DEBUG` for debugging
- **Handle empty batches**: Check for null/empty batches
- **Use CUDA streams**: All GPU operations should use the provided stream

### Step 4: Add Planner Support

**File**: `src/planner/sirius_plan_distinct.cpp`

```cpp
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "op/sirius_physical_distinct.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalDistinct& op)
{
    D_ASSERT(op.children.size() == 1);

    // Step 1: Create plan for child operator
    auto child_plan = create_plan(*op.children[0]);

    // Step 2: Create DISTINCT operator
    auto distinct_op = duckdb::make_uniq<sirius::op::sirius_physical_distinct>(
        op.types,
        op.estimated_cardinality
    );

    // Step 3: Set child
    distinct_op->children.push_back(std::move(child_plan));

    return distinct_op;
}

}  // namespace sirius::planner
```

**Register in Planner:**

Update `src/planner/sirius_physical_plan_generator.cpp` or `.hpp` to include your new planner function:

```cpp
class sirius_physical_plan_generator {
public:
    // ... existing methods ...
    duckdb::unique_ptr<sirius::op::sirius_physical_operator>
        create_plan(duckdb::LogicalDistinct& op);
};
```

### Step 5: Update CMakeLists.txt

Add your new files to the build system:

**File**: `CMakeLists.txt` (or relevant subdirectory CMakeLists.txt)

```cmake
# Operator source files
set(SIRIUS_OP_SOURCES
    src/op/sirius_physical_filter.cpp
    src/op/sirius_physical_projection.cpp
    # ... existing operators ...
    src/op/sirius_physical_distinct.cpp  # ← Add your operator
)

# Planner source files
set(SIRIUS_PLANNER_SOURCES
    src/planner/sirius_plan_filter.cpp
    src/planner/sirius_plan_projection.cpp
    # ... existing planners ...
    src/planner/sirius_plan_distinct.cpp  # ← Add your planner
)
```

---

## Example: Adding a DISTINCT Operator

Let's walk through adding a **DISTINCT** operator step-by-step.

### 1. Design Considerations

**SQL Semantics:**

```sql
SELECT DISTINCT column1, column2 FROM table;
```

**Operator Behavior:**

- **Input**: Batches of rows (possibly with duplicates)
- **Output**: Batches with duplicate rows removed
- **Global State**: Needs to track rows seen across all batches
- **Pipeline Behavior**: Should be a **sink operator** to accumulate all batches before removing duplicates

**Alternative Design (Streaming):**

- Process each batch independently (remove duplicates within batch)
- Simpler but less accurate (duplicates across batches remain)
- Suitable for approximate DISTINCT or when memory is constrained

### 2. Implementation (Sink Operator)

**Header**: `src/include/op/sirius_physical_distinct.hpp`

```cpp
#pragma once

#include "op/sirius_physical_operator.hpp"
#include <cucascade/data/data_repository.hpp>
#include <unordered_set>

namespace sirius {
namespace op {

class sirius_physical_distinct : public sirius_physical_operator {
public:
    static constexpr const SiriusPhysicalOperatorType TYPE =
        SiriusPhysicalOperatorType::DISTINCT;

    sirius_physical_distinct(duckdb::vector<duckdb::LogicalType> types,
                             duckdb::idx_t estimated_cardinality);

    std::string get_name() const override { return "DISTINCT"; }

    // Sink interface: accumulate batches
    void sink(std::shared_ptr<cucascade::data_batch> batch,
              rmm::cuda_stream_view stream) override;

    bool is_sink() const override { return true; }

    // Source interface: emit distinct rows after all batches sinked
    task_creation_hint get_next_task_hint() override;

    std::shared_ptr<cucascade::data_batch> get_next_task_input_batch(
        rmm::cuda_stream_view stream) override;

    bool is_source() const override { return true; }

    // Pipeline building
    void build_pipelines(pipeline::sirius_pipeline& current,
                        pipeline::sirius_meta_pipeline& meta_pipeline) override;

private:
    // Accumulated batches
    std::vector<std::shared_ptr<cucascade::data_batch>> accumulated_batches_;

    // Whether finalization is complete
    bool finalized_ = false;

    // Distinct result
    std::shared_ptr<cucascade::data_batch> distinct_result_;

    // Remove duplicates from accumulated batches
    void finalize(rmm::cuda_stream_view stream);
};

}  // namespace op
}  // namespace sirius
```

**Implementation**: `src/op/sirius_physical_distinct.cpp`

```cpp
#include "op/sirius_physical_distinct.hpp"
#include "log/logging.hpp"
#include <cudf/stream_compaction.hpp>
#include <cudf/concatenate.hpp>

namespace sirius {
namespace op {

sirius_physical_distinct::sirius_physical_distinct(
    duckdb::vector<duckdb::LogicalType> types,
    duckdb::idx_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::DISTINCT,
      std::move(types),
      estimated_cardinality)
{
}

void sirius_physical_distinct::sink(
    std::shared_ptr<cucascade::data_batch> batch,
    rmm::cuda_stream_view stream)
{
    if (!batch || batch->get_row_count() == 0) {
        return;
    }

    SIRIUS_LOG_DEBUG("DISTINCT: Sinking batch with {} rows", batch->get_row_count());
    accumulated_batches_.push_back(batch);
}

task_creation_hint sirius_physical_distinct::get_next_task_hint()
{
    if (!finalized_) {
        // Need to finalize: remove duplicates across all batches
        return task_creation_hint{TaskCreationHint::READY, nullptr};
    }

    // Finalization complete, no more tasks
    return task_creation_hint{TaskCreationHint::NO_MORE_TASKS, nullptr};
}

std::shared_ptr<cucascade::data_batch>
sirius_physical_distinct::get_next_task_input_batch(rmm::cuda_stream_view stream)
{
    if (!finalized_) {
        // Finalize: concatenate all batches and remove duplicates
        finalize(stream);
        finalized_ = true;
    }

    // Return distinct result
    auto result = distinct_result_;
    distinct_result_ = nullptr;  // Only return once
    return result;
}

void sirius_physical_distinct::finalize(rmm::cuda_stream_view stream)
{
    if (accumulated_batches_.empty()) {
        SIRIUS_LOG_DEBUG("DISTINCT: No batches to finalize");
        return;
    }

    SIRIUS_LOG_DEBUG("DISTINCT: Finalizing {} batches", accumulated_batches_.size());

    // Step 1: Concatenate all batches
    std::vector<cudf::table_view> table_views;
    for (auto const& batch : accumulated_batches_) {
        table_views.push_back(batch->to_cudf_table()->view());
    }

    auto concatenated = cudf::concatenate(table_views, stream);

    SIRIUS_LOG_DEBUG("DISTINCT: Concatenated {} rows", concatenated->num_rows());

    // Step 2: Remove duplicates
    auto distinct_table = cudf::unique(
        concatenated->view(),
        std::vector<cudf::size_type>{},  // Consider all columns
        cudf::duplicate_keep_option::KEEP_FIRST,
        cudf::null_equality::EQUAL,
        stream
    );

    SIRIUS_LOG_DEBUG("DISTINCT: {} rows after deduplication", distinct_table->num_rows());

    // Step 3: Convert to data_batch
    distinct_result_ = cucascade::data_batch::from_cudf_table(
        std::move(distinct_table),
        stream
    );

    // Free accumulated batches
    accumulated_batches_.clear();
}

void sirius_physical_distinct::build_pipelines(
    pipeline::sirius_pipeline& current,
    pipeline::sirius_meta_pipeline& meta_pipeline)
{
    // DISTINCT is a sink operator: breaks the pipeline
    auto& state = meta_pipeline.get_state();
    state.set_pipeline_sink(current, *this, 1);

    // Build child operator (continues current pipeline)
    children[0]->build_pipelines(current, meta_pipeline);
}

}  // namespace op
}  // namespace sirius
```

### 3. Testing

**Unit Test** (`test/cpp/operator/test_distinct.cpp`):

```cpp
#include "op/sirius_physical_distinct.hpp"
#include <gtest/gtest.h>

TEST(DistinctTest, BasicDistinct) {
    // Create input batch with duplicates
    auto input = create_test_batch({
        {"id", {1, 2, 2, 3, 1, 3}},
        {"name", {"A", "B", "B", "C", "A", "C"}}
    });

    // Create DISTINCT operator
    sirius::op::sirius_physical_distinct distinct_op(
        {duckdb::LogicalType::INTEGER, duckdb::LogicalType::VARCHAR},
        6
    );

    // Sink input batch
    distinct_op.sink(input, rmm::cuda_stream_default);

    // Get distinct result
    auto hint = distinct_op.get_next_task_hint();
    ASSERT_EQ(hint.hint, TaskCreationHint::READY);

    auto result = distinct_op.get_next_task_input_batch(rmm::cuda_stream_default);

    // Verify result
    ASSERT_EQ(result->get_row_count(), 3);  // 3 distinct rows
    // Verify data: {1, "A"}, {2, "B"}, {3, "C"}
}
```

**SQL Integration Test** (`test/sql/distinct.test`):

```sql
# test/sql/distinct.test

statement ok
CREATE TABLE test (id INTEGER, name VARCHAR);

statement ok
INSERT INTO test VALUES (1, 'A'), (2, 'B'), (2, 'B'), (3, 'C'), (1, 'A');

query II
SELECT DISTINCT * FROM gpu_execution('SELECT * FROM test ORDER BY id');
----
1	A
2	B
3	C

query I
SELECT DISTINCT id FROM gpu_execution('SELECT * FROM test ORDER BY id');
----
1
2
3
```

---

## Legacy Mode Operator Checklist

For completeness, here's a brief overview of adding Legacy Mode operators.

### Steps

1. **Add to GPUPhysicalOperator Type Enum**
2. **Create Operator Class**: Inherit from `GPUPhysicalOperator`
3. **Implement Methods**:
   - `GetData()` for source operators
   - `Execute()` for intermediate operators
   - `Sink()` and `CombineFinalize()` for sink operators
4. **Add Planner Support**: Implement `create_plan()` in `GPUPhysicalPlanGenerator`
5. **Write Tests**

**Example Header** (`src/include/operator/gpu_physical_distinct.hpp`):

```cpp
#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalDistinct : public GPUPhysicalOperator {
public:
    static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::DISTINCT;

    GPUPhysicalDistinct(vector<LogicalType> types, idx_t estimated_cardinality);

    OperatorResultType Execute(GPUIntermediateRelation& input_relation,
                               GPUIntermediateRelation& output_relation) const override;
};

}  // namespace duckdb
```

**Note:** Legacy Mode is deprecated. Focus on New Mode for new operators.

---

## Testing Your Operator

### 1. Unit Tests (C++)

**Location**: `test/cpp/operator/test_<operator_name>.cpp`

**Example**: `test/cpp/operator/test_distinct.cpp`

```cpp
#include "op/sirius_physical_distinct.hpp"
#include <gtest/gtest.h>

TEST(DistinctTest, EmptyBatch) {
    sirius::op::sirius_physical_distinct distinct_op(...);
    auto result = distinct_op.execute({}, rmm::cuda_stream_default);
    ASSERT_TRUE(result.empty());
}

TEST(DistinctTest, NoDuplicates) {
    // Test case: input has no duplicates
}

TEST(DistinctTest, AllDuplicates) {
    // Test case: all rows are duplicates
}

TEST(DistinctTest, WithNulls) {
    // Test case: input contains NULL values
}
```

**Run Unit Tests:**

```bash
cd build
make test_distinct
./test/cpp/operator/test_distinct
```

### 2. SQL Integration Tests

**Location**: `test/sql/<operator_name>.test`

**Example**: `test/sql/distinct.test`

```sql
# Basic DISTINCT test
statement ok
CREATE TABLE t (a INTEGER, b VARCHAR);

statement ok
INSERT INTO t VALUES (1, 'A'), (1, 'A'), (2, 'B');

query II
SELECT DISTINCT * FROM gpu_execution('SELECT * FROM t ORDER BY a');
----
1	A
2	B

# DISTINCT with NULL
statement ok
INSERT INTO t VALUES (NULL, 'C'), (NULL, 'C');

query II
SELECT DISTINCT * FROM gpu_execution('SELECT * FROM t WHERE a IS NULL');
----
NULL	C

# DISTINCT on single column
query I
SELECT DISTINCT a FROM gpu_execution('SELECT * FROM t ORDER BY a');
----
NULL
1
2
```

**Run SQL Tests:**

```bash
cd build
make sqllogictest
./test/sql/sqllogictest test/sql/distinct.test
```

### 3. Performance Benchmarks

**Location**: `benchmark/<operator_name>_benchmark.cpp`

```cpp
#include <benchmark/benchmark.h>
#include "op/sirius_physical_distinct.hpp"

static void BM_Distinct_1M_Rows(benchmark::State& state) {
    auto input = create_large_test_batch(1000000);  // 1M rows

    sirius::op::sirius_physical_distinct distinct_op(...);

    for (auto _ : state) {
        auto result = distinct_op.execute({input}, rmm::cuda_stream_default);
        cudaStreamSynchronize(rmm::cuda_stream_default);
    }
}

BENCHMARK(BM_Distinct_1M_Rows);
```

**Run Benchmarks:**

```bash
cd build
make distinct_benchmark
./benchmark/distinct_benchmark
```

---

## Common Pitfalls

### 1. Forgetting to Handle Empty Batches

**Problem:**

```cpp
auto result = input_batches[0]->to_cudf_table();  // ❌ Crashes if empty
```

**Solution:**

```cpp
if (input_batches.empty() || !input_batches[0]) {
    return {};  // Return empty output
}
```

### 2. Not Using CUDA Streams

**Problem:**

```cpp
auto result = cudf::filter(table, predicate);  // ❌ Uses default stream
```

**Solution:**

```cpp
auto result = cudf::filter(table, predicate, stream);  // ✅ Uses provided stream
```

### 3. Memory Leaks

**Problem:**

```cpp
auto* ptr = cudaMalloc(...);  // ❌ Never freed
```

**Solution:**

```cpp
// Use data_batch (manages memory automatically)
auto batch = cucascade::data_batch::from_cudf_table(...);
```

### 4. Incorrect Pipeline Building

**Problem:**

```cpp
void build_pipelines(...) {
    // ❌ Forgot to call child->build_pipelines()
}
```

**Solution:**

```cpp
void build_pipelines(pipeline::sirius_pipeline& current, ...) {
    auto& state = meta_pipeline.get_state();
    state.add_pipeline_operator(current, *this);
    children[0]->build_pipelines(current, meta_pipeline);  // ✅
}
```

### 5. Not Logging Execution

**Problem:**

```cpp
auto result = process_data(input);  // ❌ No logging
```

**Solution:**

```cpp
SIRIUS_LOG_DEBUG("Processing {} rows", input->get_row_count());
auto result = process_data(input);
SIRIUS_LOG_DEBUG("Produced {} rows", result->get_row_count());
```

---

## Next Steps

**Related Documentation:**

- **[New Mode Operators](../04-new-mode/operators.md)**: Understand existing operator patterns
- **[Task Creation](../04-new-mode/task-creation.md)**: How to implement get_next_task_hint()
- **[Expression Executor](../05-core-components/expression-executor.md)**: Evaluating expressions
- **[Testing Guide](testing-guide.md)**: Comprehensive testing strategies

**Tools:**

- **[Debugging Guide](debugging.md)**: Debugging your operator
- **[Building and Testing](building-and-testing.md)**: Setting up the development environment

**Examples:**

- Study existing operators in `src/op/sirius_physical_*.cpp`
- Look at planner implementations in `src/planner/sirius_plan_*.cpp`
- Review tests in `test/cpp/operator/` and `test/sql/`
