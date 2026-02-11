# New Mode Operators

Comprehensive guide to operators in Sirius New Mode, covering the `sirius_physical_operator` base class, operator types, and implementation patterns.

---

## Overview

New Mode operators inherit from `sirius_physical_operator` and implement a **task-driven execution model** with dynamic task creation.

**Key Differences from Legacy Mode**:

| Aspect | Legacy Mode | New Mode |
|--------|-------------|----------|
| **Base Class** | `GPUPhysicalOperator` | `sirius_physical_operator` |
| **Execution** | Pull-based (`GetData()`) | Push-based (`execute()`/`sink()`) |
| **Task Model** | Static | Dynamic (hint-based) |
| **Communication** | Direct state | Port-based repositories |
| **Data Unit** | `GPUIntermediateRelation` | `cucascade::data_batch` |

---

## Operator Base Class

### sirius_physical_operator

**Definition**: `src/include/op/sirius_physical_operator.hpp`

```cpp
class sirius_physical_operator {
protected:
    // Operator metadata
    SiriusPhysicalOperatorType type;
    std::string name;

    // Pipeline connections
    std::vector<std::shared_ptr<shared_data_repository>> input_ports;
    std::vector<std::shared_ptr<shared_data_repository>> output_ports;

    // Children operators
    std::vector<std::unique_ptr<sirius_physical_operator>> children;

    // Execution context
    SiriusContext& context;

public:
    // Constructor
    sirius_physical_operator(
        SiriusPhysicalOperatorType type,
        SiriusContext& ctx
    );

    // Operator identification
    SiriusPhysicalOperatorType get_type() const { return type; }
    std::string get_name() const { return name; }

    // Task creation interface
    virtual TaskCreationHint get_next_task_hint();
    virtual data_batch get_next_task_input_batch();

    // Execution interface
    virtual data_batch execute(data_batch&& input);
    virtual void sink(data_batch&& input);
    virtual void finalize();

    // Port management
    void add_input_port(std::shared_ptr<shared_data_repository> port);
    void add_output_port(std::shared_ptr<shared_data_repository> port);

    // Child management
    void add_child(std::unique_ptr<sirius_physical_operator> child);
    std::vector<sirius_physical_operator*> get_children();

    // Utilities
    virtual std::string to_string() const;
    virtual void print(size_t indent = 0) const;
};
```

### Task Creation Interface

**Purpose**: Tell task creator when tasks can be created.

**get_next_task_hint()**

Returns a hint about task availability:

```cpp
enum class TaskCreationHint {
    READY,                   // Task can be created now
    WAITING_FOR_INPUT_DATA,  // Waiting for input repository
    NO_MORE_TASKS           // Operator complete
};
```

**Default Implementation** (`src/op/sirius_physical_operator.cpp:50-80`):

```cpp
TaskCreationHint sirius_physical_operator::get_next_task_hint() {
    // Default: check input ports
    if (input_ports.empty()) {
        // No inputs: ready if not exhausted
        return has_more_work() ? TaskCreationHint::READY
                                : TaskCreationHint::NO_MORE_TASKS;
    }

    // Check first input port
    auto& input_repo = input_ports[0];

    if (input_repo->has_data()) {
        return TaskCreationHint::READY;
    }

    if (input_repo->is_complete()) {
        if (input_repo->is_empty()) {
            return TaskCreationHint::NO_MORE_TASKS;
        }
        return TaskCreationHint::READY; // Pull remaining data
    }

    // Waiting for producer
    return TaskCreationHint::WAITING_FOR_INPUT_DATA;
}
```

**get_next_task_input_batch()**

Returns input for next task:

```cpp
data_batch sirius_physical_operator::get_next_task_input_batch() {
    if (input_ports.empty()) {
        throw InternalException("No input ports");
    }

    auto& input_repo = input_ports[0];

    // Pull batch from repository (may block)
    auto batch_opt = input_repo->pull_batch();

    if (!batch_opt.has_value()) {
        throw InternalException("No input batch available");
    }

    return std::move(batch_opt.value());
}
```

### Execution Interface

**execute()**

Transform one input batch to one output batch:

```cpp
virtual data_batch execute(data_batch&& input) {
    // Default: pass through
    return std::move(input);
}
```

**Example**: Filter operator

```cpp
data_batch sirius_physical_filter::execute(data_batch&& input) {
    // Apply predicate using cuDF
    auto mask = evaluate_predicate(input.table, filter_expr);
    auto filtered = cudf::apply_boolean_mask(input.table, mask);

    return data_batch{
        .table = std::move(filtered),
        .tier = MemoryTier::GPU,
        .num_rows = filtered->num_rows()
    };
}
```

**sink()**

Consume input batch (no output):

```cpp
virtual void sink(data_batch&& input) {
    // Default: no-op
    throw InternalException("Operator does not support sink");
}
```

**Example**: Aggregate operator

```cpp
void sirius_physical_hash_group_by::sink(data_batch&& input) {
    // Accumulate into hash table
    hash_table->aggregate(input.table);

    // Check if should flush
    if (should_flush()) {
        auto result = hash_table->finalize();
        publish_output(std::move(result));
        hash_table->reset();
    }
}
```

**finalize()**

Complete operator execution:

```cpp
virtual void finalize() {
    // Default: no-op
}
```

**Example**: Aggregate operator

```cpp
void sirius_physical_hash_group_by::finalize() {
    // Flush remaining data
    auto final_batch = hash_table->finalize();

    if (final_batch.num_rows > 0) {
        publish_output(std::move(final_batch));
    }

    // Mark output complete
    if (!output_ports.empty()) {
        output_ports[0]->mark_complete();
    }
}
```

---

## Operator Type Enumeration

**Definition**: `src/include/op/sirius_physical_operator_type.hpp`

```cpp
enum class SiriusPhysicalOperatorType {
    // Scans (source operators)
    TABLE_SCAN,              // Read from table
    DUCKDB_SCAN,             // Scan from DuckDB
    DUMMY_SCAN,              // Pass-through scan
    COLUMN_DATA_SCAN,        // Scan from column data

    // Filters
    FILTER,                  // Apply predicate

    // Projections
    PROJECTION,              // Select columns / expressions

    // Aggregates
    UNGROUPED_AGGREGATE,     // Single-group aggregate
    HASH_GROUP_BY,           // Multi-group aggregate
    PERFECT_HASH_AGGREGATE,  // Optimized for dense keys

    // Sorting
    ORDER_BY,                // Sort all columns
    TOP_N,                   // Sort + limit
    MERGE_SORT,              // Merge sorted streams

    // Joins
    HASH_JOIN,               // Hash-based join
    NESTED_LOOP_JOIN,        // Nested loop join
    PIECEWISE_MERGE_JOIN,    // Merge join

    // Partitioning
    PARTITION,               // Partition data
    SORT_PARTITION,          // Sort within partitions

    // Limits
    LIMIT,                   // Limit rows

    // Output
    RESULT_COLLECTOR,        // Collect final results

    // Utility
    EMPTY_RESULT             // Return empty result
};
```

### Operator Categories

#### Source Operators (Create Data)

- **TABLE_SCAN**: Read from Parquet/CSV files
- **DUCKDB_SCAN**: Read from DuckDB tables
- **DUMMY_SCAN**: Pass-through for pipeline breaks
- **COLUMN_DATA_SCAN**: Read from in-memory columns

#### Processing Operators (Transform Data)

- **FILTER**: Apply WHERE predicates
- **PROJECTION**: Apply SELECT expressions
- **LIMIT**: Apply LIMIT clause

#### Aggregate Operators (Reduce Data)

- **UNGROUPED_AGGREGATE**: No GROUP BY (e.g., `SELECT COUNT(*)`)
- **HASH_GROUP_BY**: With GROUP BY (e.g., `SELECT category, SUM(price) GROUP BY category`)
- **PERFECT_HASH_AGGREGATE**: Optimized for integer keys

#### Sort Operators (Order Data)

- **ORDER_BY**: Full sort
- **TOP_N**: Sort + limit optimization
- **MERGE_SORT**: Merge pre-sorted batches

#### Join Operators (Combine Data)

- **HASH_JOIN**: Hash-based equi-join
- **NESTED_LOOP_JOIN**: Cross join / complex predicates
- **PIECEWISE_MERGE_JOIN**: Merge join

#### Partition Operators (Distribute Data)

- **PARTITION**: Partition by hash/range
- **SORT_PARTITION**: Sort within partitions

#### Sink Operators (Consume Data)

- **RESULT_COLLECTOR**: Final output

---

## Operator Implementation Patterns

### Pattern 1: Stateless Transformation

**Characteristics**:
- No internal state
- One input batch → one output batch
- Fully parallelizable

**Example**: Filter

**Code** (`src/op/sirius_physical_filter.cpp`):

```cpp
class sirius_physical_filter : public sirius_physical_operator {
private:
    // Filter expression
    std::unique_ptr<Expression> filter_expr;

public:
    sirius_physical_filter(
        std::unique_ptr<Expression> expr,
        SiriusContext& ctx
    ) : sirius_physical_operator(SiriusPhysicalOperatorType::FILTER, ctx),
        filter_expr(std::move(expr)) {}

    // Task creation
    TaskCreationHint get_next_task_hint() override {
        // Use default implementation (check input ports)
        return sirius_physical_operator::get_next_task_hint();
    }

    data_batch get_next_task_input_batch() override {
        // Use default implementation (pull from input port)
        return sirius_physical_operator::get_next_task_input_batch();
    }

    // Execution
    data_batch execute(data_batch&& input) override {
        if (input.num_rows == 0) {
            return std::move(input); // Empty input
        }

        // Evaluate predicate
        auto mask = evaluate_filter(input.table, filter_expr.get());

        // Apply mask
        auto filtered = cudf::apply_boolean_mask(
            input.table->view(),
            mask->view()
        );

        return data_batch{
            .table = std::move(filtered),
            .tier = MemoryTier::GPU,
            .num_rows = filtered->num_rows()
        };
    }

private:
    std::unique_ptr<cudf::column> evaluate_filter(
        const std::unique_ptr<cudf::table>& table,
        Expression* expr
    ) {
        // Use expression executor
        ExpressionExecutor executor(context);
        return executor.evaluate_boolean(table, expr);
    }
};
```

**Usage**:
```
Input:  [100K rows]
    ↓ execute()
Output: [60K rows] (60% selectivity)
```

### Pattern 2: Stateful Accumulation

**Characteristics**:
- Maintains internal state
- Multiple input batches → single output batch
- Requires finalization

**Example**: Hash Group By

**Code** (`src/op/sirius_physical_hash_group_by.cpp`):

```cpp
class sirius_physical_hash_group_by : public sirius_physical_operator {
private:
    // Grouping columns
    std::vector<size_t> group_indices;

    // Aggregate specifications
    std::vector<AggregateFunction> aggregates;

    // State: hash table
    std::unique_ptr<cudf::groupby::groupby> hash_table;
    size_t rows_accumulated = 0;

    // Flushing
    bool should_flush() const {
        return rows_accumulated > context.config.aggregate_flush_threshold;
    }

public:
    sirius_physical_hash_group_by(
        std::vector<size_t> groups,
        std::vector<AggregateFunction> aggs,
        SiriusContext& ctx
    ) : sirius_physical_operator(SiriusPhysicalOperatorType::HASH_GROUP_BY, ctx),
        group_indices(std::move(groups)),
        aggregates(std::move(aggs)) {}

    // Execution (sink mode)
    void sink(data_batch&& input) override {
        if (input.num_rows == 0) return;

        // Extract group keys
        auto keys = input.table->select(group_indices);

        // Extract aggregate values
        std::vector<cudf::groupby::aggregation_request> requests;
        for (const auto& agg : aggregates) {
            requests.push_back(make_aggregation_request(input.table, agg));
        }

        // Update hash table
        if (!hash_table) {
            // First batch: create hash table
            hash_table = std::make_unique<cudf::groupby::groupby>(keys);
        }

        hash_table->aggregate(requests);
        rows_accumulated += input.num_rows;

        // Flush if threshold reached
        if (should_flush()) {
            flush_partial();
        }
    }

    void finalize() override {
        // Flush remaining data
        if (hash_table) {
            auto result = hash_table->get_result();
            publish_output(to_data_batch(result));
        }

        // Mark output complete
        if (!output_ports.empty()) {
            output_ports[0]->mark_complete();
        }
    }

private:
    void flush_partial() {
        auto result = hash_table->get_result();
        publish_output(to_data_batch(result));

        // Reset for next batch
        hash_table.reset();
        rows_accumulated = 0;
    }

    void publish_output(data_batch&& batch) {
        if (!output_ports.empty() && batch.num_rows > 0) {
            output_ports[0]->push_data_batch(std::move(batch));
        }
    }
};
```

**Usage**:
```
Input batch 0:  [100K rows] → sink() → accumulate
Input batch 1:  [100K rows] → sink() → accumulate
Input batch 2:  [100K rows] → sink() → accumulate
    ...
Input batch 9:  [100K rows] → sink() → accumulate
    ↓ finalize()
Output: [1K unique groups]
```

### Pattern 3: Buffering Operator

**Characteristics**:
- Buffers all input
- Single output after all input consumed
- Pipeline break

**Example**: Order By

**Code** (`src/op/sirius_physical_order_by.cpp`):

```cpp
class sirius_physical_order_by : public sirius_physical_operator {
private:
    // Sort specifications
    std::vector<SortColumn> sort_columns;

    // Buffered data
    std::vector<data_batch> buffered_batches;
    bool all_input_received = false;
    bool has_emitted = false;

public:
    sirius_physical_order_by(
        std::vector<SortColumn> sorts,
        SiriusContext& ctx
    ) : sirius_physical_operator(SiriusPhysicalOperatorType::ORDER_BY, ctx),
        sort_columns(std::move(sorts)) {}

    // Task creation
    TaskCreationHint get_next_task_hint() override {
        if (has_emitted) {
            return TaskCreationHint::NO_MORE_TASKS;
        }

        if (!all_input_received) {
            // Still collecting input
            auto& input_repo = input_ports[0];

            if (input_repo->has_data()) {
                return TaskCreationHint::READY;
            }

            if (input_repo->is_complete()) {
                all_input_received = true;
                return TaskCreationHint::READY; // Ready to sort
            }

            return TaskCreationHint::WAITING_FOR_INPUT_DATA;
        }

        // All input received, ready to sort
        return TaskCreationHint::READY;
    }

    data_batch get_next_task_input_batch() override {
        if (!all_input_received) {
            // Pull and buffer
            return sirius_physical_operator::get_next_task_input_batch();
        } else {
            // Return dummy batch (will sort in execute)
            return data_batch{};
        }
    }

    // Execution
    data_batch execute(data_batch&& input) override {
        if (!all_input_received) {
            // Buffering phase
            if (input.num_rows > 0) {
                buffered_batches.push_back(std::move(input));
            }
            return data_batch{}; // No output yet
        } else {
            // Sorting phase
            if (has_emitted) {
                return data_batch{}; // Already emitted
            }

            // Concatenate all buffered batches
            auto combined = concatenate_batches(buffered_batches);
            buffered_batches.clear(); // Free memory

            // Sort
            auto sorted = cudf::sort(
                combined.table->view(),
                to_cudf_sort_columns(sort_columns)
            );

            has_emitted = true;

            return data_batch{
                .table = std::move(sorted),
                .tier = MemoryTier::GPU,
                .num_rows = sorted->num_rows()
            };
        }
    }

    void finalize() override {
        // Mark output complete
        if (!output_ports.empty()) {
            output_ports[0]->mark_complete();
        }
    }
};
```

**Usage**:
```
Phase 1: Buffering
  Input batch 0 → execute() → buffer
  Input batch 1 → execute() → buffer
  ...
  Input batch 9 → execute() → buffer
  Input complete

Phase 2: Sorting
  execute() with dummy input
    ↓
  Concatenate all batches
    ↓
  Sort combined table
    ↓
  Output: [1M rows, sorted]
```

### Pattern 4: Two-Phase Operator

**Characteristics**:
- Build phase (sink)
- Probe phase (execute)
- Requires coordination

**Example**: Hash Join

**Code** (`src/op/sirius_physical_hash_join.cpp`):

```cpp
class sirius_physical_hash_join : public sirius_physical_operator {
private:
    // Join configuration
    JoinType join_type;
    std::vector<JoinCondition> conditions;

    // Build side state
    std::unique_ptr<cudf::hash_join> hash_table;
    std::unique_ptr<cudf::table> build_payload;

    // Phase tracking
    bool build_complete = false;
    std::mutex build_mutex;
    std::condition_variable build_cv;

public:
    sirius_physical_hash_join(
        JoinType type,
        std::vector<JoinCondition> conds,
        SiriusContext& ctx
    ) : sirius_physical_operator(SiriusPhysicalOperatorType::HASH_JOIN, ctx),
        join_type(type),
        conditions(std::move(conds)) {}

    // Phase 1: Build (runs in separate pipeline)
    void sink_build(data_batch&& batch) {
        // Extract build keys
        auto keys = extract_keys(batch.table, conditions, BuildSide);

        // Extract payload
        auto payload = extract_payload(batch.table, BuildSide);

        // Build hash table
        if (!hash_table) {
            hash_table = std::make_unique<cudf::hash_join>(
                keys->view(),
                cudf::nullable_join::YES
            );
            build_payload = std::move(payload);
        } else {
            hash_table->append(keys->view());
            build_payload = cudf::concatenate({build_payload, payload});
        }
    }

    void finalize_build() {
        std::lock_guard<std::mutex> lock(build_mutex);

        if (hash_table) {
            hash_table->finalize();
        }

        build_complete = true;
        build_cv.notify_all(); // Wake probe tasks
    }

    // Phase 2: Probe (runs in different pipeline)
    TaskCreationHint get_next_task_hint() override {
        // Wait for build to complete
        if (!build_complete) {
            return TaskCreationHint::WAITING_FOR_INPUT_DATA;
        }

        // Use default implementation
        return sirius_physical_operator::get_next_task_hint();
    }

    data_batch execute(data_batch&& probe_batch) override {
        // Wait for build
        {
            std::unique_lock<std::mutex> lock(build_mutex);
            build_cv.wait(lock, [this]() {
                return build_complete;
            });
        }

        if (probe_batch.num_rows == 0) {
            return std::move(probe_batch);
        }

        // Extract probe keys
        auto keys = extract_keys(probe_batch.table, conditions, ProbeSide);

        // Probe hash table
        auto [left_indices, right_indices] = hash_table->probe(keys->view());

        // Gather matching rows
        auto left_result = cudf::gather(
            probe_batch.table->view(),
            left_indices->view()
        );

        auto right_result = cudf::gather(
            build_payload->view(),
            right_indices->view()
        );

        // Concatenate columns
        auto joined = cudf::concatenate_tables({left_result, right_result});

        return data_batch{
            .table = std::move(joined),
            .tier = MemoryTier::GPU,
            .num_rows = joined->num_rows()
        };
    }
};
```

**Usage**:
```
Pipeline 1 (Build):
  Input batch 0 (build side) → sink_build()
  Input batch 1 (build side) → sink_build()
  ...
  finalize_build() → notify probe

Pipeline 2 (Probe):
  Wait for build complete
    ↓
  Input batch 0 (probe side) → execute() → output
  Input batch 1 (probe side) → execute() → output
  ...
```

---

## Operator-Specific Details

### TABLE_SCAN

**Purpose**: Read data from Parquet/CSV files

**Implementation**: `src/op/sirius_physical_table_scan.cpp`

**Key Methods**:
```cpp
TaskCreationHint get_next_task_hint() override {
    if (current_batch_idx >= total_batches) {
        return TaskCreationHint::NO_MORE_TASKS;
    }
    return TaskCreationHint::READY;
}

data_batch get_next_task_input_batch() override {
    // No input needed (source operator)
    return data_batch{};
}

data_batch execute(data_batch&&) override {
    // Read next batch from file
    auto batch = read_parquet_batch(
        file_path,
        current_batch_idx,
        context.config.scan_batch_size
    );

    current_batch_idx++;

    return batch;
}
```

### HASH_GROUP_BY

**Purpose**: Aggregate with GROUP BY

**Implementation**: `src/op/sirius_physical_hash_group_by.cpp`

**Aggregation Functions**:
- `SUM`, `COUNT`, `AVG`, `MIN`, `MAX`
- `STDDEV`, `VARIANCE`
- `FIRST`, `LAST`

**Example**:
```sql
SELECT category, SUM(amount), COUNT(*)
FROM sales
GROUP BY category
```

**cuDF Operations**:
```cpp
// Create groupby object
auto groupby = cudf::groupby::groupby(
    keys_table.select({"category"})
);

// Aggregate requests
std::vector<cudf::groupby::aggregation_request> requests;
requests.push_back(cudf::groupby::aggregation_request{
    .values = values_table.column("amount"),
    .aggregations = {cudf::make_sum_aggregation()}
});
requests.push_back(cudf::groupby::aggregation_request{
    .values = values_table.column("id"), // Any column
    .aggregations = {cudf::make_count_aggregation()}
});

// Execute
auto [result_keys, result_aggs] = groupby.aggregate(requests);
```

### ORDER_BY

**Purpose**: Sort rows

**Implementation**: `src/op/sirius_physical_order_by.cpp`

**Sort Specifications**:
```cpp
struct SortColumn {
    size_t column_idx;
    OrderType order;  // ASCENDING or DESCENDING
    NullOrder null_order;  // NULLS_FIRST or NULLS_LAST
};
```

**cuDF Operation**:
```cpp
auto sorted = cudf::sort(
    table->view(),
    sort_columns,  // Column indices
    sort_orders,   // ASC/DESC per column
    null_precedence  // NULLS_FIRST/LAST per column
);
```

### HASH_JOIN

**Purpose**: Equi-join using hash table

**Implementation**: `src/op/sirius_physical_hash_join.cpp`

**Join Types**:
- `INNER`: Only matching rows
- `LEFT`: All left + matching right
- `RIGHT`: All right + matching left
- `FULL`: All rows from both sides
- `SEMI`: Left rows with matches (no right columns)
- `ANTI`: Left rows without matches

**cuDF Operation**:
```cpp
// Build phase
auto hash_join = cudf::hash_join(
    build_keys,
    cudf::nullable_join::YES
);

// Probe phase
auto [left_indices, right_indices] = hash_join.inner_join(probe_keys);

// Gather
auto result_left = cudf::gather(probe_table, left_indices);
auto result_right = cudf::gather(build_table, right_indices);
```

---

## Operator Lifecycle

### Creation

**By Planner** (`src/planner/sirius_physical_plan_generator.cpp`):

```cpp
std::unique_ptr<sirius_physical_operator>
CreateOperatorFromLogical(LogicalOperator& logical_op) {
    switch (logical_op.type) {
        case LogicalOperatorType::GET:
            return std::make_unique<sirius_physical_table_scan>(...);

        case LogicalOperatorType::FILTER:
            return std::make_unique<sirius_physical_filter>(...);

        case LogicalOperatorType::AGGREGATE_AND_GROUP_BY:
            return std::make_unique<sirius_physical_hash_group_by>(...);

        case LogicalOperatorType::ORDER_BY:
            return std::make_unique<sirius_physical_order_by>(...);

        case LogicalOperatorType::COMPARISON_JOIN:
            return std::make_unique<sirius_physical_hash_join>(...);

        // ... other operators ...
    }
}
```

### Execution

**By Task Executor** (`src/pipeline/sirius_pipeline_itask.cpp`):

```cpp
void sirius_pipeline_itask::compute_task() {
    // 1. Get input (if needed)
    data_batch input;
    if (operator needs input) {
        input = operator->get_next_task_input_batch();
    }

    // 2. Execute
    data_batch output;
    if (operator has execute()) {
        output = operator->execute(std::move(input));
    } else if (operator has sink()) {
        operator->sink(std::move(input));
    }

    // 3. Publish output (if any)
    if (output.num_rows > 0 && !output_ports.empty()) {
        for (auto& port : output_ports) {
            port->push_data_batch(output.clone());
        }
    }
}
```

### Cleanup

**By Engine** (`src/sirius_engine.cpp:finalize()`):

```cpp
void sirius_engine::finalize() {
    // Finalize all operators
    for (auto& pipeline : pipelines) {
        for (auto& op : pipeline.operators) {
            op->finalize();
        }
    }

    // Mark all repositories complete
    for (auto& repo : repositories) {
        if (!repo->is_complete()) {
            repo->mark_complete();
        }
    }
}
```

---

## Adding New Operators

### Step-by-Step Guide

**1. Create Header** (`src/include/op/sirius_physical_myop.hpp`):

```cpp
#pragma once

#include "op/sirius_physical_operator.hpp"

class sirius_physical_myop : public sirius_physical_operator {
public:
    sirius_physical_myop(/* params */, SiriusContext& ctx);

    // Override necessary methods
    TaskCreationHint get_next_task_hint() override;
    data_batch execute(data_batch&& input) override;
    // or: void sink(data_batch&& input) override;
    void finalize() override;

private:
    // Operator-specific state
};
```

**2. Implement** (`src/op/sirius_physical_myop.cpp`):

```cpp
#include "op/sirius_physical_myop.hpp"

sirius_physical_myop::sirius_physical_myop(/* params */, SiriusContext& ctx)
    : sirius_physical_operator(SiriusPhysicalOperatorType::MYOP, ctx) {
    // Initialize
}

TaskCreationHint sirius_physical_myop::get_next_task_hint() {
    // Implement task hint logic
}

data_batch sirius_physical_myop::execute(data_batch&& input) {
    // Implement transformation
}

void sirius_physical_myop::finalize() {
    // Cleanup and mark complete
}
```

**3. Add to Enum** (`src/include/op/sirius_physical_operator_type.hpp`):

```cpp
enum class SiriusPhysicalOperatorType {
    // ... existing types ...
    MYOP,  // Add new type
};
```

**4. Update Planner** (`src/planner/sirius_plan_*.cpp`):

```cpp
std::unique_ptr<sirius_physical_operator>
CreateMyOp(LogicalOperator& logical_op, SiriusContext& ctx) {
    // Extract parameters from logical operator
    auto params = ExtractParams(logical_op);

    // Create physical operator
    return std::make_unique<sirius_physical_myop>(params, ctx);
}
```

**5. Write Tests** (`test/cpp/operator/test_myop.cpp`):

```cpp
TEST_CASE("MyOp basic functionality") {
    SiriusContext ctx;
    auto op = std::make_unique<sirius_physical_myop>(params, ctx);

    // Test execution
    data_batch input = create_test_batch();
    auto output = op->execute(std::move(input));

    REQUIRE(output.num_rows == expected_rows);
    // ... more assertions ...
}
```

---

## See Also

- [New Mode Overview](overview.md) - Introduction to New Mode
- [Entry Points](entry-points.md) - How operators are invoked
- [Operator Guide](operator-guide.md) - Detailed operator reference
- [Pipeline Execution](pipeline-execution.md) - How operators execute in pipelines
- [Task Creation](task-creation.md) - Task creation hints
- [Cucascade Integration](cucascade-integration.md) - Data batch details
- [New Data Flow](../06-data-flow/new-data-flow.md) - Complete data flow
