# Operator Guide (New Mode)

Comprehensive reference guide for all Sirius New Mode operators, organized by category with detailed explanations, examples, and usage patterns.

---

## Overview

This guide provides **practical, operator-specific information** for developers working with Sirius New Mode operators.

**Categories**:
1. **Scan Operators** - Read data from sources
2. **Filter Operators** - Apply predicates
3. **Projection Operators** - Transform columns
4. **Aggregate Operators** - Group and aggregate
5. **Sort Operators** - Order data
6. **Join Operators** - Combine tables
7. **Partition Operators** - Distribute data
8. **Limit Operators** - Restrict rows
9. **Output Operators** - Collect results

---

## 1. Scan Operators

### TABLE_SCAN

**Purpose**: Read data from Parquet or CSV files

**File**: `src/op/sirius_physical_table_scan.cpp`

**SQL Examples**:
```sql
-- Parquet file
SELECT * FROM 'data/sales.parquet';

-- CSV file
SELECT * FROM 'data/customers.csv';

-- With filters (pushed down)
SELECT * FROM 'data/orders.parquet' WHERE year = 2024;
```

**Key Features**:
- **Batched reading**: Reads `scan_batch_size` rows at a time (default 100K)
- **Projection pushdown**: Only reads required columns
- **Filter pushdown**: Applies predicates during scan (when possible)
- **Parallel scans**: Multiple tasks can scan different batches concurrently

**Implementation Details**:

```cpp
class sirius_physical_table_scan : public sirius_physical_operator {
private:
    std::string file_path;
    std::vector<size_t> column_indices;  // Columns to read
    std::unique_ptr<Expression> filter;  // Optional pushdown filter

    size_t current_batch_idx = 0;
    size_t total_batches;
    size_t scan_batch_size;

public:
    TaskCreationHint get_next_task_hint() override {
        return (current_batch_idx < total_batches)
            ? TaskCreationHint::READY
            : TaskCreationHint::NO_MORE_TASKS;
    }

    data_batch execute(data_batch&&) override {
        // Read batch from Parquet
        auto batch = read_parquet_batch(
            file_path,
            current_batch_idx,
            scan_batch_size,
            column_indices,
            filter  // Optional pushdown
        );

        current_batch_idx++;
        return batch;
    }
};
```

**Performance Tips**:
- Adjust `scan_batch_size` based on table width:
  - Narrow tables (< 10 columns): 200K rows
  - Wide tables (> 50 columns): 50K rows
- Use projection to read only needed columns
- Enable filter pushdown when possible

**Example Configuration**:
```ini
[execution]
scan_batch_size = 100000  # Default
```

---

### DUCKDB_SCAN

**Purpose**: Read data from DuckDB tables (CPU side)

**File**: `src/op/sirius_physical_duckdb_scan.cpp`

**SQL Examples**:
```sql
-- Scan from existing DuckDB table
SELECT * FROM gpu_execution('SELECT * FROM my_table');

-- Join GPU and CPU tables
SELECT * FROM gpu_execution('
    SELECT a.id, a.gpu_col, b.cpu_col
    FROM gpu_table a
    JOIN cpu_table b ON a.id = b.id
');
```

**Key Features**:
- **CPU → GPU transfer**: Reads from DuckDB, transfers to GPU
- **Batched transfer**: Transfers in chunks to avoid memory pressure
- **Type conversion**: Handles DuckDB → cuDF type mapping

**Implementation**:

```cpp
data_batch execute(data_batch&&) override {
    // Read batch from DuckDB
    auto duckdb_chunk = duckdb_scan_state->FetchChunk();

    if (duckdb_chunk.size() == 0) {
        return data_batch{};  // No more data
    }

    // Convert DuckDB DataChunk → cuDF table
    auto cudf_table = convert_duckdb_to_cudf(duckdb_chunk);

    // Transfer to GPU
    return data_batch{
        .table = std::move(cudf_table),
        .tier = MemoryTier::GPU,
        .num_rows = duckdb_chunk.size()
    };
}
```

**Performance Considerations**:
- **Bottleneck**: CPU → GPU transfer (PCIe bandwidth)
- **Cost**: ~0.15ms per 5MB batch (A100, PCIe Gen4)
- **Optimization**: Minimize CPU table scans, prefer GPU tables

---

### DUMMY_SCAN

**Purpose**: Pass-through operator for pipeline breaks

**File**: `src/op/sirius_physical_dummy_scan.cpp`

**Use Case**: Reading data from repository (already on GPU)

**SQL Example**:
```sql
-- ORDER BY creates pipeline break
SELECT * FROM gpu_execution('
    SELECT * FROM sales
    GROUP BY category  -- Pipeline break here
    ORDER BY total
');

-- DUMMY_SCAN reads from group-by result repository
```

**Implementation**:

```cpp
data_batch execute(data_batch&& input) override {
    // Pass through input (already pulled from repository)
    return std::move(input);
}
```

**Note**: DUMMY_SCAN is created automatically by the planner, not explicitly by users.

---

## 2. Filter Operators

### FILTER

**Purpose**: Apply WHERE predicates

**File**: `src/op/sirius_physical_filter.cpp`

**SQL Examples**:
```sql
-- Simple predicate
WHERE price > 100

-- Complex predicate
WHERE (category = 'Electronics' AND price > 500)
   OR (category = 'Books' AND price > 20)

-- Multiple filters (combined by AND)
WHERE year = 2024
  AND amount > 1000
  AND status = 'completed'
```

**Supported Predicates**:
- **Comparisons**: `=`, `!=`, `<`, `<=`, `>`, `>=`
- **Logical**: `AND`, `OR`, `NOT`
- **Null checks**: `IS NULL`, `IS NOT NULL`
- **String operations**: `LIKE`, `ILIKE` (case-insensitive)
- **In list**: `IN (val1, val2, ...)`
- **Between**: `BETWEEN low AND high`

**Implementation**:

```cpp
class sirius_physical_filter : public sirius_physical_operator {
private:
    std::unique_ptr<Expression> filter_expr;

public:
    data_batch execute(data_batch&& input) override {
        if (input.num_rows == 0) {
            return std::move(input);
        }

        // Evaluate filter expression → boolean mask
        auto mask = evaluate_filter(input.table, filter_expr.get());

        // Apply mask using cuDF
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
};
```

**cuDF Operations**:

```cpp
// Simple comparison: price > 100
auto price_col = table->column("price");
auto mask = cudf::binary_operation(
    price_col,
    cudf::numeric_scalar<double>(100.0),
    cudf::binary_operator::GREATER
);

// Apply mask
auto filtered = cudf::apply_boolean_mask(table->view(), mask->view());
```

**Performance**:
- **Selectivity matters**: Higher selectivity → more rows filtered → better performance
- **Early filters**: Apply selective filters early to reduce data size
- **Cost**: ~0.1ms per 100K rows (simple predicate)

**Example**:
```sql
-- Selective filter (1% pass rate)
WHERE rare_condition = true  -- Fast (99% filtered)

-- Non-selective filter (99% pass rate)
WHERE common_condition = true  -- Slower (only 1% filtered)
```

---

## 3. Projection Operators

### PROJECTION

**Purpose**: Select columns and evaluate expressions

**File**: `src/op/sirius_physical_projection.cpp`

**SQL Examples**:
```sql
-- Select columns
SELECT id, name, price

-- Computed columns
SELECT id, price * 1.1 AS price_with_tax

-- Expressions
SELECT
    UPPER(name) AS name_upper,
    price / quantity AS unit_price,
    CASE
        WHEN status = 'A' THEN 'Active'
        WHEN status = 'I' THEN 'Inactive'
        ELSE 'Unknown'
    END AS status_text
```

**Supported Operations**:
- **Column selection**: Pick specific columns
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **String functions**: `UPPER`, `LOWER`, `SUBSTRING`, `CONCAT`
- **Date functions**: `YEAR`, `MONTH`, `DAY`, `DATE_ADD`
- **Type casts**: `CAST(col AS type)`
- **CASE expressions**: Conditional logic

**Implementation**:

```cpp
class sirius_physical_projection : public sirius_physical_operator {
private:
    std::vector<std::unique_ptr<Expression>> expressions;
    std::vector<std::string> output_names;

public:
    data_batch execute(data_batch&& input) override {
        std::vector<std::unique_ptr<cudf::column>> output_columns;

        // Evaluate each expression
        for (size_t i = 0; i < expressions.size(); i++) {
            auto col = evaluate_expression(
                input.table,
                expressions[i].get()
            );
            output_columns.push_back(std::move(col));
        }

        // Create output table
        auto output_table = std::make_unique<cudf::table>(
            std::move(output_columns)
        );

        return data_batch{
            .table = std::move(output_table),
            .tier = MemoryTier::GPU,
            .num_rows = input.num_rows
        };
    }
};
```

**cuDF Operations**:

```cpp
// Arithmetic: price * 1.1
auto price_col = table->column("price");
auto result = cudf::binary_operation(
    price_col,
    cudf::numeric_scalar<double>(1.1),
    cudf::binary_operator::MUL
);

// String uppercase: UPPER(name)
auto name_col = table->column("name");
auto upper = cudf::strings::to_upper(name_col);

// CASE expression: implemented as combination of masks and gather
```

**Performance**:
- **Simple expressions**: ~0.05ms per 100K rows
- **Complex expressions**: ~0.5ms per 100K rows
- **String operations**: Slower than numeric (10x)

---

## 4. Aggregate Operators

### UNGROUPED_AGGREGATE

**Purpose**: Single-group aggregation (no GROUP BY)

**File**: `src/op/sirius_physical_ungrouped_aggregate.cpp`

**SQL Examples**:
```sql
-- Single aggregate
SELECT COUNT(*) FROM orders;

-- Multiple aggregates
SELECT
    COUNT(*) as order_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    MAX(amount) as max_amount
FROM orders;
```

**Supported Functions**:
- `COUNT(*)`, `COUNT(column)`
- `SUM(column)`
- `AVG(column)`
- `MIN(column)`, `MAX(column)`
- `STDDEV(column)`, `VARIANCE(column)`
- `FIRST(column)`, `LAST(column)`

**Implementation**:

```cpp
class sirius_physical_ungrouped_aggregate : public sirius_physical_operator {
private:
    std::vector<AggregateFunction> aggregates;
    std::vector<cudf::column_view> accumulated_results;

public:
    void sink(data_batch&& input) override {
        // Accumulate aggregates
        for (size_t i = 0; i < aggregates.size(); i++) {
            auto& agg = aggregates[i];
            auto col = input.table->column(agg.column_idx);

            // Compute aggregate for this batch
            auto partial_result = compute_aggregate(col, agg.function);

            // Combine with accumulated result
            if (accumulated_results[i].empty()) {
                accumulated_results[i] = partial_result;
            } else {
                accumulated_results[i] = combine_aggregates(
                    accumulated_results[i],
                    partial_result,
                    agg.function
                );
            }
        }
    }

    void finalize() override {
        // Create output batch with single row
        auto output = create_single_row_batch(accumulated_results);
        publish_output(std::move(output));
    }
};
```

**cuDF Operations**:

```cpp
// COUNT(*)
auto count = input_table.num_rows();

// SUM(amount)
auto amount_col = input_table.column("amount");
auto sum = cudf::reduce(
    amount_col,
    cudf::make_sum_aggregation<cudf::reduce_aggregation>()
);

// AVG(amount) = SUM(amount) / COUNT(amount)
auto sum = cudf::reduce(amount_col, cudf::make_sum_aggregation());
auto count = cudf::reduce(amount_col, cudf::make_count_aggregation());
auto avg = sum / count;
```

**Performance**:
- **Cost**: ~0.2ms per 100K rows
- **Memory**: O(1) - single row output

---

### HASH_GROUP_BY

**Purpose**: Multi-group aggregation with GROUP BY

**File**: `src/op/sirius_physical_hash_group_by.cpp`

**SQL Examples**:
```sql
-- Single group key
SELECT category, COUNT(*), SUM(price)
FROM products
GROUP BY category;

-- Multiple group keys
SELECT year, month, category,
       COUNT(*) as count,
       SUM(sales) as total_sales
FROM transactions
GROUP BY year, month, category;
```

**Supported Aggregates**: Same as UNGROUPED_AGGREGATE

**Implementation**:

```cpp
class sirius_physical_hash_group_by : public sirius_physical_operator {
private:
    std::vector<size_t> group_indices;
    std::vector<AggregateFunction> aggregates;
    std::unique_ptr<cudf::groupby::groupby> hash_table;

public:
    void sink(data_batch&& input) override {
        // Extract group keys
        auto keys = input.table->select(group_indices);

        // Extract aggregate columns
        std::vector<cudf::groupby::aggregation_request> requests;
        for (const auto& agg : aggregates) {
            auto col = input.table->column(agg.column_idx);
            requests.push_back({
                .values = col,
                .aggregations = {make_cudf_aggregation(agg.function)}
            });
        }

        // Update hash table
        if (!hash_table) {
            hash_table = std::make_unique<cudf::groupby::groupby>(keys);
        }
        hash_table->aggregate(requests);

        // Flush if threshold reached
        if (should_flush()) {
            flush_partial();
        }
    }

    void finalize() override {
        auto result = hash_table->get_result();
        publish_output(to_data_batch(result));
    }

private:
    bool should_flush() const {
        return accumulated_rows > flush_threshold;
    }
};
```

**cuDF Operations**:

```cpp
// Create groupby object
auto keys_table = input_table.select({"category"});
auto groupby = cudf::groupby::groupby(keys_table);

// Aggregate requests
std::vector<cudf::groupby::aggregation_request> requests;
requests.push_back({
    .values = input_table.column("price"),
    .aggregations = {
        cudf::make_count_aggregation(),
        cudf::make_sum_aggregation()
    }
});

// Execute
auto [result_keys, result_aggs] = groupby.aggregate(requests);
```

**Performance**:
- **Cost**: ~1ms per 100K input rows
- **Memory**: O(num_groups) - grows with unique groups
- **Flushing**: Periodic flushes prevent memory overflow

**Configuration**:
```ini
[execution]
aggregate_flush_threshold = 1000000  # Flush after 1M accumulated rows
```

---

## 5. Sort Operators

### ORDER_BY

**Purpose**: Sort rows

**File**: `src/op/sirius_physical_order_by.cpp`

**SQL Examples**:
```sql
-- Single column
ORDER BY amount DESC

-- Multiple columns
ORDER BY year ASC, month ASC, amount DESC

-- With nulls handling
ORDER BY name ASC NULLS FIRST
```

**Sort Specifications**:

```cpp
struct SortColumn {
    size_t column_idx;
    OrderType order;      // ASCENDING or DESCENDING
    NullOrder null_order; // NULLS_FIRST or NULLS_LAST
};
```

**Implementation**:

```cpp
class sirius_physical_order_by : public sirius_physical_operator {
private:
    std::vector<SortColumn> sort_columns;
    std::vector<data_batch> buffered_batches;
    bool all_input_received = false;

public:
    TaskCreationHint get_next_task_hint() override {
        if (!all_input_received) {
            // Buffering phase
            if (input_repo->has_data()) return READY;
            if (input_repo->is_complete()) {
                all_input_received = true;
                return READY;  // Ready to sort
            }
            return WAITING_FOR_INPUT_DATA;
        }
        // Sorting phase
        return has_emitted ? NO_MORE_TASKS : READY;
    }

    data_batch execute(data_batch&& input) override {
        if (!all_input_received) {
            // Buffer input
            buffered_batches.push_back(std::move(input));
            return data_batch{};
        } else {
            // Sort and emit
            auto combined = concatenate_batches(buffered_batches);
            auto sorted = cudf::sort(combined.table, sort_columns);
            has_emitted = true;
            return to_data_batch(sorted);
        }
    }
};
```

**cuDF Operations**:

```cpp
// Sort single column descending
auto sorted = cudf::sort(
    table->view(),
    {0},  // Column indices
    {cudf::order::DESCENDING},
    {cudf::null_order::AFTER}
);

// Sort multiple columns
auto sorted = cudf::sort(
    table->view(),
    {0, 1, 2},  // year, month, amount
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING},
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::AFTER}
);
```

**Performance**:
- **Cost**: ~2ms per 100K rows (single column)
- **Memory**: O(N) - buffers all input
- **Pipeline break**: Required (must see all data)

---

### TOP_N

**Purpose**: Sort + LIMIT optimization

**File**: `src/op/sirius_physical_top_n.cpp`

**SQL Example**:
```sql
-- Top 10 by amount
SELECT * FROM orders
ORDER BY amount DESC
LIMIT 10;
```

**Key Optimization**: Only maintains top N rows, not full sort

**Implementation**:

```cpp
class sirius_physical_top_n : public sirius_physical_operator {
private:
    size_t n;
    std::vector<SortColumn> sort_columns;
    std::vector<data_batch> top_candidates;

public:
    void sink(data_batch&& input) override {
        // Merge with existing candidates
        top_candidates.push_back(std::move(input));

        if (top_candidates.size() > threshold) {
            // Sort and keep top N
            auto combined = concatenate_batches(top_candidates);
            auto sorted = cudf::sort(combined.table, sort_columns);
            auto top = extract_first_n_rows(sorted, n);

            top_candidates.clear();
            top_candidates.push_back(std::move(top));
        }
    }

    void finalize() override {
        auto combined = concatenate_batches(top_candidates);
        auto sorted = cudf::sort(combined.table, sort_columns);
        auto top = extract_first_n_rows(sorted, n);
        publish_output(std::move(top));
    }
};
```

**Performance**:
- **Cost**: ~1ms per 100K rows (vs. 2ms for full sort)
- **Memory**: O(N) instead of O(input_size)
- **Speedup**: 2-10x faster than ORDER_BY + LIMIT

---

## 6. Join Operators

### HASH_JOIN

**Purpose**: Hash-based equi-join

**File**: `src/op/sirius_physical_hash_join.cpp`

**SQL Examples**:
```sql
-- INNER JOIN
SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.id;

-- LEFT JOIN
SELECT *
FROM orders o
LEFT JOIN products p ON o.product_id = p.id;

-- Multiple conditions
SELECT *
FROM orders o
JOIN shipments s
  ON o.order_id = s.order_id
 AND o.warehouse_id = s.warehouse_id;
```

**Join Types**:
- `INNER`: Only matching rows
- `LEFT`: All left + matching right (nulls for non-matches)
- `RIGHT`: All right + matching left
- `FULL`: All rows from both sides
- `SEMI`: Left rows with matches (no right columns)
- `ANTI`: Left rows without matches

**Two-Phase Implementation**:

**Phase 1: Build** (separate pipeline):
```cpp
void sink_build(data_batch&& batch) {
    auto keys = extract_keys(batch);
    auto payload = extract_payload(batch);

    if (!hash_table) {
        hash_table = std::make_unique<cudf::hash_join>(keys);
        payload_table = payload;
    } else {
        hash_table->append(keys);
        payload_table = concatenate(payload_table, payload);
    }
}

void finalize_build() {
    hash_table->finalize();
    build_complete = true;
    cv.notify_all();  // Wake probe tasks
}
```

**Phase 2: Probe** (separate pipeline):
```cpp
TaskCreationHint get_next_task_hint() override {
    if (!build_complete) {
        return WAITING_FOR_INPUT_DATA;
    }
    return check_input_repository();
}

data_batch execute(data_batch&& probe_batch) override {
    auto keys = extract_keys(probe_batch);
    auto [left_idx, right_idx] = hash_table->inner_join(keys);

    auto left_result = cudf::gather(probe_batch.table, left_idx);
    auto right_result = cudf::gather(payload_table, right_idx);

    return concatenate_columns(left_result, right_result);
}
```

**cuDF Operations**:

```cpp
// Build phase
auto build_keys = build_table.select({"customer_id"});
auto hash_join = cudf::hash_join(
    build_keys,
    cudf::nullable_join::YES
);

// Probe phase
auto probe_keys = probe_table.select({"customer_id"});
auto [left_indices, right_indices] = hash_join.inner_join(probe_keys);

// Gather results
auto left_result = cudf::gather(probe_table, left_indices);
auto right_result = cudf::gather(build_table, right_indices);
```

**Performance**:
- **Build cost**: ~1ms per 100K build rows
- **Probe cost**: ~0.5ms per 100K probe rows
- **Memory**: O(build_size) for hash table
- **Selectivity**: Affects output size, not probe speed

**Best Practices**:
- **Build smaller table**: Put smaller table on build side
- **Filter before join**: Reduce table sizes with WHERE clauses
- **Multiple small builds**: Better than one large build

---

## 7. Partition Operators

### PARTITION

**Purpose**: Partition data by hash or range

**File**: `src/op/sirius_physical_partition.cpp`

**Use Case**: Parallel aggregation, distributed joins

**SQL Example**:
```sql
-- Not directly exposed in SQL
-- Used internally for parallel GROUP BY
```

**Implementation**:

```cpp
data_batch execute(data_batch&& input) override {
    // Hash partition keys
    auto keys = input.table->select(partition_column_indices);
    auto hash_values = cudf::hash(keys);

    // Compute partition IDs
    auto partition_ids = compute_partition_ids(hash_values, num_partitions);

    // Partition table
    auto partitioned = cudf::partition(
        input.table->view(),
        partition_ids,
        num_partitions
    );

    // Return partitioned batch
    return to_data_batch(partitioned);
}
```

**Performance**:
- **Cost**: ~0.3ms per 100K rows
- **Use sparingly**: Adds overhead

---

## 8. Limit Operators

### LIMIT

**Purpose**: Restrict number of rows

**File**: `src/op/sirius_physical_limit.cpp`

**SQL Examples**:
```sql
-- Simple limit
SELECT * FROM orders LIMIT 100;

-- With offset
SELECT * FROM orders LIMIT 100 OFFSET 50;
```

**Implementation**:

```cpp
class sirius_physical_limit : public sirius_physical_operator {
private:
    size_t limit;
    size_t offset;
    size_t rows_emitted = 0;

public:
    data_batch execute(data_batch&& input) override {
        // Skip offset rows
        if (rows_emitted < offset) {
            size_t to_skip = std::min(offset - rows_emitted, input.num_rows);
            rows_emitted += to_skip;

            if (to_skip == input.num_rows) {
                return data_batch{};  // All rows skipped
            }

            // Slice input to remove skipped rows
            input = slice_batch(input, to_skip, input.num_rows);
        }

        // Apply limit
        size_t remaining = limit - (rows_emitted - offset);
        if (input.num_rows > remaining) {
            input = slice_batch(input, 0, remaining);
        }

        rows_emitted += input.num_rows;

        return input;
    }
};
```

**Optimization**: Combine with ORDER BY → TOP_N operator

---

## 9. Output Operators

### RESULT_COLLECTOR

**Purpose**: Collect final query results

**File**: `src/op/sirius_physical_result_collector.cpp`

**Implementation**:

```cpp
class sirius_physical_result_collector : public sirius_physical_operator {
private:
    std::vector<data_batch> collected_batches;

public:
    void sink(data_batch&& input) override {
        // Collect batch
        collected_batches.push_back(std::move(input));
    }

    void finalize() override {
        // Concatenate all batches
        auto combined = concatenate_batches(collected_batches);

        // Transfer to host
        auto host_batch = transfer_to_host(combined);

        // Convert to DuckDB format
        auto result = convert_to_duckdb_result(host_batch);

        // Store for retrieval
        query_result = std::move(result);
    }
};
```

**Performance**:
- **GPU → HOST transfer**: ~0.15ms per 5MB
- **Conversion overhead**: ~0.1ms per 100K rows

---

## See Also

- [Operators](operators.md) - Operator base class and patterns
- [New Mode Overview](overview.md) - Introduction to New Mode
- [New Data Flow](../06-data-flow/new-data-flow.md) - Data flow details
- [Pipeline Execution](pipeline-execution.md) - Pipeline structure
- [Performance Tips](../appendices/performance-tips.md) - Optimization guide
