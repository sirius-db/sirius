# Legacy Mode Operators

This document provides a comprehensive guide to the **GPUPhysicalOperator** hierarchy used in Sirius Legacy Mode (`gpu_processing`).

## Table of Contents

1. [Overview](#overview)
2. [Base Operator Architecture](#base-operator-architecture)
3. [Operator Categories](#operator-categories)
4. [Source Operators](#source-operators)
5. [Intermediate Operators](#intermediate-operators)
6. [Sink Operators](#sink-operators)
7. [Complete Operator Catalog](#complete-operator-catalog)
8. [Execution Model](#execution-model)
9. [Next Steps](#next-steps)

---

## Overview

In Legacy Mode, all physical operators inherit from **`GPUPhysicalOperator`**, which provides the interface for:

- **Source operations**: Reading data into GPU memory (`GetData()`)
- **Intermediate operations**: Processing data (`Execute()`)
- **Sink operations**: Accumulating results (`Sink()`)
- **Pipeline construction**: Building execution graphs

Each operator is responsible for manipulating **GPUIntermediateRelation** objects, which contain vectors of **GPUColumn** structures representing columnar data on the GPU.

**Key Characteristics:**

- **Pull-based execution model**: Operators pull data from their children
- **DuckDB PhysicalOperatorType enum**: Operators use standard DuckDB types
- **Manual memory management**: Operators work with GPUBufferManager
- **Single-threaded per pipeline**: Each pipeline executes on one thread

---

## Base Operator Architecture

### GPUPhysicalOperator Base Class

**File**: `src/include/gpu_physical_operator.hpp:48-180`

```cpp
class GPUPhysicalOperator {
public:
    // Constructor
    GPUPhysicalOperator(PhysicalOperatorType type,
                        vector<LogicalType> types,
                        idx_t estimated_cardinality);

    // Core properties
    PhysicalOperatorType type;                           // Operator type (FILTER, JOIN, etc.)
    vector<unique_ptr<GPUPhysicalOperator>> children;    // Child operators
    vector<LogicalType> types;                           // Output column types
    idx_t estimated_cardinality;                         // Row count estimate

    // State management
    unique_ptr<GlobalSinkState> sink_state;              // Sink state (for accumulation)
    unique_ptr<GlobalOperatorState> op_state;            // Operator state

    // Source interface
    virtual SourceResultType GetData(GPUIntermediateRelation& output_relation) const;
    virtual bool IsSource() const { return false; }

    // Operator interface
    virtual OperatorResultType Execute(GPUIntermediateRelation& input_relation,
                                       GPUIntermediateRelation& output_relation) const;

    // Sink interface
    virtual SinkResultType Sink(GPUIntermediateRelation& input_relation) const;
    virtual SinkFinalizeType CombineFinalize(
        vector<shared_ptr<GPUIntermediateRelation>>& input,
        GPUIntermediateRelation& output) const;
    virtual bool IsSink() const { return false; }

    // Pipeline construction
    virtual void BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline);
};
```

### Key Methods

| Method | Purpose | Used By |
|--------|---------|---------|
| `GetData()` | Pull data from source (e.g., scan table) | Source operators |
| `Execute()` | Process input relation to produce output | Intermediate operators |
| `Sink()` | Accumulate data into global state | Sink operators |
| `CombineFinalize()` | Merge accumulated data (e.g., hash table) | Join build, aggregates |
| `BuildPipelines()` | Construct pipeline graph | All operators |

---

## Operator Categories

Legacy Mode operators fall into three categories based on their role in the pipeline:

### 1. Source Operators

- **Purpose**: Generate initial data for the pipeline
- **Method**: Override `GetData()` and `IsSource() = true`
- **Examples**: TABLE_SCAN, COLUMN_DATA_SCAN, DUMMY_SCAN
- **Data flow**: No input → GPU data output

### 2. Intermediate Operators

- **Purpose**: Transform data in-place
- **Method**: Override `Execute()`
- **Examples**: FILTER, PROJECTION, ORDER_BY
- **Data flow**: GPU input → GPU output

### 3. Sink Operators

- **Purpose**: Accumulate data across batches
- **Method**: Override `Sink()`, `CombineFinalize()`, and `IsSink() = true`
- **Examples**: HASH_JOIN (build side), HASH_GROUP_BY, RESULT_COLLECTOR
- **Data flow**: GPU input → accumulate in sink_state

---

## Source Operators

Source operators initiate pipeline execution by reading data from various sources.

### TABLE_SCAN

**File**: `src/operator/gpu_physical_table_scan.cpp`

**Purpose**: Scans DuckDB tables and transfers data to GPU memory.

**Key Features:**
- Uses DuckDB's table function API
- Transfers data from CPU → GPU in batches
- Supports filter pushdown for early filtering
- Caches frequently accessed columns

**Implementation:**

```cpp
class GPUPhysicalTableScan : public GPUPhysicalOperator {
public:
    TableFunction function;                // DuckDB table function
    unique_ptr<FunctionData> bind_data;    // Bind data from planning
    vector<ColumnIndex> column_ids;        // Columns to scan
    vector<LogicalType> returned_types;    // Column types
    unique_ptr<TableFilterSet> table_filters;  // Pushdown filters

    SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;
};
```

**Execution Flow:**

1. Call DuckDB table function to get CPU DataChunks
2. Allocate GPU memory via GPUBufferManager
3. Transfer data using `cudaMemcpy(HostToDevice)`
4. Apply filter pushdown (if applicable)
5. Return GPUIntermediateRelation with GPU columns

**Performance Notes:**
- Memory transfer is the bottleneck (~5-10 GB/s PCIe bandwidth)
- Uses CUDA streams for overlapping transfer + computation (when `Config::USE_OPT_TABLE_SCAN = true`)
- Caching reduces redundant transfers for repeated scans

### COLUMN_DATA_SCAN

**File**: `src/operator/gpu_physical_column_data_scan.cpp`

**Purpose**: Scans in-memory column data collections (e.g., CTEs, subquery results).

**Key Features:**
- Reads from `ColumnDataCollection` (DuckDB's in-memory format)
- Used for materialized CTEs and intermediate results
- Faster than TABLE_SCAN since data is already in memory

### DUMMY_SCAN

**File**: `src/operator/gpu_physical_dummy_scan.cpp`

**Purpose**: Generates empty relations or constant values.

**Key Features:**
- Used for queries like `SELECT 1` or `SELECT COUNT(*) FROM empty_table`
- Minimal GPU memory allocation
- Returns constant values or zero rows

---

## Intermediate Operators

Intermediate operators transform data in transit through the pipeline.

### FILTER

**File**: `src/operator/gpu_physical_filter.cpp:52-66`

**Purpose**: Filters rows based on a boolean predicate.

**Implementation:**

```cpp
class GPUPhysicalFilter : public GPUPhysicalOperator {
public:
    unique_ptr<Expression> expression;  // Filter predicate (e.g., "age > 25")

    OperatorResultType Execute(GPUIntermediateRelation& input_relation,
                               GPUIntermediateRelation& output_relation) const override {
        // Use GPU expression executor to evaluate predicate
        sirius::GpuExpressionExecutor gpu_expression_executor(*expression.get());
        gpu_expression_executor.Select(input_relation, output_relation);

        return OperatorResultType::FINISHED;
    }
};
```

**Execution Flow:**

1. Receive input GPUIntermediateRelation with N rows
2. Evaluate filter expression on GPU → produce boolean selection vector
3. Compact columns using selection vector (remove filtered rows)
4. Return output relation with M rows (M ≤ N)

**Performance:**
- Highly efficient on GPU (parallel predicate evaluation)
- Selection vector compaction uses cuDF gather operations
- Typical speedup: 5-20x vs CPU for large datasets

### PROJECTION

**File**: `src/operator/gpu_physical_projection.cpp`

**Purpose**: Computes new columns from expressions (e.g., `a + b AS c`).

**Key Features:**
- Evaluates scalar expressions on GPU
- Supports arithmetic, string operations, casts
- Can add or remove columns from relation

**Example:**

```sql
-- Query: SELECT age * 2 AS double_age, name FROM users WHERE age > 25
-- Pipeline: TABLE_SCAN → FILTER → PROJECTION
```

**Expression Evaluation:**

Uses **GpuExpressionExecutor** (see [Expression Executor](../05-core-components/expression-executor.md)) to:

1. Parse expression tree (e.g., `age * 2`)
2. Generate cuDF operations
3. Execute on GPU
4. Add new column to output relation

### ORDER_BY

**File**: `src/operator/gpu_physical_order.cpp`

**Purpose**: Sorts data by one or more columns.

**Key Features:**
- Uses cuDF's GPU-accelerated sort
- Supports multi-column sorting with custom collations
- Generates sort indices, then gathers data

**Implementation:**

```cpp
OperatorResultType GPUPhysicalOrder::Execute(
    GPUIntermediateRelation& input_relation,
    GPUIntermediateRelation& output_relation) const {

    // Step 1: Create cuDF table from GPU columns
    auto cudf_table = ConvertToCudfTable(input_relation);

    // Step 2: Sort to get indices
    auto sorted_indices = cudf::sorted_order(
        cudf_table->view(),
        sort_orders,        // ASC/DESC per column
        null_precedence     // NULLS FIRST/LAST
    );

    // Step 3: Gather data using sorted indices
    auto sorted_table = cudf::gather(cudf_table->view(), sorted_indices->view());

    // Step 4: Convert back to GPUIntermediateRelation
    output_relation = ConvertFromCudfTable(sorted_table);

    return OperatorResultType::FINISHED;
}
```

**Performance:**
- GPU sorting is 10-50x faster than CPU for large datasets
- Bottleneck: memory bandwidth for gather operation

### LIMIT

**File**: `src/operator/gpu_physical_limit.cpp`

**Purpose**: Limits output to first N rows (with optional OFFSET).

**Key Features:**
- Simple slice operation on GPU
- Can terminate early if limit is reached
- Used for `LIMIT` and `TOP N` queries

### TOP_N

**File**: `src/operator/gpu_physical_top_n.cpp`

**Purpose**: Optimized operator for `ORDER BY ... LIMIT N`.

**Key Features:**
- More efficient than separate ORDER_BY + LIMIT
- Uses cuDF's top-k algorithm (heap-based selection)
- Avoids sorting entire dataset

**Performance:**
- O(N log K) vs O(N log N) for full sort
- Significant speedup for small K and large N

---

## Sink Operators

Sink operators accumulate data across multiple batches before producing output.

### RESULT_COLLECTOR

**File**: `src/operator/gpu_physical_result_collector.cpp:33-100`

**Purpose**: Final operator that collects results and converts them to DuckDB format.

**Key Responsibilities:**

1. **Accumulate batches**: Collect all GPU batches from child pipeline
2. **Late materialization**: Apply any pending row_id indirections
3. **Transfer to host**: Move data from GPU → CPU memory
4. **Convert format**: Transform GPUColumns → DuckDB DataChunks
5. **Return QueryResult**: Package as DuckDB MaterializedQueryResult

**Implementation:**

```cpp
class GPUPhysicalMaterializedCollector : public GPUPhysicalResultCollector {
private:
    unique_ptr<GPUResultCollection> result_collection;  // Accumulated batches

public:
    SinkResultType Sink(GPUIntermediateRelation& input_relation) const override {
        // Accumulate batch into result collection
        result_collection->Append(input_relation);
        return SinkResultType::NEED_MORE_INPUT;
    }

    void FinalMaterialize(GPUIntermediateRelation& output_relation) {
        // Step 1: Apply late materialization (if needed)
        for (size_t col = 0; col < input_relation.columns.size(); col++) {
            if (input_relation.checkLateMaterialization(col)) {
                // Gather data using row_ids
                materializeExpression<T>(data, materialized, row_ids, count, mask, out_mask);
            }
        }

        // Step 2: Transfer to host memory
        cudaMemcpy(host_buffer, device_buffer, size, cudaMemcpyDeviceToHost);

        // Step 3: Convert to DuckDB DataChunks
        ConvertGPUColumnsToDuckDB(output_relation, duckdb_chunks);

        // Step 4: Package as QueryResult
        return make_uniq<MaterializedQueryResult>(statement_type, properties, names,
                                                  std::move(duckdb_chunks), context);
    }
};
```

**Performance:**
- GPU → HOST transfer: ~10 GB/s (PCIe bandwidth limit)
- Format conversion: Depends on data types (strings are slowest)
- See [Result Collection](../05-core-components/result-collection.md) for details

### HASH_JOIN (Build Side)

**File**: `src/operator/gpu_physical_hash_join.cpp`

**Purpose**: Builds a hash table from the inner (build) side of a join.

**Key Features:**
- Accumulates all build-side batches
- Constructs GPU hash table using cuckoo hashing
- Supports INNER, LEFT, RIGHT, SEMI, ANTI joins
- Stores hash table in global sink state

**Hash Table Construction:**

```cpp
SinkResultType GPUPhysicalHashJoin::Sink(GPUIntermediateRelation& input_relation) const {
    auto& gstate = sink_state->Cast<GPUHashJoinGlobalState>();

    // Accumulate build-side data
    gstate.build_collection.Append(input_relation);

    return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType GPUPhysicalHashJoin::CombineFinalize(
    vector<shared_ptr<GPUIntermediateRelation>>& input,
    GPUIntermediateRelation& output) const {

    // Step 1: Concatenate all build batches
    auto combined_build = ConcatenateBatches(input);

    // Step 2: Build hash table on GPU
    uint64_t ht_size = NextPrime(combined_build.row_count * 1.5);  // Load factor 0.67
    auto hash_table = BuildCuckooHashTable(combined_build.columns, build_keys, ht_size);

    // Step 3: Store in global state
    auto& gstate = sink_state->Cast<GPUHashJoinGlobalState>();
    gstate.hash_table = hash_table;
    gstate.build_data = combined_build;

    return SinkFinalizeType::READY;
}
```

**Probe Side Execution:**

After the build phase completes, the probe side executes as an intermediate operator:

```cpp
OperatorResultType GPUPhysicalHashJoin::Execute(
    GPUIntermediateRelation& input_relation,  // Probe side
    GPUIntermediateRelation& output_relation) const {

    auto& gstate = sink_state->Cast<GPUHashJoinGlobalState>();

    // Probe hash table
    vector<shared_ptr<GPUColumn>> probe_keys = ExtractProbeKeys(input_relation);

    // Launch GPU kernel to find matches
    uint64_t* row_ids_left;   // Indices into probe side
    uint64_t* row_ids_right;  // Indices into build side
    uint64_t match_count;

    ProbeHashTable<int32_t>(probe_keys, &match_count, &row_ids_left, &row_ids_right,
                           gstate.hash_table, gstate.ht_size, conditions, join_type);

    // Gather matched rows
    auto probe_columns = GatherColumns(input_relation.columns, row_ids_left, match_count);
    auto build_columns = GatherColumns(gstate.build_data.columns, row_ids_right, match_count);

    // Combine probe + build columns
    output_relation.columns.insert(output_relation.columns.end(),
                                   probe_columns.begin(), probe_columns.end());
    output_relation.columns.insert(output_relation.columns.end(),
                                   build_columns.begin(), build_columns.end());

    return OperatorResultType::FINISHED;
}
```

**Join Type Support:**

| Join Type | Build Behavior | Probe Behavior |
|-----------|----------------|----------------|
| INNER | Build hash table | Return only matches |
| LEFT | Build hash table | Return all probe + matches (NULLs for non-matches) |
| RIGHT | Build hash table | Mark matched build rows, emit unmatched at end |
| SEMI | Build hash table | Return probe rows with at least one match |
| ANTI | Build hash table | Return probe rows with no matches |

### HASH_GROUP_BY

**File**: `src/operator/gpu_physical_grouped_aggregate.cpp`

**Purpose**: Computes grouped aggregations (e.g., `GROUP BY`, `SUM`, `AVG`).

**Key Features:**
- Uses cuDF's groupby API
- Supports multiple aggregation functions per column
- Handles large group counts efficiently on GPU

**Implementation:**

```cpp
SinkResultType GPUPhysicalGroupedAggregate::Sink(
    GPUIntermediateRelation& input_relation) const {

    auto& gstate = sink_state->Cast<GPUAggregateGlobalState>();

    // Accumulate input batches
    gstate.input_collection.Append(input_relation);

    return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType GPUPhysicalGroupedAggregate::CombineFinalize(
    vector<shared_ptr<GPUIntermediateRelation>>& input,
    GPUIntermediateRelation& output) const {

    // Step 1: Combine all batches
    auto combined = ConcatenateBatches(input);

    // Step 2: Convert to cuDF table
    auto cudf_table = ConvertToCudfTable(combined);

    // Step 3: Perform groupby aggregation
    cudf::groupby::groupby gb_obj(cudf_table.select(group_columns));
    auto requests = CreateAggregationRequests(aggregates);  // SUM, COUNT, AVG, etc.
    auto results = gb_obj.aggregate(requests);

    // Step 4: Convert back to GPUIntermediateRelation
    output = ConvertFromCudfTable(results.first, results.second);

    return SinkFinalizeType::READY;
}
```

**Supported Aggregates:**
- `COUNT`, `COUNT(DISTINCT)`
- `SUM`, `AVG`, `MIN`, `MAX`
- `FIRST`, `LAST`
- String aggregates (via cuDF)

---

## Complete Operator Catalog

### Source Operators

| Operator Type | File | Purpose | Key Features |
|---------------|------|---------|--------------|
| **TABLE_SCAN** | `gpu_physical_table_scan.cpp` | Scan DuckDB tables | Filter pushdown, caching, CUDA streams |
| **COLUMN_DATA_SCAN** | `gpu_physical_column_data_scan.cpp` | Scan in-memory column data | CTE materialization, subqueries |
| **DUMMY_SCAN** | `gpu_physical_dummy_scan.cpp` | Generate empty/constant data | Zero-row queries, constant expressions |

### Intermediate Operators

| Operator Type | File | Purpose | Key Features |
|---------------|------|---------|--------------|
| **FILTER** | `gpu_physical_filter.cpp` | Filter rows by predicate | GPU expression evaluation, selection vectors |
| **PROJECTION** | `gpu_physical_projection.cpp` | Compute new columns | Scalar expressions, column arithmetic |
| **ORDER_BY** | `gpu_physical_order.cpp` | Sort by columns | Multi-column sort, custom collations |
| **LIMIT** | `gpu_physical_limit.cpp` | Limit rows (TOP N) | Early termination, offset support |
| **TOP_N** | `gpu_physical_top_n.cpp` | Optimized top-k | Heap-based selection, faster than full sort |
| **PARTITION** | `gpu_physical_partition.cpp` | Partition data for window functions | Row number, rank, dense rank |
| **NESTED_LOOP_JOIN** | `gpu_physical_nested_loop_join.cpp` | Cross join / theta join | Cartesian product, inequality joins |
| **CONCAT** | `gpu_physical_concat.cpp` | Union all (concatenate relations) | Vertical concatenation |
| **SUBSTRING** | `gpu_physical_substring.cpp` | Extract substrings | GPU string slicing |
| **STRINGS_MATCHING** | `gpu_physical_strings_matching.cpp` | String LIKE / regex matching | Pattern matching on GPU |

### Sink Operators

| Operator Type | File | Purpose | Key Features |
|---------------|------|---------|--------------|
| **RESULT_COLLECTOR** | `gpu_physical_result_collector.cpp` | Collect final results | GPU→CPU transfer, format conversion |
| **HASH_JOIN** | `gpu_physical_hash_join.cpp` | Build hash table (inner side) | Cuckoo hashing, multi-join types |
| **HASH_GROUP_BY** | `gpu_physical_grouped_aggregate.cpp` | Grouped aggregation | cuDF groupby, multiple aggregates |
| **UNGROUPED_AGGREGATE** | `gpu_physical_ungrouped_aggregate.cpp` | Global aggregation (no GROUP BY) | Single-row output (e.g., `SELECT COUNT(*)`) |
| **DELIM_JOIN** | `gpu_physical_delim_join.cpp` | Delimited join (correlated subquery) | Mark join, duplicate elimination |
| **CTE** | `gpu_physical_cte.cpp` | Materialize CTE | Store intermediate result for reuse |
| **EMPTY_RESULT** | `gpu_physical_empty_result.cpp` | Produce empty result | Used for unsupported operators |

---

## Execution Model

### Pull-Based Execution

Legacy Mode uses a **pull-based execution model**:

1. **Result Collector** (sink) requests data from its child
2. **Child operator** pulls data from *its* children recursively
3. **Source operator** returns batch of GPU data
4. **Intermediate operators** transform data as it flows back up
5. **Sink operator** accumulates transformed data

**Example: Simple Filter Query**

```sql
SELECT * FROM users WHERE age > 25;
```

**Pipeline:**

```
RESULT_COLLECTOR (sink)
    ↑ (pull)
FILTER (intermediate, expression: age > 25)
    ↑ (pull)
TABLE_SCAN (source, table: users)
```

**Execution Trace:**

1. `RESULT_COLLECTOR::Sink()` → pulls from FILTER
2. `FILTER::Execute()` → pulls from TABLE_SCAN
3. `TABLE_SCAN::GetData()` → reads batch from DuckDB, transfers to GPU
4. `FILTER::Execute()` → evaluates `age > 25`, compacts rows
5. `RESULT_COLLECTOR::Sink()` → accumulates filtered batch
6. Repeat until TABLE_SCAN returns `SOURCE_FINISHED`
7. `RESULT_COLLECTOR::FinalMaterialize()` → transfers to CPU, converts to DuckDB format

### Pipeline Splitting

Operators that require global state (sinks) split pipelines:

**Example: Join Query**

```sql
SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id;
```

**Pipeline Graph:**

```
Pipeline 1 (Build):
  TABLE_SCAN (customers)
      ↓
  HASH_JOIN (sink, build side)

Pipeline 2 (Probe):
  TABLE_SCAN (orders)
      ↓
  HASH_JOIN (execute, probe side)
      ↓
  RESULT_COLLECTOR (sink)
```

**Execution Order:**

1. Execute Pipeline 1 completely → build hash table
2. Execute Pipeline 2 → probe hash table batch-by-batch
3. Collect results

See [Pipeline Execution](pipeline-execution.md) for details on pipeline scheduling.

---

## Next Steps

**Related Documentation:**

- **[Pipeline Execution](pipeline-execution.md)**: How operators are assembled into pipelines
- **[Memory Management](memory-management.md)**: GPUBufferManager and GPU memory allocation
- **[Data Structures](data-structures.md)**: GPUColumn and GPUIntermediateRelation internals
- **[Expression Executor](../05-core-components/expression-executor.md)**: How expressions are evaluated on GPU

**For Developers:**

- **[Adding Operators](../07-development/adding-operators.md)**: How to implement new operators
- **[Debugging](../07-development/debugging.md)**: Debugging operator execution

**Comparison:**

- **[New Mode Operators](../04-new-mode/operators.md)**: Compare with the new sirius_physical_operator design
- **[Execution Modes](../02-architecture/execution-modes.md)**: Understand the trade-offs between Legacy and New modes
