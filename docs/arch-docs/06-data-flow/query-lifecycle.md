# Query Lifecycle

This document traces a complete SQL query through Sirius from submission to results, covering all major stages of execution.

## Overview

A query in Sirius goes through these phases:

```
User SQL Query
     ↓
1. Parse (DuckDB)
     ↓
2. Logical Planning (DuckDB)
     ↓
3. Physical Planning (Sirius)
     ↓
4. Pipeline Construction (Sirius)
     ↓
5. Task Creation (Sirius)
     ↓
6. GPU Execution (Sirius + cuDF)
     ↓
7. Result Collection (Sirius)
     ↓
8. Return to User (DuckDB)
```

---

## Example Query

Let's trace this query end-to-end:

```sql
SELECT * FROM gpu_execution('
    SELECT category,
           COUNT(*) as order_count,
           SUM(amount) as total_amount,
           AVG(amount) as avg_amount
    FROM orders
    WHERE order_date >= ''2024-01-01''
      AND status = ''completed''
    GROUP BY category
    HAVING COUNT(*) > 100
    ORDER BY total_amount DESC
    LIMIT 10
');
```

---

## Phase 1: Parse (DuckDB)

**Duration**: ~1-5ms
**Location**: DuckDB core

### What Happens

DuckDB's parser converts SQL text to Abstract Syntax Tree (AST):

```
SQL String:
"SELECT category, COUNT(*) as order_count, ..."

↓ Parser

AST:
SelectStatement
├─ SELECT clause:
│  ├─ category
│  ├─ COUNT(*) AS order_count
│  ├─ SUM(amount) AS total_amount
│  └─ AVG(amount) AS avg_amount
├─ FROM clause: orders
├─ WHERE clause:
│  ├─ order_date >= '2024-01-01'
│  └─ status = 'completed'
├─ GROUP BY clause: category
├─ HAVING clause: COUNT(*) > 100
├─ ORDER BY clause: total_amount DESC
└─ LIMIT clause: 10
```

### Validation

- Syntax checking
- Keyword validation
- String escaping ('' in SQL string)

---

## Phase 2: Logical Planning (DuckDB)

**Duration**: ~5-20ms
**Location**: DuckDB planner

### What Happens

DuckDB's planner creates a logical operator tree:

```
LogicalLimit (10)
     ↓
LogicalOrder (total_amount DESC)
     ↓
LogicalFilter (COUNT(*) > 100)  [HAVING clause]
     ↓
LogicalProjection (category, COUNT(*), SUM(amount), AVG(amount))
     ↓
LogicalAggregate (GROUP BY category)
├─ Aggregates: COUNT(*), SUM(amount), AVG(amount)
└─ Group: category
     ↓
LogicalFilter (order_date >= '2024-01-01' AND status = 'completed')
     ↓
LogicalGet (orders table)
```

### Steps

1. **Binding**: Resolve table/column names
2. **Type Resolution**: Determine column types
3. **Optimization**: Apply logical optimizations
   - Filter pushdown
   - Projection pushdown
   - Expression simplification

### Output

Logical plan + type information:

```cpp
vector<LogicalType> types = {
    LogicalType::VARCHAR,  // category
    LogicalType::BIGINT,   // order_count
    LogicalType::DECIMAL,  // total_amount
    LogicalType::DOUBLE    // avg_amount
};

vector<string> names = {
    "category", "order_count", "total_amount", "avg_amount"
};
```

---

## Phase 3: Physical Planning (Sirius)

**Duration**: ~10-50ms
**Location**: `src/planner/sirius_physical_plan_generator.cpp`

### What Happens

Sirius converts logical plan to GPU-executable physical operators:

```
RESULT_COLLECTOR
     ↓
LIMIT (10)
     ↓
ORDER_BY (total_amount DESC)
     ↓
FILTER (COUNT(*) > 100)  [Post-aggregate filter]
     ↓
HASH_GROUP_BY
├─ Group key: category
├─ Aggregates:
│  ├─ COUNT(*)
│  ├─ SUM(amount)
│  └─ AVG(amount)
     ↓
FILTER (order_date >= '2024-01-01' AND status = 'completed')
     ↓
DUCKDB_SCAN (orders table)
```

### Key Decisions

**Operator Selection**:
- Aggregate → `HASH_GROUP_BY` (hash-based grouping)
- Sort → `ORDER_BY` (full sort, not TOP_N since HAVING exists)
- Scan → `DUCKDB_SCAN` (read from DuckDB table)

**Type Adjustments**:
```cpp
// DuckDB uses HUGEINT for COUNT(*), downcast to BIGINT
if (type == LogicalType::HUGEINT) {
    type = LogicalType::BIGINT;
}
```

### Output

Tree of `sirius_physical_operator` objects with:
- Operator types
- Expressions (filters, projections)
- Schema (input/output types)

---

## Phase 4: Pipeline Construction (Sirius)

**Duration**: ~5-20ms
**Location**: `src/include/pipeline/sirius_pipeline_build_state.hpp`

### What Happens

Physical operators are organized into pipelines:

```
Pipeline 1: SCAN → FILTER → HASH_GROUP_BY (sink)
           Data flows directly, no materialization
           HASH_GROUP_BY accumulates into hash table

           [Pipeline Break - materialize hash table]

Pipeline 2: HASH_GROUP_BY (source) → FILTER (having) → ORDER_BY (sink)
           Pull aggregated data, apply HAVING, sort

           [Pipeline Break - materialize sorted data]

Pipeline 3: ORDER_BY (source) → LIMIT → RESULT_COLLECTOR
           Pull sorted data, apply limit, collect results
```

### Pipeline Breaks

**Why break pipelines?**

1. **HASH_GROUP_BY**: Must see all input before producing output
2. **ORDER_BY**: Requires complete dataset to sort

### Data Repositories

Create repositories for inter-pipeline communication:

```
Repository 1: Pipeline 1 output → Pipeline 2 input
├─ Storage: GPU/HOST/DISK tiers
├─ Capacity: Based on gpu_memory_limit
└─ Type: Multi-producer, single-consumer

Repository 2: Pipeline 2 output → Pipeline 3 input
├─ Storage: GPU/HOST/DISK tiers
├─ Capacity: Based on gpu_memory_limit
└─ Type: Single-producer, single-consumer
```

### Port Connections

```
Pipeline 1:
├─ Input ports: none (source pipeline)
└─ Output ports: [port0] → Repository 1

Pipeline 2:
├─ Input ports: [port0] ← Repository 1
└─ Output ports: [port0] → Repository 2

Pipeline 3:
├─ Input ports: [port0] ← Repository 2
└─ Output ports: none (sink pipeline)
```

---

## Phase 5: Task Creation (Sirius)

**Duration**: Ongoing throughout execution
**Location**: `src/parallel/task_creator.cpp`

### What Happens

`task_creator` dynamically generates tasks based on hints:

```
Iteration 1:
├─ Check Pipeline 1: get_next_task_hint() → READY
├─ Check Pipeline 2: get_next_task_hint() → WAITING_FOR_INPUT_DATA
└─ Check Pipeline 3: get_next_task_hint() → WAITING_FOR_INPUT_DATA
└─ Create tasks for Pipeline 1 only

Tasks Created for Pipeline 1:
├─ Task 1.1: Scan batch 1 (rows 0-100K) → Filter → Aggregate
├─ Task 1.2: Scan batch 2 (rows 100K-200K) → Filter → Aggregate
├─ Task 1.3: Scan batch 3 (rows 200K-300K) → Filter → Aggregate
└─ ... (one task per batch)

Iteration 2 (after Pipeline 1 completes):
├─ Pipeline 1: COMPLETED
├─ Pipeline 2: get_next_task_hint() → READY (Repository 1 has data)
└─ Pipeline 3: WAITING_FOR_INPUT_DATA

Tasks Created for Pipeline 2:
└─ Task 2.1: Pull from Repository 1 → Apply HAVING → Sort → Push to Repository 2

Iteration 3 (after Pipeline 2 completes):
├─ Pipeline 2: COMPLETED
└─ Pipeline 3: get_next_task_hint() → READY (Repository 2 has data)

Tasks Created for Pipeline 3:
└─ Task 3.1: Pull from Repository 2 → Apply LIMIT → Collect Results
```

### Task Hint Logic

```cpp
task_creation_hint sirius_pipeline::get_next_task_hint() {
    // Check if input data available
    for (auto& input_port : input_ports) {
        if (!input_port->has_data()) {
            return {TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
        }
    }

    // Check if more work to do
    if (has_more_work()) {
        return {TaskCreationHint::READY, nullptr};
    }

    return {TaskCreationHint::COMPLETED, nullptr};
}
```

---

## Phase 6: GPU Execution (Sirius + cuDF)

**Duration**: Varies by data size (typically 10ms - 10s)
**Location**: `src/parallel/pipeline_executor.cpp` + cuDF kernels

### What Happens

`pipeline_executor` runs tasks on GPU using CUDA streams:

#### Pipeline 1 Execution

**Task 1.1**: Process first batch

```
1. DUCKDB_SCAN:
   ├─ Allocate GPU memory (cuDF tables)
   ├─ Read rows 0-100K from DuckDB
   ├─ Transfer CPU → GPU (~10ms for 100K rows)
   └─ Output: cuDF table (100K rows)

2. FILTER:
   ├─ Input: 100K rows
   ├─ Apply predicate on GPU:
   │  └─ (order_date >= '2024-01-01') AND (status = 'completed')
   ├─ Vectorized evaluation (all rows in parallel)
   ├─ Duration: ~1ms
   └─ Output: ~30K rows (assuming 30% pass filter)

3. HASH_GROUP_BY (sink):
   ├─ Input: 30K rows
   ├─ Build hash table on GPU:
   │  ├─ Hash by category
   │  ├─ Accumulate: COUNT(*), SUM(amount), compute intermediate for AVG
   │  └─ Use cuDF groupby operations
   ├─ Duration: ~5ms
   └─ Output: Partial aggregates in hash table
```

**Repeat for all batches** (Task 1.2, 1.3, etc.)

**Finalize Pipeline 1**:
```
HASH_GROUP_BY finalize:
├─ Combine partial aggregates
├─ Compute final AVG (sum / count)
├─ Duration: ~10ms
├─ Output: Final aggregates (~1000 categories)
└─ Push to Repository 1 (GPU memory)
```

#### Pipeline 2 Execution

**Task 2.1**: Sort aggregated results

```
1. HASH_GROUP_BY (source):
   ├─ Pull from Repository 1
   └─ Output: ~1000 rows (aggregated by category)

2. FILTER (HAVING):
   ├─ Input: 1000 rows
   ├─ Apply predicate: COUNT(*) > 100
   ├─ Duration: <1ms
   └─ Output: ~300 rows

3. ORDER_BY (sink):
   ├─ Input: 300 rows
   ├─ Sort by total_amount DESC on GPU
   ├─ Use cuDF sort operations
   ├─ Duration: ~2ms (small dataset)
   └─ Output: Sorted 300 rows
   └─ Push to Repository 2
```

#### Pipeline 3 Execution

**Task 3.1**: Apply limit and collect

```
1. ORDER_BY (source):
   ├─ Pull from Repository 2
   └─ Output: 300 rows (sorted)

2. LIMIT:
   ├─ Input: 300 rows
   ├─ Take first 10 rows
   ├─ Duration: <1ms
   └─ Output: 10 rows

3. RESULT_COLLECTOR:
   ├─ Input: 10 rows (on GPU)
   ├─ Convert cuDF → DuckDB format
   ├─ Transfer GPU → CPU
   ├─ Duration: <1ms (tiny dataset)
   └─ Output: DuckDB DataChunk (10 rows)
```

### CUDA Streams

Tasks execute on different CUDA streams for concurrency:

```
CUDA Stream 0: Task 1.1 ──────────────┐
                                       ├─→ Task 1.3 ──────→
CUDA Stream 1:         Task 1.2 ──────┘

(Pipeline 1 tasks can execute concurrently)
```

---

## Phase 7: Result Collection (Sirius)

**Duration**: ~1-10ms
**Location**: `src/op/sirius_physical_result_collector.cpp`

### What Happens

Results transferred from GPU to CPU:

```
GPU (cuDF format):
┌──────────┬─────────────┬──────────────┬────────────┐
│ category │ order_count │ total_amount │ avg_amount │
├──────────┼─────────────┼──────────────┼────────────┤
│ "A"      │ 5000        │ 125000.50    │ 25.00      │
│ "B"      │ 4500        │ 112500.25    │ 25.00      │
│ ...      │ ...         │ ...          │ ...        │
└──────────┴─────────────┴──────────────┴────────────┘

Transfer GPU → CPU (DMA)

CPU (DuckDB DataChunk):
┌──────────┬─────────────┬──────────────┬────────────┐
│ category │ order_count │ total_amount │ avg_amount │
│ VARCHAR  │ BIGINT      │ DECIMAL      │ DOUBLE     │
├──────────┼─────────────┼──────────────┼────────────┤
│ "A"      │ 5000        │ 125000.50    │ 25.00      │
│ "B"      │ 4500        │ 112500.25    │ 25.00      │
│ ...      │ ...         │ ...          │ ...        │
└──────────┴─────────────┴──────────────┴────────────┘
```

### Type Conversion

cuDF → DuckDB type mapping:

| cuDF Type | DuckDB Type |
|-----------|-------------|
| INT64 | BIGINT |
| FLOAT64 | DOUBLE |
| STRING | VARCHAR |
| DATE32 | DATE |

---

## Phase 8: Return to User (DuckDB)

**Duration**: <1ms
**Location**: DuckDB core

### What Happens

Results returned through DuckDB:

```
DuckDB QueryResult
     ↓
Fetch() called repeatedly
     ↓
DataChunks returned to user
```

**User receives**:
```
┌──────────┬─────────────┬──────────────┬────────────┐
│ category │ order_count │ total_amount │ avg_amount │
│ varchar  │ int64       │ decimal(18,2)│ double     │
├──────────┼─────────────┼──────────────┼────────────┤
│ A        │        5000 │    125000.50 │      25.00 │
│ B        │        4500 │    112500.25 │      25.00 │
│ C        │        3200 │     80000.00 │      25.00 │
│ ...      │         ... │          ... │        ... │
└──────────┴─────────────┴──────────────┴────────────┘
10 rows returned
```

---

## Timeline Summary

Typical execution timeline for this query on SF10 dataset (10GB), NVIDIA A100:

| Phase | Duration | Percentage |
|-------|----------|------------|
| 1. Parse | 2ms | 0.3% |
| 2. Logical Planning | 10ms | 1.5% |
| 3. Physical Planning | 20ms | 3% |
| 4. Pipeline Construction | 10ms | 1.5% |
| 5. Task Creation | 5ms | 0.8% |
| 6. GPU Execution | 600ms | 90% |
| 7. Result Collection | 5ms | 0.8% |
| 8. Return to User | <1ms | 0.1% |
| **Total** | **~650ms** | **100%** |

**Key Insight**: GPU execution dominates (90% of time). Optimization efforts should focus here.

---

## Data Flow

### Memory Transfers

```
Orders Table (Disk)
     ↓ DUCKDB_SCAN
Host Memory (CPU)
     ↓ Transfer
GPU Memory (cuDF)
     ↓ Process (FILTER, AGGREGATE, etc.)
GPU Memory (Results)
     ↓ Transfer
Host Memory (DuckDB)
     ↓ Return
User Application
```

### Memory Usage

For 10GB input table:

| Stage | Location | Size | Notes |
|-------|----------|------|-------|
| Input | Disk | 10GB | Parquet compressed |
| Scan | CPU | ~300MB | Batched (not all in memory) |
| GPU Processing | GPU | ~500MB | Intermediate aggregates |
| Results | GPU | ~50KB | 10 rows |
| Results | CPU | ~50KB | Final output |

**Peak Memory**: ~800MB GPU, ~500MB CPU

---

## Error Scenarios

### Planning Error

```
Phase 3: Physical Planning
├─ Unsupported operator detected
├─ Error: "WINDOW function not supported"
└─ Fallback to DuckDB (if enabled)
```

### Execution Error

```
Phase 6: GPU Execution
├─ Out of GPU memory
├─ Trigger spilling to HOST memory
├─ Continue execution
└─ (Or fallback if spilling fails)
```

### Type Error

```
Phase 3: Physical Planning
├─ Incompatible type: HUGEINT
├─ Downcast to BIGINT
└─ Warning: Possible overflow
```

---

## Optimization Opportunities

### Query-Level

1. **Filter Pushdown**: Already done by DuckDB
2. **Projection Pushdown**: Select only needed columns
3. **LIMIT Pushdown**: Could use TOP_N instead of full sort

### Execution-Level

1. **Operator Fusion**: Combine SCAN + FILTER into single kernel
2. **Pipeline Parallelism**: Run multiple pipelines concurrently
3. **Batch Size Tuning**: Adjust batch size based on data width

---

## Monitoring Query Execution

### Enable Logging

```sql
SET sirius_log_level = 'DEBUG';
SELECT * FROM gpu_execution('...');
```

### View Execution Stats

```sql
SET sirius_enable_monitoring = true;
SELECT * FROM gpu_execution('...');
SELECT * FROM sirius_execution_stats();
```

**Output**:
```
┌─────────────────┬──────────┬────────────┬─────────────┐
│ pipeline_id     │ duration │ rows_in    │ rows_out    │
├─────────────────┼──────────┼────────────┼─────────────┤
│ Pipeline 1      │ 520ms    │ 10,000,000 │ 1,000       │
│ Pipeline 2      │ 80ms     │ 1,000      │ 300         │
│ Pipeline 3      │ 5ms      │ 300        │ 10          │
└─────────────────┴──────────┴────────────┴─────────────┘
```

---

## See Also

- [New Data Flow](new-data-flow.md) - New Mode specific flow
- [Legacy Data Flow](legacy-data-flow.md) - Legacy Mode flow
- [Inter-Pipeline Communication](inter-pipeline-communication.md) - Data repositories
- [System Overview](../02-architecture/system-overview.md) - Architecture context
