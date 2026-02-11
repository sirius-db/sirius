# Legacy Mode Data Flow

Detailed explanation of data flow in Sirius Legacy Mode (`gpu_processing`), covering the GPUIntermediateRelation model, pull-based execution, and traditional pipeline structure.

---

## Overview

Legacy Mode uses a **pull-based execution model** where operators request data from their children. This contrasts with New Mode's push-based, port-driven approach.

| Aspect | Legacy Mode | New Mode (for comparison) |
|--------|-------------|---------------------------|
| **Data Unit** | `GPUIntermediateRelation` | `cucascade::data_batch` |
| **Execution Model** | Pull-based (GetData) | Push-based (publish) |
| **Pipeline Communication** | Direct method calls | Port-based repositories |
| **Task Model** | Static (fixed pipeline) | Dynamic (hint-based) |
| **Memory Management** | GPUBufferManager | Multi-tier (GPU/HOST/DISK) |
| **Synchronization** | Implicit (call stack) | Explicit (repositories) |

**Key Characteristic**: Legacy Mode uses a **simpler, more direct** data flow model suitable for straightforward queries, but less flexible for complex pipeline dependencies.

---

## Core Data Structures

### 1. GPUIntermediateRelation

The fundamental data container in Legacy Mode.

**Definition**: `src/include/gpu_intermediate_relation.hpp`

```cpp
class GPUIntermediateRelation {
public:
    // Column data
    std::vector<GPUColumn> columns;

    // Row count
    size_t num_rows;

    // Schema information
    std::vector<LogicalType> types;
    std::vector<std::string> names;

    // Memory management
    std::vector<void*> gpu_buffers;
    bool owns_memory;

    // Constructors
    GPUIntermediateRelation();
    GPUIntermediateRelation(size_t num_rows, std::vector<GPUColumn> cols);

    // Utilities
    void Clear();
    GPUIntermediateRelation Clone();
};
```

**Key Properties**:
- **Mutable**: Can be modified in-place (unlike New Mode's immutable batches)
- **Owned**: Typically owns its GPU memory
- **Synchronous**: Created and consumed immediately
- **No Caching**: Not stored in repositories

### 2. GPUColumn

Represents a single column of data on the GPU.

**Definition**: `src/include/gpu_column.hpp`

```cpp
class GPUColumn {
public:
    // Data pointer (GPU memory)
    void* data;

    // Validity mask (nullability)
    void* validity;

    // Size in bytes
    size_t size_bytes;

    // Type information
    LogicalType type;

    // String dictionary (for string columns)
    StringDictionary* dict;

    // Constructors
    GPUColumn(void* data, void* validity, size_t size, LogicalType type);

    // Memory management
    void Free();
};
```

**Memory Layout**:
```
┌──────────────────────────────────────────┐
│ Data Array                               │
│ [value0][value1][value2]...[valueN]      │
│ (GPU memory, type-specific size)         │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ Validity Mask (optional)                 │
│ [bit0][bit1][bit2]...[bitN]              │
│ (1 bit per value, 1=valid, 0=null)      │
└──────────────────────────────────────────┘
```

### 3. GPUBufferManager

Manages GPU memory allocation and caching.

**Definition**: `src/gpu_buffer_manager.cpp:20-80`

```cpp
class GPUBufferManager {
private:
    // Singleton instance
    static GPUBufferManager* instance;

    // Memory pools
    std::unordered_map<size_t, std::vector<void*>> free_buffers;

    // Allocation tracking
    size_t total_allocated;
    size_t peak_usage;

    std::mutex allocation_mutex;

public:
    // Singleton access
    static GPUBufferManager& Get();

    // Allocation
    void* Allocate(size_t size);
    void Free(void* ptr, size_t size);

    // Caching
    void* AllocateOrReuse(size_t size);
    void ReturnToPool(void* ptr, size_t size);

    // Statistics
    size_t GetTotalAllocated();
    size_t GetPeakUsage();
};
```

**Caching Strategy**:
- Pool allocations by size
- Reuse buffers to avoid cudaMalloc overhead
- No automatic spilling (OOM if GPU full)

---

## Data Flow Patterns

### Pattern 1: Simple Scan + Filter

**Query**:
```sql
SELECT * FROM gpu_processing('SELECT * FROM data.parquet WHERE x > 100');
```

**Physical Plan**:
```
RESULT_COLLECTOR (sink)
    ↑
  FILTER (intermediate)
    ↑
TABLE_SCAN (source)
```

**Data Flow** (Pull-Based):

```
1. RESULT_COLLECTOR calls GetData()
     ↓
2. FILTER calls children[0]->GetData()
     ↓
3. TABLE_SCAN::GetData()
     - Read Parquet batch (100K rows)
     - Allocate GPU buffers via GPUBufferManager
     - Copy data to GPU
     - Return GPUIntermediateRelation
     ↑
4. FILTER::Execute(input)
     - Apply predicate (x > 100) using cuDF
     - Create filtered GPUIntermediateRelation
     - Free input buffers (return to pool)
     - Return filtered result
     ↑
5. RESULT_COLLECTOR::Sink(input)
     - Accumulate into result
     - Convert to DuckDB DataChunk
     - Free input buffers
```

**Code Example** (`src/operator/gpu_physical_filter.cpp:60-95`):

```cpp
GPUIntermediateRelation GPUPhysicalFilter::GetData() {
    // Pull data from child
    auto input = children[0]->GetData();

    if (input.num_rows == 0) {
        return input; // Empty result
    }

    // Execute filter predicate
    auto filtered = ExecuteFilter(input);

    // Free input memory (return to pool)
    for (auto& col : input.columns) {
        GPUBufferManager::Get().ReturnToPool(col.data, col.size_bytes);
        if (col.validity) {
            GPUBufferManager::Get().ReturnToPool(col.validity,
                                                  (input.num_rows + 7) / 8);
        }
    }

    return filtered;
}

GPUIntermediateRelation GPUPhysicalFilter::ExecuteFilter(
    const GPUIntermediateRelation& input) {

    // Convert to cuDF table
    auto cudf_table = ConvertToTable(input);

    // Apply predicate
    auto mask = EvaluatePredicate(cudf_table, filter_expr);

    // Apply mask using cuDF
    auto filtered_table = cudf::apply_boolean_mask(cudf_table, mask);

    // Convert back to GPUIntermediateRelation
    return ConvertFromTable(filtered_table);
}
```

### Pattern 2: Aggregation (Pipeline Break)

**Query**:
```sql
SELECT * FROM gpu_processing('
    SELECT category, SUM(amount) as total
    FROM sales.parquet
    GROUP BY category
');
```

**Physical Plan**:
```
Pipeline 2:
  RESULT_COLLECTOR
      ↑
  HASH_GROUP_BY (source)

Pipeline 1:
  HASH_GROUP_BY (sink)
      ↑
    SCAN
```

**Data Flow**:

```
Phase 1: Build Aggregation (Pipeline 1)
─────────────────────────────────────────

Loop:
  1. HASH_GROUP_BY (sink) calls children[0]->GetData()
       ↓
  2. SCAN::GetData()
       - Read batch 0 (100K rows)
       - Return GPUIntermediateRelation
       ↑
  3. HASH_GROUP_BY::Sink(batch_0)
       - Update hash table (group by category)
       - Accumulate SUM(amount)
       - Free batch_0 memory

  Repeat for batches 1-9...

  4. SCAN::GetData() returns empty
       ↓
  5. HASH_GROUP_BY::Finalize()
       - Finalize hash table
       - Store result in operator state

Phase 2: Emit Results (Pipeline 2)
──────────────────────────────────

  1. RESULT_COLLECTOR calls GetData()
       ↓
  2. HASH_GROUP_BY (source)::GetData()
       - Return finalized aggregation result
       - GPUIntermediateRelation with grouped data
       ↑
  3. RESULT_COLLECTOR::Sink(result)
       - Convert to DataChunk
       - Return to user
```

**Code Example** (`src/operator/gpu_physical_hash_aggregate.cpp:120-180`):

```cpp
// Sink side (Phase 1): Accumulate data
void GPUPhysicalHashAggregate::Sink(GPUIntermediateRelation& input) {
    if (input.num_rows == 0) return;

    // Convert to cuDF table
    auto cudf_table = ConvertToTable(input);

    // Extract group-by keys
    auto keys = cudf_table.select(groupby_column_indices);

    // Extract aggregate columns
    auto aggs = cudf_table.select(aggregate_column_indices);

    // Update hash table
    if (!hash_table) {
        // First batch: create hash table
        hash_table = std::make_unique<cudf::groupby::groupby>(keys);
        aggregate_state = aggs;
    } else {
        // Subsequent batches: merge into hash table
        hash_table->aggregate(keys, aggs, aggregate_operations);
    }

    // Free input memory
    FreeIntermediateRelation(input);

    // Track total rows processed
    total_input_rows += input.num_rows;
}

void GPUPhysicalHashAggregate::Finalize() {
    // Finalize aggregation
    if (hash_table) {
        auto result_table = hash_table->get_result();
        finalized_result = ConvertFromTable(result_table);
    }
}

// Source side (Phase 2): Emit result
GPUIntermediateRelation GPUPhysicalHashAggregate::GetData() {
    if (has_emitted) {
        return GPUIntermediateRelation(); // Empty
    }

    has_emitted = true;
    return std::move(finalized_result);
}
```

### Pattern 3: Hash Join

**Query**:
```sql
SELECT * FROM gpu_processing('
    SELECT o.order_id, o.amount, c.name
    FROM orders o
    JOIN customers c ON o.customer_id = c.id
');
```

**Physical Plan**:
```
Pipeline 3:
  RESULT_COLLECTOR
      ↑
  HASH_JOIN (probe, source)

Pipeline 2:
  HASH_JOIN (probe, sink)
      ↑
    SCAN orders

Pipeline 1:
  HASH_JOIN (build, sink)
      ↑
    SCAN customers
```

**Data Flow**:

```
Phase 1: Build Hash Table (Pipeline 1)
───────────────────────────────────────

Loop:
  1. HASH_JOIN (build sink) calls children[0]->GetData()
       ↓
  2. SCAN customers::GetData()
       - Read batch (100K rows, 5MB)
       - Return GPUIntermediateRelation
       ↑
  3. HASH_JOIN::SinkBuild(batch)
       - Extract join key (customer.id)
       - Extract payload (customer.name)
       - Insert into hash table
       - Free batch memory

  4. SCAN returns empty
       ↓
  5. HASH_JOIN::FinalizeBuild()
       - Finalize hash table structure
       - Hash table now ready (~8MB GPU memory)

Phase 2: Probe Hash Table (Pipeline 2)
───────────────────────────────────────

Loop:
  1. HASH_JOIN (probe sink) calls children[0]->GetData()
       ↓
  2. SCAN orders::GetData()
       - Read batch 0 (100K rows, 5MB)
       - Return GPUIntermediateRelation
       ↑
  3. HASH_JOIN::SinkProbe(batch)
       - Extract join key (order.customer_id)
       - Probe hash table
       - For matches: gather joined rows
       - Store in probe_results vector
       - Free batch memory

  Repeat for all order batches...

  4. SCAN returns empty
       ↓
  5. HASH_JOIN::FinalizeProbe()
       - Concatenate all probe results
       - Store in final_result

Phase 3: Emit Results (Pipeline 3)
───────────────────────────────────

  1. RESULT_COLLECTOR calls GetData()
       ↓
  2. HASH_JOIN (source)::GetData()
       - Return final_result (joined data)
       ↑
  3. RESULT_COLLECTOR::Sink(result)
       - Convert to DataChunk
       - Return to user
```

**Code Example** (`src/operator/gpu_physical_hash_join.cpp:200-280`):

```cpp
// Phase 1: Build side
void GPUPhysicalHashJoin::SinkBuild(GPUIntermediateRelation& input) {
    // Convert to cuDF table
    auto cudf_table = ConvertToTable(input);

    // Extract build key columns
    auto keys = cudf_table.select(build_key_indices);

    // Extract payload columns
    auto payload = cudf_table.select(build_payload_indices);

    // Build hash table
    if (!hash_table) {
        // Create hash table on first batch
        hash_table = std::make_unique<cudf::hash_join>(
            keys,
            cudf::nullable_join::YES
        );
        payload_table = payload;
    } else {
        // Append to existing hash table
        hash_table->append(keys);
        payload_table = cudf::concatenate({payload_table, payload});
    }

    // Free input
    FreeIntermediateRelation(input);

    build_row_count += input.num_rows;
}

void GPUPhysicalHashJoin::FinalizeBuild() {
    if (hash_table) {
        hash_table->finalize();
        build_complete = true;
    }
}

// Phase 2: Probe side
void GPUPhysicalHashJoin::SinkProbe(GPUIntermediateRelation& input) {
    // Ensure build is complete
    if (!build_complete) {
        throw InternalException("Build must complete before probe");
    }

    // Convert to cuDF table
    auto cudf_table = ConvertToTable(input);

    // Extract probe key columns
    auto keys = cudf_table.select(probe_key_indices);

    // Probe hash table
    auto [left_indices, right_indices] = hash_table->probe(keys);

    // Gather matching rows from both sides
    auto left_result = cudf::gather(cudf_table, left_indices);
    auto right_result = cudf::gather(payload_table, right_indices);

    // Concatenate columns
    auto joined = cudf::concatenate_columns({left_result, right_result});

    // Store result
    probe_results.push_back(ConvertFromTable(joined));

    // Free input
    FreeIntermediateRelation(input);

    probe_row_count += input.num_rows;
}

void GPUPhysicalHashJoin::FinalizeProbe() {
    // Concatenate all probe results
    if (!probe_results.empty()) {
        final_result = ConcatenateRelations(probe_results);
        probe_results.clear(); // Free intermediate results
    }
}

// Phase 3: Source
GPUIntermediateRelation GPUPhysicalHashJoin::GetData() {
    if (has_emitted) {
        return GPUIntermediateRelation(); // Empty
    }

    has_emitted = true;
    return std::move(final_result);
}
```

---

## Pipeline Structure

### GPUPipeline

**Definition**: `src/include/gpu_pipeline.hpp`

```cpp
class GPUPipeline {
public:
    // Source operator (produces data)
    GPUPhysicalOperator* source;

    // Intermediate operators (transform data)
    std::vector<GPUPhysicalOperator*> operators;

    // Sink operator (consumes data)
    GPUPhysicalOperator* sink;

    // Execution
    void Execute();
};
```

**Execution Logic** (`src/gpu_pipeline.cpp:50-85`):

```cpp
void GPUPipeline::Execute() {
    if (RequiresSourceSink()) {
        // Pattern: source → sink (e.g., scan → aggregate sink)
        ExecuteSourceSink();
    } else if (source) {
        // Pattern: source → result (e.g., scan → filter → result)
        ExecuteSourceToResult();
    } else {
        throw InternalException("Invalid pipeline configuration");
    }
}

void GPUPipeline::ExecuteSourceSink() {
    // Pull-based execution loop
    while (true) {
        // Pull batch from source
        auto batch = source->GetData();

        if (batch.num_rows == 0) {
            break; // No more data
        }

        // Apply intermediate operators
        for (auto& op : operators) {
            batch = op->Execute(batch);
            if (batch.num_rows == 0) {
                break; // Filtered out
            }
        }

        // Sink the batch
        if (batch.num_rows > 0 && sink) {
            sink->Sink(batch);
        }
    }

    // Finalize sink
    if (sink) {
        sink->Finalize();
    }
}

void GPUPipeline::ExecuteSourceToResult() {
    // Simple pull from top of tree
    auto result = source->GetData();

    // Apply operators (should include RESULT_COLLECTOR at top)
    for (auto& op : operators) {
        result = op->Execute(result);
    }
}
```

### GPUMetaPipeline

Orchestrates multiple pipelines with dependencies.

**Definition**: `src/include/gpu_meta_pipeline.hpp`

```cpp
class GPUMetaPipeline {
public:
    // All pipelines
    std::vector<GPUPipeline> pipelines;

    // Dependency graph
    std::vector<std::vector<size_t>> dependencies;

    // Execution
    void Execute();
    std::vector<size_t> TopologicalSort();
};
```

**Execution** (`src/gpu_meta_pipeline.cpp:40-70`):

```cpp
void GPUMetaPipeline::Execute() {
    // Topological sort of pipelines
    auto execution_order = TopologicalSort();

    // Execute pipelines in order
    for (size_t pipeline_idx : execution_order) {
        pipelines[pipeline_idx].Execute();
    }
}

std::vector<size_t> GPUMetaPipeline::TopologicalSort() {
    std::vector<size_t> result;
    std::vector<size_t> in_degree(pipelines.size(), 0);

    // Calculate in-degrees
    for (const auto& deps : dependencies) {
        for (size_t dep : deps) {
            in_degree[dep]++;
        }
    }

    // BFS
    std::queue<size_t> ready;
    for (size_t i = 0; i < pipelines.size(); i++) {
        if (in_degree[i] == 0) {
            ready.push(i);
        }
    }

    while (!ready.empty()) {
        size_t current = ready.front();
        ready.pop();
        result.push_back(current);

        for (size_t dep : dependencies[current]) {
            in_degree[dep]--;
            if (in_degree[dep] == 0) {
                ready.push(dep);
            }
        }
    }

    return result;
}
```

---

## Memory Management

### GPUBufferManager

**Allocation Strategy**:

```cpp
void* GPUBufferManager::AllocateOrReuse(size_t size) {
    std::lock_guard<std::mutex> lock(allocation_mutex);

    // Round up to next power of 2 for pooling
    size_t pool_size = NextPowerOfTwo(size);

    // Check if we have a free buffer of this size
    auto it = free_buffers.find(pool_size);
    if (it != free_buffers.end() && !it->second.empty()) {
        // Reuse existing buffer
        void* ptr = it->second.back();
        it->second.pop_back();
        return ptr;
    }

    // Allocate new buffer
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, pool_size);
    if (err != cudaSuccess) {
        throw OutOfMemoryException("GPU allocation failed");
    }

    total_allocated += pool_size;
    peak_usage = std::max(peak_usage, total_allocated);

    return ptr;
}

void GPUBufferManager::ReturnToPool(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(allocation_mutex);

    size_t pool_size = NextPowerOfTwo(size);

    // Add to free pool
    free_buffers[pool_size].push_back(ptr);
}
```

**No Automatic Spilling**:
- If GPU runs out of memory → OOM error
- No HOST or DISK fallback
- User must reduce batch size or query complexity

**Memory Lifetime**:
```
Allocate (cudaMalloc or pool)
    ↓
Use in operator
    ↓
Free (return to pool)
    ↓
Reuse in next batch
    ↓
Eventually: cudaFree on shutdown
```

---

## Concrete Example: Full Query Trace

**Query**:
```sql
SELECT * FROM gpu_processing('
    SELECT category, AVG(price) as avg_price
    FROM products
    WHERE price > 50
    GROUP BY category
    ORDER BY avg_price DESC
');
```

**Physical Plan**:
```
Pipeline 4: ORDER_BY (source) → RESULT_COLLECTOR

Pipeline 3: ORDER_BY (sink)

Pipeline 2: HASH_GROUP_BY (source) → ORDER_BY (sink)

Pipeline 1: SCAN → FILTER → HASH_GROUP_BY (sink)
```

### Execution Trace

```
Time: 0ms - Start Pipeline 1
──────────────────────────────

[Pipeline 1: SCAN → FILTER → HASH_GROUP_BY (sink)]

  HASH_GROUP_BY (sink) calls GetData()
    ↓
  FILTER calls children[0]->GetData()
    ↓
  SCAN::GetData()
    - Read Parquet batch 0 (100K rows, 5MB)
    - cudaMalloc or reuse pool buffer
    - Copy data to GPU
    - Return GPUIntermediateRelation
    ↑
  FILTER::Execute(batch_0)
    - Apply predicate: price > 50
    - Result: 80K rows (80% pass)
    - Free input batch (return to pool)
    - Return filtered batch
    ↑
  HASH_GROUP_BY::Sink(filtered_batch_0)
    - Extract category (group key)
    - Extract price (aggregate value)
    - Update hash table: category → {sum, count}
    - Free batch

  Loop for batches 1-9...

  SCAN::GetData() returns empty
    ↓
  HASH_GROUP_BY::Finalize()
    - Compute AVG = sum / count
    - Store result: ~1K unique categories

Time: 100ms - Pipeline 1 Complete
──────────────────────────────

Time: 100ms - Start Pipeline 2
──────────────────────────────

[Pipeline 2: HASH_GROUP_BY (source) → ORDER_BY (sink)]

  ORDER_BY (sink) calls GetData()
    ↓
  HASH_GROUP_BY (source)::GetData()
    - Return finalized result (1K rows)
    ↑
  ORDER_BY::Sink(aggregated_result)
    - Sort by avg_price DESC
    - Store sorted result

Time: 120ms - Pipeline 2 Complete
──────────────────────────────

Time: 120ms - Start Pipeline 3
──────────────────────────────

[Pipeline 3: ORDER_BY (source) → RESULT_COLLECTOR]

  RESULT_COLLECTOR calls GetData()
    ↓
  ORDER_BY (source)::GetData()
    - Return sorted result (1K rows)
    ↑
  RESULT_COLLECTOR::Sink(sorted_result)
    - Copy to host memory
    - Convert to DuckDB DataChunk
    - Free GPU memory

Time: 125ms - Query Complete
─────────────────────────────

Total Time: 125ms
- Pipeline 1 (scan+filter+agg): 100ms (80%)
- Pipeline 2 (sort): 20ms (16%)
- Pipeline 3 (collect): 5ms (4%)
```

### Memory Usage Timeline

```
Time    Operation                   GPU Memory      Pool Size
────    ─────────                   ──────────      ─────────
0ms     Initial                     0MB             0MB
10ms    Batch 0 allocated           5MB             5MB
15ms    Batch 0 filtered            4MB (freed 5MB) 5MB (reused)
20ms    Batch 0 aggregated          4MB (hash)      5MB
30ms    Batch 1 allocated           9MB (reused)    5MB
...     Process batches 1-9         4MB (hash)      5MB
100ms   Aggregation finalized       5MB (1K rows)   5MB
105ms   Sort allocated              10MB            10MB
120ms   Sort complete               5MB (sorted)    10MB
125ms   Result collected (host)     0MB (freed)     10MB

Peak GPU Usage: 10MB (during sort)
Pool Efficiency: 5MB allocated, reused 10x → 50MB virtual
```

---

## Limitations of Legacy Mode

### 1. No Automatic Spilling

**Problem**: Fixed GPU memory, no fallback

**Example**:
```sql
-- Query with 20GB intermediate result on 16GB GPU
SELECT * FROM gpu_processing('
    SELECT customer_id, COUNT(*) as order_count
    FROM huge_orders_table  -- 10 billion rows
    GROUP BY customer_id
');
```

**Result**: `OutOfMemoryException: GPU allocation failed`

**Workaround**:
- Reduce batch size
- Add more selective filters
- Split query into smaller parts
- Use New Mode with multi-tier memory

### 2. Synchronous Execution

**Problem**: Each pipeline waits for previous to complete

**Example**:
```sql
SELECT * FROM gpu_processing('
    SELECT * FROM t1
    JOIN t2 ON t1.id = t2.id
    ORDER BY t1.value
');
```

**Execution**:
```
Pipeline 1 (build): ████████████░░░░░░░░░░ (60% time)
Pipeline 2 (probe):             ███████░░░░ (30% time)
Pipeline 3 (sort):                     ████ (10% time)

Total: 100% sequential, no overlap
```

**New Mode Improvement**: Task-based execution allows overlap

### 3. Limited Parallelism

**Problem**: Single pipeline executes sequentially

**Legacy Mode**:
```
Batch 0: SCAN → FILTER → AGGREGATE
Batch 1: SCAN → FILTER → AGGREGATE
Batch 2: SCAN → FILTER → AGGREGATE
...
```

**New Mode**:
```
Batch 0: SCAN ──┐
Batch 1: SCAN ──┼─→ FILTER ──┐
Batch 2: SCAN ──┘             ├─→ AGGREGATE
Batch 3: SCAN ────→ FILTER ──┘

Multiple batches in flight simultaneously
```

### 4. Memory Inefficiency

**Problem**: Intermediate results stored in full

**Example** (10 batch join):
```
Build Phase:
  Batch 0 → hash table (keep)
  Batch 1 → hash table (keep)
  ...
  Batch 9 → hash table (keep)
  Total: 10 batches in memory

Probe Phase:
  Batch 0 → result 0 (store)
  Batch 1 → result 1 (store)
  ...
  Batch 9 → result 9 (store)
  Total: 10 results + hash table = 20 batches worth

Peak Memory: 20 batches
```

**New Mode**: Stream results through repositories, only keep active batches

---

## Performance Characteristics

### Typical Overhead

| Operation | Legacy Mode Cost | New Mode Cost | Difference |
|-----------|------------------|---------------|------------|
| Function call overhead | ~100ns/call | ~50ns/task | 2x slower |
| Memory allocation | ~50μs (pool hit) | ~10μs (pre-allocated) | 5x slower |
| Pipeline transition | ~1ms (synchronous) | ~100μs (async) | 10x slower |
| Data copy | Same | Same | Equal |
| GPU kernel | Same | Same | Equal |

**Bottleneck**: Pipeline transitions and memory management

### Benchmark: TPC-H Q1 (10GB)

**Legacy Mode**:
```
Phase               Time    % Total
────────────────    ────    ───────
Scan                400ms   50%
Aggregate           300ms   37.5%
Pipeline transition 100ms   12.5%
────────────────────────────────────
Total               800ms   100%
```

**New Mode**:
```
Phase               Time    % Total
────────────────    ────    ───────
Scan                400ms   57%
Aggregate           280ms   40%
Pipeline transition 20ms    3%
────────────────────────────────────
Total               700ms   100%

Speedup: 1.14x (mainly from reduced transitions)
```

---

## Debugging Legacy Mode

### Enable Debug Logging

```bash
export SIRIUS_LOG_LEVEL=DEBUG
export SIRIUS_LOG_FILE=/tmp/sirius_legacy.log
```

### Trace Data Flow

Add logging to operators:

```cpp
GPUIntermediateRelation GPUPhysicalFilter::GetData() {
    LOG_DEBUG("Filter::GetData() called");

    auto input = children[0]->GetData();
    LOG_DEBUG("Filter received {} rows", input.num_rows);

    auto output = ExecuteFilter(input);
    LOG_DEBUG("Filter produced {} rows", output.num_rows);

    return output;
}
```

**Log Output**:
```
[DEBUG] SCAN::GetData() called
[DEBUG] SCAN read 100000 rows from Parquet
[DEBUG] Filter::GetData() called
[DEBUG] Filter received 100000 rows
[DEBUG] Filter produced 80000 rows (80% selectivity)
[DEBUG] AGGREGATE::Sink called with 80000 rows
[DEBUG] AGGREGATE updated hash table, now 1523 groups
```

### Memory Profiling

```cpp
// Add to GPUBufferManager
void GPUBufferManager::PrintStats() {
    printf("Total Allocated: %zu MB\n", total_allocated / 1024 / 1024);
    printf("Peak Usage: %zu MB\n", peak_usage / 1024 / 1024);
    printf("Pool Sizes:\n");
    for (const auto& [size, buffers] : free_buffers) {
        printf("  %zu bytes: %zu buffers\n", size, buffers.size());
    }
}
```

**Output**:
```
Total Allocated: 245 MB
Peak Usage: 87 MB
Pool Sizes:
  1048576 bytes: 12 buffers
  2097152 bytes: 8 buffers
  4194304 bytes: 4 buffers
```

---

## Migration to New Mode

### Why Migrate?

1. **Better Memory Management**: Multi-tier spilling
2. **Improved Parallelism**: Task-based execution
3. **Lower Latency**: Reduced pipeline transitions
4. **Active Development**: New features and optimizations

### How to Migrate

**Old (Legacy Mode)**:
```sql
SELECT * FROM gpu_processing('SELECT * FROM data.parquet WHERE x > 10');
```

**New (New Mode)**:
```sql
SELECT * FROM gpu_execution('SELECT * FROM data.parquet WHERE x > 10');
```

### Known Issues During Migration

1. **Different SQL Support**: New Mode may not support all Legacy Mode features yet
2. **Different Error Messages**: Error handling differs
3. **Performance Variation**: Some queries faster, some slower (being optimized)

---

## See Also

- [Legacy Mode Overview](../03-legacy-mode/overview.md) - Introduction to Legacy Mode
- [New Data Flow](new-data-flow.md) - New Mode data flow for comparison
- [Query Lifecycle](query-lifecycle.md) - Complete query execution
- [Execution Modes](../02-architecture/execution-modes.md) - Mode comparison
- [Memory Management](../05-core-components/memory-management.md) - Memory details
- [Performance Tips](../appendices/performance-tips.md) - Optimization guide
