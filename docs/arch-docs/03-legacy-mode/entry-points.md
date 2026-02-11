# Legacy Mode Entry Points

This document details how queries enter and execute through Legacy Mode, focusing on the `gpu_processing()` table function and its implementation.

## Table Function: gpu_processing()

### Overview

The `gpu_processing()` table function is the primary entry point for Legacy Mode GPU execution in Sirius.

**Declaration**: `src/sirius_extension.cpp:598-602`

```cpp
TableFunction gpu_processing(
    "gpu_processing",
    {LogicalType::VARCHAR},        // Input: SQL query string
    GPUProcessingFunction,          // Execution function
    GPUProcessingBind              // Bind function
);
```

**Usage**:
```sql
SELECT * FROM gpu_processing('
    SELECT column1, SUM(column2)
    FROM my_table
    WHERE condition
    GROUP BY column1
');
```

---

## Bind Phase: GPUProcessingBind()

### Purpose

The bind phase analyzes the query and prepares for execution:
1. Parse the SQL query string
2. Create logical plan via DuckDB
3. Generate GPU physical plan
4. Set up return types and column names

### Implementation

**File**: `src/sirius_extension.cpp:240-339`

```cpp
unique_ptr<FunctionData> SiriusExtension::GPUProcessingBind(
    ClientContext& context,
    TableFunctionBindInput& input,
    vector<LogicalType>& return_types,
    vector<string>& names)
{
    auto result = make_uniq<GPUTableFunctionData>();

    // 1. Extract query string
    result->query = input.inputs[0].ToString();

    // 2. Create new connection for isolated execution
    result->conn = make_uniq<Connection>(*context.db);

    // 3. Prepare connection with custom settings
    result->PrepareConnection(context);

    // 4. Parse query
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(result->query);

    // 5. Create logical plan
    Planner planner(context);
    auto statement_type = parser.statements[0]->type;
    planner.CreatePlan(std::move(parser.statements[0]));

    // 6. Create prepared statement data
    auto prepared = make_shared_ptr<PreparedStatementData>(statement_type);
    prepared->names = planner.names;
    prepared->types = planner.types;
    prepared->value_map = std::move(planner.value_map);

    // 7. Generate GPU physical plan
    unique_ptr<LogicalOperator> logical_plan = result->ExtractPlan(context);
    auto gpu_physical_plan = GPUGeneratePhysicalPlan(context, logical_plan);

    // 8. Store prepared plan
    result->gpu_prepared = make_shared_ptr<GPUPreparedStatementData>(
        std::move(prepared), std::move(gpu_physical_plan));

    // 9. Set output schema
    for (auto& name : planner.names) {
        names.emplace_back(name);
    }
    for (auto& type : planner.types) {
        return_types.emplace_back(type);
    }

    return std::move(result);
}
```

### Key Steps Explained

#### Step 1: Extract Query String

```cpp
result->query = input.inputs[0].ToString();
```

Extracts the SQL query string from the table function parameter.

#### Step 2-3: Create Isolated Connection

```cpp
result->conn = make_uniq<Connection>(*context.db);
result->PrepareConnection(context);
```

Creates a new DuckDB connection to avoid interfering with the main query execution. Disables certain optimizations:

**File**: `src/sirius_extension.cpp:72-96`

```cpp
void GPUTableFunctionData::PrepareConnection(ClientContext& context)
{
    // Save original configuration
    original_config = context.config;
    original_disabled_optimizers = DBConfig::GetConfig(context)
        .options.disabled_optimizers;

    // Configure connection for GPU execution
    context.config.enable_optimizer = enable_optimizer;
    context.config.use_replacement_scans = false;

    // Disable problematic optimizations
    set<OptimizerType> disabled_optimizers =
        DBConfig::GetConfig(context).options.disabled_optimizers;

    // IN clause rewriter creates mark joins (not GPU-friendly)
    disabled_optimizers.insert(OptimizerType::IN_CLAUSE);

    // Compressed materialization is DuckDB-specific
    disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);

    DBConfig::GetConfig(context).options.disabled_optimizers =
        disabled_optimizers;
}
```

**Why Disable Optimizations?**
- `IN_CLAUSE`: Creates mark joins that are complex on GPU
- `COMPRESSED_MATERIALIZATION`: DuckDB-specific compression not supported

#### Step 4-5: Parse and Plan

```cpp
Parser parser(context.GetParserOptions());
parser.ParseQuery(result->query);

Planner planner(context);
planner.CreatePlan(std::move(parser.statements[0]));
```

Uses DuckDB's native parser and planner to create a logical plan.

#### Step 7: Generate GPU Physical Plan

```cpp
auto gpu_physical_plan = GPUGeneratePhysicalPlan(context, logical_plan);
```

Converts the logical plan to GPU physical operators. See [GPU Physical Plan Generator](#gpu-physical-plan-generator) below.

---

## Execution Phase: GPUProcessingFunction()

### Purpose

The execution phase runs the GPU plan and returns results to DuckDB.

### Implementation

**File**: `src/sirius_extension.cpp:560-596`

```cpp
void SiriusExtension::GPUProcessingFunction(
    ClientContext& context,
    TableFunctionInput& data_p,
    DataChunk& output)
{
    auto& data = (GPUTableFunctionData&)*data_p.bind_data;

    // Check if already finished
    if (data.finished) {
        return;
    }

    // First call: execute query on GPU
    if (!data.res) {
        auto start = std::chrono::high_resolution_clock::now();

        try {
            // Execute GPU query
            data.res = GPUExecuteQuery(
                context,
                data.query,
                data.gpu_prepared,
                {}
            );
        } catch (std::exception& e) {
            // Log error
            SIRIUS_LOG_ERROR("GPU execution failed: {}", e.what());

            // Optional fallback to CPU
            if (Config::ENABLE_FALLBACK) {
                data.res = data.conn->Query(data.query);
            } else {
                throw;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);
        SIRIUS_LOG_INFO("Query execution time: {:.2f} ms",
                       duration.count() / 1000.0);
    }

    // Fetch next chunk of results
    auto result_chunk = data.res->Fetch();
    if (result_chunk == nullptr) {
        output.SetCardinality(0);
        data.finished = true;

        // Cleanup connection
        data.CleanupConnection(context);
        return;
    }

    // Return chunk to DuckDB
    output.Reference(*result_chunk);
}
```

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ First Call: Execute Query                                    │
│                                                              │
│  if (!data.res) {                                           │
│      data.res = GPUExecuteQuery(context, query, plan, {});  │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Subsequent Calls: Fetch Results                             │
│                                                              │
│  while (has_more_chunks) {                                  │
│      chunk = data.res->Fetch();                             │
│      output.Reference(*chunk);  // Return to DuckDB         │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Final Call: Cleanup                                          │
│                                                              │
│  if (no_more_chunks) {                                      │
│      output.SetCardinality(0);  // Signal completion        │
│      CleanupConnection(context);                            │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

### Error Handling and Fallback

```cpp
try {
    data.res = GPUExecuteQuery(context, data.query, data.gpu_prepared, {});
} catch (std::exception& e) {
    SIRIUS_LOG_ERROR("GPU execution failed: {}", e.what());

    if (Config::ENABLE_FALLBACK) {
        // Fallback to DuckDB CPU execution
        data.res = data.conn->Query(data.query);
    } else {
        throw;  // Propagate error to user
    }
}
```

**Fallback Scenarios**:
- Unsupported operators encountered
- GPU out of memory
- CUDA errors
- Type conversion failures

---

## GPU Physical Plan Generator

### Entry Point

**File**: `src/gpu_physical_plan_generator.cpp`

```cpp
unique_ptr<GPUPhysicalOperator> GPUGeneratePhysicalPlan(
    ClientContext& context,
    unique_ptr<LogicalOperator>& logical_plan)
{
    GPUPhysicalPlanGenerator generator(context);
    return generator.CreatePlan(logical_plan);
}
```

### Logical to Physical Mapping

| Logical Operator | Physical Operator | Notes |
|------------------|-------------------|-------|
| LogicalGet | GPUPhysicalTableScan | Read from DuckDB tables |
| LogicalFilter | GPUPhysicalFilter | WHERE clause predicates |
| LogicalProjection | GPUPhysicalProjection | Column selection/computation |
| LogicalAggregate | GPUPhysicalHashGroupBy | GROUP BY aggregation |
| LogicalAggregate (no group) | GPUPhysicalUngroupedAggregate | No GROUP BY |
| LogicalJoin (Hash) | GPUPhysicalHashJoin | Equi-join via hash table |
| LogicalJoin (NestedLoop) | GPUPhysicalNestedLoopJoin | Non-equi join |
| LogicalOrder | GPUPhysicalOrderBy | ORDER BY clause |
| LogicalLimit | GPUPhysicalLimit | LIMIT clause |
| LogicalTopN | GPUPhysicalTopN | Combined ORDER BY + LIMIT |

### Planning Process

```cpp
class GPUPhysicalPlanGenerator {
public:
    unique_ptr<GPUPhysicalOperator> CreatePlan(
        unique_ptr<LogicalOperator>& logical_op)
    {
        // Recursively plan children first
        for (auto& child : logical_op->children) {
            auto physical_child = CreatePlan(child);
            physical_children.push_back(std::move(physical_child));
        }

        // Map logical operator to physical operator
        unique_ptr<GPUPhysicalOperator> physical_op;
        switch (logical_op->type) {
            case LogicalOperatorType::LOGICAL_GET:
                physical_op = PlanGet(logical_op);
                break;
            case LogicalOperatorType::LOGICAL_FILTER:
                physical_op = PlanFilter(logical_op);
                break;
            case LogicalOperatorType::LOGICAL_AGGREGATE:
                physical_op = PlanAggregate(logical_op);
                break;
            case LogicalOperatorType::LOGICAL_JOIN:
                physical_op = PlanJoin(logical_op);
                break;
            // ... more operators
        }

        // Attach children
        physical_op->children = std::move(physical_children);

        return physical_op;
    }
};
```

---

## GPUExecuteQuery Function

### Overview

**File**: `src/gpu_executor.cpp:50-150` (approximate)

```cpp
unique_ptr<QueryResult> GPUExecuteQuery(
    ClientContext& context,
    const string& query,
    shared_ptr<GPUPreparedStatementData> prepared,
    vector<Value> parameters)
{
    // 1. Create GPU executor
    GPUExecutor executor(context);

    // 2. Build pipelines from physical plan
    executor.BuildPipelines(prepared->physical_plan);

    // 3. Initialize operator states
    executor.InitializeOperatorStates();

    // 4. Execute pipelines in dependency order
    executor.Execute();

    // 5. Collect results
    auto result = executor.GetQueryResult();

    return result;
}
```

### Detailed Flow

```
┌───────────────────────────────────────────────────────────┐
│ 1. Build Pipelines                                         │
│    • Traverse operator tree                                │
│    • Identify pipeline breaks                              │
│    • Create GPUPipeline objects                            │
│    • Build GPUMetaPipeline DAG                             │
└───────────────────────────────────────────────────────────┘
                         ↓
┌───────────────────────────────────────────────────────────┐
│ 2. Initialize States                                       │
│    • GlobalSourceState for sources                         │
│    • GlobalOperatorState for operators                     │
│    • GlobalSinkState for sinks                             │
└───────────────────────────────────────────────────────────┘
                         ↓
┌───────────────────────────────────────────────────────────┐
│ 3. Execute Pipelines                                       │
│    For each pipeline in topological order:                 │
│      • Run source: GetData()                               │
│      • Run operators: Execute()                            │
│      • Run sink: Sink()                                    │
│      • Finalize: CombineFinalize()                         │
└───────────────────────────────────────────────────────────┘
                         ↓
┌───────────────────────────────────────────────────────────┐
│ 4. Collect Results                                         │
│    • Extract final data from result collector              │
│    • Convert from GPU format to DuckDB format              │
│    • Create MaterializedQueryResult                        │
└───────────────────────────────────────────────────────────┘
```

---

## Data Structures

### GPUTableFunctionData

**File**: `src/sirius_extension.cpp:57-100`

```cpp
struct GPUTableFunctionData : public TableFunctionData {
    // Query string
    string query;

    // Isolated connection
    unique_ptr<Connection> conn;

    // Prepared plan
    shared_ptr<GPUPreparedStatementData> gpu_prepared;

    // Result set
    unique_ptr<QueryResult> res;

    // Execution state
    bool finished = false;

    // Configuration
    bool enable_optimizer;
    ClientConfig original_config;
    set<OptimizerType> original_disabled_optimizers;

    // Setup/cleanup
    void PrepareConnection(ClientContext& context);
    void CleanupConnection(ClientContext& context);

    // Extract logical plan
    unique_ptr<LogicalOperator> ExtractPlan(ClientContext& context);
};
```

### GPUPreparedStatementData

**File**: `src/include/gpu_prepared_statement.hpp`

```cpp
struct GPUPreparedStatementData {
    // DuckDB prepared statement (for schema info)
    shared_ptr<PreparedStatementData> prepared;

    // GPU physical plan
    unique_ptr<GPUPhysicalOperator> physical_plan;

    GPUPreparedStatementData(
        shared_ptr<PreparedStatementData> prep,
        unique_ptr<GPUPhysicalOperator> plan)
        : prepared(std::move(prep)),
          physical_plan(std::move(plan))
    {}
};
```

---

## Complete Example Walkthrough

Let's trace a simple query end-to-end:

```sql
SELECT * FROM gpu_processing('
    SELECT category, COUNT(*) as count
    FROM products
    WHERE price > 50
    GROUP BY category
');
```

### Step 1: User Invokes Table Function

DuckDB parser recognizes `gpu_processing()` and calls `GPUProcessingBind()`.

### Step 2: Bind Phase

```
GPUProcessingBind():
  1. Extract query: "SELECT category, COUNT(*) FROM products WHERE price > 50 GROUP BY category"
  2. Create isolated connection
  3. Parse → AST
  4. Plan → Logical operators:
     LogicalAggregate [category, COUNT(*)]
          ↓
     LogicalFilter [price > 50]
          ↓
     LogicalGet [products]

  5. Generate physical plan:
     GPUPhysicalHashGroupBy [category, COUNT(*)]
          ↓
     GPUPhysicalFilter [price > 50]
          ↓
     GPUPhysicalTableScan [products]

  6. Return types: [VARCHAR, BIGINT]
     Names: ["category", "count"]
```

### Step 3: Execution Phase (First Call)

```
GPUProcessingFunction():
  1. data.res = GPUExecuteQuery(context, query, gpu_prepared, {})

  GPUExecuteQuery():
    a. Build pipelines:
       Pipeline 1: SCAN → FILTER → HASH_AGGREGATE (sink)
       Pipeline 2: HASH_AGGREGATE (source) → RESULT

    b. Execute Pipeline 1:
       - SCAN.GetData() → products table (batch 1)
       - FILTER.Execute() → apply price > 50
       - HASH_AGGREGATE.Sink() → build hash table
       - Repeat for all batches
       - HASH_AGGREGATE.CombineFinalize() → finalize aggregates

    c. Execute Pipeline 2:
       - HASH_AGGREGATE.GetData() → aggregated results
       - RESULT.Sink() → collect results

    d. Return QueryResult

  2. Fetch first chunk
  3. output.Reference(*chunk) → return to DuckDB
```

### Step 4: Subsequent Calls

```
GPUProcessingFunction():
  1. data.res->Fetch() → get next chunk
  2. output.Reference(*chunk) → return to DuckDB
  3. Repeat until no more chunks
```

### Step 5: Final Call

```
GPUProcessingFunction():
  1. data.res->Fetch() → nullptr (no more chunks)
  2. output.SetCardinality(0) → signal completion
  3. CleanupConnection() → restore configuration
  4. data.finished = true
```

---

## Configuration Options

### Query-Level Options

```sql
-- Enable optimizer (default: true)
SELECT * FROM gpu_processing(
    'SELECT ...',
    enable_optimizer := true
);

-- Disable optimizer for testing
SELECT * FROM gpu_processing(
    'SELECT ...',
    enable_optimizer := false
);
```

### Global Configuration

```sql
-- Enable fallback to CPU on errors
SET enable_gpu_fallback = true;

-- GPU memory limit
SET gpu_memory_limit = 8192;  -- MB

-- Batch size for scanning
SET gpu_scan_batch_size = 100000;
```

---

## Performance Considerations

### Optimization Disabled

Some DuckDB optimizations are disabled for GPU compatibility:
- IN clause rewriter (creates mark joins)
- Compressed materialization (DuckDB-specific)

**Impact**: CPU fallback may be slower than native DuckDB execution.

### Connection Overhead

Each `gpu_processing()` call creates a new connection:
- Overhead: ~1-5ms per call
- Necessary for isolation
- Amortized over query execution time

### Type Conversion

Some types require conversion:
- HUGEINT → BIGINT (precision loss for large values)
- DECIMAL → DOUBLE (for unsupported precision)
- Complex types → Fallback to CPU

---

## Debugging

### Enable Logging

```cpp
// In code or via config
SIRIUS_LOG_DEBUG("Query plan: {}", plan->ToString());
SIRIUS_LOG_INFO("Executing on GPU");
```

### Inspect Plans

```sql
-- Enable query profiling
PRAGMA enable_profiling;

SELECT * FROM gpu_processing('SELECT ...');

-- View profile
PRAGMA profile_output;
```

### Fallback Testing

```sql
-- Force fallback to compare results
SET enable_gpu_fallback = true;
SET gpu_force_fallback = true;  -- Testing only

SELECT * FROM gpu_processing('SELECT ...');
```

---

## Next Steps

- **Operators**: [Legacy Mode Operators](operators.md) - Detailed operator implementations
- **Pipelines**: [Pipeline Execution](pipeline-execution.md) - Pipeline construction and execution
- **Memory**: [Memory Management](memory-management.md) - GPUBufferManager details

For modern development, see [New Mode Entry Points](../04-new-mode/entry-points.md).
