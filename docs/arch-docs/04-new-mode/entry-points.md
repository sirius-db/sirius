# New Mode Entry Points

This document details how queries enter and execute through New Mode, focusing on the `gpu_execution()` table function and the sirius_interface.

## Table Function: gpu_execution()

### Overview

The `gpu_execution()` table function is the primary entry point for New Mode GPU execution in Sirius.

**Declaration**: `src/sirius_extension.cpp:604-610`

```cpp
TableFunction gpu_execution(
    "gpu_execution",
    {LogicalType::VARCHAR},           // Input: SQL query string
    GPUExecutionFunction,              // Execution function
    SiriusExtension::GPUExecutionBind  // Bind function
);
```

**Usage**:
```sql
SELECT * FROM gpu_execution('
    SELECT customer_id, SUM(amount) as total
    FROM orders
    WHERE date >= ''2024-01-01''
    GROUP BY customer_id
    ORDER BY total DESC
');
```

---

## Bind Phase: GPUExecutionBind()

### Purpose

The bind phase analyzes the query and prepares execution:
1. Parse SQL query string
2. Create logical plan via DuckDB
3. Generate Sirius physical plan
4. Set up return types and column names

### Implementation

**File**: `src/sirius_extension.cpp:353-409`

```cpp
unique_ptr<FunctionData> SiriusExtension::GPUExecutionBind(
    ClientContext& context,
    TableFunctionBindInput& input,
    vector<LogicalType>& return_types,
    vector<string>& names)
{
    auto result = make_uniq<SiriusTableFunctionData>();

    // 1. Extract query and setup
    result->conn = make_uniq<Connection>(*context.db);
    result->query = input.inputs[0].ToString();
    result->enable_optimizer = true;
    result->sirius_iface = make_uniq<::sirius::sirius_interface>(context);

    // 2. Validate input
    if (input.inputs[0].IsNull()) {
        throw BinderException("gpu_execution cannot be called with a NULL parameter");
    }

    // 3. Parse query
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(result->query);

    Planner planner(context);
    auto statement_type = parser.statements[0]->type;
    planner.CreatePlan(std::move(parser.statements[0]));
    D_ASSERT(planner.plan);

    // 4. Type handling - cuDF doesn't support HUGEINT
    for (auto& type : planner.types) {
        if (type == LogicalType::HUGEINT) {
            type = LogicalType::BIGINT;  // Downcast
        }
    }

    // 5. Create prepared statement
    auto prepared = make_shared_ptr<PreparedStatementData>(statement_type);
    prepared->names = planner.names;
    prepared->types = planner.types;
    prepared->value_map = std::move(planner.value_map);

    // 6. Generate Sirius physical plan
    unique_ptr<LogicalOperator> query_plan = result->ExtractPlan(context);
    SIRIUS_LOG_DEBUG("Query plan:\n{}", query_plan->ToString());

    try {
        auto sirius_physical_plan = SiriusGeneratePhysicalPlan(context, query_plan);
        SIRIUS_LOG_DEBUG("Done generating sirius physical plan");

        auto gpu_prepared = make_shared_ptr<::sirius::sirius_prepared_statement_data>(
            std::move(prepared), std::move(sirius_physical_plan));
        result->gpu_prepared = gpu_prepared;
    } catch (std::exception& e) {
        ErrorData error(e);
        SIRIUS_LOG_ERROR("Error in SiriusGeneratePhysicalPlan: {}", error.RawMessage());
        result->plan_error = true;
    }

    // 7. Set output schema
    for (auto& column : planner.names) {
        names.emplace_back(column);
    }
    for (auto& type : planner.types) {
        return_types.emplace_back(type);
    }

    return std::move(result);
}
```

### Key Steps Explained

#### Step 1: Initialize Sirius Interface

```cpp
result->sirius_iface = make_uniq<::sirius::sirius_interface>(context);
```

Creates the `sirius_interface` - the main coordinator for New Mode execution.

**File**: `src/sirius_interface.hpp`

```cpp
class sirius_interface {
public:
    sirius_interface(ClientContext& context);

    // Main execution method
    unique_ptr<QueryResult> sirius_execute_query(
        ClientContext& context,
        const string& query,
        shared_ptr<sirius_prepared_statement_data> prepared,
        vector<Value> parameters);
};
```

#### Step 2: Type Downcasting

```cpp
// cuDF does not support HUGEINT (int128)
for (auto& type : planner.types) {
    if (type == LogicalType::HUGEINT) {
        type = LogicalType::BIGINT;
    }
}
```

**Why?** DuckDB widens aggregates like `SUM(int32)` to `HUGEINT`, but cuDF only supports up to 64-bit integers.

**Impact**: Potential overflow for very large sums.

**Workaround**: Use `DECIMAL` for high-precision aggregates.

#### Step 3: Generate Physical Plan

```cpp
auto sirius_physical_plan = SiriusGeneratePhysicalPlan(context, query_plan);
```

Converts logical plan to `sirius_physical_operator` tree.

**File**: `src/planner/sirius_physical_plan_generator.cpp`

---

## Execution Phase: GPUExecutionFunction()

### Purpose

Execute the query on GPU and return results to DuckDB.

### Implementation

**File**: `src/sirius_extension.cpp:411-452`

```cpp
void SiriusExtension::GPUExecutionFunction(
    ClientContext& context,
    TableFunctionInput& data_p,
    DataChunk& output)
{
    auto& data = (SiriusTableFunctionData&)*data_p.bind_data;

    // Check if finished
    if (data.finished) {
        return;
    }

    // First call: execute query
    if (!data.res) {
        auto start = std::chrono::high_resolution_clock::now();

        if (data.plan_error) {
            // Planning failed, fallback to DuckDB
            printf("Error in SiriusExecuteQuery, fallback to DuckDB\n");
            data.res = data.conn->Query(data.query);
        } else {
            // Execute on GPU via sirius_interface
            data.res = data.sirius_iface->sirius_execute_query(
                context, data.query, data.gpu_prepared, {});

            if (data.res->HasError()) {
                SIRIUS_LOG_ERROR("SiriusExecuteQuery error: {}", data.res->GetError());

                if (!Config::ENABLE_FALLBACK_CHECK) {
                    // Fallback to DuckDB
                    printf("Error in SiriusExecuteQuery, fallback to DuckDB\n");
                    data.res = data.conn->Query(data.query);
                }
                // With ENABLE_FALLBACK_CHECK, error propagates
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        SIRIUS_LOG_INFO("Execute query time: {:.2f} ms", duration.count() / 1000.0);
    }

    // Fetch next chunk
    auto result_chunk = data.res->Fetch();
    if (result_chunk == nullptr) {
        output.SetCardinality(0);
        return;
    }

    output.Reference(*result_chunk);
}
```

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ First Call: Execute Query on GPU                            │
│                                                              │
│  if (!data.res) {                                           │
│      data.res = sirius_iface->sirius_execute_query(...);   │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Subsequent Calls: Fetch Result Chunks                       │
│                                                              │
│  while (has_more_chunks) {                                  │
│      chunk = data.res->Fetch();                             │
│      output.Reference(*chunk);  // Return to DuckDB         │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Final Call: Signal Completion                               │
│                                                              │
│  if (no_more_chunks) {                                      │
│      output.SetCardinality(0);                              │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Sirius Interface

### Main Execution Method

**File**: `src/sirius_interface.cpp`

```cpp
unique_ptr<QueryResult> sirius_interface::sirius_execute_query(
    ClientContext& context,
    const string& query,
    shared_ptr<sirius_prepared_statement_data> prepared,
    vector<Value> parameters)
{
    // 1. Create sirius_engine
    auto engine = make_unique<sirius_engine>(context);

    // 2. Initialize engine with physical plan
    engine->initialize(prepared->physical_plan);

    // 3. Execute query
    engine->execute();

    // 4. Get results
    auto result = engine->get_query_result();

    return result;
}
```

### Sirius Engine Lifecycle

```
sirius_interface::sirius_execute_query()
     ↓
1. Create sirius_engine
     ↓
2. engine->initialize(physical_plan)
   ├─ Build sirius_pipelines
   ├─ Create data_repositories
   ├─ Establish port connections
   └─ Initialize task_executors
     ↓
3. engine->execute()
   ├─ task_creator: Generate tasks
   ├─ pipeline_executor: Run tasks on GPU
   └─ Tasks execute when inputs ready
     ↓
4. engine->get_query_result()
   ├─ Collect results from result_collector
   ├─ Convert cuDF → DuckDB format
   └─ Transfer GPU → CPU
     ↓
Return QueryResult to DuckDB
```

---

## Sirius Engine

### Engine Structure

**File**: `src/include/sirius_engine.hpp`

```cpp
class sirius_engine {
public:
    sirius_engine(ClientContext& context);

    // Initialize with physical plan
    void initialize(unique_ptr<sirius_physical_operator> plan);

    // Execute query
    void execute();

    // Get results
    unique_ptr<QueryResult> get_query_result();

private:
    // Execution state
    unique_ptr<sirius_meta_pipeline> meta_pipeline;
    vector<shared_ptr<data_repository>> repositories;
    vector<unique_ptr<task_executor>> executors;

    // Configuration
    sirius_config& config;
    SiriusContext& context;
};
```

### Initialize Phase

**File**: `src/sirius_engine.cpp`

```cpp
void sirius_engine::initialize(unique_ptr<sirius_physical_operator> plan) {
    // 1. Build pipelines from operator tree
    sirius_pipeline_build_state build_state;
    meta_pipeline = build_state.build_pipelines(std::move(plan));

    // 2. Create data repositories for inter-pipeline communication
    for (auto& pipeline : meta_pipeline->pipelines) {
        for (auto& output_port : pipeline->output_ports) {
            auto repo = make_shared<data_repository>();

            // Configure memory tiers
            repo->configure_tiers(
                config.gpu_memory_limit,
                config.host_memory_limit,
                config.disk_memory_limit
            );

            repositories.push_back(repo);
            output_port.connect(repo);
        }
    }

    // 3. Connect input ports to repositories
    meta_pipeline->connect_ports();

    // 4. Initialize task executors
    executors.push_back(make_unique<pipeline_executor>(config.pipeline_executor_threads));
    executors.push_back(make_unique<task_creator>(config.task_creator_threads));
    executors.push_back(make_unique<downgrade_executor>(config.downgrade_executor_threads));
    executors.push_back(make_unique<duckdb_scan_executor>(config.duckdb_scan_executor_threads));

    // 5. Start executors
    for (auto& executor : executors) {
        executor->start();
    }
}
```

### Execute Phase

```cpp
void sirius_engine::execute() {
    // 1. Submit initial tasks to task_creator
    for (auto& pipeline : meta_pipeline->pipelines) {
        auto hint = pipeline->get_task_hint();
        if (hint.hint == TaskCreationHint::READY) {
            task_creator->submit_pipeline(pipeline);
        }
    }

    // 2. Task executors work asynchronously
    // - task_creator generates tasks when pipelines ready
    // - pipeline_executor runs tasks on GPU
    // - Tasks publish results to data repositories
    // - Dependent pipelines become READY and create tasks

    // 3. Wait for completion
    meta_pipeline->wait_for_completion();

    // 4. Shutdown executors
    for (auto& executor : executors) {
        executor->shutdown();
    }
}
```

---

## Data Structures

### SiriusTableFunctionData

**File**: `src/sirius_extension.cpp:103-150`

```cpp
struct SiriusTableFunctionData : public TableFunctionData {
    // Query string
    string query;

    // Connection for fallback
    unique_ptr<Connection> conn;

    // Sirius interface
    unique_ptr<::sirius::sirius_interface> sirius_iface;

    // Prepared plan
    shared_ptr<::sirius::sirius_prepared_statement_data> gpu_prepared;

    // Result set
    unique_ptr<QueryResult> res;

    // Execution state
    bool finished = false;
    bool plan_error = false;
    bool enable_optimizer;
};
```

### sirius_prepared_statement_data

**File**: `src/include/sirius_prepared_statement.hpp`

```cpp
struct sirius_prepared_statement_data {
    // DuckDB prepared statement (schema info)
    shared_ptr<PreparedStatementData> prepared;

    // Sirius physical plan
    unique_ptr<sirius_physical_operator> physical_plan;

    sirius_prepared_statement_data(
        shared_ptr<PreparedStatementData> prep,
        unique_ptr<sirius_physical_operator> plan)
        : prepared(std::move(prep)),
          physical_plan(std::move(plan))
    {}
};
```

---

## Complete Example Walkthrough

### Query

```sql
SELECT * FROM gpu_execution('
    SELECT category, COUNT(*) as count, AVG(price) as avg_price
    FROM products
    WHERE price > 50
    GROUP BY category
    ORDER BY count DESC
');
```

### Step 1: Bind Phase

```
GPUExecutionBind():
  1. Parse: "SELECT category, COUNT(*) as count, AVG(price)..."
  2. Logical Plan:
     LogicalOrder [count DESC]
          ↓
     LogicalProjection [category, COUNT(*), AVG(price)]
          ↓
     LogicalAggregate [GROUP BY category]
          ↓
     LogicalFilter [price > 50]
          ↓
     LogicalGet [products]

  3. Physical Plan (SiriusGeneratePhysicalPlan):
     RESULT_COLLECTOR
          ↓
     ORDER_BY [count DESC]
          ↓
     HASH_GROUP_BY [category, COUNT(*), AVG(price)]
          ↓
     FILTER [price > 50]
          ↓
     DUCKDB_SCAN [products]

  4. Return: types=[VARCHAR, BIGINT, DOUBLE], names=["category", "count", "avg_price"]
```

### Step 2: First Execution Call

```
GPUExecutionFunction():
  data.res = sirius_iface->sirius_execute_query(context, query, gpu_prepared, {})
    ↓
  sirius_interface::sirius_execute_query():
    engine = make_unique<sirius_engine>(context)
    engine->initialize(physical_plan)
    engine->execute()
    result = engine->get_query_result()
```

### Step 3: Engine Initialize

```
sirius_engine::initialize():
  1. Build pipelines:
     Pipeline 1: DUCKDB_SCAN → FILTER → HASH_GROUP_BY (sink)
     Pipeline 2: HASH_GROUP_BY (source) → ORDER_BY (sink)
     Pipeline 3: ORDER_BY (source) → RESULT_COLLECTOR

  2. Create repositories:
     Repository 1: Pipeline 1 output → Pipeline 2 input
     Repository 2: Pipeline 2 output → Pipeline 3 input

  3. Connect ports:
     Pipeline 1.output_port[0] → Repository 1
     Pipeline 2.input_port[0] ← Repository 1
     Pipeline 2.output_port[0] → Repository 2
     Pipeline 3.input_port[0] ← Repository 2

  4. Start executors: pipeline_executor, task_creator, etc.
```

### Step 4: Engine Execute

```
sirius_engine::execute():
  1. task_creator checks hints:
     - Pipeline 1: READY (no dependencies)
     - Pipeline 2: WAITING_FOR_INPUT_DATA (needs Repository 1)
     - Pipeline 3: WAITING_FOR_INPUT_DATA (needs Repository 2)

  2. task_creator submits Pipeline 1 tasks to pipeline_executor:
     Task 1.1: Scan batch 1 → Filter → Aggregate (partial)
     Task 1.2: Scan batch 2 → Filter → Aggregate (partial)
     ...
     Task 1.N: Finalize aggregate → push to Repository 1

  3. Pipeline 1 completes, Repository 1 has data:
     - Pipeline 2 hint becomes READY
     - task_creator submits Pipeline 2 tasks

  4. Task 2.1: Pull from Repository 1 → Sort → push to Repository 2

  5. Pipeline 2 completes, Repository 2 has data:
     - Pipeline 3 hint becomes READY
     - task_creator submits Pipeline 3 tasks

  6. Task 3.1: Pull from Repository 2 → Collect results

  7. All pipelines complete
```

### Step 5: Fetch Results

```
GPUExecutionFunction() (subsequent calls):
  chunk = data.res->Fetch()  // Get next chunk from result collector
  output.Reference(*chunk)   // Return to DuckDB

  Repeat until Fetch() returns nullptr
  Then: output.SetCardinality(0) // Signal completion
```

---

## Error Handling

### Planning Errors

```cpp
try {
    auto sirius_physical_plan = SiriusGeneratePhysicalPlan(context, query_plan);
    result->gpu_prepared = make_shared_ptr<sirius_prepared_statement_data>(...);
} catch (std::exception& e) {
    SIRIUS_LOG_ERROR("Error in planning: {}", e.what());
    result->plan_error = true;  // Will trigger fallback
}
```

### Execution Errors

```cpp
data.res = data.sirius_iface->sirius_execute_query(...);

if (data.res->HasError()) {
    SIRIUS_LOG_ERROR("Execution error: {}", data.res->GetError());

    if (Config::ENABLE_FALLBACK) {
        // Fallback to DuckDB CPU execution
        data.res = data.conn->Query(data.query);
    } else {
        // Propagate error to user
        throw;
    }
}
```

---

## Performance Considerations

### Cold Start

First query initializes:
- CUDA context (~100-500ms)
- cuDF libraries
- Task executors
- Memory pools

**Mitigation**: Warm up with dummy query

### Query Complexity

**Simple queries** (< 100K rows):
- May be slower than CPU due to GPU overhead
- Use CPU for very small queries

**Complex queries** (> 1M rows):
- GPU excels
- Benefits from parallel execution

---

## Configuration

### Query-Level Options

```sql
-- Enable/disable optimizer
SELECT * FROM gpu_execution('...', enable_optimizer := false);
```

### Global Configuration

```sql
SET sirius_enable_fallback = true;
SET sirius_log_level = 'DEBUG';
```

---

## Next Steps

- **Operators**: [New Mode Operators](operators.md) - Operator details
- **Cucascade**: [Cucascade Integration](cucascade-integration.md) - Data repositories
- **Pipelines**: [Pipeline Execution](pipeline-execution.md) - Task model
- **Task Creation**: [Task Creation](task-creation.md) - Dynamic scheduling

For comparison, see [Legacy Mode Entry Points](../03-legacy-mode/entry-points.md).
