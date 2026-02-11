# DuckDB Integration

This document explains how Sirius integrates with DuckDB as an extension, covering the extension API, data flow between Sirius and DuckDB, and the integration points.

## Overview

Sirius operates as a **DuckDB extension**, which means it:
- Loads dynamically into the DuckDB process
- Registers custom table functions for GPU execution
- Leverages DuckDB for SQL parsing, optimization, and result management
- Seamlessly interoperates with DuckDB's native execution

## Extension Architecture

```
┌──────────────────────────────────────────────────────┐
│              DuckDB Core                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │   Parser   │→ │  Planner   │→ │  Executor  │    │
│  └────────────┘  └────────────┘  └────────────┘    │
│         ↓               ↓                             │
│  ┌──────────────────────────────────────────┐       │
│  │       Extension API                       │       │
│  └──────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────┐
│            Sirius Extension                           │
│  ┌────────────────────────────────────────┐          │
│  │  Extension Registration                │          │
│  │  - SiriusExtension::Load()             │          │
│  │  - Register table functions            │          │
│  └────────────────────────────────────────┘          │
│  ┌────────────────────────────────────────┐          │
│  │  Table Functions                       │          │
│  │  - gpu_processing()  (legacy)          │          │
│  │  - gpu_execution()   (new)             │          │
│  └────────────────────────────────────────┘          │
│  ┌────────────────────────────────────────┐          │
│  │  GPU Execution Engine                  │          │
│  └────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────┘
```

---

## Extension Registration

### Loading the Extension

The extension is loaded via the `LOAD` command:

```sql
LOAD 'sirius';
```

This triggers the extension initialization process:

**File**: `src/sirius_extension.cpp:794-833`

```cpp
void SiriusExtension::Load(DuckDB& db) {
    // Register configuration options
    auto& config = DBConfig::GetConfig(*db.instance);

    // Register table functions
    Connection con(db);
    con.BeginTransaction();

    // Register gpu_processing (legacy mode)
    auto gpu_processing_function = GPUProcessingTableFunction();
    CreateTableFunctionInfo gpu_processing_info(gpu_processing_function);
    auto& catalog = Catalog::GetSystemCatalog(*con.context);
    catalog.CreateTableFunction(*con.context, &gpu_processing_info);

    // Register gpu_execution (new mode)
    auto gpu_execution_function = GPUExecutionTableFunction();
    CreateTableFunctionInfo gpu_execution_info(gpu_execution_function);
    catalog.CreateTableFunction(*con.context, &gpu_execution_info);

    con.Commit();

    // Initialize GPU resources
    InitializeGPUResources();
}
```

### Extension Metadata

**File**: `src/sirius_extension.cpp:830-833`

```cpp
extern "C" {
DUCKDB_EXTENSION_API void sirius_init(duckdb::DatabaseInstance& db) {
    DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<SiriusExtension>();
}

DUCKDB_EXTENSION_API const char* sirius_version() {
    return duckdb::DuckDB::LibraryVersion();
}
}
```

---

## Table Functions

Table functions are the primary integration mechanism. They allow SQL queries to invoke custom execution logic.

### Table Function Interface

DuckDB table functions require four components:

1. **Bind Function**: Analyze query and prepare execution
2. **Init Function**: Initialize per-thread state
3. **Function**: Execute and produce results
4. **Cardinality**: Estimate result size (optional)

### Legacy Mode: gpu_processing()

**Declaration**: `src/sirius_extension.cpp:598-620`

```cpp
TableFunction GPUProcessingTableFunction() {
    TableFunction func("gpu_processing",
                      {LogicalType::VARCHAR},  // Input: SQL query string
                      GPUProcessingFunction,   // Execution function
                      GPUProcessingBind);      // Bind function
    func.name = "gpu_processing";
    return func;
}
```

**Usage**:
```sql
SELECT * FROM gpu_processing('
    SELECT customer_id, SUM(total)
    FROM orders
    GROUP BY customer_id
');
```

#### Bind Phase (GPUProcessingBind)

**File**: `src/sirius_extension.cpp:240-339`

The bind function:
1. Extracts the query string
2. Creates a new DuckDB connection
3. Parses and plans the query
4. Generates GPU physical plan
5. Returns function data with prepared statement

```cpp
unique_ptr<FunctionData> GPUProcessingBind(
    ClientContext& context,
    TableFunctionBindInput& input,
    vector<LogicalType>& return_types,
    vector<string>& names)
{
    auto result = make_uniq<GPUTableFunctionData>();
    result->query = input.inputs[0].ToString();

    // Parse query using DuckDB
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(result->query);

    // Create logical plan
    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));

    // Generate GPU physical plan
    unique_ptr<LogicalOperator> logical_plan = /* extract plan */;
    auto gpu_physical_plan = GPUGeneratePhysicalPlan(context, logical_plan);

    result->gpu_prepared = make_shared_ptr<GPUPreparedStatementData>(
        std::move(prepared), std::move(gpu_physical_plan));

    // Set return types and column names
    return_types = planner.types;
    names = planner.names;

    return result;
}
```

#### Execution Phase (GPUProcessingFunction)

**File**: `src/sirius_extension.cpp:560-596`

```cpp
void GPUProcessingFunction(ClientContext& context,
                          TableFunctionInput& data_p,
                          DataChunk& output)
{
    auto& data = (GPUTableFunctionData&)*data_p.bind_data;

    if (!data.res) {
        // First call: execute query on GPU
        data.res = GPUExecuteQuery(context, data.query, data.gpu_prepared);
    }

    // Fetch next chunk of results
    auto result_chunk = data.res->Fetch();
    if (result_chunk) {
        output.Reference(*result_chunk);
    } else {
        output.SetCardinality(0);  // Signal completion
    }
}
```

### New Mode: gpu_execution()

**Declaration**: `src/sirius_extension.cpp:740-762`

```cpp
TableFunction GPUExecutionTableFunction() {
    TableFunction func("gpu_execution",
                      {LogicalType::VARCHAR},
                      GPUExecutionFunction,
                      GPUExecutionBind);
    func.name = "gpu_execution";
    return func;
}
```

**Usage**:
```sql
SELECT * FROM gpu_execution('
    SELECT category, COUNT(*) as count
    FROM products
    WHERE price > 100
    GROUP BY category
');
```

#### Bind Phase (GPUExecutionBind)

**File**: `src/sirius_extension.cpp:353-409`

Similar to legacy mode, but uses new mode infrastructure:

```cpp
unique_ptr<FunctionData> GPUExecutionBind(
    ClientContext& context,
    TableFunctionBindInput& input,
    vector<LogicalType>& return_types,
    vector<string>& names)
{
    auto result = make_uniq<SiriusTableFunctionData>();
    result->query = input.inputs[0].ToString();
    result->sirius_iface = make_uniq<sirius::sirius_interface>(context);

    // Parse and plan query
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(result->query);
    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));

    // Generate Sirius physical plan
    unique_ptr<LogicalOperator> query_plan = result->ExtractPlan(context);
    auto sirius_physical_plan = SiriusGeneratePhysicalPlan(context, query_plan);

    result->gpu_prepared = make_shared_ptr<sirius::sirius_prepared_statement_data>(
        std::move(prepared), std::move(sirius_physical_plan));

    return_types = planner.types;
    names = planner.names;

    return result;
}
```

#### Execution Phase (GPUExecutionFunction)

**File**: `src/sirius_extension.cpp:411-452`

```cpp
void GPUExecutionFunction(ClientContext& context,
                         TableFunctionInput& data_p,
                         DataChunk& output)
{
    auto& data = (SiriusTableFunctionData&)*data_p.bind_data;

    if (!data.res) {
        // Execute using Sirius engine
        data.res = data.sirius_iface->sirius_execute_query(
            context, data.query, data.gpu_prepared, {});

        if (data.res->HasError()) {
            // Optional fallback to DuckDB
            if (Config::ENABLE_FALLBACK) {
                data.res = data.conn->Query(data.query);
            }
        }
    }

    auto result_chunk = data.res->Fetch();
    if (result_chunk) {
        output.Reference(*result_chunk);
    } else {
        output.SetCardinality(0);
    }
}
```

---

## Data Flow Between DuckDB and Sirius

### Input: Logical Plan Extraction

Sirius receives logical plans from DuckDB:

```
DuckDB Parser
     ↓
DuckDB Planner
     ↓
Logical Operator Tree ──→ Sirius Physical Planner
                               ↓
                         Physical Operator Tree
```

**Key Classes**:
- `duckdb::LogicalOperator` - DuckDB's logical operators
- `duckdb::Planner` - DuckDB's logical planner
- `sirius::sirius_physical_plan_generator` - Sirius physical planner

### Output: Query Results

Sirius returns results to DuckDB via `QueryResult`:

```
GPU Execution
     ↓
Result Collector (GPU → CPU transfer)
     ↓
Convert to DuckDB DataChunk
     ↓
QueryResult ──→ DuckDB
     ↓
User Application
```

**Data Chunk Format**:
- `duckdb::DataChunk` - Row-oriented batch
- `duckdb::Vector` - Column vector
- Supports all DuckDB types (with some limitations for GPU)

### Type System Integration

Sirius maps DuckDB types to GPU-compatible types:

| DuckDB Type | GPU Type | Notes |
|-------------|----------|-------|
| BOOLEAN | BOOL8 | Byte-sized boolean |
| TINYINT | INT8 | 8-bit integer |
| SMALLINT | INT16 | 16-bit integer |
| INTEGER | INT32 | 32-bit integer |
| BIGINT | INT64 | 64-bit integer |
| HUGEINT | INT64 | Downcasted (limited precision) |
| FLOAT | FLOAT32 | 32-bit float |
| DOUBLE | FLOAT64 | 64-bit float |
| DATE | DATE32 | Days since epoch |
| TIMESTAMP | TIMESTAMP64 | Microseconds since epoch |
| VARCHAR | STRING | Variable-length strings |
| DECIMAL | DECIMAL64/128 | Fixed-precision decimal |

**Type Conversion**: `src/data/sirius_converter_registry.cpp`

---

## DuckDB APIs Used by Sirius

### 1. Catalog API

Used to register table functions and access table metadata.

```cpp
#include "duckdb/catalog/catalog.hpp"

Catalog& catalog = Catalog::GetSystemCatalog(context);
catalog.CreateTableFunction(context, &function_info);
```

### 2. Parser and Planner API

Used to parse SQL and create logical plans.

```cpp
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"

Parser parser(context.GetParserOptions());
parser.ParseQuery(query_string);

Planner planner(context);
planner.CreatePlan(std::move(parser.statements[0]));
```

### 3. Execution API

Used for fallback execution and accessing table data.

```cpp
#include "duckdb/main/connection.hpp"

Connection conn(db);
auto result = conn.Query(query);
```

### 4. Type System API

Used for type resolution and conversion.

```cpp
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"

LogicalType type = LogicalType::INTEGER;
DataChunk chunk;
chunk.Initialize(types);
```

---

## Configuration Integration

Sirius adds custom configuration options to DuckDB:

### Configuration Options

**File**: `src/sirius_config.cpp:50-100`

```cpp
struct SiriusConfigOptions {
    // Path to configuration file
    static constexpr const char* CONFIG_PATH = "sirius_config_path";

    // Thread pool sizes
    static constexpr const char* PIPELINE_THREADS = "sirius_pipeline_threads";
    static constexpr const char* TASK_CREATOR_THREADS = "sirius_task_creator_threads";

    // Memory limits
    static constexpr const char* GPU_MEMORY_LIMIT = "sirius_gpu_memory_limit";
    static constexpr const char* HOST_MEMORY_LIMIT = "sirius_host_memory_limit";

    // Logging
    static constexpr const char* LOG_LEVEL = "sirius_log_level";

    // Behavior
    static constexpr const char* ENABLE_FALLBACK = "sirius_enable_fallback";
};
```

### Usage in SQL

```sql
-- Set configuration options
SET sirius_config_path = '/path/to/sirius.cfg';
SET sirius_log_level = 'DEBUG';
SET sirius_gpu_memory_limit = 8192;  -- MB

-- Query options
SELECT * FROM duckdb_settings() WHERE name LIKE 'sirius%';
```

---

## Error Handling and Fallback

Sirius provides fallback to DuckDB CPU execution on errors:

```cpp
try {
    // Attempt GPU execution
    result = sirius_execute_query(context, query, prepared);
} catch (std::exception& e) {
    if (Config::ENABLE_FALLBACK) {
        // Fallback to DuckDB
        SIRIUS_LOG_WARNING("GPU execution failed, falling back to CPU: {}", e.what());
        result = duckdb_connection->Query(query);
    } else {
        // Propagate error
        throw;
    }
}
```

**Fallback Scenarios**:
- Unsupported operators
- Out of GPU memory (and spilling failed)
- CUDA errors (device failure, etc.)
- Type conversion errors

---

## Connection and Context Management

### Per-Connection State

Each DuckDB connection gets a `SiriusContext`:

```cpp
class SiriusContext {
public:
    // Associated DuckDB context
    ClientContext& duckdb_context;

    // Sirius-specific state
    unique_ptr<sirius_engine> engine;
    unique_ptr<sirius_config> config;

    // Memory management
    shared_ptr<memory_reservation_manager> mem_mgr;

    // Active queries
    vector<shared_ptr<sirius_prepared_statement_data>> active_queries;
};
```

**File**: `src/sirius_context.cpp`

### Context Lifecycle

```
DuckDB Connection Created
     ↓
SiriusContext Created (lazy)
     ↓
Query Execution (multiple queries)
     ↓
DuckDB Connection Closed
     ↓
SiriusContext Destroyed
     ↓
GPU Resources Cleaned Up
```

---

## Transaction Handling

Sirius operates within DuckDB's transaction model:

- **Read-only Transactions**: Sirius queries don't modify data
- **Snapshot Isolation**: Sees consistent snapshot of data
- **No Write Support**: Sirius doesn't support INSERT/UPDATE/DELETE (yet)

```cpp
// DuckDB handles transaction boundaries
con.BeginTransaction();
auto result = con.Query("SELECT * FROM gpu_execution('...')");
con.Commit();
```

---

## Interoperability Patterns

### Pattern 1: GPU Acceleration for Subqueries

```sql
-- Use GPU for heavy lifting, DuckDB for final processing
SELECT customer_id, total, rank
FROM (
    SELECT * FROM gpu_execution('
        SELECT customer_id, SUM(price) as total
        FROM orders
        GROUP BY customer_id
    ')
) t
QUALIFY ROW_NUMBER() OVER (ORDER BY total DESC) <= 10;
```

### Pattern 2: Hybrid Execution

```sql
-- GPU for large table, DuckDB for small table join
SELECT *
FROM gpu_execution('SELECT * FROM large_orders WHERE date > ''2024-01-01''') o
JOIN small_customer_metadata c ON o.customer_id = c.id;
```

### Pattern 3: Result Materialization

```sql
-- Materialize GPU results for further CPU processing
CREATE TABLE gpu_results AS
SELECT * FROM gpu_execution('
    SELECT category, product, SUM(sales)
    FROM large_sales
    GROUP BY category, product
');

-- CPU-based post-processing
SELECT * FROM gpu_results WHERE category = 'Electronics';
```

---

## Performance Considerations

### Extension Overhead

- **Function Call Overhead**: ~1-5μs per table function invocation
- **Type Conversion**: ~10-50μs depending on schema complexity
- **Plan Extraction**: ~100-500μs for complex queries

### Data Transfer

- **CPU → GPU**: ~10-50 ms per GB (depends on PCIe generation)
- **GPU → CPU**: Similar to CPU → GPU
- **Minimize transfers**: Keep intermediate results on GPU

### Optimization Disabling

Sirius disables certain DuckDB optimizations that aren't GPU-compatible:

```cpp
// Disabled optimizers
disabled_optimizers.insert(OptimizerType::IN_CLAUSE);           // Mark joins
disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
```

This ensures the logical plan is GPU-friendly but may reduce CPU fallback performance.

---

## Debugging Integration

### Logging

```cpp
// Sirius logging macros
SIRIUS_LOG_DEBUG("Query plan: {}", query_plan->ToString());
SIRIUS_LOG_INFO("Executing query on GPU");
SIRIUS_LOG_WARNING("Memory pressure detected");
SIRIUS_LOG_ERROR("GPU execution failed: {}", error.what());
```

### DuckDB Profiling

```sql
-- Enable DuckDB profiling
PRAGMA enable_profiling;
PRAGMA profiling_mode='detailed';

-- Run query
SELECT * FROM gpu_execution('...');

-- View profile
PRAGMA profile_output;
```

---

## Next Steps

- **Execution Modes**: [Execution Modes Comparison](execution-modes.md)
- **Legacy Mode Details**: [Legacy Mode Overview](../03-legacy-mode/overview.md)
- **New Mode Details**: [New Mode Overview](../04-new-mode/overview.md)
- **Development**: [Building and Testing](../07-development/building-and-testing.md)
