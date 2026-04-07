# main

**Status**: USED
**Path**: `duckdb/src/include/duckdb/main/`
**Headers we include**:
- `duckdb/main/client_context.hpp`
- `duckdb/main/config.hpp`
- `duckdb/main/connection.hpp`
- `duckdb/main/database.hpp`
- `duckdb/main/prepared_statement_data.hpp`
- `duckdb/main/query_result.hpp`
- `duckdb/main/relation.hpp`
- `duckdb/main/settings.hpp`

## Summary

The `main` module provides the core database infrastructure: database instances, client contexts (sessions), connections, configuration, and query result handling. Sirius uses it extensively for extension registration, query execution orchestration, and accessing database/client configuration.

## API Reference

### ClientContext

**Header**: `duckdb/main/client_context.hpp`
**Signature**:
```cpp
class ClientContext : public enable_shared_from_this<ClientContext> {
public:
    explicit ClientContext(shared_ptr<DatabaseInstance> db);
    shared_ptr<DatabaseInstance> db;
    atomic<bool> interrupted;
    ClientConfig config;
    // Query execution
    unique_ptr<PendingQueryResult> PendingQuery(const string &query, bool allow_stream_result);
    unique_ptr<QueryResult> Query(unique_ptr<SQLStatement> statement, bool allow_stream_result);
    shared_ptr<PreparedStatementData> CreatePreparedStatement(ClientContextLock &lock, const string &query, ...);
    // Catalog access
    Catalog &GetCatalog();
    DatabaseInstance &GetDatabase();
};
```

**Description**: Represents a client session. Holds per-session state (config, transaction context, query profiler). Primary entry point for executing queries and accessing the catalog.

**Our usage**:
- `src/sirius_extension.cpp` — Accessed in table function callbacks to get database config, run queries, access catalog
- `src/op/scan/duckdb_scan_task.cpp` — Used to create execution contexts for DuckDB table scans
- `src/plan/gpu_plan_aggregate.cpp` — Accessed for client config settings (e.g., optimizer flags)

### Connection

**Header**: `duckdb/main/connection.hpp`
**Signature**:
```cpp
class Connection {
public:
    explicit Connection(DuckDB &database);
    explicit Connection(DatabaseInstance &database);
    shared_ptr<ClientContext> context;
    unique_ptr<QueryResult> Query(const string &query);
    unique_ptr<QueryResult> Query(unique_ptr<SQLStatement> statement);
    shared_ptr<Relation> Table(const string &table_name);
};
```

**Description**: Represents a connection to a DuckDB database. Wraps a ClientContext and provides a simplified query API.

**Our usage**:
- `src/sirius_extension.cpp` — Creates internal connections for query execution
- `test/cpp/pipeline/test_modified_pipeline.cpp` — Used in tests for setting up test database connections

### DBConfig

**Header**: `duckdb/main/config.hpp`
**Signature**:
```cpp
class DBConfig {
public:
    static DBConfig &GetConfig(ClientContext &context);
    static DBConfig &GetConfig(DatabaseInstance &db);
    template <class T> static T GetSetting(ClientContext &context);
    // Extension callbacks
    vector<ExtensionCallback> extension_callbacks;
};
```

**Description**: Database-level configuration. Sirius reads settings to check optimizer flags and register extension callbacks.

**Our usage**:
- `src/sirius_extension.cpp` — `DBConfig::GetConfig()` to register extension callbacks and read settings
- `src/plan/gpu_plan_aggregate.cpp` — `DBConfig::GetSetting<>()` to check optimizer-related settings

### PreparedStatementData

**Header**: `duckdb/main/prepared_statement_data.hpp`
**Signature**:
```cpp
class PreparedStatementData {
public:
    unique_ptr<LogicalOperator> plan;
    vector<LogicalType> types;
    vector<string> names;
    StatementProperties properties;
};
```

**Description**: Holds the logical plan and metadata for a prepared statement. Sirius accesses this to get the logical plan for GPU translation.

**Our usage**:
- `src/sirius_extension.cpp` — Extracts logical plan from prepared statements for GPU physical plan generation
- `src/operator/gpu_physical_result_collector.cpp` — Accesses statement types/names for result collection

### QueryResult / MaterializedQueryResult

**Header**: `duckdb/main/query_result.hpp`
**Signature**:
```cpp
class QueryResult {
public:
    vector<LogicalType> types;
    vector<string> names;
    virtual unique_ptr<DataChunk> Fetch();
    bool HasError() const;
    ErrorData &GetErrorObject();
    string ToBox(ClientContext &context, const BoxRendererConfig &config);
};

class MaterializedQueryResult : public QueryResult {
public:
    ColumnDataCollection &Collection();
};
```

**Description**: Represents query execution results. Sirius uses it to display results and check for errors.

**Our usage**:
- `src/sirius_extension.cpp` — Result display via `ToBox()`, error checking via `HasError()`

### Relation

**Header**: `duckdb/main/relation.hpp`

**Description**: Represents a relational query (table, projection, filter, etc.) that can be lazily executed.

**Our usage**:
- `src/sirius_extension.cpp` — Used for query result handling

### Settings (PreserveInsertionOrderSetting, etc.)

**Header**: `duckdb/main/settings.hpp`
**Signature**:
```cpp
struct PreserveInsertionOrderSetting { static constexpr const char *Name = "preserve_insertion_order"; };
struct PreferRangeJoinsSetting { static constexpr const char *Name = "prefer_range_joins"; };
struct NestedLoopJoinThresholdSetting { static constexpr const char *Name = "nested_loop_join_threshold"; };
struct MergeJoinThresholdSetting { static constexpr const char *Name = "merge_join_threshold"; };
```

**Description**: Type-safe settings accessors used with `DBConfig::GetSetting<T>()`.

**Our usage**:
- `src/plan/gpu_plan_join.cpp` — Checks join threshold settings to match DuckDB's join selection behavior
- `src/plan/gpu_plan_aggregate.cpp` — Checks `PreserveInsertionOrderSetting`

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `Appender` | `appender.hpp` | Bulk data insertion |
| `QueryProfiler` | `query_profiler.hpp` | Query profiling and timing |
| `ExtensionHelper` | `extension_helper.hpp` | Extension loading utilities |
| `DatabaseManager` | `database_manager.hpp` | Multi-database management |
| `PendingQueryResult` | `pending_query_result.hpp` | Async query execution (indirectly used via ClientContext) |
