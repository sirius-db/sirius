# function

**Status**: USED
**Path**: `duckdb/src/include/duckdb/function/`
**Headers we include**:
- `duckdb/function/table_function.hpp`
- `duckdb/function/function_binder.hpp`
- `duckdb/function/scalar_function.hpp`
- `duckdb/function/table/table_scan.hpp`

## Summary

The `function` module defines DuckDB's function system — table functions, scalar functions, and aggregate functions. Sirius uses table functions as its primary extension interface (`gpu_execution`) and reads scalar/aggregate function metadata to map them to GPU equivalents.

## API Reference

### TableFunction

**Header**: `duckdb/function/table_function.hpp`
**Signature**:
```cpp
class TableFunction : public SimpleNamedParameterFunction {
public:
    TableFunction(string name, vector<LogicalType> arguments,
                  table_function_t function, table_function_bind_t bind,
                  table_function_init_global_t init_global = nullptr,
                  table_function_init_local_t init_local = nullptr);

    table_function_bind_t bind;
    table_function_init_global_t init_global;
    table_function_init_local_t init_local;
    table_function_t function;           // Main execute callback
    table_function_to_string_t to_string;
    table_function_partition_t get_partition_info;

    // Pushdown support
    bool projection_pushdown;
    bool filter_pushdown;
    bool filter_prune;
};
```

**Description**: Defines a table-valued function. This is the primary mechanism for Sirius to hook into DuckDB — `gpu_execution`, `start_profiling`, and `stop_profiling` are all registered as table functions.

**Our usage**:
- `src/sirius_extension.cpp` — Register all Sirius table functions
- `src/op/scan/duckdb_scan_task.cpp` — Call DuckDB's table scan functions (parquet reader)
- `test/cpp/operator/test_physical_table_scan.cpp` — Test table scan function interface

### TableFunctionBindInput / TableFunctionInitInput / TableFunctionInput

**Header**: `duckdb/function/table_function.hpp`
**Signature**:
```cpp
struct TableFunctionBindInput {
    vector<Value> &inputs;
    named_parameter_map_t &named_parameters;
    vector<LogicalType> &input_table_types;
    vector<string> &input_table_names;
    ClientContext &context;
};

struct TableFunctionInitInput {
    const TableFunction &function;
    const TableFunctionData &bind_data;
    const vector<column_t> &column_ids;
    optional_ptr<TableFilterSet> filters;
};

struct TableFunctionInput {
    const TableFunctionData &bind_data;
    LocalTableFunctionState &local_state;
    GlobalTableFunctionState &global_state;
};
```

**Our usage**:
- `src/sirius_extension.cpp` — Used in bind/init/execute callbacks for Sirius table functions
- `src/op/scan/duckdb_scan_task.cpp` — Construct these to call DuckDB's scan functions

### GlobalTableFunctionState / LocalTableFunctionState / TableFunctionData

**Header**: `duckdb/function/table_function.hpp`
**Signature**:
```cpp
struct GlobalTableFunctionState {
    virtual ~GlobalTableFunctionState();
    virtual idx_t MaxThreads() const;
    template <class TARGET> TARGET &Cast();
};

struct LocalTableFunctionState {
    virtual ~LocalTableFunctionState();
    template <class TARGET> TARGET &Cast();
};

struct TableFunctionData {
    virtual ~TableFunctionData();
    template <class TARGET> TARGET &Cast();
};
```

**Description**: State objects for table function lifecycle. `TableFunctionData` is created in bind, global/local states in init.

**Our usage**:
- `src/sirius_extension.cpp` — Sirius defines its own subclasses (`GpuProcessingData`, `GpuExecutionData`, etc.)
- `src/op/scan/duckdb_scan_task.cpp` — Access DuckDB's scan states

### FunctionBinder

**Header**: `duckdb/function/function_binder.hpp`
**Signature**:
```cpp
class FunctionBinder {
public:
    explicit FunctionBinder(ClientContext &context);
    void BindSortedAggregate(ClientContext &context, BoundAggregateExpression &expr,
                             const vector<unique_ptr<Expression>> &groups);
};
```

**Our usage**:
- `src/plan/gpu_plan_aggregate.cpp` — Bind sorted aggregates when translating aggregate plans

### ScalarFunction

**Header**: `duckdb/function/scalar_function.hpp`

**Description**: Defines a scalar (row-level) function. Sirius reads function names and types to map to GPU implementations.

**Our usage**:
- `src/expression_executor/gpu_expression_translator.cpp` — Read function name/type from `BoundFunctionExpression::function`
- `test/cpp/operator/aggregate/aggregate_test_utils.hpp` — Create test aggregate functions

### TableScanFunction

**Header**: `duckdb/function/table/table_scan.hpp`

**Description**: DuckDB's built-in table scan function.

**Our usage**:
- `test/cpp/scan/test_scan_executor.cpp` — Access table scan function for testing

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `AggregateFunction` | `aggregate_function.hpp` | Aggregate function definition (accessed indirectly via BoundAggregateExpression) |
| `PragmaFunction` | `pragma/pragma_function.hpp` | PRAGMA command functions |
| `CastFunction` | `cast/cast_function_set.hpp` | Type cast definitions |
| `WindowFunction` | `window/` | Window function infrastructure |
| `CompressionFunction` | `compression/` | Storage compression functions |
