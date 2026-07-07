# parser

**Status**: USED
**Path**: `duckdb/src/include/duckdb/parser/`
**Headers we include**:
- `duckdb/parser/parser.hpp`
- `duckdb/parser/parsed_data/create_table_function_info.hpp`
- `duckdb/parser/expression/constant_expression.hpp`
- `duckdb/parser/expression/function_expression.hpp`
- `duckdb/parser/tableref/table_function_ref.hpp`

## Summary

The `parser` module provides SQL parsing and unbound AST types. Sirius uses it to parse user SQL queries and to register table functions in the catalog.

## API Reference

### Parser

**Header**: `duckdb/parser/parser.hpp`
**Signature**:
```cpp
class Parser {
public:
    explicit Parser(ParserOptions options = ParserOptions());
    void ParseQuery(const string &query);
    vector<unique_ptr<SQLStatement>> statements;
};
```

**Description**: SQL parser that produces a list of `SQLStatement` objects from a query string.

**Our usage**:
- `src/sirius_extension.cpp` — Parse the SQL string passed to `gpu_execution()`
- `test/cpp/pipeline/test_modified_pipeline.cpp` — Parse test queries
- `test/cpp/integration/test_tpcds_plan_translation.cpp` — Parse TPC-DS queries

### CreateTableFunctionInfo

**Header**: `duckdb/parser/parsed_data/create_table_function_info.hpp`
**Signature**:
```cpp
struct CreateTableFunctionInfo : public CreateFunctionInfo {
    explicit CreateTableFunctionInfo(TableFunction function);
    explicit CreateTableFunctionInfo(TableFunctionSet set);
    vector<TableFunction> functions;
};
```

**Description**: Information needed to register a table function in the catalog.

**Our usage**:
- `src/sirius_extension.cpp` — Create registration info for `gpu_execution`, profiling functions
- `test/cpp/integration/test_tpcds_plan_translation.cpp` — Register test functions

### ConstantExpression / FunctionExpression / TableFunctionRef

**Headers**: `duckdb/parser/expression/*.hpp`, `duckdb/parser/tableref/table_function_ref.hpp`

**Description**: Unbound (pre-planning) expression and table reference types. Used to programmatically construct queries.

**Our usage**:
- `test/cpp/scan/test_parquet_scan_task.cpp` — Construct parquet scan function calls programmatically

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `SelectStatement` | `statement/select_statement.hpp` | SELECT query AST |
| `CreateStatement` | `statement/create_statement.hpp` | DDL CREATE AST |
| `Transformer` | `transformer.hpp` | PG parse tree → DuckDB AST |
| `QueryNode` | `query_node/` | Query structure (SELECT, UNION, etc.) |
| `Constraints` | `constraints/` | Table constraint definitions |
