# DuckDB — Module Reference

**Version**: v1.4.3 (commit d1dc88f950)
**Location**: `./duckdb/` (git submodule)
**Namespace**: `duckdb`
**Headers**: `duckdb/src/include/duckdb/`

## Module Map

| Module | Status | Description | Key APIs Used |
|--------|--------|-------------|---------------|
| main | USED | Database instance, client context, connections, config, query results | `ClientContext`, `Connection`, `DBConfig`, `QueryResult`, `PreparedStatementData` |
| common | USED | Core types, data chunks, vectors, validity masks, enums, exceptions | `DataChunk`, `Vector`, `LogicalType`, `Value`, `ValidityMask`, `FlatVector` |
| planner | USED | Bound expressions, logical operators, table filters, plan optimization | `BoundReferenceExpression`, `BoundConstantExpression`, `LogicalOperator`, `TableFilterSet` |
| execution | USED | Physical operators, execution context, column binding resolver | `PhysicalOperator`, `ExecutionContext`, `ColumnBindingResolver` |
| function | USED | Table functions, scalar functions, function binding | `TableFunction`, `FunctionBinder`, `ScalarFunction` |
| parser | USED | SQL parser, parsed data, expression types | `Parser`, `CreateTableFunctionInfo`, `ConstantExpression` |
| optimizer | USED | Query optimizer, optimizer types | `Optimizer`, `OptimizerType` |
| parallel | USED | Thread context, task scheduling, meta pipelines | `ThreadContext`, `TaskScheduler`, `TaskExecutor` |
| catalog | USED | Catalog access, table/function catalog entries | `Catalog`, `TableCatalogEntry`, `TableFunctionCatalogEntry` |
| storage | USED (minimal) | Storage engine, buffer management, compression | `DataTable` |
| transaction | UNUSED | Transaction management | — |
| logging | UNUSED | DuckDB internal logging | — |
| verification | UNUSED | Query verification / testing infrastructure | — |

## Our Usage Summary

We use **10 of 13** modules. Primary integration points:

- **Extension registration**: Sirius registers as a DuckDB extension via `TableFunction` + `CreateTableFunctionInfo` in `sirius_extension.cpp`
- **Query interception**: Parse SQL via `Parser` → plan via `Planner` → optimize via `Optimizer` → resolve bindings via `ColumnBindingResolver` → translate physical plan to GPU operators
- **Data exchange**: Convert DuckDB `DataChunk`/`Vector` to cuDF GPU tables and back via `FlatVector::GetData()`, `ValidityMask`, etc.
- **Expression translation**: Map DuckDB bound expressions (`BoundReferenceExpression`, `BoundComparisonExpression`, etc.) to GPU expression evaluators
- **Physical plan generation**: Walk DuckDB's `PhysicalOperator` tree to create corresponding GPU operator pipelines
- **Table scanning**: Use DuckDB's parquet/table scan infrastructure via `TableFunction` bind/init/execute callbacks

## Files That Reference DuckDB (Key Files)

| Source File | Modules Used | Key APIs |
|-------------|-------------|----------|
| `src/sirius_extension.cpp` | main, parser, optimizer, planner, execution, function, catalog | `ClientContext`, `Parser`, `Planner`, `Optimizer`, `TableFunction`, `Connection` |
| `src/sirius_interface.cpp` | main, common, function | `ClientContext`, `TableFunction`, `DataChunk` |
| `src/gpu_physical_plan_generator.cpp` | execution, planner, common | `PhysicalOperator`, `LogicalOperator`, `LogicalType` |
| `src/planner/sirius_physical_plan_generator.cpp` | execution, planner, common | `PhysicalOperator`, `LogicalOperator`, `PhysicalOperatorType` |
| `src/expression_executor/gpu_expression_translator.cpp` | planner, common | `BoundReferenceExpression`, `BoundComparisonExpression`, `BoundFunctionExpression` |
| `src/op/scan/duckdb_scan_task.cpp` | execution, parallel, function, common | `ExecutionContext`, `ThreadContext`, `TableFunction`, `DataChunk` |
| `src/op/result/host_table_chunk_reader.cpp` | common | `DataChunk`, `Vector`, `FlatVector`, `ValidityMask` |
| `src/gpu_columns.cpp` | common | `LogicalType`, `Value`, `DecimalType` |
| `src/plan/gpu_plan_aggregate.cpp` | execution, function, main, planner | `PhysicalHashAggregate`, `FunctionBinder`, `BoundAggregateExpression` |
| `src/operator/gpu_physical_table_scan.cpp` | execution, parallel, planner, common | `ExecutionContext`, `ThreadContext`, `TaskScheduler`, `TableFilterSet` |
