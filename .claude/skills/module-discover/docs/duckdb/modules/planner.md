# planner

**Status**: USED
**Path**: `duckdb/src/include/duckdb/planner/`
**Headers we include**:
- `duckdb/planner/planner.hpp`
- `duckdb/planner/expression.hpp`
- `duckdb/planner/expression/bound_aggregate_expression.hpp`
- `duckdb/planner/expression/bound_between_expression.hpp`
- `duckdb/planner/expression/bound_case_expression.hpp`
- `duckdb/planner/expression/bound_cast_expression.hpp`
- `duckdb/planner/expression/bound_comparison_expression.hpp`
- `duckdb/planner/expression/bound_conjunction_expression.hpp`
- `duckdb/planner/expression/bound_constant_expression.hpp`
- `duckdb/planner/expression/bound_function_expression.hpp`
- `duckdb/planner/expression/bound_operator_expression.hpp`
- `duckdb/planner/expression/bound_parameter_expression.hpp`
- `duckdb/planner/expression/bound_reference_expression.hpp`
- `duckdb/planner/bound_result_modifier.hpp`
- `duckdb/planner/filter/conjunction_filter.hpp`
- `duckdb/planner/filter/constant_filter.hpp`
- `duckdb/planner/filter/dynamic_filter.hpp`
- `duckdb/planner/logical_operator.hpp`
- `duckdb/planner/operator/logical_aggregate.hpp`
- `duckdb/planner/operator/logical_comparison_join.hpp`
- `duckdb/planner/operator/logical_cteref.hpp`
- `duckdb/planner/operator/logical_get.hpp`
- `duckdb/planner/operator/logical_order.hpp`
- `duckdb/planner/table_filter.hpp`

## Summary

The `planner` module is the most heavily used DuckDB module in Sirius. It provides the bound expression hierarchy (the typed/resolved form of SQL expressions), logical operators (the logical plan tree), and table filters. Sirius walks these structures to translate DuckDB's query plan into GPU-executable operators and expressions.

## API Reference

### Planner

**Header**: `duckdb/planner/planner.hpp`
**Signature**:
```cpp
class Planner {
public:
    explicit Planner(ClientContext &context);
    void CreatePlan(unique_ptr<SQLStatement> statement);
    unique_ptr<LogicalOperator> plan;
    vector<LogicalType> types;
    vector<string> names;
};
```

**Description**: Converts parsed SQL statements into a logical plan tree.

**Our usage**:
- `src/sirius_extension.cpp` — Creates logical plan from parsed SQL for GPU translation

### LogicalOperator

**Header**: `duckdb/planner/logical_operator.hpp`
**Signature**:
```cpp
class LogicalOperator {
public:
    LogicalOperatorType type;
    vector<unique_ptr<LogicalOperator>> children;
    vector<unique_ptr<Expression>> expressions;
    vector<LogicalType> types;
    virtual vector<ColumnBinding> GetColumnBindings();
    idx_t EstimateCardinality(ClientContext &context);
};
```

**Description**: Base class for all logical operators in the query plan tree. Sirius walks this tree to decide which operators can run on GPU.

**Our usage**:
- `src/gpu_physical_plan_generator.cpp` — Walk logical plan to create GPU physical operators
- `src/planner/sirius_physical_plan_generator.cpp` — Same for new code path
- `src/fallback.cpp` — Inspect logical operators for fallback decisions

### BoundReferenceExpression

**Header**: `duckdb/planner/expression/bound_reference_expression.hpp`
**Signature**:
```cpp
class BoundReferenceExpression : public Expression {
public:
    BoundReferenceExpression(LogicalType type, idx_t index);
    idx_t index;      // Column index in the input
    idx_t depth = 0;  // Subquery depth
};
```

**Description**: References a column by index in the input. Most common expression type — every column access in a query creates one.

**Our usage**:
- Used in virtually every operator translation file to map column references to GPU column indices
- `src/expression_executor/gpu_expression_translator.cpp` — Map to GPU column references
- `src/operator/gpu_physical_hash_join.cpp` — Join key column indices
- `src/plan/gpu_plan_get.cpp` — Table scan column indices

### BoundComparisonExpression

**Header**: `duckdb/planner/expression/bound_comparison_expression.hpp`
**Signature**:
```cpp
class BoundComparisonExpression : public Expression {
public:
    BoundComparisonExpression(ExpressionType type, unique_ptr<Expression> left, unique_ptr<Expression> right);
    unique_ptr<Expression> left;
    unique_ptr<Expression> right;
};
```

**Description**: Binary comparison (=, !=, <, >, <=, >=, IS DISTINCT FROM, etc.).

**Our usage**:
- `src/expression_executor/gpu_expression_translator.cpp` — Translate to GPU comparison operations
- `test/cpp/operator/test_physical_filter.cpp` — Build test filter expressions

### BoundConjunctionExpression

**Header**: `duckdb/planner/expression/bound_conjunction_expression.hpp`
**Signature**:
```cpp
class BoundConjunctionExpression : public Expression {
public:
    BoundConjunctionExpression(ExpressionType type);  // AND or OR
    vector<unique_ptr<Expression>> children;
};
```

**Description**: AND/OR combination of boolean expressions.

**Our usage**:
- `src/operator/gpu_physical_filter.cpp` — Decompose conjunction into individual filter conditions
- `src/expression_executor/gpu_expression_translator.cpp` — Translate AND/OR to GPU

### BoundConstantExpression

**Header**: `duckdb/planner/expression/bound_constant_expression.hpp`
**Signature**:
```cpp
class BoundConstantExpression : public Expression {
public:
    BoundConstantExpression(Value value);
    Value value;
};
```

**Description**: A constant literal value in an expression.

**Our usage**:
- `src/expression_executor/gpu_expression_translator.cpp` — Extract constant values for GPU scalar operations

### BoundCastExpression

**Header**: `duckdb/planner/expression/bound_cast_expression.hpp`
**Signature**:
```cpp
class BoundCastExpression : public Expression {
public:
    unique_ptr<Expression> child;
    LogicalType return_type;
    bool try_cast;  // TRY_CAST vs CAST
};
```

**Our usage**:
- `src/expression_executor/gpu_expression_translator.cpp` — Translate type casts to cuDF cast operations

### BoundFunctionExpression

**Header**: `duckdb/planner/expression/bound_function_expression.hpp`
**Signature**:
```cpp
class BoundFunctionExpression : public Expression {
public:
    ScalarFunction function;
    vector<unique_ptr<Expression>> children;
    unique_ptr<FunctionData> bind_info;
    bool is_operator;
};
```

**Description**: A bound scalar function call (e.g., `SUBSTRING`, `YEAR`, arithmetic operators).

**Our usage**:
- `src/expression_executor/gpu_expression_translator.cpp` — Map DuckDB scalar functions to GPU implementations
- `src/expression_executor/specializations/` — Specialized GPU implementations for specific functions

### BoundAggregateExpression

**Header**: `duckdb/planner/expression/bound_aggregate_expression.hpp`
**Signature**:
```cpp
class BoundAggregateExpression : public Expression {
public:
    AggregateFunction function;
    vector<unique_ptr<Expression>> children;
    unique_ptr<Expression> filter;
    unique_ptr<BoundOrderModifier> order_bys;
    AggregateType aggr_type;  // DISTINCT or NON_DISTINCT
    bool IsDistinct() const;
};
```

**Our usage**:
- `src/plan/gpu_plan_aggregate.cpp` — Extract aggregate functions for GPU grouped/ungrouped aggregation
- `src/operator/gpu_physical_grouped_aggregate.cpp` — Map aggregate functions to cuDF aggregation ops

### BoundCaseExpression / BoundBetweenExpression / BoundOperatorExpression

**Headers**: Respective `bound_*_expression.hpp`

**Our usage**:
- `src/expression_executor/gpu_expression_translator.cpp` — Translate CASE/WHEN, BETWEEN, and operators (COALESCE, NOT, IS NULL, IS NOT NULL) to GPU

### TableFilterSet / ConstantFilter / ConjunctionFilter / DynamicFilter

**Headers**: `duckdb/planner/table_filter.hpp`, `duckdb/planner/filter/constant_filter.hpp`, etc.
**Signature**:
```cpp
class TableFilterSet {
public:
    unordered_map<idx_t, unique_ptr<TableFilter>> filters;  // column_index -> filter
};

class ConstantFilter : public TableFilter {
public:
    ExpressionType comparison_type;
    Value constant;
};
```

**Description**: Pushed-down filters for table scans. Sirius translates these to GPU filter operations that run during data loading.

**Our usage**:
- `src/operator/gpu_physical_table_scan.cpp` — Apply pushed-down filters during scan
- `test/cpp/operator/test_physical_table_scan.cpp` — Test filter pushdown

### Logical Operators (LogicalGet, LogicalAggregate, LogicalOrder, LogicalComparisonJoin, LogicalCTERef)

**Headers**: `duckdb/planner/operator/logical_*.hpp`

**Description**: Specific logical operator types in the plan tree. Sirius accesses their properties (e.g., `LogicalGet::table_filters`, `LogicalAggregate::groups`, `LogicalComparisonJoin::conditions`).

**Our usage**:
- `src/plan/gpu_plan_get.cpp` — Access `LogicalGet` for table scan configuration
- `src/plan/gpu_plan_aggregate.cpp` — Access `LogicalAggregate` for grouping/aggregate info
- `src/plan/gpu_plan_join.cpp` — Access `LogicalComparisonJoin` for join conditions
- `src/plan/gpu_plan_recursive_cte.cpp` — Access `LogicalCTERef` for CTE handling

### BoundResultModifier (BoundOrderModifier)

**Header**: `duckdb/planner/bound_result_modifier.hpp`

**Our usage**:
- `test/cpp/operator/test_physical_top_n.cpp` — Build ORDER BY specifications for tests
- `src/plan/gpu_plan_order.cpp` — Access sort specifications

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `BoundSubqueryExpression` | `expression/bound_subquery_expression.hpp` | Correlated/uncorrelated subqueries |
| `BoundWindowExpression` | `expression/bound_window_expression.hpp` | Window function expressions |
| `ExpressionBinder` | `expression_binder/` | Expression binding infrastructure |
| `LogicalWindow` | `operator/logical_window.hpp` | Window function operator |
| `LogicalSample` | `operator/logical_sample.hpp` | TABLESAMPLE operator |
