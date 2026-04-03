# execution

**Status**: USED
**Path**: `duckdb/src/include/duckdb/execution/`
**Headers we include**:
- `duckdb/execution/column_binding_resolver.hpp`
- `duckdb/execution/execution_context.hpp`
- `duckdb/execution/physical_operator.hpp`
- `duckdb/execution/physical_plan_generator.hpp`
- `duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp`
- `duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp`

## Summary

The `execution` module provides the physical operator hierarchy and execution infrastructure. Sirius reads DuckDB's physical plan tree (`PhysicalOperator` subclasses) to understand operator configuration, then creates corresponding GPU operators. It also uses `ColumnBindingResolver` to resolve column references before GPU translation.

## API Reference

### PhysicalOperator

**Header**: `duckdb/execution/physical_operator.hpp`
**Signature**:
```cpp
class PhysicalOperator {
public:
    PhysicalOperator(PhysicalPlan &physical_plan, PhysicalOperatorType type,
                     vector<LogicalType> types, idx_t estimated_cardinality);
    PhysicalOperatorType type;
    vector<LogicalType> types;
    idx_t estimated_cardinality;
    ArenaLinkedList<reference<PhysicalOperator>> children;

    virtual string GetName() const;
    const vector<LogicalType> &GetTypes() const;
    vector<const_reference<PhysicalOperator>> GetChildren() const;
};
```

**Description**: Base class for all physical operators. Sirius walks the physical plan tree to map operators to GPU equivalents.

**Our usage**:
- `src/gpu_physical_plan_generator.cpp` — Read operator type, children, types to create GPU operators
- `src/planner/sirius_physical_plan_generator.cpp` — Same for new code path
- `test/cpp/pipeline/test_modified_pipeline.cpp` — Build test physical plans

### ExecutionContext

**Header**: `duckdb/execution/execution_context.hpp`
**Signature**:
```cpp
struct ExecutionContext {
    ClientContext &client;
    ThreadContext &thread;
    optional_ptr<Pipeline> pipeline;
};
```

**Description**: Bundles the client context, thread context, and current pipeline for operator execution.

**Our usage**:
- `src/operator/gpu_physical_table_scan.cpp` — Passed to DuckDB scan functions
- `src/op/scan/duckdb_scan_task.cpp` — Created for DuckDB-side table scans
- `test/cpp/config/test_context.cpp` — Test execution context creation

### ColumnBindingResolver

**Header**: `duckdb/execution/column_binding_resolver.hpp`
**Signature**:
```cpp
class ColumnBindingResolver : public LogicalOperatorVisitor {
public:
    ColumnBindingResolver();
    void VisitOperator(LogicalOperator &op) override;
};
```

**Description**: Resolves `ColumnBinding` references in logical plans to flat column indices (`BoundReferenceExpression`). Must be run before physical plan generation.

**Our usage**:
- `src/sirius_extension.cpp` — Run on logical plan before GPU physical plan generation
- `test/cpp/pipeline/test_modified_pipeline.cpp` — Resolve bindings in test plans
- `test/cpp/integration/test_tpcds_plan_translation.cpp` — Resolve bindings for TPC-DS plans

### PhysicalPlanGenerator

**Header**: `duckdb/execution/physical_plan_generator.hpp`
**Signature**:
```cpp
class PhysicalPlanGenerator {
public:
    explicit PhysicalPlanGenerator(ClientContext &context);
    unique_ptr<PhysicalOperator> CreatePlan(unique_ptr<LogicalOperator> logical);
};
```

**Description**: Converts a logical plan to a physical plan. Sirius uses a modified version to generate GPU physical operators.

**Our usage**:
- `src/plan/gpu_plan_aggregate.cpp` — Reference DuckDB's aggregate physical plan generation
- `src/plan/gpu_plan_recursive_cte.cpp` — Reference DuckDB's CTE handling

### PhysicalHashAggregate / PhysicalPerfectHashAggregate

**Headers**: `duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp`, `physical_perfecthash_aggregate.hpp`

**Description**: DuckDB's concrete aggregate operators. Sirius reads their configuration (groups, aggregates, filter) but replaces execution with GPU.

**Our usage**:
- `src/plan/gpu_plan_aggregate.cpp` — Read aggregate configuration from DuckDB's physical plan

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `Executor` | `executor.hpp` | DuckDB's main executor (Sirius replaces this) |
| `Pipeline` / `MetaPipeline` | `pipeline.hpp` | DuckDB's pipeline execution (Sirius has its own) |
| `PhysicalHashJoin` | `operator/join/physical_hash_join.hpp` | DuckDB's hash join (replaced by GPU) |
| `PhysicalOrder` | `operator/order/physical_order.hpp` | DuckDB's sort (replaced by GPU) |
| `Expression Executor` | `expression_executor.hpp` | DuckDB's expression evaluator (replaced by GPU) |
