# optimizer

**Status**: USED
**Path**: `duckdb/src/include/duckdb/optimizer/`
**Headers we include**:
- `duckdb/optimizer/optimizer.hpp`

## Summary

The `optimizer` module provides query optimization passes. Sirius runs DuckDB's optimizer on the logical plan before translating to GPU, selectively disabling certain optimization passes that interfere with GPU execution.

## API Reference

### Optimizer

**Header**: `duckdb/optimizer/optimizer.hpp`
**Signature**:
```cpp
class Optimizer {
public:
    explicit Optimizer(Binder &binder, ClientContext &context);
    unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> plan);
    // Optimizer pass management
    ClientContext &context;
};
```

**Description**: Runs optimization passes on a logical plan (predicate pushdown, join ordering, filter/projection elimination, etc.).

**Our usage**:
- `src/sirius_extension.cpp` — Optimize logical plan before GPU physical plan generation. Certain optimizer passes are disabled via `DBConfig::GetSetting<OptimizerType>()`.
- `test/cpp/pipeline/test_modified_pipeline.cpp` — Optimize test plans
- `test/cpp/integration/test_tpcds_plan_translation.cpp` — Optimize TPC-DS plans

### OptimizerType (enum)

**Header**: `duckdb/common/enums/optimizer_type.hpp` (used alongside optimizer)

**Key values used**:
```cpp
OptimizerType::IN_CLAUSE                  // IN-list optimization
OptimizerType::COMPRESSED_MATERIALIZATION // Compressed materialization pass
OptimizerType::COLUMN_LIFETIME            // Column lifetime analysis
OptimizerType::MATERIALIZED_CTE           // CTE materialization decisions
```

**Our usage**:
- `src/sirius_extension.cpp` — Disable `COMPRESSED_MATERIALIZATION` and `IN_CLAUSE` optimizations that interfere with GPU execution

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `JoinOrderOptimizer` | `join_order/` | Join ordering heuristics |
| `Rule` | `rule/` | Individual optimizer rules |
| `Matcher` | `matcher/` | Expression pattern matching for rules |
| `FilterPullup/Pushdown` | Various | Filter movement optimizations |
