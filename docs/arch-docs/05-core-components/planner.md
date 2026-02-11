# Physical Planner

Comprehensive guide to Sirius physical plan generation, covering how logical plans are converted to GPU-executable physical plans with operator selection and pipeline breaks.

---

## Overview

The **Physical Planner** transforms DuckDB logical plans into Sirius physical plans optimized for GPU execution.

**Location**: `src/planner/sirius_physical_plan_generator.cpp`

**Key Responsibilities**:
1. **Operator Translation**: Logical → Physical operators
2. **Type Resolution**: DuckDB → cuDF type mapping
3. **Pipeline Breaking**: Identify blocking operators
4. **Optimization**: GPU-specific optimizations

---

## Planning Pipeline

### Complete Flow

```
DuckDB Logical Plan
       ↓
Type Analysis
       ↓
Operator Translation
       ↓
Pipeline Breaking
       ↓
Optimization
       ↓
Sirius Physical Plan
```

### Entry Point

**Function**: `sirius_physical_plan_generator::Generate()`

**Location**: `src/planner/sirius_physical_plan_generator.cpp:50-100`

```cpp
std::unique_ptr<SiriusPhysicalPlan>
sirius_physical_plan_generator::Generate(
    LogicalOperator& logical_root,
    SiriusContext& context
) {
    LOG_INFO("Physical planner: starting plan generation");

    // Create plan structure
    auto physical_plan = std::make_unique<SiriusPhysicalPlan>();

    // Visit logical operator tree
    auto root_op = Visit(logical_root, context);

    // Set root operator
    physical_plan->root_operator = std::move(root_op);

    // Identify pipeline breaks
    IdentifyPipelineBreaks(*physical_plan);

    // Build pipeline graph
    BuildPipelines(*physical_plan, context);

    // Apply optimizations
    ApplyOptimizations(*physical_plan);

    LOG_INFO("Physical planner: plan generation complete");

    return physical_plan;
}
```

---

## Operator Translation

### Visit Pattern

**Function**: `sirius_physical_plan_generator::Visit()`

```cpp
std::unique_ptr<sirius_physical_operator>
sirius_physical_plan_generator::Visit(
    LogicalOperator& logical_op,
    SiriusContext& context
) {
    switch (logical_op.type) {
        case LogicalOperatorType::GET:
            return PlanTableScan(logical_op, context);

        case LogicalOperatorType::FILTER:
            return PlanFilter(logical_op, context);

        case LogicalOperatorType::PROJECTION:
            return PlanProjection(logical_op, context);

        case LogicalOperatorType::AGGREGATE_AND_GROUP_BY:
            return PlanAggregate(logical_op, context);

        case LogicalOperatorType::ORDER_BY:
            return PlanOrderBy(logical_op, context);

        case LogicalOperatorType::COMPARISON_JOIN:
            return PlanJoin(logical_op, context);

        case LogicalOperatorType::LIMIT:
            return PlanLimit(logical_op, context);

        // ... other operators ...

        default:
            throw NotImplementedException(
                "Unsupported logical operator: " +
                LogicalOperatorToString(logical_op.type)
            );
    }
}
```

### Example: TABLE_SCAN

**Function**: `PlanTableScan()`

**Location**: `src/planner/sirius_plan_table_scan.cpp:20-80`

```cpp
std::unique_ptr<sirius_physical_operator>
sirius_physical_plan_generator::PlanTableScan(
    LogicalOperator& logical_op,
    SiriusContext& context
) {
    auto& get_op = logical_op.Cast<LogicalGet>();

    // Extract scan information
    std::string table_name = get_op.table_name;
    auto& column_ids = get_op.column_ids;
    auto& filters = get_op.table_filters;

    // Determine file path
    std::string file_path;
    if (get_op.function.name == "parquet_scan") {
        file_path = ExtractParquetPath(get_op);
    } else if (get_op.function.name == "csv_scan") {
        file_path = ExtractCSVPath(get_op);
    } else {
        throw NotImplementedException(
            "Unsupported scan function: " + get_op.function.name
        );
    }

    // Extract column projections
    std::vector<size_t> column_indices;
    for (auto col_id : column_ids) {
        column_indices.push_back(col_id);
    }

    // Convert filters (pushdown)
    std::unique_ptr<Expression> filter_expr;
    if (!filters.filters.empty()) {
        filter_expr = ConvertFiltersToExpression(filters);
    }

    // Create physical operator
    return std::make_unique<sirius_physical_table_scan>(
        file_path,
        column_indices,
        std::move(filter_expr),
        context
    );
}
```

### Example: FILTER

**Function**: `PlanFilter()`

**Location**: `src/planner/sirius_plan_filter.cpp:20-60`

```cpp
std::unique_ptr<sirius_physical_operator>
sirius_physical_plan_generator::PlanFilter(
    LogicalOperator& logical_op,
    SiriusContext& context
) {
    auto& filter_op = logical_op.Cast<LogicalFilter>();

    // Recursively plan child
    auto child = Visit(*logical_op.children[0], context);

    // Convert filter expression
    auto filter_expr = ConvertExpression(filter_op.expressions[0]);

    // Create filter operator
    auto filter = std::make_unique<sirius_physical_filter>(
        std::move(filter_expr),
        context
    );

    // Set child
    filter->add_child(std::move(child));

    return filter;
}
```

### Example: HASH_GROUP_BY

**Function**: `PlanAggregate()`

**Location**: `src/planner/sirius_plan_aggregate.cpp:20-120`

```cpp
std::unique_ptr<sirius_physical_operator>
sirius_physical_plan_generator::PlanAggregate(
    LogicalOperator& logical_op,
    SiriusContext& context
) {
    auto& agg_op = logical_op.Cast<LogicalAggregate>();

    // Recursively plan child
    auto child = Visit(*logical_op.children[0], context);

    // Extract group-by expressions
    std::vector<size_t> group_indices;
    for (auto& group_expr : agg_op.groups) {
        // Resolve to column index
        size_t col_idx = ResolveColumnIndex(group_expr, child.get());
        group_indices.push_back(col_idx);
    }

    // Extract aggregate functions
    std::vector<AggregateFunction> aggregates;
    for (auto& expr : agg_op.expressions) {
        if (expr->type == ExpressionType::AGGREGATE) {
            auto agg_expr = expr->Cast<AggregateExpression>();

            AggregateFunction agg_func;
            agg_func.function = ConvertAggregateType(agg_expr.function);
            agg_func.column_idx = ResolveColumnIndex(
                agg_expr.children[0],
                child.get()
            );
            agg_func.distinct = agg_expr.distinct;

            aggregates.push_back(agg_func);
        }
    }

    // Determine aggregate type
    if (group_indices.empty()) {
        // Ungrouped aggregate (single group)
        return std::make_unique<sirius_physical_ungrouped_aggregate>(
            aggregates,
            context
        );
    } else {
        // Hash group-by (multiple groups)
        return std::make_unique<sirius_physical_hash_group_by>(
            group_indices,
            aggregates,
            context
        );
    }
}
```

### Example: HASH_JOIN

**Function**: `PlanJoin()`

**Location**: `src/planner/sirius_plan_join.cpp:20-150`

```cpp
std::unique_ptr<sirius_physical_operator>
sirius_physical_plan_generator::PlanJoin(
    LogicalOperator& logical_op,
    SiriusContext& context
) {
    auto& join_op = logical_op.Cast<LogicalComparisonJoin>();

    // Plan left and right children
    auto left_child = Visit(*logical_op.children[0], context);
    auto right_child = Visit(*logical_op.children[1], context);

    // Extract join conditions
    std::vector<JoinCondition> conditions;
    for (auto& cond : join_op.conditions) {
        JoinCondition condition;

        // Left key
        condition.left_column_idx = ResolveColumnIndex(
            cond.left,
            left_child.get()
        );

        // Right key
        condition.right_column_idx = ResolveColumnIndex(
            cond.right,
            right_child.get()
        );

        // Comparison type
        condition.comparison = ConvertComparisonType(cond.comparison);

        conditions.push_back(condition);
    }

    // Convert join type
    JoinType join_type = ConvertJoinType(join_op.join_type);

    // Determine build/probe sides
    // Heuristic: smaller table on build side
    bool swap_sides = EstimateCardinality(right_child.get()) <
                      EstimateCardinality(left_child.get());

    if (swap_sides) {
        std::swap(left_child, right_child);
        // Adjust column indices
        for (auto& cond : conditions) {
            std::swap(cond.left_column_idx, cond.right_column_idx);
        }
    }

    // Create hash join operator
    auto join = std::make_unique<sirius_physical_hash_join>(
        join_type,
        conditions,
        context
    );

    // Set children (build = left, probe = right)
    join->add_child(std::move(left_child));   // Build side
    join->add_child(std::move(right_child));  // Probe side

    return join;
}
```

---

## Type Resolution

### Type Mapping

**DuckDB → cuDF Type Conversion**

```cpp
cudf::data_type ConvertType(const LogicalType& duckdb_type) {
    switch (duckdb_type.id()) {
        case LogicalTypeId::BOOLEAN:
            return cudf::data_type(cudf::type_id::BOOL8);

        case LogicalTypeId::TINYINT:
            return cudf::data_type(cudf::type_id::INT8);

        case LogicalTypeId::SMALLINT:
            return cudf::data_type(cudf::type_id::INT16);

        case LogicalTypeId::INTEGER:
            return cudf::data_type(cudf::type_id::INT32);

        case LogicalTypeId::BIGINT:
            return cudf::data_type(cudf::type_id::INT64);

        case LogicalTypeId::HUGEINT:
            // cuDF doesn't support 128-bit int, downgrade
            LOG_WARN("Downgrading HUGEINT to BIGINT");
            return cudf::data_type(cudf::type_id::INT64);

        case LogicalTypeId::FLOAT:
            return cudf::data_type(cudf::type_id::FLOAT32);

        case LogicalTypeId::DOUBLE:
            return cudf::data_type(cudf::type_id::FLOAT64);

        case LogicalTypeId::VARCHAR:
        case LogicalTypeId::BLOB:
            return cudf::data_type(cudf::type_id::STRING);

        case LogicalTypeId::DATE:
            return cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);

        case LogicalTypeId::TIMESTAMP:
            return cudf::data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);

        case LogicalTypeId::DECIMAL:
            // cuDF supports DECIMAL64
            auto decimal_type = duckdb_type.Cast<DecimalType>();
            if (decimal_type.width <= 18) {
                return cudf::data_type(
                    cudf::type_id::DECIMAL64,
                    -decimal_type.scale
                );
            }
            throw NotImplementedException(
                "DECIMAL wider than 18 digits not supported"
            );

        default:
            throw NotImplementedException(
                "Unsupported type: " + duckdb_type.ToString()
            );
    }
}
```

### Type Compatibility Check

```cpp
bool IsTypeSupported(const LogicalType& type) {
    // Check if type can be executed on GPU
    switch (type.id()) {
        case LogicalTypeId::BOOLEAN:
        case LogicalTypeId::TINYINT:
        case LogicalTypeId::SMALLINT:
        case LogicalTypeId::INTEGER:
        case LogicalTypeId::BIGINT:
        case LogicalTypeId::FLOAT:
        case LogicalTypeId::DOUBLE:
        case LogicalTypeId::VARCHAR:
        case LogicalTypeId::DATE:
        case LogicalTypeId::TIMESTAMP:
            return true;

        case LogicalTypeId::DECIMAL:
            auto decimal = type.Cast<DecimalType>();
            return decimal.width <= 18;  // DECIMAL64 limit

        // Complex types not supported
        case LogicalTypeId::LIST:
        case LogicalTypeId::STRUCT:
        case LogicalTypeId::MAP:
        case LogicalTypeId::UNION:
            return false;

        default:
            return false;
    }
}
```

---

## Pipeline Breaking

### Identification

**Blocking Operators** (require all input before producing output):
- `ORDER_BY`: Must see all data to sort
- `HASH_GROUP_BY` (finalize): Must aggregate all groups
- `HASH_JOIN` (build): Must build complete hash table
- `TOP_N` (when large): May need to buffer
- `WINDOW` functions: Partition-dependent

**Function**: `IdentifyPipelineBreaks()`

```cpp
void IdentifyPipelineBreaks(SiriusPhysicalPlan& plan) {
    std::vector<PipelineBreak> breaks;

    // Traverse operator tree
    VisitOperators(plan.root_operator.get(), [&](auto* op) {
        if (IsBlockingOperator(op)) {
            PipelineBreak break_point;
            break_point.operator_id = op->get_id();
            break_point.operator_type = op->get_type();
            break_point.blocking_reason = GetBlockingReason(op);

            breaks.push_back(break_point);

            LOG_DEBUG("Pipeline break at {}: {}",
                      op->get_name(),
                      break_point.blocking_reason);
        }
    });

    plan.pipeline_breaks = std::move(breaks);
}

bool IsBlockingOperator(sirius_physical_operator* op) {
    switch (op->get_type()) {
        case SiriusPhysicalOperatorType::ORDER_BY:
        case SiriusPhysicalOperatorType::TOP_N:
        case SiriusPhysicalOperatorType::MERGE_SORT:
            return true;

        case SiriusPhysicalOperatorType::HASH_GROUP_BY:
            // Finalize phase is blocking
            return true;

        case SiriusPhysicalOperatorType::HASH_JOIN:
            // Build phase is blocking
            return IsHashJoinBuild(op);

        default:
            return false;
    }
}
```

### Pipeline Construction

**Function**: `BuildPipelines()`

```cpp
void BuildPipelines(SiriusPhysicalPlan& plan, SiriusContext& context) {
    // Split operator tree at pipeline breaks
    auto pipeline_segments = SplitAtBreaks(
        plan.root_operator.get(),
        plan.pipeline_breaks
    );

    // Create pipeline for each segment
    for (size_t i = 0; i < pipeline_segments.size(); i++) {
        auto& segment = pipeline_segments[i];

        auto pipeline = std::make_unique<sirius_pipeline>(i, "pipeline_" + std::to_string(i));

        // Set source operator (entry point)
        pipeline->set_source(segment.source);

        // Set intermediate operators
        for (auto* op : segment.intermediate_operators) {
            pipeline->add_intermediate(op);
        }

        // Set sink operator (if any)
        if (segment.sink) {
            pipeline->set_sink(segment.sink);
        }

        plan.pipelines.push_back(std::move(pipeline));
    }

    LOG_INFO("Created {} pipelines", plan.pipelines.size());
}
```

---

## Optimization

### Push-Down Optimizations

#### Filter Pushdown

```cpp
void PushDownFilters(sirius_physical_operator* op) {
    if (op->get_type() == SiriusPhysicalOperatorType::FILTER) {
        auto* filter_op = static_cast<sirius_physical_filter*>(op);

        // Check if child is TABLE_SCAN
        if (op->children[0]->get_type() ==
            SiriusPhysicalOperatorType::TABLE_SCAN) {

            auto* scan_op = static_cast<sirius_physical_table_scan*>(
                op->children[0].get()
            );

            // Push filter into scan
            if (CanPushDownFilter(filter_op->get_filter(), scan_op)) {
                scan_op->add_filter(filter_op->get_filter());

                LOG_INFO("Pushed down filter to TABLE_SCAN");

                // Remove filter operator (bypass)
                ReplaceOperatorWithChild(filter_op);
            }
        }
    }

    // Recurse to children
    for (auto& child : op->children) {
        PushDownFilters(child.get());
    }
}

bool CanPushDownFilter(Expression* filter, sirius_physical_table_scan* scan) {
    // Check if filter only references scanned columns
    auto referenced_columns = ExtractReferencedColumns(filter);

    for (auto col_idx : referenced_columns) {
        if (!scan->projects_column(col_idx)) {
            return false;  // References non-projected column
        }
    }

    // Check if Parquet/CSV reader supports filter type
    return IsSimplePredicate(filter);  // =, <, >, AND, OR
}
```

#### Projection Pushdown

```cpp
void PushDownProjections(sirius_physical_operator* op) {
    // Identify columns actually used by query
    auto required_columns = IdentifyRequiredColumns(op);

    // Propagate requirements down to scans
    PropagateColumnRequirements(op, required_columns);
}

std::unordered_set<size_t> IdentifyRequiredColumns(
    sirius_physical_operator* op
) {
    std::unordered_set<size_t> required;

    switch (op->get_type()) {
        case SiriusPhysicalOperatorType::FILTER:
            // Filter needs columns in predicate
            auto* filter = static_cast<sirius_physical_filter*>(op);
            auto cols = ExtractReferencedColumns(filter->get_filter());
            required.insert(cols.begin(), cols.end());
            break;

        case SiriusPhysicalOperatorType::PROJECTION:
            // Projection needs columns in expressions
            auto* proj = static_cast<sirius_physical_projection*>(op);
            for (auto& expr : proj->get_expressions()) {
                auto cols = ExtractReferencedColumns(expr.get());
                required.insert(cols.begin(), cols.end());
            }
            break;

        case SiriusPhysicalOperatorType::HASH_GROUP_BY:
            // Group-by needs group keys + aggregate columns
            auto* agg = static_cast<sirius_physical_hash_group_by*>(op);
            for (auto idx : agg->get_group_indices()) {
                required.insert(idx);
            }
            for (auto& agg_func : agg->get_aggregates()) {
                required.insert(agg_func.column_idx);
            }
            break;

        // ... other operators ...
    }

    return required;
}
```

### Join Reordering

```cpp
void ReorderJoins(SiriusPhysicalPlan& plan) {
    // Simple heuristic: smaller table on build side
    VisitOperators(plan.root_operator.get(), [](auto* op) {
        if (op->get_type() == SiriusPhysicalOperatorType::HASH_JOIN) {
            auto* join = static_cast<sirius_physical_hash_join*>(op);

            auto left_card = EstimateCardinality(join->children[0].get());
            auto right_card = EstimateCardinality(join->children[1].get());

            if (right_card < left_card) {
                // Swap sides (right should be build)
                std::swap(join->children[0], join->children[1]);

                LOG_INFO("Swapped join sides (build cardinality: {} vs {})",
                         right_card, left_card);
            }
        }
    });
}

size_t EstimateCardinality(sirius_physical_operator* op) {
    // Simple heuristics
    switch (op->get_type()) {
        case SiriusPhysicalOperatorType::TABLE_SCAN:
            auto* scan = static_cast<sirius_physical_table_scan*>(op);
            return scan->get_row_count();

        case SiriusPhysicalOperatorType::FILTER:
            // Assume 10% selectivity
            return EstimateCardinality(op->children[0].get()) * 0.1;

        case SiriusPhysicalOperatorType::HASH_GROUP_BY:
            // Assume 10% unique groups
            return EstimateCardinality(op->children[0].get()) * 0.1;

        default:
            return EstimateCardinality(op->children[0].get());
    }
}
```

---

## Expression Conversion

### DuckDB Expression → Sirius Expression

**Function**: `ConvertExpression()`

```cpp
std::unique_ptr<Expression> ConvertExpression(
    const duckdb::Expression& duckdb_expr
) {
    switch (duckdb_expr.GetExpressionType()) {
        case ExpressionType::BOUND_COLUMN_REF:
            auto& col_ref = duckdb_expr.Cast<BoundColumnRefExpression>();
            return std::make_unique<ColumnRefExpression>(
                col_ref.binding.column_index
            );

        case ExpressionType::COMPARE_EQUAL:
        case ExpressionType::COMPARE_NOTEQUAL:
        case ExpressionType::COMPARE_LESSTHAN:
        case ExpressionType::COMPARE_GREATERTHAN:
        case ExpressionType::COMPARE_LESSTHANOREQUALTO:
        case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            auto& comp = duckdb_expr.Cast<BoundComparisonExpression>();
            return std::make_unique<ComparisonExpression>(
                ConvertComparisonType(comp.type),
                ConvertExpression(*comp.left),
                ConvertExpression(*comp.right)
            );

        case ExpressionType::CONJUNCTION_AND:
        case ExpressionType::CONJUNCTION_OR:
            auto& conj = duckdb_expr.Cast<BoundConjunctionExpression>();
            return std::make_unique<ConjunctionExpression>(
                ConvertConjunctionType(conj.type),
                ConvertExpressions(conj.children)
            );

        case ExpressionType::VALUE_CONSTANT:
            auto& const_expr = duckdb_expr.Cast<BoundConstantExpression>();
            return std::make_unique<ConstantExpression>(
                ConvertValue(const_expr.value)
            );

        case ExpressionType::OPERATOR_ADD:
        case ExpressionType::OPERATOR_SUBTRACT:
        case ExpressionType::OPERATOR_MULTIPLY:
        case ExpressionType::OPERATOR_DIVIDE:
            auto& arith = duckdb_expr.Cast<BoundFunctionExpression>();
            return std::make_unique<ArithmeticExpression>(
                ConvertArithmeticType(arith.function.name),
                ConvertExpressions(arith.children)
            );

        // ... other expression types ...

        default:
            throw NotImplementedException(
                "Expression type not supported: " +
                ExpressionTypeToString(duckdb_expr.GetExpressionType())
            );
    }
}
```

---

## Complete Example

### Query

```sql
SELECT category, SUM(price) as total
FROM products
WHERE price > 100
GROUP BY category
ORDER BY total DESC
LIMIT 10;
```

### Planning Steps

**Step 1: Logical Plan** (from DuckDB)

```
LIMIT (10)
  ↓
ORDER_BY (total DESC)
  ↓
AGGREGATE_AND_GROUP_BY (category, SUM(price))
  ↓
FILTER (price > 100)
  ↓
GET (products)
```

**Step 2: Physical Plan** (from Sirius Planner)

```
LIMIT (10)
  ↓
ORDER_BY (total DESC)  ← Pipeline Break
  ↓
HASH_GROUP_BY (category, SUM(price))  ← Pipeline Break
  ↓
FILTER (price > 100)
  ↓
TABLE_SCAN (products.parquet)
```

**Step 3: Pipeline Construction**

```
Pipeline 0:
  TABLE_SCAN (products.parquet)
  → FILTER (price > 100)
  → HASH_GROUP_BY (sink)
  → Output: Repository A

Pipeline 1:
  DUMMY_SCAN (from Repo A)
  → ORDER_BY (buffer all, sort)
  → LIMIT (10)
  → RESULT_COLLECTOR
```

**Step 4: Optimizations**

- **Filter pushdown**: Move `price > 100` into TABLE_SCAN
- **Projection pushdown**: Only read `category` and `price` columns
- **TOP_N optimization**: Combine ORDER_BY + LIMIT → TOP_N operator

**Final Plan**:

```
Pipeline 0:
  TABLE_SCAN (products.parquet)
    - Columns: [category, price]
    - Filter: price > 100 (pushdown)
  → HASH_GROUP_BY (sink)
  → Output: Repository A

Pipeline 1:
  DUMMY_SCAN (from Repo A)
  → TOP_N (10, DESC)  ← Optimized ORDER_BY + LIMIT
  → RESULT_COLLECTOR
```

---

## Debugging

### Enable Planner Logging

```bash
export SIRIUS_LOG_LEVEL=DEBUG
export SIRIUS_LOG_FILE=/tmp/sirius_planner.log
```

**Log Output**:

```
[DEBUG] Physical planner: starting plan generation
[DEBUG] Visiting logical operator: GET
[DEBUG] Created TABLE_SCAN operator
[DEBUG] Visiting logical operator: FILTER
[DEBUG] Created FILTER operator
[DEBUG] Visiting logical operator: AGGREGATE_AND_GROUP_BY
[DEBUG] Created HASH_GROUP_BY operator
[DEBUG] Pipeline break at HASH_GROUP_BY: blocking finalize
[DEBUG] Visiting logical operator: ORDER_BY
[DEBUG] Created ORDER_BY operator
[DEBUG] Pipeline break at ORDER_BY: blocking sort
[INFO] Created 2 pipelines
[DEBUG] Applying filter pushdown
[INFO] Pushed down filter to TABLE_SCAN
[DEBUG] Applying TOP_N optimization
[INFO] Optimized ORDER_BY + LIMIT → TOP_N
[INFO] Physical planner: plan generation complete
```

### Print Physical Plan

```cpp
void PrintPhysicalPlan(const SiriusPhysicalPlan& plan) {
    printf("Physical Plan:\n");
    printf("  Pipelines: %zu\n", plan.pipelines.size());
    printf("  Pipeline Breaks: %zu\n\n", plan.pipeline_breaks.size());

    for (size_t i = 0; i < plan.pipelines.size(); i++) {
        printf("Pipeline %zu:\n", i);
        plan.pipelines[i]->print();
        printf("\n");
    }
}
```

---

## See Also

- [New Mode Overview](../04-new-mode/overview.md) - New Mode architecture
- [Legacy Mode Overview](../03-legacy-mode/overview.md) - Legacy Mode architecture
- [Operators](../04-new-mode/operators.md) - Operator implementations
- [Expression Executor](expression-executor.md) - Expression evaluation
- [Query Lifecycle](../06-data-flow/query-lifecycle.md) - Complete execution flow
- [Adding Operators](../07-development/adding-operators.md) - Extend planner
