# Expression Executor

Comprehensive guide to Sirius expression evaluation system, covering how SQL expressions are evaluated on GPU using cuDF operations.

---

## Overview

The **Expression Executor** evaluates SQL expressions (filters, projections, aggregates) on GPU using cuDF operations.

**Key Responsibilities**:
1. **Expression Evaluation**: Convert expressions to cuDF operations
2. **Type Handling**: Manage type conversions and casting
3. **Null Handling**: Propagate nulls correctly
4. **Optimization**: Reuse intermediate results

**Location**: `src/expression/expression_executor.cpp`

---

## Expression Types

### Hierarchy

```
Expression (base)
├─ ColumnRefExpression
├─ ConstantExpression
├─ ComparisonExpression
├─ ConjunctionExpression (AND/OR)
├─ ArithmeticExpression (+, -, *, /)
├─ FunctionExpression
└─ CaseExpression
```

### Base Class

**Location**: `src/include/expression/expression.hpp`

```cpp
class Expression {
protected:
    ExpressionType type_;
    LogicalType return_type_;

public:
    Expression(ExpressionType type, LogicalType return_type)
        : type_(type), return_type_(return_type) {}

    virtual ~Expression() = default;

    ExpressionType get_type() const { return type_; }
    LogicalType get_return_type() const { return return_type_; }

    // Convert to string for debugging
    virtual std::string to_string() const = 0;
};

enum class ExpressionType {
    COLUMN_REF,
    CONSTANT,
    COMPARISON,
    CONJUNCTION,
    ARITHMETIC,
    FUNCTION,
    CASE
};
```

---

## Expression Executor

### Core Class

```cpp
class ExpressionExecutor {
private:
    SiriusContext& context_;

public:
    ExpressionExecutor(SiriusContext& context)
        : context_(context) {}

    // Evaluate to column
    std::unique_ptr<cudf::column> evaluate(
        const cudf::table_view& table,
        const Expression& expr
    );

    // Evaluate to boolean mask
    std::unique_ptr<cudf::column> evaluate_boolean(
        const cudf::table_view& table,
        const Expression& expr
    );

    // Evaluate to scalar
    cudf::scalar_type_t<T> evaluate_scalar(
        const cudf::table_view& table,
        const Expression& expr
    );

private:
    // Type-specific evaluation
    std::unique_ptr<cudf::column> evaluate_column_ref(
        const cudf::table_view& table,
        const ColumnRefExpression& expr
    );

    std::unique_ptr<cudf::column> evaluate_constant(
        const ConstantExpression& expr,
        size_t num_rows
    );

    std::unique_ptr<cudf::column> evaluate_comparison(
        const cudf::table_view& table,
        const ComparisonExpression& expr
    );

    std::unique_ptr<cudf::column> evaluate_conjunction(
        const cudf::table_view& table,
        const ConjunctionExpression& expr
    );

    std::unique_ptr<cudf::column> evaluate_arithmetic(
        const cudf::table_view& table,
        const ArithmeticExpression& expr
    );

    std::unique_ptr<cudf::column> evaluate_function(
        const cudf::table_view& table,
        const FunctionExpression& expr
    );

    std::unique_ptr<cudf::column> evaluate_case(
        const cudf::table_view& table,
        const CaseExpression& expr
    );
};
```

### Main Evaluation

```cpp
std::unique_ptr<cudf::column> ExpressionExecutor::evaluate(
    const cudf::table_view& table,
    const Expression& expr
) {
    switch (expr.get_type()) {
        case ExpressionType::COLUMN_REF:
            return evaluate_column_ref(
                table,
                static_cast<const ColumnRefExpression&>(expr)
            );

        case ExpressionType::CONSTANT:
            return evaluate_constant(
                static_cast<const ConstantExpression&>(expr),
                table.num_rows()
            );

        case ExpressionType::COMPARISON:
            return evaluate_comparison(
                table,
                static_cast<const ComparisonExpression&>(expr)
            );

        case ExpressionType::CONJUNCTION:
            return evaluate_conjunction(
                table,
                static_cast<const ConjunctionExpression&>(expr)
            );

        case ExpressionType::ARITHMETIC:
            return evaluate_arithmetic(
                table,
                static_cast<const ArithmeticExpression&>(expr)
            );

        case ExpressionType::FUNCTION:
            return evaluate_function(
                table,
                static_cast<const FunctionExpression&>(expr)
            );

        case ExpressionType::CASE:
            return evaluate_case(
                table,
                static_cast<const CaseExpression&>(expr)
            );

        default:
            throw NotImplementedException(
                "Expression type not supported: " +
                std::to_string(static_cast<int>(expr.get_type()))
            );
    }
}
```

---

## Expression Evaluation

### Column Reference

**Example**: `SELECT price FROM products`

**Expression**:
```cpp
class ColumnRefExpression : public Expression {
private:
    size_t column_index_;
    std::string column_name_;

public:
    ColumnRefExpression(size_t column_index, std::string column_name)
        : Expression(ExpressionType::COLUMN_REF, LogicalType::UNKNOWN),
          column_index_(column_index),
          column_name_(column_name) {}

    size_t get_column_index() const { return column_index_; }
    std::string get_column_name() const { return column_name_; }
};
```

**Evaluation**:
```cpp
std::unique_ptr<cudf::column> ExpressionExecutor::evaluate_column_ref(
    const cudf::table_view& table,
    const ColumnRefExpression& expr
) {
    // Simply return a copy of the referenced column
    size_t col_idx = expr.get_column_index();

    if (col_idx >= table.num_columns()) {
        throw InternalException(
            "Column index out of bounds: " + std::to_string(col_idx)
        );
    }

    return std::make_unique<cudf::column>(table.column(col_idx));
}
```

### Constant

**Example**: `WHERE price > 100`

**Expression**:
```cpp
class ConstantExpression : public Expression {
private:
    Value value_;

public:
    ConstantExpression(Value value)
        : Expression(ExpressionType::CONSTANT, value.type()),
          value_(value) {}

    const Value& get_value() const { return value_; }
};
```

**Evaluation**:
```cpp
std::unique_ptr<cudf::column> ExpressionExecutor::evaluate_constant(
    const ConstantExpression& expr,
    size_t num_rows
) {
    const Value& value = expr.get_value();

    // Convert value to cuDF scalar
    auto scalar = value_to_cudf_scalar(value);

    // Create column of repeated scalars
    return cudf::make_column_from_scalar(*scalar, num_rows);
}

std::unique_ptr<cudf::scalar> value_to_cudf_scalar(const Value& value) {
    switch (value.type().id()) {
        case LogicalTypeId::BOOLEAN:
            return std::make_unique<cudf::numeric_scalar<bool>>(
                value.GetValue<bool>()
            );

        case LogicalTypeId::INTEGER:
            return std::make_unique<cudf::numeric_scalar<int32_t>>(
                value.GetValue<int32_t>()
            );

        case LogicalTypeId::BIGINT:
            return std::make_unique<cudf::numeric_scalar<int64_t>>(
                value.GetValue<int64_t>()
            );

        case LogicalTypeId::FLOAT:
            return std::make_unique<cudf::numeric_scalar<float>>(
                value.GetValue<float>()
            );

        case LogicalTypeId::DOUBLE:
            return std::make_unique<cudf::numeric_scalar<double>>(
                value.GetValue<double>()
            );

        case LogicalTypeId::VARCHAR:
            return std::make_unique<cudf::string_scalar>(
                value.GetValue<std::string>()
            );

        default:
            throw NotImplementedException(
                "Unsupported constant type: " + value.type().ToString()
            );
    }
}
```

### Comparison

**Example**: `WHERE price > 100`

**Expression**:
```cpp
class ComparisonExpression : public Expression {
private:
    ExpressionComparisonType comparison_type_;
    std::unique_ptr<Expression> left_;
    std::unique_ptr<Expression> right_;

public:
    ComparisonExpression(
        ExpressionComparisonType type,
        std::unique_ptr<Expression> left,
        std::unique_ptr<Expression> right
    ) : Expression(ExpressionType::COMPARISON, LogicalType::BOOLEAN),
        comparison_type_(type),
        left_(std::move(left)),
        right_(std::move(right)) {}

    // Getters
    ExpressionComparisonType get_comparison_type() const { return comparison_type_; }
    Expression* get_left() const { return left_.get(); }
    Expression* get_right() const { return right_.get(); }
};

enum class ExpressionComparisonType {
    EQUAL,
    NOT_EQUAL,
    LESS_THAN,
    LESS_THAN_OR_EQUAL,
    GREATER_THAN,
    GREATER_THAN_OR_EQUAL
};
```

**Evaluation**:
```cpp
std::unique_ptr<cudf::column> ExpressionExecutor::evaluate_comparison(
    const cudf::table_view& table,
    const ComparisonExpression& expr
) {
    // Evaluate left and right sides
    auto left_col = evaluate(table, *expr.get_left());
    auto right_col = evaluate(table, *expr.get_right());

    // Convert comparison type
    cudf::binary_operator op;
    switch (expr.get_comparison_type()) {
        case ExpressionComparisonType::EQUAL:
            op = cudf::binary_operator::EQUAL;
            break;
        case ExpressionComparisonType::NOT_EQUAL:
            op = cudf::binary_operator::NOT_EQUAL;
            break;
        case ExpressionComparisonType::LESS_THAN:
            op = cudf::binary_operator::LESS;
            break;
        case ExpressionComparisonType::LESS_THAN_OR_EQUAL:
            op = cudf::binary_operator::LESS_EQUAL;
            break;
        case ExpressionComparisonType::GREATER_THAN:
            op = cudf::binary_operator::GREATER;
            break;
        case ExpressionComparisonType::GREATER_THAN_OR_EQUAL:
            op = cudf::binary_operator::GREATER_EQUAL;
            break;
    }

    // Execute comparison using cuDF
    return cudf::binary_operation(
        left_col->view(),
        right_col->view(),
        op,
        cudf::data_type(cudf::type_id::BOOL8)
    );
}
```

**cuDF Example**:
```cpp
// price > 100
auto price_col = table.column("price");  // [50, 150, 200, 75]
auto const_100 = cudf::numeric_scalar<int32_t>(100);

auto result = cudf::binary_operation(
    price_col,
    const_100,
    cudf::binary_operator::GREATER,
    cudf::data_type(cudf::type_id::BOOL8)
);
// Result: [false, true, true, false]
```

### Conjunction (AND/OR)

**Example**: `WHERE (price > 100) AND (category = 'Electronics')`

**Expression**:
```cpp
class ConjunctionExpression : public Expression {
private:
    ConjunctionType conjunction_type_;
    std::vector<std::unique_ptr<Expression>> children_;

public:
    ConjunctionExpression(
        ConjunctionType type,
        std::vector<std::unique_ptr<Expression>> children
    ) : Expression(ExpressionType::CONJUNCTION, LogicalType::BOOLEAN),
        conjunction_type_(type),
        children_(std::move(children)) {}

    ConjunctionType get_conjunction_type() const { return conjunction_type_; }
    const std::vector<std::unique_ptr<Expression>>& get_children() const {
        return children_;
    }
};

enum class ConjunctionType {
    AND,
    OR
};
```

**Evaluation**:
```cpp
std::unique_ptr<cudf::column> ExpressionExecutor::evaluate_conjunction(
    const cudf::table_view& table,
    const ConjunctionExpression& expr
) {
    const auto& children = expr.get_children();

    if (children.empty()) {
        throw InternalException("Conjunction with no children");
    }

    // Evaluate first child
    auto result = evaluate(table, *children[0]);

    // Combine with remaining children
    for (size_t i = 1; i < children.size(); i++) {
        auto child_result = evaluate(table, *children[i]);

        cudf::binary_operator op;
        switch (expr.get_conjunction_type()) {
            case ConjunctionType::AND:
                op = cudf::binary_operator::LOGICAL_AND;
                break;
            case ConjunctionType::OR:
                op = cudf::binary_operator::LOGICAL_OR;
                break;
        }

        result = cudf::binary_operation(
            result->view(),
            child_result->view(),
            op,
            cudf::data_type(cudf::type_id::BOOL8)
        );
    }

    return result;
}
```

**cuDF Example**:
```cpp
// (price > 100) AND (category = 'Electronics')
auto mask1 = price > 100;         // [false, true, true, false]
auto mask2 = category == "Electronics";  // [true, true, false, false]

auto result = cudf::binary_operation(
    mask1,
    mask2,
    cudf::binary_operator::LOGICAL_AND,
    cudf::data_type(cudf::type_id::BOOL8)
);
// Result: [false, true, false, false]
```

### Arithmetic

**Example**: `SELECT price * 1.1 AS price_with_tax`

**Expression**:
```cpp
class ArithmeticExpression : public Expression {
private:
    ArithmeticType arithmetic_type_;
    std::unique_ptr<Expression> left_;
    std::unique_ptr<Expression> right_;

public:
    ArithmeticExpression(
        ArithmeticType type,
        std::unique_ptr<Expression> left,
        std::unique_ptr<Expression> right,
        LogicalType return_type
    ) : Expression(ExpressionType::ARITHMETIC, return_type),
        arithmetic_type_(type),
        left_(std::move(left)),
        right_(std::move(right)) {}

    // Getters
    ArithmeticType get_arithmetic_type() const { return arithmetic_type_; }
    Expression* get_left() const { return left_.get(); }
    Expression* get_right() const { return right_.get(); }
};

enum class ArithmeticType {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    MODULO
};
```

**Evaluation**:
```cpp
std::unique_ptr<cudf::column> ExpressionExecutor::evaluate_arithmetic(
    const cudf::table_view& table,
    const ArithmeticExpression& expr
) {
    // Evaluate left and right sides
    auto left_col = evaluate(table, *expr.get_left());
    auto right_col = evaluate(table, *expr.get_right());

    // Convert arithmetic type
    cudf::binary_operator op;
    switch (expr.get_arithmetic_type()) {
        case ArithmeticType::ADD:
            op = cudf::binary_operator::ADD;
            break;
        case ArithmeticType::SUBTRACT:
            op = cudf::binary_operator::SUB;
            break;
        case ArithmeticType::MULTIPLY:
            op = cudf::binary_operator::MUL;
            break;
        case ArithmeticType::DIVIDE:
            op = cudf::binary_operator::DIV;
            break;
        case ArithmeticType::MODULO:
            op = cudf::binary_operator::MOD;
            break;
    }

    // Determine result type
    auto result_type = convert_logical_type_to_cudf(expr.get_return_type());

    // Execute arithmetic using cuDF
    return cudf::binary_operation(
        left_col->view(),
        right_col->view(),
        op,
        result_type
    );
}
```

**cuDF Example**:
```cpp
// price * 1.1
auto price_col = table.column("price");  // [100.0, 200.0, 300.0]
auto multiplier = cudf::numeric_scalar<double>(1.1);

auto result = cudf::binary_operation(
    price_col,
    multiplier,
    cudf::binary_operator::MUL,
    cudf::data_type(cudf::type_id::FLOAT64)
);
// Result: [110.0, 220.0, 330.0]
```

### CASE Expression

**Example**: `CASE WHEN price > 100 THEN 'expensive' ELSE 'cheap' END`

**Expression**:
```cpp
class CaseExpression : public Expression {
private:
    struct WhenClause {
        std::unique_ptr<Expression> condition;  // Boolean expression
        std::unique_ptr<Expression> result;     // Result if condition true
    };

    std::vector<WhenClause> when_clauses_;
    std::unique_ptr<Expression> else_result_;

public:
    CaseExpression(
        std::vector<WhenClause> when_clauses,
        std::unique_ptr<Expression> else_result,
        LogicalType return_type
    ) : Expression(ExpressionType::CASE, return_type),
        when_clauses_(std::move(when_clauses)),
        else_result_(std::move(else_result)) {}
};
```

**Evaluation**:
```cpp
std::unique_ptr<cudf::column> ExpressionExecutor::evaluate_case(
    const cudf::table_view& table,
    const CaseExpression& expr
) {
    size_t num_rows = table.num_rows();

    // Initialize result with ELSE value
    auto result = expr.get_else_result()
        ? evaluate(table, *expr.get_else_result())
        : create_null_column(expr.get_return_type(), num_rows);

    // Process WHEN clauses in reverse order (last to first)
    // This allows us to overwrite earlier results
    for (int i = expr.get_when_clauses().size() - 1; i >= 0; i--) {
        const auto& when_clause = expr.get_when_clauses()[i];

        // Evaluate condition
        auto condition = evaluate(table, *when_clause.condition);

        // Evaluate result
        auto when_result = evaluate(table, *when_clause.result);

        // Use copy_if_else to merge
        result = cudf::copy_if_else(
            when_result->view(),
            result->view(),
            condition->view()
        );
    }

    return result;
}
```

**cuDF Example**:
```cpp
// CASE WHEN price > 100 THEN 'expensive' ELSE 'cheap' END
auto price = table.column("price");  // [50, 150, 200, 75]
auto condition = price > 100;        // [false, true, true, false]

auto expensive = cudf::make_column_from_scalar(
    cudf::string_scalar("expensive"),
    price.size()
);

auto cheap = cudf::make_column_from_scalar(
    cudf::string_scalar("cheap"),
    price.size()
);

auto result = cudf::copy_if_else(
    expensive->view(),
    cheap->view(),
    condition->view()
);
// Result: ["cheap", "expensive", "expensive", "cheap"]
```

---

## Function Evaluation

### String Functions

**UPPER**:
```cpp
// UPPER(name)
auto name_col = table.column("name");  // ["alice", "BOB", "Charlie"]
auto upper = cudf::strings::to_upper(name_col);
// Result: ["ALICE", "BOB", "CHARLIE"]
```

**LOWER**:
```cpp
// LOWER(name)
auto name_col = table.column("name");  // ["Alice", "BOB", "Charlie"]
auto lower = cudf::strings::to_lower(name_col);
// Result: ["alice", "bob", "charlie"]
```

**SUBSTRING**:
```cpp
// SUBSTRING(name, 1, 3)
auto name_col = table.column("name");  // ["Alice", "Bob", "Charlie"]
auto substr = cudf::strings::slice_strings(
    name_col,
    cudf::numeric_scalar<int32_t>(0),  // Start (0-indexed)
    cudf::numeric_scalar<int32_t>(3),  // End (exclusive)
    cudf::numeric_scalar<int32_t>(1)   // Step
);
// Result: ["Ali", "Bob", "Cha"]
```

**CONCAT**:
```cpp
// CONCAT(first_name, ' ', last_name)
auto first = table.column("first_name");  // ["Alice", "Bob"]
auto last = table.column("last_name");    // ["Smith", "Jones"]

std::vector<cudf::column_view> cols = {first, last};
std::vector<cudf::string_scalar> separators = {
    cudf::string_scalar(" ")
};

auto concat = cudf::strings::concatenate(cols, separators[0]);
// Result: ["Alice Smith", "Bob Jones"]
```

### Date Functions

**YEAR**:
```cpp
// YEAR(date_col)
auto date_col = table.column("date");  // [2024-01-15, 2024-06-30]
auto year = cudf::datetime::extract_year(date_col);
// Result: [2024, 2024]
```

**MONTH**:
```cpp
// MONTH(date_col)
auto date_col = table.column("date");  // [2024-01-15, 2024-06-30]
auto month = cudf::datetime::extract_month(date_col);
// Result: [1, 6]
```

**DATE_ADD**:
```cpp
// DATE_ADD(date_col, INTERVAL 7 DAY)
auto date_col = table.column("date");  // [2024-01-15, 2024-06-30]
auto days = cudf::numeric_scalar<int32_t>(7);

auto result = cudf::datetime::add_calendrical_months(
    date_col,
    days
);
// Result: [2024-01-22, 2024-07-07]
```

### Aggregate Functions

**SUM**:
```cpp
// SUM(price)
auto price_col = table.column("price");  // [100, 200, 300]
auto sum = cudf::reduce(
    price_col,
    cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
    cudf::data_type(cudf::type_id::INT64)
);
// Result: 600
```

**AVG**:
```cpp
// AVG(price)
auto price_col = table.column("price");  // [100, 200, 300]
auto avg = cudf::reduce(
    price_col,
    cudf::make_mean_aggregation<cudf::reduce_aggregation>(),
    cudf::data_type(cudf::type_id::FLOAT64)
);
// Result: 200.0
```

**MIN/MAX**:
```cpp
// MIN(price), MAX(price)
auto price_col = table.column("price");  // [100, 200, 300]

auto min_val = cudf::reduce(
    price_col,
    cudf::make_min_aggregation<cudf::reduce_aggregation>(),
    price_col.type()
);
// Result: 100

auto max_val = cudf::reduce(
    price_col,
    cudf::make_max_aggregation<cudf::reduce_aggregation>(),
    price_col.type()
);
// Result: 300
```

---

## Null Handling

### Null Propagation

**Example**: `price + tax` where some values are NULL

```cpp
auto price = column([100, NULL, 200, 300]);
auto tax = column([10, 20, NULL, 30]);

auto result = cudf::binary_operation(
    price,
    tax,
    cudf::binary_operator::ADD,
    cudf::data_type(cudf::type_id::INT64)
);
// Result: [110, NULL, NULL, 330]

// NULLs propagate automatically in cuDF operations
```

### IS NULL / IS NOT NULL

```cpp
// WHERE price IS NULL
auto price_col = table.column("price");  // [100, NULL, 200]
auto is_null_mask = cudf::is_null(price_col);
// Result: [false, true, false]

// WHERE price IS NOT NULL
auto is_not_null_mask = cudf::is_valid(price_col);
// Result: [true, false, true]
```

### COALESCE

```cpp
// COALESCE(price, default_price)
auto price = column([100, NULL, 200]);
auto default_price = column([50, 75, 80]);

// Use copy_if_else with validity mask
auto result = cudf::copy_if_else(
    price->view(),
    default_price->view(),
    cudf::is_valid(price)  // Use price if valid, else default
);
// Result: [100, 75, 200]
```

---

## Performance Optimization

### Expression Caching

**Problem**: Repeated evaluation of same expression

```sql
SELECT
    price * 1.1 as price_with_tax,
    price * 1.1 * 0.95 as discounted_price
FROM products;
```

**Optimization**: Cache `price * 1.1`

```cpp
class ExpressionExecutor {
private:
    std::unordered_map<std::string, std::unique_ptr<cudf::column>> cache_;

public:
    std::unique_ptr<cudf::column> evaluate_with_cache(
        const cudf::table_view& table,
        const Expression& expr
    ) {
        std::string expr_key = expr.to_string();

        // Check cache
        auto it = cache_.find(expr_key);
        if (it != cache_.end()) {
            LOG_TRACE("Expression cache hit: {}", expr_key);
            return std::make_unique<cudf::column>(*it->second);
        }

        // Evaluate and cache
        auto result = evaluate(table, expr);
        cache_[expr_key] = std::make_unique<cudf::column>(*result);

        return result;
    }
};
```

### Constant Folding

**Example**: `WHERE 10 + 5 > price`

**Optimization**: Evaluate `10 + 5` at planning time, not runtime

```cpp
// At planning time:
auto const_expr = evaluate_constant_expression(10 + 5);  // 15
// Rewrite to: WHERE 15 > price
```

### Common Subexpression Elimination

**Example**:
```sql
WHERE (price * quantity) > 1000 AND (price * quantity) < 10000
```

**Optimization**: Compute `price * quantity` once

```cpp
auto price_times_qty = price * quantity;  // Compute once
auto mask = (price_times_qty > 1000) AND (price_times_qty < 10000);
```

---

## Debugging

### Expression Tracing

```cpp
std::unique_ptr<cudf::column> ExpressionExecutor::evaluate(
    const cudf::table_view& table,
    const Expression& expr
) {
    LOG_TRACE("Evaluating expression: {}", expr.to_string());

    auto start = std::chrono::steady_clock::now();

    auto result = evaluate_impl(table, expr);

    auto end = std::chrono::steady_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start
    ).count();

    LOG_TRACE("Expression evaluated in {} μs, result rows: {}",
              duration_us, result->size());

    return result;
}
```

**Log Output**:
```
[TRACE] Evaluating expression: (price > 100)
[TRACE] Expression evaluated in 45 μs, result rows: 100000
[TRACE] Evaluating expression: (category = 'Electronics')
[TRACE] Expression evaluated in 120 μs, result rows: 100000
[TRACE] Evaluating expression: ((price > 100) AND (category = 'Electronics'))
[TRACE] Expression evaluated in 30 μs, result rows: 100000
```

---

## See Also

- [Operators](../04-new-mode/operators.md) - Operator implementations
- [Planner](planner.md) - Expression conversion
- [Operator Guide](../04-new-mode/operator-guide.md) - Usage examples
- [New Data Flow](../06-data-flow/new-data-flow.md) - Data flow details
