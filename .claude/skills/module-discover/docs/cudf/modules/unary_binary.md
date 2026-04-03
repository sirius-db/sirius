# Unary & Binary Operations

**Status**: USED
**Path**: `cudf/unary.hpp`, `cudf/binaryop.hpp`
**Headers we include**: `cudf/unary.hpp`, `cudf/binaryop.hpp`

## Summary

The most frequently called cuDF APIs in Sirius. Every SQL expression involving arithmetic, comparison, or logical operations translates to `binary_operation` or `unary_operation`. Type casting via `cudf::cast` is also in this module.

## API Reference

### `cudf::binary_operation`

**Header**: `cudf/binaryop.hpp`
```cpp
// Column op Column
std::unique_ptr<column> binary_operation(column_view const& lhs, column_view const& rhs,
                                          binary_operator op, data_type output_type, ...);
// Scalar op Column
std::unique_ptr<column> binary_operation(scalar const& lhs, column_view const& rhs,
                                          binary_operator op, data_type output_type, ...);
// Column op Scalar
std::unique_ptr<column> binary_operation(column_view const& lhs, scalar const& rhs,
                                          binary_operator op, data_type output_type, ...);
```

**Description**: Element-wise binary operation producing a new column.

**Our usage** (40+ call sites):
- `src/expression_executor/specializations/gpu_execute_operator.cpp` — Arithmetic ops (+, -, *, /, %)
- `src/expression_executor/specializations/gpu_execute_comparison.cpp` — Comparison ops (<, >, =, etc.)
- `src/expression_executor/specializations/gpu_execute_between.cpp` — BETWEEN as two comparisons + AND
- `src/expression_executor/specializations/gpu_execute_conjunction.cpp` — AND/OR logical ops
- `src/expression_executor/specializations/gpu_execute_function.cpp` — Date arithmetic

### `cudf::binary_operator` (enum)

**Header**: `cudf/binaryop.hpp`

**Values used by Sirius**:
- Arithmetic: `ADD`, `SUB`, `MUL`, `DIV`, `TRUE_DIV`, `FLOOR_DIV`, `MOD`, `PYMOD`
- Comparison: `EQUAL`, `NOT_EQUAL`, `LESS`, `GREATER`, `LESS_EQUAL`, `GREATER_EQUAL`, `NULL_EQUALS`
- Logical: `LOGICAL_AND`, `LOGICAL_OR`
- Bitwise: `BITWISE_AND`, `BITWISE_OR`

### `cudf::unary_operation`

**Header**: `cudf/unary.hpp`
```cpp
std::unique_ptr<column> unary_operation(column_view const& input, unary_operator op, ...);
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_operator.cpp` — NOT, ABS, negation
- `src/cuda/expression_executor/gpu_dispatch_string.cu` — IS_NULL checks

### `cudf::cast`

**Header**: `cudf/unary.hpp`
```cpp
std::unique_ptr<column> cast(column_view const& input, data_type out_type, ...);
```

**Description**: Type casting between cuDF data types.

**Our usage** (10+ call sites):
- `src/expression_executor/specializations/gpu_execute_cast.cpp:20` — SQL CAST expressions
- `src/operator/gpu_physical_ungrouped_aggregate.cpp` — Cast before aggregation
- `src/op/sirius_physical_hash_join.cpp:25` — Type alignment before join
- `src/op/partition/gpu_partition_impl.cpp:22` — Cast for partitioning

### `cudf::is_null` / `cudf::is_valid`

**Header**: `cudf/unary.hpp`
```cpp
std::unique_ptr<column> is_null(column_view const& input, ...);
std::unique_ptr<column> is_valid(column_view const& input, ...);
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_operator.cpp` — IS NULL / IS NOT NULL

### `cudf::replace_nulls`

**Header**: `cudf/replace.hpp` (also accessible via unary patterns)
```cpp
std::unique_ptr<column> replace_nulls(column_view const& input, scalar const& replacement, ...);
std::unique_ptr<column> replace_nulls(column_view const& input, column_view const& replacement, ...);
```

**Our usage**:
- `src/operator/gpu_physical_table_scan.cpp` — Replace nulls in boolean filter masks

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::unary_operator::SIN/COS/TAN/...` | `unary.hpp` | Trigonometric functions |
| `cudf::unary_operator::EXP/LOG/SQRT` | `unary.hpp` | Mathematical functions |
| `cudf::binary_operator::POW` | `binaryop.hpp` | Exponentiation |
| `cudf::binary_operator::LOG_BASE` | `binaryop.hpp` | Logarithm with base |
| `cudf::binary_operator::NULL_MIN/NULL_MAX` | `binaryop.hpp` | Null-aware min/max |
