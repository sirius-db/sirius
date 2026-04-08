# AST (Abstract Syntax Tree)

**Status**: USED
**Path**: `cudf/ast/`
**Headers we include**: `cudf/ast/expressions.hpp`, `cudf/ast/ast_operator.hpp`

## Summary

Sirius translates DuckDB expression trees to cuDF AST for conditional joins and expression evaluation. The AST is used as input to `conditional_join` and `mixed_join` operations, enabling complex non-equi join predicates on GPU.

## API Reference

### `cudf::ast::tree`

**Header**: `cudf/ast/expressions.hpp`
```cpp
class tree {
    template <typename T, typename... Args>
    T& emplace(Args&&... args);  // Construct expression node in tree
    expression& back();           // Last added expression
    expression& front();          // First expression
};
```

**Our usage**:
- `src/expression_executor/gpu_expression_translator.cpp` — Building AST trees from DuckDB expressions

### `cudf::ast::expression` (base class)

**Header**: `cudf/ast/expressions.hpp`

Subclasses:
- `cudf::ast::operation` — Binary/unary operation node
- `cudf::ast::literal` — Constant value node
- `cudf::ast::column_reference` — Reference to table column

### `cudf::ast::operation`

**Header**: `cudf/ast/expressions.hpp`
```cpp
class operation : public expression {
    operation(ast_operator op, expression const& input);           // Unary
    operation(ast_operator op, expression const& lhs, expression const& rhs); // Binary
};
```

**Our usage**:
- `src/expression_executor/gpu_expression_translator.cpp` — Constructing comparison and logical operations

### `cudf::ast::column_reference`

**Header**: `cudf/ast/expressions.hpp`
```cpp
class column_reference : public expression {
    column_reference(cudf::size_type column_index, table_reference table_source = table_reference::LEFT);
};
```

**Our usage**:
- `src/expression_executor/gpu_expression_translator.cpp` — Referencing left/right table columns in join conditions
- `src/op/sirius_physical_nested_loop_join.cpp` — Column references for conditional join predicates

### `cudf::ast::table_reference` (enum)

**Header**: `cudf/ast/expressions.hpp`
**Values**: `LEFT`, `RIGHT`

**Our usage**:
- `src/expression_executor/gpu_expression_translator.cpp` — Distinguishing left vs right table columns in join ASTs

### `cudf::ast::ast_operator` (enum)

**Header**: `cudf/ast/ast_operator.hpp`

**Values used by Sirius**:
- Comparison: `EQUAL`, `NOT_EQUAL`, `LESS`, `GREATER`, `LESS_EQUAL`, `GREATER_EQUAL`
- Logical: `LOGICAL_AND`, `LOGICAL_OR`, `NOT`
- Arithmetic: `ADD`, `SUB`, `MUL`, `DIV`, `MOD`
- Null: `IS_NULL`
- Cast: `CAST_TO_INT64`, `CAST_TO_UINT64`, `CAST_TO_FLOAT64`

**Our usage**:
- `src/include/expression_executor/gpu_expression_translator.hpp:36` — Operator mapping from DuckDB to cuDF

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::ast::literal` with device scalar | `expressions.hpp` | Literal values in AST (Sirius uses column_reference pattern instead) |
| `cudf::compute_column()` with AST | `transform.hpp` | Evaluate AST expression on a table to produce column |
