# Transform

**Status**: USED
**Path**: `cudf/transform.hpp`
**Headers we include**: `cudf/transform.hpp`

## Summary

Element-wise transformation using cuDF AST expressions. Used in expression evaluation and testing.

## API Reference

### `cudf::compute_column`

**Header**: `cudf/transform.hpp`
```cpp
std::unique_ptr<column> compute_column(table_view const& table,
                                        ast::expression const& expr, ...);
```

**Description**: Evaluates an AST expression against a table, producing a result column.

**Our usage**:
- `src/include/expression_executor/regex/regex_playground.hpp:21` — Expression evaluation
- `test/cpp/expression_executor/test_gpu_expression_translator.cpp:31` — Testing AST translation correctness
