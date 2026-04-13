# Expression Executor

This document covers the GPU expression execution subsystem used by FILTER and PROJECTION operators.

## Overview

**File:** `src/include/expression_executor/gpu_expression_executor.hpp`

`GpuExpressionExecutor` evaluates DuckDB expressions on the GPU. It provides two execution modes:

| Method | Purpose | Used By |
|--------|---------|---------|
| `execute(batch, stream)` | Projects: evaluates expressions and returns result columns with all rows | PROJECTION |
| `select(batch, stream)` | Filters: evaluates a boolean expression and returns only rows that pass | FILTER |

Both methods accept a `data_batch` and return a new `data_batch` with the result.

## Supported Expression Types

| Expression Type | Class | Example |
|----------------|-------|---------|
| Column reference | `BoundReferenceExpression` | `column #3` |
| Constant | `BoundConstantExpression` | `42`, `'hello'` |
| Comparison | `BoundComparisonExpression` | `a > b`, `x = 10`, `a IS NOT DISTINCT FROM b` |
| Conjunction | `BoundConjunctionExpression` | `a AND b`, `x OR y` |
| Arithmetic/logical | `BoundOperatorExpression` | `a + b`, `NOT x` |
| Function call | `BoundFunctionExpression` | `UPPER(name)`, `YEAR(date)` |
| Type cast | `BoundCastExpression` | `CAST(x AS DOUBLE)` |
| CASE/WHEN | `BoundCaseExpression` | `CASE WHEN x > 0 THEN 'pos' ELSE 'neg' END` |
| BETWEEN | `BoundBetweenExpression` | `x BETWEEN 10 AND 20` |

## State Management

**File:** `src/include/expression_executor/gpu_expression_executor_state.hpp`

Expression evaluation maintains a state tree that mirrors the expression tree:

```
GpuExpressionExecutorState
└── GpuExpressionState (root)
    ├── GpuExpressionState (child 0)
    │   └── GpuExpressionState (leaf)
    └── GpuExpressionState (child 1)
        └── GpuExpressionState (leaf)
```

Each `GpuExpressionState` holds:
- Reference to the `Expression` being evaluated
- `child_states` — recursively mirrors expression tree structure
- `types` — cuDF data types for child results

States are initialized once and reused across batches within the same operator.

## GPU Expression Translator

**File:** `src/include/expression_executor/gpu_expression_translator.hpp`

The `gpu_expression_translator` converts DuckDB expressions into cuDF AST trees for operators that need compiled expression evaluation (primarily mixed joins).

```cpp
struct translated_expression {
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> owned_literals;
};
```

### Supported Translations

| Category | Operations |
|----------|-----------|
| Arithmetic | `+`, `-`, `*`, `/` |
| Comparison | `=`, `!=`, `<`, `>`, `<=`, `>=` |
| Logical | `AND`, `OR` |
| BETWEEN | Translated to `(val >= lower) AND (val <= upper)` |
| Casting | Fixed-width types (INT, FLOAT, DOUBLE) |
| Column references | With LEFT/RIGHT table reference tracking |

### IS NOT DISTINCT FROM

`IS NOT DISTINCT FROM` is supported in GPU comparison execution via `cudf::binary_operator::NULL_EQUALS`, which treats NULLs as equal (unlike standard comparisons where NULL comparisons yield NULL). This enables full GPU execution for IS NOT DISTINCT FROM predicates.

### Unsupported Translations

These return `nullopt`, causing the caller to fall back to row-by-row evaluation:
- CASE expressions
- COALESCE, TRY
- CAST with non-fixed-width types (e.g., VARCHAR)
- Parameter expressions
- DISTINCT operators (IS DISTINCT FROM throws `NotImplementedException`)

### Join Condition Translation

The translator provides specialized methods for join conditions:

- `translate_join_condition(condition)` — translates a single equality or inequality condition
- `translate_join_conditions(conditions, start, end, swap_sides)` — combines multiple conditions with AND, optionally swapping LEFT/RIGHT table references for RIGHT/OUTER joins

This is used by `sirius_physical_hash_join` in MIXED_JOIN mode to pass inequality conditions to `cudf::mixed_join()` as a cuDF AST expression.

## Key Files

| File | Purpose |
|------|---------|
| `src/include/expression_executor/gpu_expression_executor.hpp` | Main executor class |
| `src/expression_executor/gpu_expression_executor.cpp` | Implementation |
| `src/include/expression_executor/gpu_expression_executor_state.hpp` | State tree |
| `src/include/expression_executor/gpu_expression_translator.hpp` | DuckDB → cuDF AST |
| `src/expression_executor/gpu_expression_translator.cpp` | Translator implementation |
