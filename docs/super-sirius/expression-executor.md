# Expression Executor

This document covers the GPU expression execution subsystem used by FILTER and PROJECTION operators.

## Overview

**File:** `src/include/expression_executor/gpu_expression_executor.hpp`

`gpu_expression_executor` evaluates DuckDB expressions on the GPU. It provides two execution modes:

> **API boundary:** Operator headers and the executor's public header take expressions as `sirius::expression` / `sirius::join_condition` — opaque PIMPL wrappers around `duckdb::Expression` / `duckdb::JoinCondition` defined in `src/include/expression/`. Plan builders wrap at the DuckDB boundary (`sirius::wrap`); operator `.cpp` files unwrap internally via `expression/expression_internal.hpp` to access the raw DuckDB type. This keeps `duckdb/planner/expression/...` includes out of the operator surface.

| Method | Purpose | Used By |
|--------|---------|---------|
| `execute(batch)` | Projects: evaluates expressions and returns result columns with all rows | PROJECTION |
| `select(batch)` | Filters: evaluates a boolean expression and returns only rows that pass | FILTER |

Both methods accept a `data_batch` and return a new `data_batch` with the result. The `rmm::cuda_stream_view` and memory resource are passed to the constructor and stored as members — they are not per-call arguments.

## Execution Strategies

**File:** `src/include/expression_executor/expression_executor_strategy.hpp`

The executor supports three strategies, selected via the `strategy` constructor parameter (default from `duckdb::Config::EXPRESSION_EXECUTOR_STRATEGY`):

| Strategy | How it executes | cuDF API |
|----------|-----------------|----------|
| `MATERIALIZE` | Every expression node becomes a single kernel. Intermediate results are materialized as `cudf::column`s. | `cudf::unary_operation`, `cudf::binary_operation`, etc. (one per node) |
| `AST_INTERPRET` (default) | Builds a `cudf::ast::tree` and interprets it with a single monolithic kernel. | `cudf::compute_column` |
| `AST_JIT` | Builds a `cudf::ast::tree` and JIT-compiles it into a fused kernel. | `cudf::compute_column_jit` |

### Tree of AST Trees

Not every DuckDB expression has a cuDF AST equivalent — these are called **AST breakers** (e.g. `CASE`, `LIKE`, `SUBSTRING`, unsupported `CAST` types). For AST strategies, the executor walks the DuckDB expression and greedily builds AST subtrees up to each breaker. When it hits a breaker, it materializes that subtree as a `cudf::column`, stashes it internally, and references it from the enclosing AST subtree via a `cudf::ast::column_reference`.

The result is a tree of AST trees whose edges are AST breakers. Each AST tree is evaluated by `cudf::compute_column` (or `compute_column_jit`) in `execute_ast()`.

### `min_ast_size` — per-subtree mode selection

An AST subtree with only one operator gains little from AST execution and would pay the launch overhead of `compute_column`. The `min_ast_size` constructor parameter (default `2`) sets the threshold: before adding a subtree to the AST tree, the executor calls `count_ast_ops()` on it; if the count is below `min_ast_size`, the subtree is evaluated operator-by-operator in MATERIALIZE mode instead.

This means `MATERIALIZE` strategy is effectively `AST_INTERPRET` with `min_ast_size = ∞`.

### `execution_mode` (internal)

Internal to the executor, each node is evaluated with either `execution_mode::AST` or `execution_mode::MATERIALIZE`. This is a **hint** — if a node tagged AST turns out to be a breaker, it is evaluated in MATERIALIZE mode anyway (and wrapped via `materialize_as_ast_column()` so the parent still sees an AST reference).

### Setting the strategy

The strategy is a DuckDB SET variable registered in `src/sirius_extension.cpp`:

```sql
SET expression_executor_strategy = 'ast_jit';   -- or 'ast_interpret', 'materialize'
```

## Supported Expression Types

| Expression Type | Class | Example |
|----------------|-------|---------|
| Column reference | `BoundReferenceExpression` | `column #3` |
| Constant | `BoundConstantExpression` | `42`, `'hello'` |
| Comparison | `BoundComparisonExpression` | `a > b`, `x = 10`, `a IS NOT DISTINCT FROM b` |
| Conjunction | `BoundConjunctionExpression` | `a AND b`, `x OR y` |
| Arithmetic/logical | `BoundOperatorExpression` | `a + b`, `NOT x`, `COALESCE(a, b, 0)`, `x IN (1, 2, 3)` |
| Function call | `BoundFunctionExpression` | `UPPER(name)`, `YEAR(date)` |
| Type cast | `BoundCastExpression` | `CAST(x AS DOUBLE)` |
| CASE/WHEN | `BoundCaseExpression` | `CASE WHEN x > 0 THEN 'pos' ELSE 'neg' END` |
| BETWEEN | `BoundBetweenExpression` | `x BETWEEN 10 AND 20` |

`COALESCE` is materialized via `cudf::replace_nulls` iteratively across children: the first child is materialized (scalars are lifted to a column); each subsequent child replaces the residual nulls in the running result. Children are only evaluated when the running result still has nulls. The result is wrapped via `materialize_as_ast_column` so COALESCE composes inside AST-capable parents. `count_ast_ops` returns 0 — COALESCE always takes the materialize-only path.

`COMPARE_IN` covers the full numeric set (INT8–INT64, UINT8–UINT64, BOOL8, FLOAT, DOUBLE, DECIMAL32/64/128, all TIMESTAMP precisions, DATE) plus VARCHAR. BOOL8 dispatches via `uint8_t` because `std::vector<bool>::data()` is deleted.

Per-expression-type dispatch lives in `src/expression_executor/specializations/` (one file per expression class: `gpu_execute_comparison.cpp`, `gpu_execute_case.cpp`, etc.). Each specialization decides how to emit AST nodes, materialize, or fall back based on the effective execution mode.

### AST-Eligible Operations

The following translate directly into cuDF AST nodes:

| Category | Operations |
|----------|-----------|
| Arithmetic | `+`, `-`, `*`, `/`, `//`, `%` |
| Comparison | `=`, `!=`, `<`, `>`, `<=`, `>=`, `IS NOT DISTINCT FROM` (via `NULL_EQUALS`) |
| Logical | `AND`, `OR`, `NOT` |
| BETWEEN | Translated to `(val >= lower) AND (val <= upper)` |
| Casting | Fixed-width types: `UBIGINT`, `BIGINT`, `DOUBLE` (see `supported_ast_cast_types`) |

Anything outside this set is an AST breaker and forces materialization at that node.

## GPU Expression Translator

**File:** `src/include/expression_executor/gpu_expression_translator.hpp`

`gpu_expression_translator` is a **separate** utility that converts DuckDB expressions into standalone cuDF AST trees for operators that need compiled expression evaluation outside the executor — primarily mixed joins.

```cpp
struct translated_expression {
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> owned_literals;
};
```

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
| `src/expression_executor/gpu_expression_executor.cpp` | Driver: strategy dispatch, AST tree management, temp lifetimes |
| `src/expression_executor/specializations/gpu_execute_*.cpp` | Per-expression-class dispatch (comparison, case, function, …) |
| `src/include/expression_executor/expression_executor_strategy.hpp` | `expression_executor_strategy` enum + string conversions |
| `src/include/expression_executor/gpu_expression_translator.hpp` | Standalone DuckDB → cuDF AST translator (for mixed joins) |
| `src/expression_executor/gpu_expression_translator.cpp` | Translator implementation |
