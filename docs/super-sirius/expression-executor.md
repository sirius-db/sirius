# Expression Executor

This document covers the GPU expression execution subsystem used by FILTER and PROJECTION operators.

## Overview

**File:** `src/include/expression_executor/gpu_expression_executor.hpp`

`gpu_expression_executor` evaluates expressions on the GPU using the Sirius AST type hierarchy (see [Sirius AST Type Hierarchy](#sirius-ast-type-hierarchy)). It provides two execution modes:

> **API boundary:** `sirius::ast::node` is the single expression currency at every operator, planner, and executor boundary. Operators own their expressions directly as `std::unique_ptr<sirius::ast::node>` (e.g. a projection's `select_list`, a table scan's `filter_expr`), and `sirius::join_condition` holds its `left`/`right` sides as `std::unique_ptr<sirius::ast::node>`. DuckDB expressions are translated to Sirius AST once, at the planner/scan boundary, via `sirius::ast::from_duckdb(duckdb::Expression const&)`. There is no wrapper type between DuckDB and Sirius's expression IR, so neither `duckdb/planner/expression/...` nor an opaque handle appears on the operator surface — only `sirius::ast::node`.

| Method | Purpose | Used By |
|--------|---------|---------|
| `execute(input)` | Projects: evaluates expressions and returns result columns with all rows | PROJECTION |
| `select(input)` | Filters: evaluates a boolean expression and returns only rows that pass | FILTER |

Both methods accept a `cudf::table_view` and return a new `cudf::table` with the result. The `rmm::cuda_stream_view` and memory resource are passed to the constructor and stored as members — they are not per-call arguments.

The executor can be constructed from a `duckdb::vector<std::unique_ptr<sirius::ast::node>>` (the full operator expression list), from a single `sirius::ast::node`, from a non-owning `sirius::ast::node const*`, or from a non-owning `std::vector<sirius::ast::node const*>`. The PROJECTION operator uses the non-owning vector form to pass only the entries that actually need evaluation, after pulling out pure BOUND_REF passthroughs that it exposes as zero-copy views (see [operators](operators.md)). `execute()` returns one output column per supplied expression, in order. If a slot is ever null (an unsupported expression that `from_duckdb` could not translate), the executor throws `InternalException` rather than dereferencing it.

## Sirius AST Type Hierarchy

**Files:** `src/include/expression/ast/node.hpp`, `src/include/expression/ast/*.hpp`

`sirius::ast::node` is a `std::variant`-based sum type over all Sirius expression node kinds. It is the sole expression representation passed across operator, planner, and executor boundaries, and the type the executor dispatches on via `std::visit`.

```cpp
struct node {
  using variant_t = std::variant<reference,
                                 constant,
                                 comparison,
                                 conjunction,
                                 between,
                                 case_expr,
                                 cast,
                                 unary_op,
                                 coalesce,
                                 in_list,
                                 function_call,
                                 aggregate>;
  variant_t v;
};
```

The alternative order is part of the ABI: `std::variant` indexes by position and downstream dispatch depends on it, so new alternatives are appended at the end. `node` is move-only. Children are stored as `std::unique_ptr<node>` inside each alternative struct, making the tree recursive without incomplete-type issues. Every alternative must implement `cudf_ast_op_count() const`, enforced by a `static_assert` at the variant declaration.

| Alternative | Sirius type | Typical source |
|-------------|-------------|----------------|
| `reference` | `sirius::ast::reference` | Column reference (`BoundReferenceExpression`) |
| `constant` | `sirius::ast::constant` | Literal value (`BoundConstantExpression`) — payload stored as `sirius::value` |
| `comparison` | `sirius::ast::comparison` | `=`, `!=`, `<`, `>`, `<=`, `>=`, `IS NOT DISTINCT FROM` |
| `conjunction` | `sirius::ast::conjunction` | `AND`, `OR` |
| `between` | `sirius::ast::between` | `BETWEEN … AND …` |
| `case_expr` | `sirius::ast::case_expr` | `CASE WHEN … THEN … ELSE … END` |
| `cast` | `sirius::ast::cast` | `CAST(x AS T)` |
| `unary_op` | `sirius::ast::unary_op` | `NOT x`, `-x`, arithmetic binary ops (`+`, `-`, `*`, `/`) |
| `coalesce` | `sirius::ast::coalesce` | `COALESCE(a, b, 0)` |
| `in_list` | `sirius::ast::in_list` | `x IN (1, 2, 3)` |
| `function_call` | `sirius::ast::function_call` | Named function (`YEAR`, `UPPER`, `concat`, etc.) — `sirius::function_id` enum |
| `aggregate` | `sirius::ast::aggregate` | Aggregate function (`SUM`, `COUNT`, etc.) — `sirius::aggregate_id` enum |

**Translation boundary:** `sirius::ast::from_duckdb(expr)` produces a `sirius::ast::node` from a `duckdb::Expression`. This happens once at plan time in the plan builders (and at the scan boundary for pushdown filters); the resulting node is owned by the operator and never re-translated.

**Key files:**

| File | Purpose |
|------|---------|
| `src/include/expression/ast/node.hpp` | `sirius::ast::node` variant definition |
| `src/include/expression/ast/from_duckdb.hpp` | `sirius::ast::from_duckdb` — DuckDB → Sirius AST translator |
| `src/include/expression/ast/utils.hpp` | AST tree utilities — `visit_references`, `clone`, `substitute_references` |
| `src/include/expression/value.hpp` | `sirius::value` — typed constant payload (INT8–DECIMAL128, VARCHAR, TIMESTAMP, …) |
| `src/include/expression/function_id.hpp` | `sirius::function_id` closed enum of supported functions |
| `src/include/expression/aggregate_id.hpp` | `sirius::aggregate_id` closed enum of supported aggregates |
| `src/include/expression/join_condition.hpp` | `sirius::join_condition` — `{left, right, comparison}` with AST-node sides |

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

| Expression Type | Sirius AST alternative | Example |
|----------------|------------------------|---------|
| Column reference | `sirius::ast::reference` | `column #3` |
| Constant | `sirius::ast::constant` | `42`, `'hello'` |
| Comparison | `sirius::ast::comparison` | `a > b`, `x = 10`, `a IS NOT DISTINCT FROM b` |
| Conjunction | `sirius::ast::conjunction` | `a AND b`, `x OR y` |
| Arithmetic / unary | `sirius::ast::unary_op` | `a + b`, `NOT x`, `-x` |
| COALESCE | `sirius::ast::coalesce` | `COALESCE(a, b, 0)` |
| IN-list | `sirius::ast::in_list` | `x IN (1, 2, 3)` |
| Function call | `sirius::ast::function_call` | `UPPER(name)`, `YEAR(date)`, `a \|\| b` |
| Type cast | `sirius::ast::cast` | `CAST(x AS DOUBLE)` |
| CASE/WHEN | `sirius::ast::case_expr` | `CASE WHEN x > 0 THEN 'pos' ELSE 'neg' END` |
| BETWEEN | `sirius::ast::between` | `x BETWEEN 10 AND 20` |
| Aggregate | `sirius::ast::aggregate` | `SUM(x)`, `COUNT(*)` |

`coalesce` is materialized via `cudf::replace_nulls` iteratively across children: the first child is materialized (scalars are lifted to a column); each subsequent child replaces the residual nulls in the running result. Children are only evaluated when the running result still has nulls. The result is wrapped via `materialize_as_ast_column` so COALESCE composes inside AST-capable parents. `cudf_ast_op_count` returns 0 — `coalesce` always takes the materialize-only path.

`in_list` covers the full numeric set (INT8–INT64, UINT8–UINT64, BOOL8, FLOAT, DOUBLE, DECIMAL32/64/128, all TIMESTAMP precisions, DATE) plus VARCHAR. BOOL8 dispatches via `uint8_t` because `std::vector<bool>::data()` is deleted.

### String concatenation

String concatenation — both the `||` operator (`col || ' suffix'`) and the `concat(a, b, …)` function — resolves to `function_id::concat` (DuckDB lowers `||` to a function named `"||"`, which maps to the same id as `"concat"`). It is dispatched as a materialize-only function in `gpu_execute_function.cpp`: every argument is materialized, any scalar argument is broadcast to a full-length column via `cudf::make_column_from_scalar`, and the columns are joined with `cudf::strings::concatenate` using an empty separator. The narep scalar is invalid (null), giving standard SQL null propagation — any NULL input produces a NULL output.

Per-expression-type dispatch lives in `src/expression_executor/specializations/` (one file per Sirius AST alternative: `gpu_execute_comparison.cpp`, `gpu_execute_case.cpp`, etc.). Each specialization decides how to emit cuDF AST nodes, materialize, or fall back based on the effective execution mode.

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

**File:** `src/include/expression_executor/gpu_expression_translator_internal.hpp`

`gpu_expression_translator` is a **separate** utility that converts Sirius AST expressions into standalone cuDF AST trees for operators that need compiled expression evaluation outside the executor — primarily mixed joins and parquet filter pushdown.

```cpp
struct translated_expression {
    cudf::ast::tree tree;
    std::optional<rmm::cuda_stream> owned_stream;
    std::vector<std::unique_ptr<cudf::scalar>> owned_literals;
};
```

The primary entry point takes a `sirius::ast::node const&`:

```cpp
std::optional<translated_expression> translate_expression(
    sirius::ast::node const& expr,
    cudf::ast::table_reference table_src = cudf::ast::table_reference::LEFT);
```

A second overload, `translate_expression_with_names`, produces `cudf::ast::column_name_reference` nodes instead of index-based references — used for cuDF parquet predicate pushdown.

### Unsupported Translations

These return `nullopt`, causing the caller to fall back to row-by-row evaluation:
- `case_expr` nodes
- `coalesce` nodes and `TRY`
- `cast` with non-fixed-width target types (e.g., VARCHAR)
- `IS DISTINCT FROM` (throws `NotImplementedException`)

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
| `src/expression_executor/specializations/gpu_execute_*.cpp` | Per-Sirius-AST-alternative dispatch (comparison, case, function, …) |
| `src/include/expression_executor/expression_executor_strategy.hpp` | `expression_executor_strategy` enum + string conversions |
| `src/include/expression_executor/ast_supported_types.hpp` | AST-eligible cast targets and functions (`supported_ast_cast_types`, `supported_ast_functions`) |
| `src/include/expression_executor/gpu_expression_translator_internal.hpp` | Sirius AST → cuDF AST translator (mixed joins, parquet pushdown) |
| `src/expression_executor/gpu_expression_translator.cpp` | Translator implementation |
| `src/include/expression/ast/node.hpp` | `sirius::ast::node` variant; per-alternative headers included from here |
| `src/include/expression/ast/from_duckdb.hpp` | `sirius::ast::from_duckdb` — DuckDB → Sirius AST translation |
| `src/include/expression/ast/utils.hpp` | AST tree utilities — `visit_references`, `clone`, `substitute_references` |
| `src/include/expression/join_condition.hpp` | `sirius::join_condition` — AST-node sides + comparison operator |
