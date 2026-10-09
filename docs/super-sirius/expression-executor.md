# Expression Evaluator

This document covers GPU expression evaluation for FILTER, PROJECTION, and join operands.

## Overview

**File:** `src/expression_evaluator/expression_evaluator.hpp`

`expression_evaluator` evaluates expressions on the GPU using the Sirius AST type hierarchy (see [Sirius AST Type Hierarchy](#sirius-ast-type-hierarchy)). It provides two public table operations:

> **API boundary:** `sirius::ast::node` is the single expression currency at every operator, planner, and evaluator boundary. Operators own their expressions directly as `std::unique_ptr<sirius::ast::node>` (e.g. a projection's `select_list`, a table scan's `filter_expr`), and `sirius::join_condition` holds its `left`/`right` sides as `std::unique_ptr<sirius::ast::node>`. DuckDB expressions are translated to Sirius AST once, at the planner/scan boundary, via `sirius::ast::from_duckdb(duckdb::Expression const&)`. There is no wrapper type between DuckDB and Sirius's expression IR, so neither `duckdb/planner/expression/...` nor an opaque handle appears on the operator surface — only `sirius::ast::node`.

| Method | Purpose | Used By |
|--------|---------|---------|
| `evaluate(input)` | Projects: evaluates expressions and returns result columns with all rows | PROJECTION |
| `select(input)` | Filters: evaluates a boolean expression and returns only rows that pass | FILTER |

Both methods accept a `cudf::table_view` and return a new `cudf::table` with the result. The `rmm::cuda_stream_view` and memory resource are passed to the constructor and stored as members — they are not per-call arguments.

The evaluator can be constructed from a `duckdb::vector<std::unique_ptr<sirius::ast::node>>` (the full operator expression list), from a single `sirius::ast::node`, from a non-owning `sirius::ast::node const*`, or from a non-owning `std::vector<sirius::ast::node const*>`. The PROJECTION operator uses the non-owning vector form to pass only the entries that actually need evaluation, after pulling out pure BOUND_REF passthroughs that it exposes as zero-copy views (see [operators](operators.md)). `evaluate()` returns one output column per supplied expression, in order. Unsupported expressions never reach the evaluator: the plan builders reject any expression that `from_duckdb` cannot translate during GPU plan construction (throwing `NotImplementedException`, which triggers the transparent CPU fallback — see [Physical Plan Generation](physical-plan-generation.md)). A null slot in the evaluator's expression list therefore indicates a planner bug, and the evaluator throws `sirius::internal_exception` as a defensive backstop rather than dereferencing it.

## Native expression consumers

Operators inspect and evaluate the Sirius AST directly. `node::return_type()`
provides logical types, and `ast::visit_references` visits references through all
child expressions, including casts. Reference collection preserves column indices;
consumers deduplicate them within each input's index space. Hash joins use these
sets for mixed-join reference-overlap checks and residual validity columns.
A direct-reference check is stricter than traversal: finding a reference beneath
a cast does not make the expression eligible for direct-column dynamic filtering.

DuckDB translation is an input boundary, not an execution dependency. CPU fallback
uses separately preserved DuckDB planning state and a prepared CPU plan (see
[Physical Plan Generation](physical-plan-generation.md)); it does not reconstruct
DuckDB expressions from the Sirius AST. Type, value, and comparison adapters remain
available at interfaces that require DuckDB types.

### Shared hash-join key preparation

`materialize_expression_join_keys` in
[`sirius_physical_hash_join.cpp`](../../src/op/sirius_physical_hash_join.cpp)
prepares join operands in child projections. Direct references must match their
input schema. Each operation, including a cast of a reference, becomes a hidden
projection column evaluated by the native evaluator; its condition operand becomes
a reference to that column. Projection maps keep these hidden columns out of the
visible result. Both the comparison-join planner and the hash-join constructor use
this preparation boundary, so direct native construction has the same contract.

The join publishes prepared key descriptors containing left and right column
indices, a common cuDF type, and whether the condition is a hash equality key.
Operands with different physical types are rejected. Partitioning consumes these
descriptors rather than interpreting expressions independently. It uses all prepared
equality keys, including null-safe keys routed to the mixed-join predicate; the join
hash table uses the descriptors marked as hash keys. Predicate routing retains the
distinction between ordinary equality and `IS NOT DISTINCT FROM`.

Both partitioning and hash joining use
[`restore_prepared_join_key`](../../src/op/join_key_preparation.hpp) to restore
narrowed carriers to the prepared type before hashing. This shared representation
is a correctness requirement: equal numeric values in different physical types can
hash differently, sending matching rows to different partitions. Carrier restoration
is distinct from evaluating a SQL cast; see
[Compressed Materialization](compressed-materialization.md).

### Nested-loop operands and conversion semantics

[`sirius_physical_nested_loop_join.cpp`](../../src/op/sirius_physical_nested_loop_join.cpp)
reuses a direct input column only when its physical type matches the reference's
logical type mapped to cuDF. Other operands, including references to narrowed
carriers, pass through the native evaluator on the join's stream and memory resource.
Evaluated tables and cuDF AST operand storage remain alive for the join call.

A cast is an operation with a target type, `try_cast`, and `cast_kind`, not a
transparent reference wrapper. Native evaluation preserves nested conversions and
the distinction between semantic casts and carrier restoration. In particular,
semantic timestamp conversions use the checked conversion behavior described under
[Temporal semantics](#temporal-semantics).

Each condition evaluates its own operands. Repeated expressions can therefore incur
repeated evaluation and allocation. Sharing results requires proven expression
identity or equivalence; a hash value alone cannot establish that two operands have
the same meaning.

## Sirius AST Type Hierarchy

**Files:** `src/expression/ast/node.hpp`, `src/expression/ast/*.hpp`

`sirius::ast::node` is a `std::variant`-based sum type over all Sirius expression node kinds. It is the sole expression representation passed across operator, planner, and evaluator boundaries, and the type the evaluator dispatches on via `std::visit`.

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

The alternative order is part of the ABI: `std::variant` indexes by position and downstream dispatch depends on it, so new alternatives are appended at the end. `node` is move-only. Children are stored as `std::unique_ptr<node>` inside each alternative struct, making the tree recursive without incomplete-type issues. Every alternative must implement `cudf_ast_op_count() const` and `return_type() const` (returning `sirius::logical_type`), both enforced by concepts and `static_assert`s at the variant declaration. `node::return_type()` dispatches to the active alternative via `std::visit`, so the evaluator reads each expression's result type natively from the Sirius AST — the output cuDF type in `post_process` and the BOOLEAN assertion on the filter path both come from `return_type()`, with no round-trip through DuckDB types.

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
| `src/expression/ast/node.hpp` | `sirius::ast::node` variant definition |
| `src/expression/ast/from_duckdb.hpp` | `sirius::ast::from_duckdb` — DuckDB → Sirius AST translator |
| `src/expression/ast/utils.hpp` | AST tree utilities — `visit_references`, `clone`, `substitute_references` |
| `src/expression/value.hpp` | `sirius::value` — typed constant payload (INT8–DECIMAL128, VARCHAR, TIMESTAMP, …) |
| `src/expression/function_id.hpp` | `sirius::function_id` closed enum of supported functions |
| `src/expression/aggregate_id.hpp` | `sirius::aggregate_id` closed enum of supported aggregates |
| `src/expression/join_condition.hpp` | `sirius::join_condition` — `{left, right, comparison}` with AST-node sides |

## Execution Strategies

**File:** `src/expression_evaluator/expression_evaluator_strategy.hpp`

The evaluator supports three strategies, selected via the `strategy` constructor parameter (default from `duckdb::Config::EXPRESSION_EVALUATOR_STRATEGY`):

| Strategy | How it executes | cuDF API |
|----------|-----------------|----------|
| `MATERIALIZE` | Every expression node becomes a single kernel. Intermediate results are materialized as `cudf::column`s. | `cudf::unary_operation`, `cudf::binary_operation`, etc. (one per node) |
| `AST_INTERPRET` (default) | Builds a `cudf::ast::tree` and interprets it with a single monolithic kernel. | `cudf::compute_column` |
| `AST_JIT` | Builds a `cudf::ast::tree` and JIT-compiles it into a fused kernel. | `cudf::compute_column_jit` |

### Tree of AST Trees

Not every DuckDB expression has a cuDF AST equivalent — these are called **AST breakers** (e.g. `CASE`, `LIKE`, `SUBSTRING`, unsupported `CAST` types). For AST strategies, the evaluator walks the Sirius AST and greedily builds AST subtrees up to each breaker. When it hits a breaker, it materializes that subtree as a `cudf::column`, stashes it internally, and references it from the enclosing AST subtree via a `cudf::ast::column_reference`.

The result is a tree of AST trees whose edges are AST breakers. Each AST tree is evaluated by `cudf::compute_column` (or `compute_column_jit`) in `evaluate_ast()`.

### `min_ast_size` — per-subtree mode selection

An AST subtree with only one operator gains little from AST execution and would pay the launch overhead of `compute_column`. The `min_ast_size` constructor parameter (default `2`) sets the threshold: before adding a subtree to the AST tree, the evaluator calls `cudf_ast_op_count()` on it; if the count is below `min_ast_size`, the subtree is evaluated operator-by-operator in MATERIALIZE mode instead.

This means `MATERIALIZE` strategy is effectively `AST_INTERPRET` with `min_ast_size = ∞`.

### `evaluation_mode` (internal)

Internal to the evaluator, each node is evaluated with either `evaluation_mode::AST` or `evaluation_mode::MATERIALIZE`. This is a **hint** — if a node tagged AST turns out to be a breaker, it is evaluated in MATERIALIZE mode anyway (and wrapped via `materialize_as_ast_column()` so the parent still sees an AST reference).

### Setting the strategy

The strategy is a DuckDB SET variable registered in `src/sirius_extension.cpp`:

```sql
SET expression_evaluator_strategy = 'ast_jit';   -- or 'ast_interpret', 'materialize'
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

String concatenation covers the `concat(a, b, …)` function and the `||` operator, which have **different** NULL semantics and therefore resolve to **distinct** ids: `concat()` ignores NULL arguments (`concat(NULL, 'x') = 'x'`) and maps to `function_id::concat`; the `||` operator propagates NULL (`'a' || NULL = NULL`) and maps to `function_id::concat_operator` (DuckDB lowers `||` to a function named `"||"`). Both are dispatched as a materialize-only function in `src/expression_evaluator/specializations/function.cpp`: every argument is materialized, any scalar argument is broadcast to a full-length column via `cudf::make_column_from_scalar`, and the columns are joined with `cudf::strings::concatenate` using an empty separator. The two semantics are selected by the narep scalar: a valid empty string for `concat()` (NULL → `""`, ignored) and an invalid narep for `||` (NULL → NULL, propagated).

### String case conversion

`upper()` and `lower()` materialize their input and call `cudf::strings::to_upper` and
`cudf::strings::to_lower`, respectively. Like `strlen()` and `length()`, these handlers
require column inputs and rely on DuckDB to constant-fold scalar calls before GPU evaluation.
Scalar inputs that survive constant folding are not yet supported.
NULLs propagate and empty strings remain empty. These functions are AST breakers and can
compose inside projections, filters, and materialized join-condition operands.

Case conversion follows cuDF's Unicode semantics, which differ from DuckDB
for some inputs. For example `upper('ß')` produces `SS`, and `lower('İ')`
produces `i` followed by combining dot U+0307. In contrast DuckDB produces
`ẞ` and `i`. So cuDF round-trips `upper(lower('İ'))` as the identity, where
DuckDB produces `upper(lower('İ')) == I`, in contrast cuDF produces
`lower(upper('ß')) == ss` whereas DuckDB produces `ß` for the round-trip.
See cuDF's [Unicode
limitations](https://docs.rapids.ai/api/libcudf/stable/md_doxygen_unicode)
for its supported code points and context-sensitive conversion limitations.
### Temporal semantics

`src/helper/timestamp_semantics.hpp` defines reusable GPU helpers in `sirius::temporal`.
The current contract matches DuckDB: signed epoch ticks reserve exactly `+MAX` and
`-MAX` for infinity, while `MIN` remains finite. NULLs propagate. Frontends must
normalize their temporal values to this representation before using these helpers;
the helpers do not depend on DuckDB types or a frontend identity.

`cast_to_microseconds_checked()` converts second, millisecond, and nanosecond
columns to microseconds. It checks finite values before multiplying, preserves
infinity, and truncates nanoseconds toward zero. Overflow raises
`sirius::invalid_input_exception`; reading the overflow result synchronizes the
supplied stream. The cast evaluator dispatches to this helper for semantic casts.

`finite_mask()` returns a nullable boolean column for DATE or any timestamp
precision. Both checked conversion and millisecond/microsecond extraction use it,
so sentinel handling has one implementation. The evaluator boundary regressions
are tagged `[timestamp_bounds]`; SQL extraction comparisons against DuckDB are
tagged `[timestamp_extraction]`.

### Rounding

`round(x)` and `round(x, n)` on FLOAT and DOUBLE map to `function_id::round` and run in
`round_floating_point` (`src/cuda/sirius_round_floating_point.cu`), one elementwise kernel that
reproduces DuckDB's `RoundOperatorPrecision`: the value is rounded in double precision as
`round(x * 10^n) / 10^n`, or `round(x / 10^-n) * 10^-n` for negative `n`, with ties away from
zero. A non-finite result yields the input for `n >= 0` and 0 for `n < 0`. The output is bit
identical to DuckDB. The precision must be a constant INTEGER.
DECIMAL and integer inputs and a column precision fall back to the CPU at plan time.

### Logical AND/OR NULL semantics

SQL `AND`/`OR` use Kleene three-valued logic (`TRUE OR NULL = TRUE`, `FALSE AND NULL = FALSE`), so conjunctions map to cuDF's Kleene operators — `NULL_LOGICAL_AND`/`NULL_LOGICAL_OR` as AST operators and `cudf::binary_operator::NULL_LOGICAL_*` on the materialize path — never the null-propagating `LOGICAL_*` variants. The plain `LOGICAL_AND` still appears for internal structural conjunctions where operands cannot be NULL, such as the BETWEEN lowering and the AND that combines multi-condition join predicates.

Per-expression-type dispatch lives in `src/expression_evaluator/specializations/` (one file per Sirius AST alternative: `comparison.cpp`, `case.cpp`, etc.). Each specialization decides how to emit cuDF AST nodes, materialize, or fall back based on the effective evaluation mode.

### AST-Eligible Operations

The following translate directly into cuDF AST nodes:

| Category | Operations |
|----------|-----------|
| Arithmetic | `+`, `-`, `*`, `/`, `//`, `%` |
| Comparison | `=`, `!=`, `<`, `>`, `<=`, `>=`, `IS NOT DISTINCT FROM` (via `NULL_EQUALS`) |
| Logical | `AND`, `OR` (Kleene `NULL_LOGICAL_AND`/`NULL_LOGICAL_OR`), `NOT` |
| BETWEEN | Translated to `(val >= lower) AND (val <= upper)` |
| Casting | Fixed-width types: `UBIGINT`, `BIGINT`, `DOUBLE` (see `supported_ast_cast_types`) |

Anything outside this set is an AST breaker and forces materialization at that node.

## GPU Expression Translator

**File:** `src/expression_evaluator/gpu_expression_translator_internal.hpp`

`gpu_expression_translator` is a **separate** utility that converts Sirius AST expressions into standalone cuDF AST trees for operators that need compiled expression evaluation outside the evaluator — primarily mixed joins and parquet filter pushdown.

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

This is used by `sirius_physical_hash_join` in MIXED_JOIN mode to pass the conditional predicates — inequality conditions, and null-safe `IS NOT DISTINCT FROM` keys (emitted as `NULL_EQUAL`) that were mixed with a plain `=` — to `cudf::mixed_join()` as a cuDF AST expression.

## Key Files

| File | Purpose |
|------|---------|
| `src/expression_evaluator/expression_evaluator.hpp` | Main evaluator class |
| `src/expression_evaluator/expression_evaluator.cpp` | Driver: strategy dispatch, AST tree management, temp lifetimes |
| `src/expression_evaluator/specializations/*.cpp` | Per-Sirius-AST-alternative dispatch (comparison, case, function, …) |
| `src/expression_evaluator/expression_evaluator_strategy.hpp` | `expression_evaluator_strategy` enum + string conversions |
| `src/expression_evaluator/ast_supported_types.hpp` | AST-eligible cast targets and functions (`supported_ast_cast_types`, `supported_ast_functions`) |
| `src/expression_evaluator/gpu_expression_translator_internal.hpp` | Sirius AST → cuDF AST translator (mixed joins, parquet pushdown) |
| `src/expression_evaluator/gpu_expression_translator.cpp` | Translator implementation |
| `src/expression/ast/node.hpp` | `sirius::ast::node` variant; per-alternative headers included from here |
| `src/expression/ast/from_duckdb.hpp` | `sirius::ast::from_duckdb` — DuckDB → Sirius AST translation |
| `src/expression/ast/utils.hpp` | AST tree utilities — `visit_references`, `clone`, `substitute_references` |
| `src/expression/join_condition.hpp` | `sirius::join_condition` — AST-node sides + comparison operator |
