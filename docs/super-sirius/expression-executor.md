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
[`restore_prepared_key`](../../src/op/key_preparation.hpp) to restore
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
| `unary_op` | `sirius::ast::unary_op` | `NOT x`, `x IS NULL`, `x IS NOT NULL` |
| `coalesce` | `sirius::ast::coalesce` | `COALESCE(a, b, 0)` |
| `in_list` | `sirius::ast::in_list` | `x IN (1, 2, 3)` |
| `function_call` | `sirius::ast::function_call` | Named function (`YEAR`, `UPPER`, `concat`, etc.) and arithmetic (`a + b`, `-x`) — `sirius::function_id` enum |
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

The evaluator supports four strategies, selected via `policy.strategy` in the `expression_query_policy` constructor argument:

| Strategy | How it executes | API |
|----------|-----------------|----------|
| `MATERIALIZE` | Materializes intermediate results as `cudf::column`s. Floating comparisons use small interpreted ASTs for NaN semantics; this strategy never selects AST JIT. | `cudf::unary_operation`, `cudf::binary_operation`, `cudf::compute_column` |
| `AST_INTERPRET` (default) | Builds a `cudf::ast::tree` and interprets it with a single monolithic kernel. | `cudf::compute_column` |
| `AST_JIT` | Builds a `cudf::ast::tree` and JIT-compiles it into a fused kernel. | `cudf::compute_column_jit` |
| `NVRTC_JIT` | Compiles the Sirius expression trees directly into one CUDA `__global__` kernel per projection list or filter predicate (see [NVRTC_JIT kernels](#nvrtc_jit-kernels)). | NVRTC via `codegen::jit::KernelCache` |

### NVRTC_JIT kernels

**Files:** `src/expression_evaluator/nvrtc/`. `supportability.cpp` plans what compiles;
`translation_context.cpp` emits per-row CUDA statements over `(value, valid, overflow)` Register
Values; `kernel_abi.cpp` derives both the kernel signature and the `cuLaunchKernel` argument
array from one parameter table; `sink_launch.cpp` compiles, launches and reads status back;
`project_sink.cpp` and `filter_sink.cpp` wrap the statements in the `ProjectSink` and
`FilterSink` kernels. The device runtime the kernels are compiled behind lives in
`device_runtime/*.cuh` (embedded at build time and also compiled by nvcc as a syntax check).

- **Compile planning.** Before any GPU work, `plan_project_sink` / `plan_filter_sink` walk the
  expressions once and return either no plan (the expressions take the materialize path) or a
  `compile_plan`: the roots one kernel computes (a projection's other roots, such as a VARCHAR
  output, take the materialize path), compiled directly, or after first evaluating the named
  Materialization Boundaries (`upper`, `concat`, string and temporal casts, ...) into columns
  appended to the input. Each boundary is planned as a column of its declared type, which its
  evaluated column has, so after evaluation only a STRING column's offset width is checked
  (`is_supported_boundary_column`). A projection list is planned once, whether or not every root
  compiles, and the per-subexpression retry in `evaluate(node, MATERIALIZE)` costs one host walk
  when it fails.
- **Kernel shape.** Blocks of 256 threads process a tile of rows, each thread handling several
  rows (4 per thread for projections, 16 for filters) so every thread has several memory requests
  in flight. A projection thread's rows are strided by the block width and a filter thread's by
  the warp width (a filter warp covers 512 consecutive rows, 32 per item), which keeps loads and
  stores coalesced. A projection kernel holds one copy of the row body and never checks
  `row < num_rows` around it: an item past the end of the batch evaluates the last row again,
  which stores the same bytes and raises the same Overflow Poison as that row's own thread, and
  only the validity words tell such an item apart. A filter kernel holds two copies of the row
  body: full tiles run an unguarded one and only a tile cut short by the end of the batch checks
  `row < num_rows`. Neither kernel forces a row body to be unrolled: the item loops over full
  tiles carry no unroll pragma, so unrolling is the compiler's choice and a large body (a LIKE, a
  long IN list or OR chain of string comparisons) is not forced through it once per item. The
  filter's partial-tile loop runs in at most one block per batch and is marked never to be
  unrolled. The speed of a light body rests on that choice, which was measured on NVRTC 13.4: it
  unrolls a comparison or two, and the same body kept rolled ran about twice as long, so
  re-measure the kernel time of the `LC_conj1` and `F03_rare_1pct` benchmark cases when the CUDA
  toolkit changes. It keeps a body of one to three string equalities rolled, and such a filter's
  kernels take 14 to 50 % longer than under a forced unroll. Forcing the unroll is no remedy: a
  LIKE or a long string IN list then takes several times longer to compile and to run.
- **Null masks.** Input validity bits are loaded independently of the payload (a null row's slot
  still exists). A nullable output stores its payload for every row, valid or not. Its validity
  words come from one `__ballot_sync` per warp and row group, in which an item past the end of
  the batch votes invalid. Valid rows are counted per warp in registers and reduced once per
  block, and a block adds the rows it holds that are not valid to the null count, so a block
  without nulls issues no atomic. A column with no nulls in the batch, and any expression over
  such columns and non-NULL constants, is statically non-null: no mask is allocated and no null
  count is read back.
- **Overflow Poison.** An operation sets a Register Value's `overflow` bit only by propagating an operand's poison, or from its own computation over operands that are all valid and unpoisoned; a NULL value's payload never decides overflow. Every function in `device_runtime/*.cuh` and every statement `translation_context.cpp` emits keeps this invariant (`reg_val` in `device_runtime/reg_val.cuh` states it for new code), so poison always comes from an Evaluated Operation. The arithmetic and the numeric casts of the device runtime compute every row and apply the invariant to the flags (`operation_result`) instead of returning early for a NULL or poisoned operand: a branch per operation made a chain of them slower. The payload of a NULL or poisoned value is therefore unspecified, a nullable output column holds such payloads under its NULL rows, and whatever reads a payload is defined for every bit pattern: integer arithmetic wraps, a division runs with the divisor 1 where the hardware operation is undefined, and a float-to-integer conversion out of range converts 0. A DOUBLE or FLOAT division divides 1 by 1 on a NULL row: a column load gives a NULL the payload 0, and the hardware division takes a slow path for a zero operand that every other row of the warp waits for. The `ProjectSink` and `FilterSink` kernels therefore report every poison that reaches them, whatever the value's final validity: `n + abs(x)` with `n` NULL raises, as rule 2 of [SQL errors](#sql-errors) requires. Poison is kept away only where DuckDB does not evaluate an operation: CASE and COALESCE skip a branch the row does not select, and AND or OR in a filter predicate or CASE WHEN condition skip an operand that can raise on the rows the earlier operands already decide (rule 1 of [SQL errors](#sql-errors)). TRY clears the bit, and TRY_CAST clears it only for its own conversion.
- **Division by a power of ten.** A DECIMAL to DOUBLE cast and `round(x, n)` with `0 <= n <= 38` divide by a literal power of ten. `div_by_pow10` multiplies by the reciprocal, which the compiler folds, and corrects the product with two fused multiply-adds, which gives the correctly rounded quotient, bit for bit that of the hardware division it replaces. `translation_context` prints ROUND's modifier into the source for that, so kernels that differ only in the precision are compiled separately.
- **DECIMAL to DOUBLE or FLOAT.** The cast converts the DECIMAL's integer to the destination type and then divides by 10^scale, so it rounds twice. DuckDB returns the value nearest to the exact decimal. The two agree where the integer and 10^scale are exact in the destination type (a magnitude of at most 2^53 and a scale of at most 22 for DOUBLE, 2^24 and 10 for FLOAT); beyond that the last bit can differ.
- **Floating-point results.** By default the kernels are compiled with `-fmad=false` (`CompileOptions::fuse_multiply_add`), so the compiler never contracts a multiplication and an addition or subtraction of its product into one fused multiply-add. The fused operation rounds once where DuckDB and the cuDF strategies round twice, and whether the compiler fuses depends on the shape of the kernel, so with it a DOUBLE or FLOAT result would vary with the surrounding expression and the nullability of its inputs. The `nvrtc_jit_fuse_multiply_add` setting (off by default) lets the compiler fuse: multiply-add chains over DOUBLE or FLOAT run faster, and their results, with everything computed from them, can differ from DuckDB and from the other strategies as [Setting the strategy](#setting-the-strategy) describes. The choice is a compile option, so it is part of the kernel's identity in every cache and a kernel built one way never serves the other. The explicit `fma` calls of `div_by_pow10` are fused either way.
- **Decimal addition.** A DECIMAL addition or subtraction whose operands are at the result's scale and no wider than its carrier is the checked integer operation in that carrier, with no 128-bit intermediate.
- **Limits of DECIMAL and HUGEINT `+`, `-` and `*`.** A sum, difference or product raises where its result leaves its SQL type. `translation_context` passes the lowest and the highest value of that type to `eval_dec_add`, `eval_dec_sub` and `eval_dec_mul`. A DECIMAL(P,S) holds fewer than 10^P in magnitude, which is less than its carrier holds: a DECIMAL(18,0) product of 10^18 raises although it fits 64 bits. A HUGEINT holds everything its 128-bit carrier holds, so `6e37 + 6e37` returns its 39 digits and a sum or difference raises only past the carrier. A 128-bit product follows DuckDB's `Hugeint::TryMultiply` (`checked_mul128`), which multiplies the magnitudes of the operands and raises for a product of 2^127 or more: the lowest value is a product only of itself by 1, and `-2^126 * 2` raises although the carrier holds -2^127. A cast to HUGEINT is still checked for 38 digits, which is stricter than DuckDB for a DOUBLE or FLOAT of 39 digits; CPU Replay answers it. For a HUGEINT, and for a DECIMAL whose result precision DuckDB clamped to 18 or 38 digits, these limits are exactly where DuckDB raises. DuckDB does not check any other DECIMAL result, which operands within their declared precisions cannot exceed; Sirius checks it too, so it is stricter than DuckDB only for an operand that holds more digits than its type, and CPU Replay answers that. Such operands exist: nothing holds a column to its declared DECIMAL precision, and the materialize path, a scan, and the DECIMAL division, modulo and ROUND of this evaluator all pass on whatever the carrier holds. Negation and `abs` pass them on as well: they raise only for the lowest value of the carrier, for a DECIMAL too, whose storage integer DuckDB negates without looking at the precision.
- **Division by zero.** `//` in every type, and `%` and Sirius's own `/` in the integer and DECIMAL types, raise for a zero divisor, as DuckDB does under its default `error_on_division_by_zero = true`: a row whose operands are both valid carries Overflow Poison, so CPU Replay returns DuckDB's error and TRY returns NULL. The evaluator does not read `error_on_division_by_zero`. With it off DuckDB returns NULL for such a row, and the query still gets DuckDB's answer, through CPU Replay. A DOUBLE or FLOAT `/` or `%` follows IEEE 754 and returns an infinity or NaN, as DuckDB does under its default `ieee_floating_point_ops = true`; with that setting off DuckDB raises or returns NULL instead, and Sirius, which does not read it, still returns the IEEE value. Only `NVRTC_JIT` kernels check a divisor. The `MATERIALIZE` and cuDF AST strategies hand it to cuDF, which leaves the row of a zero divisor undefined, as it lets an integer sum wrap: under them such a query returns a wrong value instead of DuckDB's error.
- **AND, OR and IN.** Only a later operand that can raise a SQL error has to run on exactly the
  rows where DuckDB evaluates it. Any other later AND or OR operand or IN list value changes no
  result wherever it runs, so `translation_context::skips_decided_rows` chooses by cost alone.
  Input loads are unconditional, so a branch around an operand saves only the work on its
  Register Values, and only in a warp whose 32 rows are all decided. An operand that only tests
  fixed-width inputs (comparisons, BETWEEN, IN and IS NULL over columns and constants, and AND,
  OR and NOT of those) has next to no such work and runs on every row without a branch: with the
  branch, the predicate kernel of the `F01_q6_pred` benchmark case took 74 us instead of 56 us at
  16M rows and that of `DS28_q28_filter` 125 us instead of 57 us. An operand that computes a
  value (any other node whose result is not BOOLEAN, such as arithmetic, a cast, a date part or a
  CASE) or reads a string, whose bytes the device runtime walks in a loop, is skipped for the
  decided rows: `LG_rare_and_costly`, whose first operand leaves 1 % of the rows undecided, takes
  202 us with the branches and 681 us without. The branch is a loss where few warps are fully
  decided: `H16_q16_filter_part` takes 245 us with its string operands skipped and 218 us with
  every operand run on every row.
- **Constants** travel as kernel parameters so one compiled kernel serves every literal value;
  only a constant's NULL-ness and the precision of a ROUND are baked into the source. Overflow
  and the null counts are read back once per evaluation through a pinned staging slab
  (`read_status_words`), which is the one host synchronization of an evaluation; a projection
  whose outputs are statically non-null and cannot overflow does not synchronize at all. Binding
  a string column reads nothing from the device: the end of its characters is derived in-kernel
  from `offsets[num_rows]`.
- **Overflow messages.** The status words say only that Overflow Poison reached the NVRTC Sink. `translation_context` records each Overflow Site it emits outside TRY, other than a TRY_CAST's own conversion, and `read_status_words` names it in the error (`Overflow in NVRTC evaluation of abs(INTEGER) -> INTEGER`), or lists up to four candidates when the kernel holds several distinct sites. The error reads `Overflow or division by zero` when one of them is a division or modulo. The record stays on the host: the kernel source, its registers and its cache key do not change.
- **String constants** are passed as a device pointer and a length (`c_N_ptr`, `c_N_len`). A
  kernel's string constants are concatenated into one pool, which the query's
  `nvrtc::string_constant_cache` (in the `expression_query_caches` shared through the pipeline
  build context) uploads once per device, on a stream of its own; each launch's stream waits for
  that upload on the device, so a batch neither allocates device memory for its constants nor
  copies them or blocks the host on them. Each copy is charged to its device's GPU memory space,
  not to the reservation of the task that first needed it, and is freed when the query ends, on
  the cache's stream once every launch that read it has finished, without a host wait.
- **Filters.** A filter warp votes once per item with `__ballot_sync`, and lane `i` keeps the
  ballot word of item `i` in a register, so the item loop writes no memory and a warp's 16 words
  cover consecutive rows. A one-tile batch writes its survivors from those registers in a single
  pass. Any other batch takes two launches. The predicate kernel stores each warp's ballot words
  (one coalesced store) and a survivor count per tile, and adds that count to the total of the
  tile's group of 256 tiles. The scatter kernel, whose warps load their 16 words back into
  registers, writes the surviving row indices in ascending order with coalesced stores: a block
  finds where its tile's survivors start by summing the totals of the groups before its own and
  the counts of the tiles before it in its group, so no tile waits for another and no launch runs
  a prefix sum over all tiles. The block of the last tile publishes the survivor total. The
  survivors feed `cudf::gather` (or are returned to late materialization).
- **Strings** are `str_view{ptr, len, limit}` registers; `limit` bounds word-at-a-time loads:
  `load_bounded_word` loads no byte at or after it and reads the buffer's last, cut-short word
  with narrower aligned loads, so the matchers need no padding after a column's characters.
  `%lit1%...%litN%` LIKE patterns and `contains` use a SWAR digram matcher (the literal
  descriptor is a kernel parameter) when the `like_swar_fastpath` setting is on, classified once
  per query through the query's shared LIKE cache and capped at eight descriptors per kernel;
  other patterns use a byte-wise backtracking matcher.
- **Compile time.** The device runtime preamble is self-contained (its own fixed-width integer
  types, limits, traits, checked 128-bit arithmetic and civil-calendar helpers; no includes),
  because parsing `<cuda/std/chrono>` and `<cuda/numeric>` dominated every kernel compile and
  even `<cuda/std/cstdint>` cost each one about 12 ms. The kernels are therefore compiled without
  named headers (`CompileOptions::embedded_headers` off), which also spares NVRTC registering the
  embedded CCCL bundle, and such a compilation passes NVRTC `--minimal`, which leaves out its
  built-in declarations of the CUDA runtime API that no kernel here uses. Compiled kernels are
  cached in memory and on disk (`SIMPATICO_JIT_CACHE_DIR`), one module per source: pipeline
  threads that need the same kernel at once wait for a single compilation.
  `SIRIUS_NVRTC_DUMP_DIR` writes every generated source for inspection.
- **Benchmarking.** `sirius_unittest "[expr_bench]"` times a fixed set of projections and
  filters across all four strategies (see the header of
  `test/cpp/expression_evaluator/bench_expression_evaluator.cpp`).

### Tree of AST Trees

Not every DuckDB expression has a cuDF AST equivalent — these are called **AST breakers** (e.g. `CASE`, `LIKE`, `SUBSTRING`, unsupported `CAST` types). For AST strategies, the evaluator walks the Sirius AST and greedily builds AST subtrees up to each breaker. When it hits a breaker, it materializes that subtree as a `cudf::column`, stashes it internally, and references it from the enclosing AST subtree via a `cudf::ast::column_reference`.

The result is a tree of AST trees whose edges are AST breakers. Each AST tree is evaluated by `cudf::compute_column` (or `compute_column_jit`) in `evaluate_ast()`.

### `min_ast_size` — per-subtree mode selection

For nodes that use size-based mode selection, the `min_ast_size` constructor parameter (default `2`) controls whether a subtree is materialized operator-by-operator or built as an AST. The evaluator uses `cudf_ast_op_count()` to estimate its size; a subtree below the threshold normally materializes unless its parent requests AST mode. This is a heuristic, not a universal rule: floating comparisons, BETWEEN, and floating membership may require NaN-aware ASTs independently of the threshold.

`MATERIALIZE` and `AST_INTERPRET` with `min_ast_size = ∞` are therefore not interchangeable. In particular, an all-constant FLOAT/DOUBLE IN list uses `cudf::contains` under `MATERIALIZE`, but follows the floating comparison AST path under `AST_INTERPRET` even when the threshold exceeds the expression size. Floating lists with nonconstant entries retain the comparison chain under either strategy. AST evaluation requested under `MATERIALIZE` is interpreted; only `AST_JIT` selects the JIT backend.

### `evaluation_mode` (internal)

Internal to the evaluator, each node is evaluated with either `evaluation_mode::AST` or `evaluation_mode::MATERIALIZE`. This is a **hint** — if a node tagged AST turns out to be a breaker, it is evaluated in MATERIALIZE mode anyway (and wrapped via `materialize_as_ast_column()` so the parent still sees an AST reference).

### Setting the strategy

The strategy is a DuckDB SET variable registered in `src/sirius_extension.cpp`:

```sql
SET expression_evaluator_strategy = 'nvrtc_jit';   -- or 'ast_jit', 'ast_interpret', 'materialize'
```

`NVRTC_JIT` has one further setting, registered next to it and off by default:

```sql
SET nvrtc_jit_fuse_multiply_add = true;
```

With it NVRTC may fuse a DOUBLE or FLOAT multiplication with an addition or subtraction of its product into one operation, which does not round the product before it adds. Multiply-add chains run faster. The sum then differs from DuckDB's and the other strategies' by up to one rounding of the product, and which operations are fused changes with the shape of the surrounding expression. Where nothing cancels that is at most the last bit of the sum. Where the terms cancel it is the whole result: with `c` holding the rounded product of `a` and `b`, `a * b - c` is 0 on DuckDB and the rounding error of the product, of either sign, when fused. Whatever is computed from such a value can then differ too: a comparison and so the rows of a filter, the branch of a CASE, a cast to an integer or DECIMAL, a join or group key. The other strategies ignore the setting.

### Policy ownership

At query initialization, Sirius snapshots the connection's settings into immutable
`operator_params`. Operators derive an `expression_query_policy` from that snapshot and
pass it to their evaluators. Scan materialization and residual filtering forward the
same policy explicitly; pin operations supply their own settings snapshot.

Each evaluator owns a constant copy of the execution strategy, the regex JIT flag and
the fused multiply-add choice. Evaluation therefore uses the settings captured at
construction, including when it runs on a different worker thread. It does not consult
thread-local settings or mutable process globals. Standalone callers may omit the policy
to use fixed defaults (`AST_INTERPRET`, regex JIT enabled, no fused multiply-add), or
pass an explicit policy.

## SQL errors

The evaluator raises `sirius::invalid_input_exception` for a SQL error it detects: an overflow, a conversion out of range, or an Overflow Poison that reaches the NVRTC Sink. Such an error is a hint. Transparent execution answers every GPU error other than a user interrupt with CPU Replay where replay is available, and DuckDB's result or error is final, because a GPU-detected error cannot in general prove that DuckDB would raise (see rules 1 and 3 below). No exception class selects that policy, so a new check throws `sirius::invalid_input_exception` and adds no handler of its own.

The evaluator's contract relative to DuckDB, which new operators and tests use as their oracle:

1. **Never raise for rows a Row-exact Guard excludes.** CASE, COALESCE and TRY never raise for a row on an operation in a branch that the row does not select. A fused NVRTC kernel also runs an AND or OR operand that can raise for a row exactly when DuckDB evaluates it: in an AND or OR reached from the root of a filter predicate or a CASE WHEN condition only through AND and OR, which DuckDB evaluates with `Select`, a later operand runs only on the rows the earlier ones leave undecided (for AND, only where all of them are TRUE); anywhere else every such operand runs on every row, as does every IN list value that can raise, in any position. Where an evaluator computes a guarded branch over every row (the `MATERIALIZE` and cuDF AST strategies, and an NVRTC kernel compiled for a subexpression of that branch), it may detect such an error; CPU Replay then returns DuckDB's value.
2. **Raise when an Evaluated Operation overflows** and no guard suppresses it, even if the final value is NULL. TRY_CAST turns only its own conversion failure into NULL; an overflow inside its input still raises.
3. **Errors that depend on DuckDB's run-time choices may return a value.** Such choices include: DuckDB's `Select` reorders the AND and OR operands of a filter predicate or CASE WHEN condition by their observed cost as it runs, and a LIMIT stops reading input early. Where DuckDB raises only because of such a choice, Sirius may return a value.

SQL integration tests check results with `GpuExecutionFixture::expect_result_matches_cpu` and errors with `GpuExecutionFixture::expect_error_matches_cpu`, each of which also asserts whether the query completed on the GPU or through CPU Replay. Tests are tagged `[sql_errors]`.

## Supported Expression Types

| Expression Type | Sirius AST alternative | Example |
|----------------|------------------------|---------|
| Column reference | `sirius::ast::reference` | `column #3` |
| Constant | `sirius::ast::constant` | `42`, `'hello'` |
| Comparison | `sirius::ast::comparison` | `a > b`, `x = 10`, `a IS NOT DISTINCT FROM b` |
| Conjunction | `sirius::ast::conjunction` | `a AND b`, `x OR y` |
| Arithmetic | `sirius::ast::function_call` | `a + b`, `-x` |
| Unary predicate | `sirius::ast::unary_op` | `NOT x`, `x IS NULL` |
| COALESCE | `sirius::ast::coalesce` | `COALESCE(a, b, 0)` |
| IN-list | `sirius::ast::in_list` | `x IN (1, 2, 3)` |
| Function call | `sirius::ast::function_call` | `UPPER(name)`, `YEAR(date)`, `a \|\| b` |
| Type cast | `sirius::ast::cast` | `CAST(x AS DOUBLE)` |
| CASE/WHEN | `sirius::ast::case_expr` | `CASE WHEN x > 0 THEN 'pos' ELSE 'neg' END` |
| BETWEEN | `sirius::ast::between` | `x BETWEEN 10 AND 20` |
| Aggregate | `sirius::ast::aggregate` | `SUM(x)`, `COUNT(*)` |

`coalesce` is materialized via `cudf::replace_nulls` iteratively across children: the first child is materialized (scalars are lifted to a column); each subsequent child replaces the residual nulls in the running result. Children are only evaluated when the running result still has nulls. The result is wrapped via `materialize_as_ast_column` so COALESCE composes inside AST-capable parents. `cudf_ast_op_count` returns 0 — `coalesce` always takes the materialize-only path.

Handlers that read an operand as a column evaluate it with `evaluate_as_column`, which broadcasts
a scalar result to the input's row count. These are the date parts (including `millisecond` and
`microsecond`), `date_trunc`, `round`, the string functions `strlen`, `length`, `upper`, `lower`,
`substring`, `contains`, `prefix`, `suffix`, `LIKE` and `regexp_replace`, the `IN` probe, `NOT`,
`IS [NOT] NULL`, and one side of constant-only arithmetic. Comparisons and the `BETWEEN` input
broadcast a constant the same way on the materialize path. DuckDB does not always fold
all-constant calls: statistics propagation reduces `coalesce(<non-null constant>, x)` to the
constant after constant folding has run, so `month(coalesce(DATE '2020-02-01', d))` reaches the
GPU as `month(DATE '2020-02-01')`.
Parameters that must be constants, such as a `LIKE` pattern or `substring` bounds, stay scalars.

`in_list` covers the full numeric set (INT8–INT64, UINT8–UINT64, BOOL8, FLOAT, DOUBLE, DECIMAL32/64/128, all TIMESTAMP precisions, DATE) plus VARCHAR. BOOL8 dispatches via `uint8_t` because `std::vector<bool>::data()` is deleted. `x IN (a, b)` follows SQL semantics: TRUE on any match, otherwise NULL if `x` or any list entry is NULL, otherwise FALSE; NOT IN negates that.

The AST lowering for IN joins per-entry equality comparisons with `NULL_LOGICAL_OR`. The standalone translator lowers NOT IN to per-entry inequality comparisons joined by `NULL_LOGICAL_AND`. For non-floating operands these use primitive `EQUAL` and `NOT_EQUAL` nodes. FLOAT/DOUBLE comparisons instead use the shared NaN-aware comparison builder: equality expands to `EQUAL OR both_nan`, and inequality to `NOT(EQUAL OR both_nan)`, where `both_nan` means both operands are valid NaNs. These expressions retain ordinary comparison NULL propagation. The pinned cuDF implementation normalizes the expanded predicate before Parquet membership pruning; the floating inequality's `NOT` node must not be confused with negating a Bloom-filter membership answer (see [Row Group Pruning](scan.md#row-group-pruning)). The row evaluator can negate its IN result because it operates on actual row values.

Under `MATERIALIZE`, all-constant lists (including FLOAT/DOUBLE) use `cudf::contains` over the non-NULL entries; this lookup treats NaNs and signed zeros as equal. When the list also has a NULL, the evaluator combines the result with a NULL scalar using `NULL_LOGICAL_OR`. Floating lists with nonconstant entries retain the NaN-aware comparison chain.

### String concatenation

String concatenation covers the `concat(a, b, …)` function and the `||` operator, which have **different** NULL semantics and therefore resolve to **distinct** ids: `concat()` ignores NULL arguments (`concat(NULL, 'x') = 'x'`) and maps to `function_id::concat`; the `||` operator propagates NULL (`'a' || NULL = NULL`) and maps to `function_id::concat_operator` (DuckDB lowers `||` to a function named `"||"`). Both are dispatched as a materialize-only function in `src/expression_evaluator/specializations/function.cpp`: every argument is materialized, any scalar argument is broadcast to a full-length column via `cudf::make_column_from_scalar`, and the columns are joined with `cudf::strings::concatenate` using an empty separator. The two semantics are selected by the narep scalar: a valid empty string for `concat()` (NULL → `""`, ignored) and an invalid narep for `||` (NULL → NULL, propagated).

### String case conversion

`upper()` and `lower()` materialize their input and call `cudf::strings::to_upper` and
`cudf::strings::to_lower`, respectively. NULLs propagate and empty strings remain empty.
These functions are AST breakers and can compose inside projections, filters, and materialized
join-condition operands.

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
infinity, and rounds nanoseconds to the nearest microsecond with ties away from the
epoch, as DuckDB's `TryCastTimestampBase` does for every precision decrease (the
`NVRTC_JIT` kernels apply the same rule to every fine-to-coarse timestamp cast). Overflow raises
`sirius::invalid_input_exception`; reading the overflow result synchronizes the
supplied stream. The cast evaluator dispatches to this helper for semantic casts.

`finite_mask()` returns a nullable boolean column for DATE or any timestamp
precision. Both checked conversion and millisecond/microsecond extraction use it,
so sentinel handling has one implementation. The evaluator boundary regressions
are tagged `[timestamp_bounds]`; SQL extraction comparisons against DuckDB are
tagged `[timestamp_extraction]`.

### HUGEINT

HUGEINT is carried as cuDF DECIMAL128 with scale 0, an exact 128-bit integer with the layout of
`duckdb::hugeint_t`; its constants are `sirius::decimal128` values with scale 0. Like DECIMAL
(`logical_type::has_fixed_point_carrier`), HUGEINT arithmetic is materialized rather than lowered
to the cuDF AST. `//` and `%` with a HUGEINT result run on the GPU only by a nonzero constant
divisor (cuDF leaves a zero divisor's row undefined, where DuckDB raises an error), and other
divisors stay on the CPU. A cast from HUGEINT to a narrower integer or DECIMAL first checks the
column's range and throws `not_implemented_exception` when a value does not fit, so the query falls
back and DuckDB raises its conversion error (or returns NULL under `TRY_CAST`). Such casts never
lower to the AST, whose integer casts wrap (`ast_supports_cast`); a cast to DOUBLE does. A cast to
VARCHAR uses `cudf::strings::from_fixed_point`. Tests are tagged `[hugeint]`.

`DATE + INTEGER`, `INTEGER + DATE`, `DATE - INTEGER`, and `DATE - DATE`
match their already-bound signatures and execute through `date_arithmetic_checked()`.
The helper uses a libcudf JIT transform with public CCCL overflow helpers and error
collection; DATE inputs are bitcast to day counts without copying. Scalar operands
use one-row columns broadcast by `cudf::scalar_column_view`, with the output row
count passed explicitly so scalar-only and empty inputs retain the input shape.
Offsets preserve NULLs and infinity and reject finite overflow or infinity-sentinel
collisions. Date differences widen both day counts before subtraction and return
BIGINT. These operations materialize under every evaluator strategy; standalone
cuDF AST translation declines them because it cannot implement the checks.
The existing reference evaluator restores compressed operands before execution.

### DECIMAL to integer casts

DECIMAL-to-integer casts round to the nearest integer, with ties away from zero, then check the
destination range. NULLs remain NULL. On overflow, `TRY_CAST` returns NULL and `CAST` replays on
the CPU. A UHUGEINT result above UINT64 also replays on the CPU, including under `TRY_CAST`.
Tests are tagged `[decimal_integer_cast]`.

### Arity and negation

`from_duckdb_function_name` resolves a (name, argument count) pair against one signature table
in `function_id.cpp`: a one-argument `-` is `function_id::negate`, a two-argument `-` is
`function_id::sub`, and an arity no GPU handler takes stays on the CPU. A one-argument `+`
translates to its operand. The `function_call` constructor throws on a mismatched arity.

`negate` runs `cudf::unary_operation(NEGATE)` on signed integers, FLOAT, DOUBLE and DECIMAL,
keeping the input type (`-0.0` matches DuckDB). Negating a signed integer's minimum raises
DuckDB's `Overflow in negation of numeric value!`; the check is a `cudf::contains` shared with
`abs` (one stream sync). Unsigned, HUGEINT and INTERVAL negation, and a result type that differs
from the operand's, stay on the CPU. `negate` is an AST breaker. Tests are tagged `[negate]`.

### Rounding

`round(x)` and `round(x, n)` on FLOAT and DOUBLE map to `function_id::round` and run in
`round_floating_point` (`src/cuda/sirius_round_floating_point.cu`), one elementwise kernel that
reproduces DuckDB's `RoundOperatorPrecision`: the value is rounded in double precision as
`round(x * 10^n) / 10^n`, or `round(x / 10^-n) * 10^-n` for negative `n`, with ties away from
zero. A non-finite result yields the input for `n >= 0` and 0 for `n < 0`. The output is bit
identical to DuckDB. The precision must be a constant INTEGER.
DECIMAL and integer inputs and a column precision fall back to the CPU at plan time.

### Absolute value

`abs(x)` on FLOAT, DOUBLE, DECIMAL and the integer types other than HUGEINT maps to `function_id::abs` and runs `cudf::unary_operation` with `ABS`. NaN and infinities are preserved and `abs(-0.0)` is `0.0`. The result has the input's type. The absolute value of the smallest value of a signed integer type wraps in cuDF, where DuckDB raises an overflow error, so the evaluator checks with `cudf::contains` whether the input holds that value and throws `sirius::invalid_input_exception`, as negation does; CPU Replay then raises DuckDB's error (see [SQL errors](#sql-errors)). Unsigned integers have no such value and are returned as they are. HUGEINT falls back at plan time.

### Logical AND/OR NULL semantics

SQL `AND`/`OR` use Kleene three-valued logic (`TRUE OR NULL = TRUE`, `FALSE AND NULL = FALSE`), so SQL conjunctions map to cuDF's Kleene operators — `NULL_LOGICAL_AND`/`NULL_LOGICAL_OR` as AST operators and `cudf::binary_operator::NULL_LOGICAL_*` on the materialize path — rather than the strictly null-propagating `LOGICAL_*` variants.

BETWEEN follows the same rule: `(val >= lower) AND (val <= upper)` uses `NULL_LOGICAL_AND` in the evaluator's AST and materialized paths and in the standalone translator. Its comparison operands can be NULL. For example, `(val, lower, upper) = (2, NULL, 1)` gives `NULL AND FALSE`, which is FALSE; reversing the conjunction to `FALSE AND NULL` also gives FALSE. NOT BETWEEN negates that result and therefore returns TRUE in this example.

Per-expression-type dispatch lives in `src/expression_evaluator/specializations/` (one file per Sirius AST alternative: `comparison.cpp`, `case.cpp`, etc.). Each specialization decides how to emit cuDF AST nodes, materialize, or fall back based on the effective evaluation mode.

### AST-Eligible Operations

The following translate directly into cuDF AST nodes:

| Category | Operations |
|----------|-----------|
| Arithmetic | `+`, `-`, `*`, `/`, `//`, `%` |
| Comparison | `=`, `!=`, `<`, `>`, `<=`, `>=`, `IS NOT DISTINCT FROM` (via `NULL_EQUALS`) |
| Logical | `AND`, `OR` (Kleene `NULL_LOGICAL_AND`/`NULL_LOGICAL_OR`), `NOT` |
| BETWEEN | Translated to `(val >= lower) AND (val <= upper)` |
| Casting | Fixed-width types: `UBIGINT`, `BIGINT`, `DOUBLE` (see `supported_ast_cast_types`); from HUGEINT only `DOUBLE` (`ast_supports_cast`) |

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
- `cast` with non-fixed-width target types (e.g., VARCHAR), and a HUGEINT cast to anything but DOUBLE
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
