# Code Review — Compressed Materialization (numeric carrier narrowing)

**Reviewed:** uncommitted working tree on `dev`, 2026-07-21
**Scope:** the `enable_compressed_materialization` feature — 16 modified files, 6 new files,
~750 changed lines plus ~540 lines of new source, test, and docs.
**Method:** four independent `reviewer` agents over separate tracks (narrowing helper + expression
evaluator; planner propagation + plan-time statistics; scan runtime + pin cache + config; engineering
quality + tests + docs). Each was instructed to refute its own findings before reporting. Three of the
four converged independently on finding B1 by different evidence paths.
**Build:** `pixi run make` clean, exit 0, no warnings from any new or changed file.
**Tests:** not run as part of this review.

File:line references are as of the review date and will drift as the change is edited.

---

## Verdict

The design is sound. Keeping `types` as the authoritative SQL schema and carrying the physical cuDF
carrier in a parallel `physical_types` sidecar is the right shape, and restoring join keys rather than
co-compressing them is the right conservative call for a first version. `numeric_narrowing.{hpp,cpp}`
is well-factored: every branch of `choose_signed` / `choose_unsigned` / `choose_decimal` was walked
against concrete values and is correct, the DECIMAL scale handling is right, `fits<T>` promotion is
sound, and `noexcept` is correctly placed.

The problems are not in the narrowing math. They are:

1. **The change is not inert when the feature is off.** Three separate code paths alter behaviour on
   the default configuration, two of them converting previously-tolerated conditions into failures.
2. **Nothing verifies that the feature works, or that its central soundness claim holds.** The one
   integration test would pass identically if the feature were deleted, and the plan-time bound is
   trusted without any runtime check.
3. **A repo-root notes file attaches benchmark and correctness claims from a different, superseded
   design to this one.**

Items B1–B3 should be fixed before this lands. Item S6 determines whether the feature keeps working.

---

## A. Blocking — regressions on the default (feature-off) path

### B1. `normalize_physical_schema` runs unconditionally

`src/op/scan/sirius_gpu_scan_operator.cpp:199-209` (call site), `:96-119` (function)

`execute()` builds `target_types` from `get_cudf_type(types[i])` whenever there is no sidecar, so the
`targets.empty()` early-out never fires and the normalization executes on every scan batch regardless
of the flag. It throws `std::runtime_error` on a column-count mismatch (`:101-103`) and on a type
mismatch that `is_allowed_narrowing` rejects (`:113-114`).

This converts a condition the codebase deliberately tolerates into a hard failure.
`src/pipeline/gpu_pipeline_task.cpp:58-88` is the pre-existing checker for exactly this condition; it
logs `SIRIUS_LOG_WARN` and returns, and carries an in-code comment stating that delim join trips it and
"there is no bug here".

Reachable today with the flag off:

| Trigger | Mechanism |
|---|---|
| `SELECT count(*) FROM read_parquet(...)` | `LogicalGet::ResolveTypes` yields a single BOOLEAN column, while `scan_plan.cpp:255` drops virtual columns so no reader projection is set and cuDF reads every column in the file. This repo's own `log/*.log` contains 84× `column count mismatch: got 16, expected 1`. |
| DATE / TIMESTAMP hive-partition key | `cudf_utils.hpp:266-272` returns `numeric_scalar<int32_t>` (type `INT32`) for `TIMESTAMP_DAYS`, against a target of `TIMESTAMP_DAYS`. DuckDB auto-detects DATE hive keys by default. |
| INT96 parquet timestamps | `set_timestamp_type` is never called anywhere in the repo, so the decoded unit can differ from `get_cudf_type(TIMESTAMP)`. |

Existing coverage that should fail: the three count-star cases in
`test/cpp/integration/test_gpu_execution_multi_format.cpp:811-820`.

**Consequence.** With DuckDB fallback on (the default) the exception is caught at
`src/transparent/physical_sirius_execution.cpp:276-279`, downgraded to a WARN, and the whole query
re-runs on CPU. The user sees no error — only an unexplained 10–100× slowdown on a query unrelated to
this feature. On the S3 and FFI paths there is no fallback, so it is a hard error.

**Fix.** Gate the entire block on `has_physical_overrides()`. When there is no sidecar, return the
table untouched and leave today's warn-and-pass-through semantics intact. The schema check is a
*feature* invariant, not a contract the scan has ever guaranteed.

---

### B2. `is_allowed_narrowing` computes both widths and never compares them

`src/op/scan/sirius_gpu_scan_operator.cpp:86-94`

```cpp
auto const source_width = numeric_carrier_width(source.id());
auto const target_width = numeric_carrier_width(target.id());
if (source_width == 0 || target_width == 0) { return false; }
if (is_signed_integer(source.id()) && is_signed_integer(target.id())) { return true; }
```

The widths are bound and then unused. The sibling predicate does the check the name implies —
`src/expression_evaluator/specializations/reference.cpp:73`:
`if (source_width == 0 || source_width >= target_width) { return false; }`.

So any same-family wider-to-narrower pair is accepted and `cudf::cast` truncates modularly with no
diagnostic. Live case: cuDF selects a decimal carrier from the parquet **physical** type
(FLBA/BYTE_ARRAY → DECIMAL128) while `get_cudf_type` selects from **precision**
(`cudf_utils.hpp:195-208`), so an FLBA-stored `DECIMAL(15,2)` decodes DECIMAL128 against a declared
DECIMAL64. `parquet_gpu_ingestible.cpp:469-491` shows that file shape is an explicitly supported path
(it disables reader-side filter pushdown specifically to handle it).

This also means the throw in B1 is not a safety net. Design invariant 2 in
`docs/super-sirius/compressed-materialization.md:114` has **zero** runtime enforcement in the
truncating direction: a stale or incorrect sidecar produces silently wrong rows, not an error.

**Fix.** Require `source_width < target_width` here too. Better, per Q1 below: delete both copies and
put one direction-explicit predicate in `helper/numeric_narrowing.hpp`.

---

### B3. The reference type-check throw contradicts a documented contract

`src/expression_evaluator/specializations/reference.cpp:85-104`

The removed comment read "Bound reference: pass column through without type check". That was not an
oversight — `src/include/expression/ast/reference.hpp:32-35` states that the executor "resolves column
types from the input `table_view` at runtime and **ignores `return_type()`**; it exists so the cuDF-AST
translator's decimal-propagation guard can see the type of a referenced decimal column."

The new code throws `internal_exception` on any mismatch that is not a strict widening restore, ungated
by the feature flag. `sirius::ast::from_duckdb` always populates the declared type
(`src/expression/from_duckdb.cpp:84-87`), so the check is live for every reference originating in user
SQL.

Producers that diverge today:

| Producer | Declared | Actual | Result |
|---|---|---|---|
| `LEFT`/`RIGHT_DELIM_JOIN` pass-through (`sirius_physical_delim_join.cpp:197-211`) | DECIMAL128 | INT64 | throws |
| `count(DISTINCT)` partial via `COLLECT_SET` (`aggregate_op_util.cpp:104-131`) | INT64 | LIST | throws |
| grouped `SUM(DECIMAL(≤9,s))` (`gpu_aggregate_impl.cpp:295-308` vs DuckDB `BindDecimalSum`) | DECIMAL128 | DECIMAL64 | silent extra cast (see S7) |

Top-level bare references are safe — `sirius_physical_projection.cpp:82-92` routes them through a
`passthrough` fast path that never reaches the evaluator. Nested references inside comparisons,
conjunctions and functions, and the entire filter path via `compute_mask`, are not.

**Fix.** Throw only when the operator actually declares physical overrides; otherwise restore what can
be restored and fall back to today's pass-through. This is what
`docs/super-sirius/compressed-materialization.md:119` (invariant 8) already promises.

---

## B. Significant

### S4. Plan-time bounds are writer-declared, not exact

`src/planner/sirius_plan_get.cpp:169-176` feeding `sirius_gpu_scan_operator.cpp:110-116`

For parquet the bound is footer metadata merged verbatim across row groups
(`duckdb/extension/parquet/parquet_reader.cpp:963-982`). DuckDB uses that statistic only to *prune* and
then re-evaluates the predicate on surviving rows, so a bad footer costs it nothing. Sirius uses it to
select a lossy carrier and never checks, and per B2 there is no runtime guard. One file with an
incorrect footer min/max — a known writer bug class — yields silently wrong answers.

The pin-time path is genuinely exact (`compute_exact_numeric_range` performs a real `cudf::minmax`).
The doc's invariant 2 conflates the two and should be split per path.

What was checked and is *not* a problem: transaction-local uncommitted appends cannot produce an
out-of-bounds value, because `TableScanStatistics` returns `nullptr` outright when the transaction has
local storage (`duckdb/src/function/table/table_scan.cpp:761-765`); DuckDB base-table stats only widen
outside a checkpoint; and `MultiFileScanStats` returns `nullptr` for a multi-file list without
`union_by_name`.

**Fix.** Verify before narrowing on the plan-time path — at minimum a debug-build
`compute_exact_numeric_range` check inside `normalize_physical_schema` — and correct the doc.

### S5. The `bytes * 8` memory branch keys on the flag, not on actual narrowing

`src/op/scan/sirius_gpu_scan_operator.cpp:224-226`

Wrong in both directions:

- **Over-reserves 8× when nothing is narrowed.** `scan_physical_schema` needs
  `op.function.statistics`; `MultiFileScanStats` returns `nullptr` for globbed multi-file parquet
  without `union_by_name`, which is the production shape and the shape the new GB10 benchmark config
  targets. So nothing is ever narrowed there, yet every resident scan reserves 8×. With
  `scan_task_batch_size: 536870912` that is 4 GB per task instead of 512 MB — roughly 8× fewer
  concurrent scan tasks and spurious downgrade pressure until history accumulates.
- **Under-reserves in the case the doc calls safe.** Pin with the flag on (chunks stored INT8), then
  `SET enable_compressed_materialization = false`, then query: no sidecar, so the operator is built
  with `false` and the estimate falls through to `max(stats.bytes, working_set)` sized on the *narrow*
  resident bytes — while `normalize_physical_schema` still widens INT8 → INT64. Result is
  `oom_reschedule_exception` churn on every scan task of that table.

Also: 8× understates the peak, since `cudf::cast` allocates the destination while the source column is
still live (~9×); the resident branch uses `max()` where the fresh-read branch at `:228-230` uses a
sum; and `8` is a bare magic number duplicated across both branches.

**Fix.** Derive the factor from the actual served chunk types versus the planned target schema, or
record the narrowing decision on the pinned cache entry. Name the constant.

### S6. The test suite cannot distinguish "narrowing happened" from "narrowing silently no-opped"

`test/cpp/integration/test_gpu_execution_tpch.cpp:631-654` calls `compare_gpu_vs_cpu`, which compares
stringified rows and transparent-execution counters. Nothing observes carrier width. Every skip path in
the feature returns "no override" and none of them logs:

- `scan_physical_schema` returns `{}` when nothing changed — `sirius_plan_get.cpp:178`
- `get_column_statistics` returns `nullptr` — `sirius_plan_get.cpp:125-140`
- both install sites drop the sidecar on a width mismatch — `sirius_plan_get.cpp:534`, `:602`
- every `break` in `propagate_compressed_schema` falls through to the go-native tail —
  `sirius_physical_plan_generator.cpp:922-927`

Untested surface: `compute_exact_numeric_range` (which drives pin-time narrowing), `range_from_statistics`,
`to_int128`, `get_column_statistics`, the entire ~250-line propagation pass, hash-join key restoration,
delim joins, root restoration, pin-time narrowing, heterogeneous cached chunks, host-pin round trip, the
dynamic-filter interaction, the feature-off path, `set_physical_types`' width throw, and both
`is_allowed_narrowing*` variants.

Concrete bug that would slip through: swapping `value(stream)` for `fixed_point_value(stream)` in
`decimal_bounds` (`numeric_narrowing.cpp:97-105`) makes DECIMAL(38,2) bounds 100× too small →
`choose_decimal` picks DECIMAL32 → unchecked wraparound in the cast → silently wrong pinned data. No
exception anywhere, and `validate_operator_output_types` compares against the *planned* schema so it
does not warn either.

**Fix, in order.**

1. **Make narrowing observable.** Add a `compressed_materialization_stats { scans_narrowed,
   columns_narrowed, restore_projections_inserted }` counter on `SiriusContext`, mirroring the existing
   `transparent_execution_stats` pattern in `test/cpp/utils/transparent_execution_test_utils.hpp:40-52`.
   Without this, no integration test can assert the feature ran.
2. **Extend `test_numeric_narrowing.cpp`:** `compute_exact_numeric_range` over real device columns
   (empty, all-null, interleaved nulls, single row, carrier ≠ `get_cudf_type(logical)`, DECIMAL64
   scale −2, DECIMAL128 scale −7); `is_narrowable_numeric_type` at DECIMAL precision 4/5/9/10/18/19/38;
   `choose_decimal` at precision 19. Add the property "any chosen target is strictly narrower than
   native **and** `is_allowed_narrowing_restore(target, native)` holds" — that single property catches B2.
3. **New `test/cpp/planner/test_compressed_materialization_plan.cpp`** — the propagation pass is pure
   plan-tree manipulation and needs no GPU. Cover: FILTER passthrough and permuted `output_columns`;
   PROJECTION pure-reference vs arithmetic; LIMIT type-mismatch guard; HASH_JOIN for INNER/SEMI/ANTI/
   MARK/RIGHT_SEMI/RIGHT_ANTI asserting keys restored, payloads left narrow, output map aligned; a key
   referenced through an expression (`ON a+1 = b`); root restoration; delim join; aggregate and ORDER BY
   boundaries; dynamic-filter consumer scan cleared vs non-consumer installed; `set_physical_types`
   width throw.
4. **Scan/pin:** `narrow_pin_chunk` per carrier; heterogeneous chunks served correctly (this is the
   doc's line-101 claim); host-pin round trip preserving width; pin-on-then-flag-off under a tight
   `usage_limit` (guards S5); and a feature-off regression case asserting zero new mismatch warnings and
   zero runtime fallbacks (guards B1).
5. **Integration:** the existing test plus a stats-delta assertion; a join where both sides narrow the
   same key to different widths; GROUP BY / ORDER BY over a narrowed column; a flag-off vs flag-on
   in-test A/B asserting byte-identical results.

Separately, `test_gpu_execution_tpch.cpp:640` runs `REQUIRE_FALSE(enabled->HasError())` **before** the
RAII reset is constructed at `:642-648`, so a failure there leaks the flag into every subsequent test on
the shared connection. Move the `SET` and its reset into a scoped fixture.

### S7. Restores are per-occurrence, and AST mode materializes them

`src/expression_evaluator/specializations/reference.cpp:99-103`

`WHERE a > 1 AND a < 10` evaluates `reference{a}` twice and produces two independent `cudf::cast`
allocations; there is no memoization keyed on `column_index`. In `evaluation_mode::AST` the restore goes
through `materialize_as_ast_column`, pushing a real column into `_temp_columns` instead of emitting a
zero-copy `cudf::ast::column_reference` — so N occurrences hold N full copies live until
`release_temporaries` runs after `compute_column`. That directly cancels the bandwidth saving the feature
exists to buy. Allocation comes from `_mr` (for projections, `get_current_device_resource_ref()`), not the
batch's memory space, so it is invisible to reservation accounting.

### S8. The pinned path can invert its own benefit

For a resident chunk `materialize_table` returns the cached view as `UNFILTERED`
(`src/op/scan/gpu_ingestible.cpp:37-60`), so pushed-down filters are evaluated against the **narrowed**
carriers in `post_filter_and_project` — which runs *before* `normalize_physical_schema`. Correctness holds
only because the reference specialization restores each referenced column, but it does so over every row
of the chunk before the filter selects survivors, which are then cast again. For exactly the columns
narrowing was meant to shrink, peak memory on the pinned path can exceed the un-narrowed baseline.

### S9. `native_physical_schema` calls a throwing function unguarded

`src/planner/sirius_physical_plan_generator.cpp:687-694`, invoked from `:697`, `:702`, `:714`, `:743`

`get_cudf_type` throws `duckdb::InvalidInputException` for `SQLNULL`, `INVALID`, and DECIMAL with
precision ≤ 4 (`cudf_utils.hpp:191-211`). Once the flag is on this runs for every column of every
FILTER / PROJECTION / LIMIT / HASH_JOIN plus the generic tail, so a plan containing such a column throws
from inside `create_plan` rather than at execution — changing where the failure occurs and which fallback
counters move. A propagation pass should never be able to turn a supported plan into a plan-time
exception. Mitigating: `validate_operator_output_types` already calls `get_cudf_type` unguarded, so such
a column likely fails elsewhere today; worth confirming with a `DECIMAL(4,2)` SQLLogic case either way.

---

## C. Unresolved — needs an author decision

### U10. The full-`data_type` comparison in `post_process`

`src/expression_evaluator/expression_evaluator.cpp:305` changes
`result_column->type().id() != cudf_return_type.id()` to `result_column->type() != cudf_return_type`.
`cudf::data_type::operator==` compares the fixed-point scale, and `get_cudf_type` encodes the DECIMAL
scale, so this brings scale into scope for every decimal-producing expression on the **default** path.

The two reviewers disagreed:

- One classified it as a global decimal-arithmetic behaviour change bundled into a flag-gated PR that
  reviewers will assume is inert — insert a rescaling `cudf::cast` where the carrier previously passed
  through, with no test coverage.
- The other traced cuDF's `binary_operation_fixed_point_scale` (MUL → sum, DIV → difference, else min)
  against DuckDB's binders — `BindDecimalMultiply` sets `result_scale += scale` and never clamps the
  scale, `BindDecimalArithmetic` uses max-scale, and decimal `/` binds to `BindBinaryFloatingPoint` →
  DOUBLE so there is no decimal divide — and could not construct an expression where the scales diverge.
  It concluded the change is a latent fix: `sirius_physical_result_collector.cpp:62-63` materializes into
  the declared DuckDB types, so a scale-divergent column was previously a silent 10^k error.

Best synthesis: probably an improvement, but it is an unrelated semantic change to shared code, it is
untested, and where it does fire `cudf::cast` fixed→fixed **truncates**, so last-digit results change.
Recommend splitting it into its own change with decimal-scale tests (`a*b`, `sum(a*b)`, cast chains at
several precisions).

---

## D. Engineering quality

### Q1. Duplicated predicates that have already diverged

`numeric_carrier_width` / `is_signed_integer` / `is_unsigned_integer` / `is_decimal` are byte-identical
copies in `src/expression_evaluator/specializations/reference.cpp:31-78` and
`src/op/scan/sirius_gpu_scan_operator.cpp:50-94`, with a third conceptual copy of the same type knowledge
in `src/helper/numeric_narrowing.cpp:24-86`. The two `is_allowed_narrowing*` functions differ by one line
(B2) with nothing documenting the asymmetry — so the drift hazard is not hypothetical, it has already
happened before the feature shipped. The next editor will "fix" one to match the other and either
reintroduce silent truncation or break the required widening.

cuDF already provides all four predicates: `cudf::is_integral`, `cudf::is_signed`, `cudf::is_unsigned`,
`cudf::is_fixed_point` (`cudf/utilities/traits.hpp:238-456`), and width via `cudf::size_of` /
`type_dispatcher`. None of the three hand-rolled tables should exist.

**Fix.** One function in `helper/numeric_narrowing.hpp` built on the cuDF traits, with two
direction-explicit wrappers (`can_narrow_to` / `can_restore_to`), used by all call sites and unit-tested.

### Q2. Raw `std` exceptions where the project has its own

`sirius_physical_operator.hpp:442` (`std::invalid_argument`), `sirius_gpu_scan_operator.cpp:103`, `:113`
(`std::runtime_error`), `pin_table.cpp:156` (`std::invalid_argument`). `sirius_physical_operator.hpp`
already includes `sirius/exception.hpp` and throws `internal_exception` at `:567` and `:577` — the new
throw is inconsistent within the same file.

Traced propagation for a scan-time throw: `run_one_operator` (no catch) → `compute_task` (catches only
`rmm::out_of_memory`) → the generic handler at `gpu_pipeline_executor.cpp:421-425` → rethrown at
`sirius_engine.cpp:148-157` → `sirius_interface.cpp:199-203` builds `duckdb::ErrorData(e)` →
`error_data.cpp:17-38` classifies it `ExceptionType::INVALID`. So a Sirius-internal invariant violation
is reported to the user under DuckDB's category for *user input* errors — or, on the default
configuration, disappears into a CPU fallback. Neither message names the column index or the two types,
so the log gives nothing to debug from.

### Q3. Zero observability

No log, telemetry field, or EXPLAIN output anywhere reports that a column was narrowed:
`numeric_narrowing.cpp`, `sirius_gpu_scan_operator.cpp`, `narrow_pin_chunk`, `scan_physical_schema` and
`install_physical_schema` contain no `SIRIUS_LOG` calls, and no `params_to_string` mentions
`physical_types`. The one signal that would have shown a divergence — the warn in
`gpu_pipeline_task.cpp:71-88` — is now suppressed for exactly the operators carrying a sidecar. Every
sibling feature logs its decisions (e.g. `dynamic_filter_merge.cpp:158`).

Add a `SIRIUS_LOG_DEBUG` per converted column in `normalize_physical_schema` and `narrow_pin_chunk`, a
debug line at each planner bail-out, and one info-level per-plan summary.

### Q4. Encapsulation and API shape

- `physical_types` is a public member (`sirius_physical_operator.hpp:405`) next to a setter enforcing a
  size invariant it cannot protect; `gpu_pipeline_task.cpp:74` then indexes it with unchecked
  `operator[]`. No in-repo path breaks it today — every mutation goes through the setter and the only
  post-construction write to `types` is in a constructor — so this is latent, not active. Make it
  private behind the existing accessors (C.9 / C.131), or at minimum use `.at()`.
- Adjacent same-typed `bool` defaults on `materialize_all_batches` / `materialize_pin_to_host`
  (`pin_table.hpp:147-148`, `:176-177`) and on the scan constructor
  (`sirius_gpu_scan_operator.hpp:79`). Both call sites pass them positionally; a transposition compiles
  clean and inverts both features at once. Prefer an options struct with designated initialisers, and
  drop the constructor default (its only consumer is one test, and its existence means the S5 estimate
  branch has no unit coverage at all).
- `int` as a byte width with `0` doubling as "not a numeric carrier" conflates "unknown type" with "not
  narrower" and makes a new cuDF carrier a silent reject rather than a compile error.
- `__int128_t` — a compiler extension with no `std::numeric_limits` specialisation under strict
  conformance, which `fits<T>` relies on — is in the public header `numeric_narrowing.hpp:32-52`.
  `to_int128` also reimplements what `duckdb::hugeint_t` already provides.
- Declaration/definition parameter-name mismatch: the header declares
  `logical_type const& logical_type` (`numeric_narrowing.hpp:69`, also shadowing the type name) while
  the definition uses `logical` (`numeric_narrowing.cpp:197`). Sibling declarations at `:57`/`:61` use
  `type`.
- `const logical_type&` and `logical_type const&` both appear in the same declaration list
  (`numeric_narrowing.hpp:57`, `:62` vs `:68`, `:69`).
- `target_types` is rebuilt per batch on the scan hot path (`sirius_gpu_scan_operator.cpp:199-207`) for a
  plan-time constant; compute once and pass a `std::span`.

### Q5. Dead and unreachable code

- `sirius_physical_plan_generator.cpp:280-282` — propagating the sidecar onto the DYNAMIC_FILTER wrapper
  is unreachable (`leaf` can only acquire overrides under `if (!dynamic_filters …)`, and this block runs
  under `if (dynamic_filters)`). Worse, it implies dynamic filters can carry narrow carriers — the
  opposite of the rule the surrounding comment states.
- `sirius_physical_plan_generator.cpp:518-521` — installing the sidecar on PARTITION and CONCAT is dead,
  because neither type appears in `propagate_compressed_schema`'s switch and both are reset by the
  default arm. This also means every scan feeding a join gets a restore projection immediately above it,
  i.e. no width savings for join-side scans — most of TPC-H.
- `sirius_plan_get.cpp:106-111` — the DECIMAL `precision <= 9` arm is unreachable (the caller gates on
  `is_narrowable_numeric_type`, which requires precision > 9). If it ever became reachable, precision ≤ 4
  is stored in an `int16_t` carrier and `GetValueUnsafe<int32_t>` would read two bytes of uninitialised
  union padding behind a release-disabled assert.
- `numeric_narrowing.cpp:36`, `:55` — the `id != TINYINT` / `id != UTINYINT` guards can never be false.
- `numeric_narrowing.cpp:209` — the two `is_valid(stream)` calls are redundant given the null-count
  early-out at `:201-203`, and each one synchronises. With the `value(stream)` calls that follow, this is
  up to four blocking device syncs per column, per chunk, on the pin's single stream.

### Q6. Latent: the non-projection-pushdown install site can land on the wrong columns

`sirius_plan_get.cpp:534-536` guards with `physical_types.size() == node->types.size()`, but
`physical_types` is computed over `op.types` (projection output order) while `node->types` in that branch
is `from_duckdb_vec(op.returned_types)` (natural table order). A size check is standing in for an order
invariant. Currently unreachable — all four supported scan functions set `projection_pushdown = true` —
but it stops guarding the moment one that does not is added. The live install site at `:602-604` is
correct: `original_types` is captured before `projection_ids` is mutated.

### Q7. Build and formatting hygiene

- `CMakeLists.txt:258` places `src/helper/numeric_narrowing.cpp` between `src/log/level.cpp` and
  `src/log/logging.cpp`, splitting the sorted `src/log/*` block; `:598` similarly misplaces the test file.
  The block is `# cmake-format: off`, so no hook will catch it.
- `reference.cpp:18-23` puts `<cudf/cudf_utils.hpp>` at the head of the `// sirius` block and leaves
  `<cudf/unary.hpp>` in an unlabelled trailing block; `sirius_plan_get.cpp:35` breaks the alphabetical
  order of its quoted include block. `.clang-format` sets `IncludeBlocks: Regroup` with `^<cudf/` at
  priority 4, so the hook will rewrite both files.
- Targets are correct and no file is missing from a build target.

---

## E. Documentation

### D1. `compressed-materialization-notes.md` should not be committed as-is

It sits at the repo root — where the only tracked markdown files are `AGENTS.md`, `CLAUDE.md`,
`CONTRIBUTING.md` — and describes the **offset-based** prototype: `CAST(col - min AS INT32)` /
`CAST(col AS INT64) + min`, union-find over join-key equivalence classes, a ~450-line single plan pass,
and mutual exclusion with `enable_dynamic_filter_pushdown`. The implementation in the tree is carrier-width
narrowing with no offset, no union-find, keys restored rather than co-compressed, and dynamic filters
handled by making participating scans native — explicitly not mutual exclusion.

Every path its "Sirius-specific pointers" section cites was verified **absent**:

- `src/planner/sirius_plan_compressed_materialization.hpp` / `.cpp`
- `test/tpch_performance/compmat_sf50_on.yaml`, `bothmat_sf50_on.yaml`
- `run_latemat_ab.sh`
- `sirius_physical_hash_join::refresh_types_from_children()` (zero matches under `src/`)

Its headline numbers (Q8 −43.6%, Q5 +22.4%, Q9 +11.7%, "fires on 9 of 22", "all 22 byte-identical to the
CPU baseline") are performance and correctness claims for a prototype that no longer exists here. Next to
a flag with the same name, it reads as validation for code that has never been benchmarked — the in-tree
evidence is four unit `TEST_CASE`s and one integration query.

**Fix.** Delete it, or move it under `docs/`, retitle it explicitly as a superseded offset-based
prototype, delete the "Sirius-specific pointers" section, add a header stating the measurements describe
a different design, and cross-link from `compressed-materialization.md`.

### D2. Claims in the design doc stronger than the code guarantees

`docs/super-sirius/compressed-materialization.md`:

| Line | Claim | Reality |
|---|---|---|
| 47-51 | "annotates all eligible outputs" | all-or-nothing per scan, and conditional on statistics availability |
| 63-64 | "pure reference projection remains narrow and zero-copy" | true only because of a passthrough fast path in a different file; nothing enforces it |
| 101 | "heterogeneous chunks are never concatenated before normalization" | true, but by accident of the serve path — name the mechanism or drop the claim |
| 102 | "changing the setting after a table was pinned is safe" | false for memory accounting (S5) |
| 104 | "static filters see correct logical semantics through reference restoration" | filters are safe because they run pre-narrowing at the scan; attributing it to reference restoration would mislead a reader into thinking filters are safe anywhere in the pipeline |
| 107-108 | "reserve for up to an 8x expansion" | under-reserves and keys on the wrong condition (S5) |
| 114 | invariant 2, "exact min/max bounds" | true of the pin-time path, false of the plan-time path (S4) |
| 116-117 | invariants 5 and 6 | asserted, never checked at runtime, never tested; the only checker warns and returns |
| 122-126 | "ensuring it remains narrow while passed through and expands when consumed" | the test asserts nothing about narrowness (S6) — the most misleading sentence in the doc |
| 68 | "The initial implementation…" | version narration; prefer "This contract favors…" |

### D3. Missing from the configuration index

`enable_compressed_materialization` appears in neither the operator-params table nor the SET-variables
section of `docs/super-sirius/configuration.md`, and has no entry in `optimizations.md` (which follows a
strict `### <Name> (PR #N)` + `**Config:**` pattern). `scan.md`'s pinned-tables section omits the new
pin-time narrowing step and the changed `pinned_column_types.clear()` semantics.
`docs/super-sirius/README.md`'s index entry is present and correct.

Also: `docs/super-sirius/scan.md:129` claims the count-star path yields "a 0-column table"; the code
returns the full-width reader batch. That inaccuracy is part of what makes B1 look safe.

### D4. Orphaned benchmark config

`test/cpp/integration/integration-gb10-compressed-materialization.yaml:2` says to keep the file identical
to `integration-gb10.yaml` except for the feature gate, but it already dropped that file's 16-line header
including the `SIRIUS_DISABLE=1` data-generation requirement and the unified-memory budget arithmetic.
Nothing in the repo references the file. Either generate it from a base plus an overlay and wire it into a
documented A/B harness, or drop it and set the flag with `SET` in the run script. Its keys do all parse,
so `reject_unknown()` will not fire.

### D5. Comment style — compliant

Every new comment was checked against the "describe the present design, never narrate how code used to
be" convention. No violations; the two forward-looking comments explain why the present design is what it
is, which is appropriate. The only version narration is in the doc (D2, line 68).

---

## F. Checked and could not break

Recorded so these are not re-litigated:

- **`projection_ids` direction** matches `LogicalGet::ResolveTypes`
  (`duckdb/src/planner/operator/logical_get.cpp:168-176`) exactly — not inverted.
- **`to_int128`** is correct for −1 and both INT128 extremes; `hugeint_t` is `{uint64_t lower; int64_t upper;}`.
- **`NumericStats::HasMinMax`** requires `has_min && has_max && Min <= Max`; both "unknown" and null-only
  columns return false.
- **Hash-join output alignment** is correct for every supported join type. `collect_left`/`collect_right`
  match `gather_join_output`, and MARK's exclusion matches the fact that it routes to
  `build_mark_join_output`, emitting lhs columns plus one BOOL8 that stays native.
- **Restore projections between a join and its child** break nothing: CONCAT/PARTITION wrap above them,
  `key_source` derivation is index-based with unchanged indices, and the delim-join shape check still holds.
- **Root wrapping is safe** — `verify()` is a debug no-op, and the result collector is constructed later
  from the prepared statement's types.
- **No later plan rewrite invalidates a sidecar.** `wrap_order_by` is the only re-typing rewrite, and
  ORDER_BY always falls to the default arm with an already-empty sidecar.
- **Operators owning children outside `children[]`**: the enumeration is complete — only
  `sirius_physical_delim_join::join` / `::distinct_root`, both handled correctly via
  `restore_native_output_in_place`, which preserves the non-owning `delim.distinct` borrow.
- **Dynamic-filter conditions are exact complements** between `propagate_compressed_schema` and
  `make_gpu_scan_leaf`; both run after the tree is built, so `has_producers()` is settled for both.
  DuckDB's own `op.dynamic_filters` channel is stored only as a route-key identity and never evaluated.
- **Heterogeneous pinned chunks never reach `cudf::concatenate`** — one cached chunk → one split → one
  `execute()` → one batch, bypassing the coalescer entirely.
- **Host-pin round trip** preserves carrier and scale (`column_metadata` stores `type_id` + `scale`;
  sizes come from `cudf::size_of`), and `base_row_count_per_chunk` is width-independent.
- **Cross-GPU transfer** sizes from `cudf::size_of(src.type())` and rebuilds with `src.type()`.
- **Static and dynamic filters both restore before comparing.** Dynamic filters run in a separate operator
  downstream of the scan, and the planner clears the sidecar for any scan with a wired producer.
- **AST temp-column lifetimes are safe** — every AST-mode specialization forwards `temp_column_indices`,
  so a restored column is never released while a live `column_reference` points at it. This was a genuine
  new hazard and it is closed.
- **No dangling views** from the new owned-column path; the "same reference twice in a select list" case
  yields two independent owned columns.
- **`choose_*` selection, `fits<T>` promotion, and `noexcept` placement** are all correct; `cudf::minmax`
  does support fixed-point for all three decimal reps.
- **Pin-time ordering** (stats captured from the native table before narrowing) is correct — and
  load-bearing, since `compute_pinned_chunk_stats` silently drops any column whose carrier is not native.
- **`narrow_pin_chunk` cannot throw on a reachable pin**; `pinned_column_types` is cleared only when both
  flags are off, in which case it is not called.
- **Flag change between plan and execute** is self-consistent: both the sidecar and the operator's flag are
  baked in at plan time.
- **Rowid / late materialization** is not a mismatch source (INT64 both sides).
- **`stats.bytes * 8` overflow** is not a real concern (`std::size_t`).

---

## G. Suggested sequence

1. Gate `normalize_physical_schema` on `has_physical_overrides()` (B1).
2. Consolidate the duplicated predicates into `helper/numeric_narrowing.hpp` with direction-explicit
   names — this fixes B2 permanently rather than locally (Q1).
3. Make the reference throw conditional on declared overrides (B3).
4. Split the `post_process` decimal-scale change into its own PR with tests (U10).
5. Add the observability counter and the planner unit-test file (S6 steps 1 and 3) — the pass is pure
   plan-tree manipulation, so this is the cheap half of the coverage gap.
6. Fix the memory estimate to key on actual served widths (S5).
7. Correct the doc claims (D2), add the configuration entries (D3), and resolve the notes file (D1).
