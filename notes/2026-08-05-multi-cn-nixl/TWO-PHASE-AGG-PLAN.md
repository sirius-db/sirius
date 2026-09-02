# Two-phase aggregation support — implementation plan

Goal: TPC-H Q6 over `FILES()` parquet returns `61567694.9502…` on the two-CN cluster with the
FE's **default** `new_planner_agg_stage` — i.e. the FE's two-phase plan (partial agg on each scan
fragment → UNPARTITIONED gather → single merge/final fragment) translates and executes. Partial
states ride the existing packed-cudf hop (nixl remote / local relay) unchanged, as ordinary cudf
columns.

Paths repo-relative to `sirius-worktrees/integration`. Translator crate =
`experimental/starrocks/crates/starrocks-plan-translator`. Base branch: `demo-multi-cn`
(continues the nixl demo stack `d6cce3ae..3473a686`).

Full research/design provenance (session-scratch, ephemeral): three research reports
(FE plan shape via live EXPLAIN VERBOSE + thrift; CN translator; engine agg path) and two
independent designs (deep-reasoner, Opus) that converged on the decisions below.

---

## 1. Background: why `new_planner_agg_stage = 1` was a workaround, and why it can go now

### Why it existed

The session-variable recipe is Matthijs's deliberate stopgap from the single-node
multi-fragment experiment: at the time there were **no real exchanges**. The compute node ran
fragments sequentially, feeding fragment 1's materialized result to fragment 2 as its source
("the setup that would typically handle these exchanges isn't in place"). A two-phase
aggregation plan is *inherently* an exchange construct — N partial states produced on N
instances must be gathered and merged — so the only executable shape was the one where the FE
emits a **single finalized aggregation instance**: `new_planner_agg_stage = 1`. The branch's
special paths were an end-to-end validation experiment (plan translation, result encoding), not
the intended architecture. The intended architecture is "do what the exchange tells us".

### Where the workaround lives (commits, classes, functions)

- **Commit `0cda6e2c`** ("demo(starrocks): multi-fragment compute node on the streaming
  branch", 2026-07-25) — the port of mbrobbel/sirius `c35223c4..dfedb07e`, which carries the
  plan translator's "exchange/aggregate handling":
  - `node_translator.rs:438-444` — `translate_aggregation`'s node-level guard: rejects
    `!need_finalize || intermediate_tuple_id != output_tuple_id` with *"only finalized
    one-phase aggregation is supported (new_planner_agg_stage=1)"*. **Live finding:** the
    tuple-id half is dead code — new-optimizer plans always set the two tuple ids equal
    (`PlanFragmentBuilder.java:2384-2387,2429-2436`), so a merge node actually *passes* this
    guard.
  - `expr_translator.rs:668-679` — the blanket rejection of any measure with
    `agg_expr.is_merge_agg` set. This is the guard that actually prevents a merge node from
    being mistranslated as a one-shot aggregate today.
  - `experimental/starrocks/DEMO.md:69,122` — the documented `SET new_planner_agg_stage = 1;`
    recipe (line 151 notes the GROUP BY limit that persists even at `agg_stage=1`).
- **Commit `3473a686`** carried the recipe forward into the two-CN nixl demo instructions.
- Related but distinct: the destination guard `compute_node_service.rs:742-747`
  ("a data stream sink with N destinations needs partitioned streaming") — that one is the
  *partitioned-output* boundary (#838), not the agg-phase boundary, and stays.

### Why the default configuration is possible now

Everything a two-phase plan needs at the transport level now exists on this branch:

1. **Real exchanges.** STREAMING_SOURCE/SINK are wired through the FFI; a fragment's output
   crosses to another fragment as native GPU batches — locally via `relay_from`, across CNs via
   `chunked_pack` → nixl WRITE → `push_packed`. Multi-sender streams with per-sender
   `close_input` already work (`engine.rs:372-382`), which is exactly the merge fragment's
   fan-in shape (one partial row per CN).
2. **Partial states are ordinary cudf columns.** For `sum/count/min/max` the engine's partial
   output shape *is* the final shape — no serialized opaque state, nothing new on the wire.
3. **The engine already merges.** Every UNGROUPED_AGGREGATE is auto-wrapped
   `MERGE_AGGREGATE → UNGROUPED_AGGREGATE` (`sirius_physical_plan_generator.cpp:384-393`), so a
   plain aggregate over the exchange stream naturally "reduces each arriving batch, concatenates,
   re-reduces" — the cross-CN merge for free.

What was missing is purely translator-side: classify the phase from thrift, translate the
partial and merge nodes, and predict the wire schema. That is this plan.

---

## 2. The two facts that shape the design (verified in research)

1. **The engine ignores phase.** `SubstraitToDuckDB::TransformAggregateOp` reads only
   groupings, `invocation()` (DISTINCT) and the function name; it ignores both
   `AggregateFunction.phase` and `Measure.output_type` (`substrait/src/from_substrait.cpp:707-783`).
   DuckDB's `AggregateRelation` / `LogicalAggregate` have no phase field. Phase semantics can
   therefore only be expressed as *which function is applied to which column* — a plan rewrite,
   not a marker. **The roadmap's "FFI marker" idea is rejected**: it would be fragment-scoped
   while phase is per-measure, would need finalize-suppression modes in two operators, and buys
   nothing since partial shape == final shape for the v1 functions.
2. **Nothing validates a pushed batch against the declared stream schema.**
   `Fragment::push_packed` (`src/sirius_ffi.cpp:629-664`) and `relay_from` (`:531-550`) check
   nothing; the receiver's declared types (`declare_input_column`, `:447-455`) are what the plan
   binds against while opaque cudf columns arrive. A declared/actual mismatch is a **silent
   wrong-results bug**. And the FE's declared intermediate type is *wrong* for Sirius: the
   translator lowers decimal `sum` args to FP64 (`expr_translator.rs:719-733`), so Q6's partial
   column is DOUBLE while the FE slot says DECIMAL128(38,2) (and avg's says VARBINARY).

So the design's center is: **predict the partial-state column type exactly, use one prediction
on both ends, and make a wrong prediction loud in the engine.**

## 3. Decisions

**(a) Phase mechanism — translator-resolved; no FFI setter; no engine phase awareness.**
The phase is fully observable in thrift with zero parser work: `TAggregationNode.need_finalize`
(`plan_nodes.rs:7408-7427`) × per-measure `agg_expr.is_merge_agg` (`exprs.rs:203-205`).
Substrait `AggregateFunction.phase` is *also* set on each measure (`InitialToIntermediate` /
`IntermediateToResult`, one field at `node_translator.rs:503-519`) as advisory metadata so dumped
plans are self-describing — the engine keeps ignoring it. Invariant to write in the code:
**substitute first, label second** — the emitted plan must be correct for a phase-ignoring
consumer.

**(b) Merge = plain `AggregateRel` with substituted functions.**
`merge(sum)=sum`, `merge(count)=sum`, `merge(min)=min`, `merge(max)=max` — verbatim the engine's
own internal merge table (`gpu_merge_impl.cpp:93-141` ungrouped, `:250-274` grouped).
`MERGE_GROUP_BY` is unreachable from Substrait and does not handle the scalar case at all (the
ungrouped merge is `MERGE_AGGREGATE`, auto-inserted around every UNGROUPED_AGGREGATE). The
substitution is mandatory, not cosmetic: `count` merged as `count` would count rows instead of
summing partial counts — silently wrong. `count→sum` also flips the URN to
`functions_arithmetic`, which is correct (FindFunction resolves by name). On the merge side the
decimal→FP64 argument cast (`expr_translator.rs:719-733`) is **skipped** — the state column is
already FP64; the FE `ret_type` lies about the child.

**(c) Intermediate schema — one pure prediction function, applied on both ends.**
New module `partial_state.rs`: `partial_state_columns(fn_name, &TFunction) -> Result<Vec<PartialColumn>>`
(`Vec` from day one so avg's sum+count drops in later):

| aggregate | FE-declared intermediate | Sirius wire column | merge fn |
|---|---|---|---|
| `sum(decimal)` | DECIMAL128(38,s) | **FP64** (decimal args cast at `expr_translator.rs:719-733`) | `sum` |
| `sum(int*)` | BIGINT | I64 (HUGEINT→BIGINT downcast, `sirius_plan_aggregate.cpp:308-330`) | `sum` |
| `sum(fp*)` | DOUBLE | FP64 | `sum` |
| `count(*/col)` | BIGINT | I64 | **`sum`** |
| `min/max(T)` | T | `map_type_desc(arg_types[0])` (identity) | `min`/`max` |
| `avg` | **VARBINARY** | *deferred* → loud error | — |

Applied (1) on the **receiver** as a `translate_plan` pre-pass: for every Merge-classified agg
node whose direct child is an EXCHANGE_NODE, override the exchange `NamedStruct` positionally
before `duckdb_type_name` derives the declared stream columns; (2) on the **sender** as the
partial measures' `output_type` (engine-inert, keeps dumped plans honest). Both ends agree by
construction: same binary, same pure function, same thrift inputs (fn name and
`TFunction.arg_types/ret_type` are serialized identically on the partial and merge nodes — the
FE's *slot* type for a state column is never trusted, so VARBINARY→BLOB never reaches the
engine). avg is deferred because it is a **cardinality** change (1 FE slot → 2 Sirius columns),
which a type override cannot express — it breaks the one-column-per-slot schema derivation and
the width checks; v2 rewrites `avg(x)` → `sum(x), count(x)` + a division ProjectRel.

**(d) Guard replacement — both guards, atomically.**
New `agg_phase.rs` classifier: `need_finalize × is_merge_agg` →
`OneShot` (today's path, untouched) / `Partial` / `Merge` / merge-serialize → error / mixed →
error. `AggregateCall` gains `is_merge: bool`; the blanket rejection at
`expr_translator.rs:668-679` becomes flag-recording (the node-level classifier owns the
decision). Allowlist for Partial/Merge: `{sum, count, min, max}`, **ungrouped only**. Every
error names the workaround (`SET new_planner_agg_stage = 1`): avg/VARBINARY (also fixes the
misleading "temporal avg" message at `expr_translator.rs:705-711`), unsupported function,
merge-not-over-exchange, 3/4-phase (merge-serialize), mixed flags, DISTINCT.
**Grouped two-phase gets an explicit translator error** ("grouped two-phase aggregation needs
partitioned streaming (#838)") rather than relying on the destination guard: on a single CN the
merge fragment has one instance → one destination → `compute_node_service.rs:742-747` never
fires, and grouped two-phase would become reachable-but-untested, which the loud-failure rule
forbids. One line to delete when partitioned output lands.

## 4. Commits (reviewable, dependency order)

Engine truth first (1-2), so the translator is written against verified behavior.

**C1 — engine: loud schema-agreement guards on both hop legs.**
Hoist the resolved `stream_input_spec`s onto `Fragment::Impl` at `build()`
(`sirius_ffi.cpp:474-489`); validate column count + `get_cudf_type(spec.types[i])`
(`cudf_utils.hpp:158`) against the actual table in `push_packed` (after `cudf::unpack`, `:655`)
and `relay_from` (`:531-550`, source sink types vs declared — metadata only). Throw naming
stream, column index, declared vs actual. Precedent: the warn-only check at
`gpu_pipeline_task.cpp:71-86`. Tests: Rust negative (declare BIGINT, push DOUBLE) beside
`packed_hop_matches_relay_hop` (`rust/crates/sirius/src/lib.rs:586`); Catch2 negative for the
relay leg. This is the safety net every later commit leans on.
Verify: `pixi run make test` + `pixi run cargo test -p sirius --lib`.

**C2 — engine tests: partial → hop → merge == one-shot (the design's oracle).**
Nothing today tests an aggregate on the *sender* side of a streaming fragment
(`test_streaming_fragment.cpp:220-285` aggregates on the receiver only). New Catch2 case: two
sender fragments over disjoint halves running `sum/count/min/max`, receiver runs the substituted
merge over `sirius_stream_source(0)`, oracle = one-shot over the union. Sections: both senders
non-empty; **one sender with empty input** (a CN with no scan ranges); a GROUP BY variant (pins
grouped merge semantics even though distributed grouped stays blocked). Plus the packed-leg Rust
variant (export_packed → push_packed → close_input ×2) with a decimal min/max case.
Empirically answers: engine-produced types for min/max(DECIMAL), min(DATE), grouped int-sum
widening; one-batch-per-sender incl. empty input.
Verify: `sirius_unittest "[streaming_fragment]"` + `pixi run cargo test -p sirius --lib packed`.

**C3 — translator: phase classifier, both guards replaced atomically, still rejecting.**
New `agg_phase.rs` + the (d) decision table; `Partial`/`Merge` still error, but with the precise
new texts. The repo never sits in the state where a merge node passes the node guard and
mistranslates as one-shot. Tests: retarget `merge_aggregation_is_rejected` (`translate.rs:2245`);
add classified/merge-serialize/mixed/grouped/avg rejection cases (helpers `aggregate_expr:2104`,
`aggregation_node:2117`, `agg_desc:2154`).
Verify: `pixi run cn-test-no-engine` (= `pixi run -e cn cargo test --workspace --no-default-features`).

**C4 — translator: the partial-state wire-type model.**
New `partial_state.rs` implementing the (c) table, doc comment citing
`gpu_aggregate_impl.cpp:66-100` and `sirius_plan_aggregate.cpp:308-330`; the riskiest ~60 lines,
reviewed alone. Unit tests per table row, incl. `decimal_sum_state_is_fp64_not_decimal` and the
avg refusal. Adjust the table if C2's empirical types disagree (narrow the allowlist rather than
guess).
Verify: `pixi run cn-test-no-engine`.

**C5 — translator: translate the Partial (sender) phase.**
`translate_aggregation` Partial branch: plain measures, `phase=InitialToIntermediate`, measure
`output_type` from the model (replacing `node_translator.rs:486-495`). After this commit the
query still fails — loudly, at the merge fragment — the intended intermediate state. Tests:
partial-translation case mirroring `aggregation_translates_to_aggregate_rel:2169`.
Verify: `pixi run cn-test-no-engine`.

**C6 — translator: translate the Merge (receiver) phase + exchange schema override.**
Pre-pass in `translate_plan` (`:319-336`) building the positional override map keyed by
exchange node id; `translate_exchange` (`:338-418`) applies it to the `NamedStruct` before the
declared stream columns are derived; Merge branch substitutes functions, sets
`phase=IntermediateToResult`, enforces the direct-child-is-EXCHANGE guard, skips the decimal
cast. **This commit makes the demo run.** Tests: merge variant of
`bound_exchange_feeds_aggregate_from_a_stream:2540` asserting measures `sum/sum/min/max`,
declared stream types DOUBLE/BIGINT (not DECIMAL/BLOB), phase; `merge_over_a_project_is_rejected`;
a determinism test asserting the partial fragment's output slot types equal the merge fragment's
declared stream types column-for-column.
Verify: `pixi run cn-test-no-engine && pixi run cn-test` (engine-linked).

**C7 — CN service + docs + e2e.**
Sharpen the destination-guard text (`compute_node_service.rs:742-747`) to name #838 and the
workaround; service test for a two-sender merge receiver (`per_exch_num_senders = 2`, harness
`:1404-2267`); update `DEMO.md:69,122` (drop the SET where two-phase now works, keep it
documented for grouped/avg). E2E on `pixi run cluster2`: Q6 **without** the session variable →
assert `|revenue − 61567694.9502| < 1e-6` (FP64 sums are order-dependent — do not assert exact
digits), logs show the hop carrying one row per sender; regressions: same query with
`agg_stage=1`, and single-CN `pixi run cluster` (first EXPLAIN to check whether AUTO still plans
two fragments there).
Verify: mysql transcript + `pixi run cn-test-no-engine` + `pixi run pre-commit run -a`.

**C8 (optional / follow-up issue) — avg.**
Partial `avg(x)` → two measures `sum(x), count(x)` (the engine already represents avg this way
internally, `aggregate_op_util.cpp:83-99`); merge = `sum/sum` + division ProjectRel. Needs a
synthetic-slot / output-name-override mechanism (width checks at `descriptor_table.rs:285-292`,
`lib.rs:270-290`). If not taken, the loud error stands and this becomes the follow-up issue.

**Final step (not a commit — local docs):** update `REVIEW-GUIDE.md` (add C1-C7 to the
per-commit review guide, refresh the riskiest-lines ranking — the wire-type model and the guard
atomicity join it) and `ROADMAP-8CN-TPCH.md` (§2 two-phase agg: scalar path DONE, grouped
still gated by #1 partitioned output; dependency table + shortest-path update; note the
FFI-marker assumption was overturned).

## 5. Non-goals (v1)

1. avg in two-phase plans (loud error; v2 = plan-shape rewrite, C8).
2. Grouped two-phase / hash-partitioned shuffle — translator-blocked; unblocking is the
   partitioned-output item (#838). Nothing here precludes it: the substitution table and wire-type
   rule are grouping-agnostic.
3. 3/4-phase DISTINCT plans, `multi_distinct_*`, opaque VARBINARY states (`ndv`, `hll`,
   `max_by`…), LIST-typed states (pack round-trip unverified).
4. Engine phase awareness / operator changes / any FFI addition beyond C1's validation.
5. Decimal-exact sums (two-phase inherits the existing FP64 lowering, doesn't change it).
6. The pre-existing failed-sender cancellation gap (`compute_node_service.rs:556-566`) — noted:
   two-phase makes the merge fragment always depend on N senders, so a one-sender failure hangs
   until the FE timeout more often. Tracked in ROADMAP §7, not fixed here.

## 6. Risks (ranked by blast radius)

1. **Wire-type model diverges from what the engine binds** → reinterpreted bits → silently wrong
   numbers on every two-phase query. C1 converts it to a loud boundary error; C2 pins the real
   types on GPU; C4's tests pin the model.
2. **Wrong merge substitution** (count merged as count) → wrong counts, no error. C2 is the
   oracle; C6 asserts the substitution in translation.
3. **Guard atomicity**: relaxing the node guard without the expression guard → merge node
   mistranslated as one-shot → double-aggregated garbage. C3 replaces both in one commit.
4. **FP64 non-associativity** → two-phase answer differs in the last ulps from one-phase (and
   run-to-run by arrival order). Every exact-equality assertion must move to tolerance/prefix.
5. **Sender-side aggregation under a STREAMING_SINK has zero coverage today** — an empty-input
   CN could hang rather than error. C2's empty-sender section runs before any translator work.

## 7. Open questions

| # | Question | Blocks | How answered |
|---|---|---|---|
| Q1 | Engine-produced types for min/max(DECIMAL64) (does `cudf_utils.hpp:198-201` reject p≤4?), min(DATE), grouped int-sum widening | C4 allowlist | C2 (read the produced types) |
| Q2 | Ungrouped agg under a STREAMING_SINK emits exactly one batch, incl. empty scan input? | C2 assertion; gates C7 | C2 empty-sender section |
| Q3 | Is the merge measure's argument always a bare SLOT_REF on the wire? (EXPLAIN says yes) | C6 pre-pass | `SIRIUS_CN_DUMP_FRAGMENTS` on the live cluster, ~10 min, before C6 |
| Q4 | Does single-CN AUTO still plan two fragments for Q6 (is the single-CN path genuinely exercised)? | C7 regression claim | EXPLAIN on `pixi run cluster` |
| Q5 | Exact two-phase digits / does the FE cast to the DECIMAL128(38,4) output slot before MySQL text encoding? | C7 assertion form only | measure during C7 |
| Q6 | avg expansion cost (synthetic slots vs `output_exprs`) | C8 only | decide before scheduling C8 |

## 8. Progress log

- 2026-08-05 **C1–C7 landed** (see `git log` / `REVIEW-GUIDE.md` Part 3): guards, engine oracle
  tests, phase classifier, wire-type model, partial + merge translation, CN service + docs. Q6
  runs two-phase without `agg_stage=1`.
- 2026-08-06 **C8 avg (translator crate; committed `bd232c40`)** — partial `avg(x)` →
  `sum(cast(x AS DOUBLE)) + count(x)` (deterministic `__count` name suffix), merge = `sum/sum`
  + finalize ProjectRel (`if count==0 then NULL else sum/count`), exchange schema expansion
  (`partial_state.rs` wire columns, positional overrides keyed by exchange node id).
- 2026-08-06 **Engine empty-partition hang fix (committed `19d7cca2`: `task_creator.cpp`,
  `task_scheduler.{hpp,cpp}`)** — the grouped-avg gate exposed a pre-existing defect: a hash
  partition owning zero keys delivers an input stream that ends with zero batches between
  `build()` and `run()`; the pipeline finishes before the query registers, no task is ever
  created, the completion handler never fires, `execute()` hangs forever, silently. Fix: an
  exhausted stream head re-checks pipeline status + new
  `task_scheduler::complete_query_if_finished()`. Two rejected attempts recorded: guarding on
  `!is_pipeline_finished()` (skips the completion signal) and draining the stream (deadlocks
  the manager thread's own pool slot). Verified: new Rust GPU tests
  `fragment_over_an_empty_input_stream_terminates` and `merge_fragment_over_an_empty_partition`
  (hung 120 s before, ~1.2 s after).
- 2026-08-06 **HUGEINT wire fix (in `bd232c40`)** — live gate exposed: the finalize projection passed the
  merged `count(*)` state through bare; DuckDB binds `sum(BIGINT)` as HUGEINT (the plan-level
  `downcast_hugeint_types` only relabels the aggregate node, not the projection above it), the
  receiving relay declared BIGINT, and the C1 guard refused the hop. Fix: `merge_projection`
  wraps **every** measure column in a throwing cast to its FE-declared output-slot type
  (`declared_measure_types`), so the wire never carries the widened type. Test
  `merge_avg_divides_the_summed_state` asserts the casts (I64 for the merged count, FP64
  around the division).
- 2026-08-06 **Engine overflow guard (committed `2c535b0e`)** — the downcast means integer sums run in cuDF's
  int64 accumulator, which wraps silently (no INT128 exists on the GPU; the projection's
  CAST is a physical no-op over an already-int64 column). New
  `throw_if_int64_sum_could_overflow` (`aggregate_op_util.{hpp,cpp}`): cheap min/max pre-check,
  refuses any INT64/UINT64-input SUM whose `valid_rows * max(|min|,|max|)` bound exceeds the
  accumulator (conservative, never lets one wrap), `sirius::invalid_input_exception` naming the
  HUGEINT→BIGINT downcast. Wired at all four live SUM sites: local ungrouped
  (`sirius_physical_ungrouped_aggregate.cpp`), local grouped (`gpu_aggregate_impl.cpp`), and
  both merges (`gpu_merge_impl.cpp`; COUNT states exempt — row counts stay far below int64).
  Rust negative `int64_sum_that_could_overflow_fails_loudly` pins the loud failure.
- 2026-08-06 **Verification** — translator 13+99, cn-test-no-engine 95+7, cn-test 98 (2
  ignored), C++ suite 2189 passed / 1 skipped (32,513,663 assertions), Rust GPU harness 19/19.
  Live gate on `cluster2` (2 CNs): the grouped
  `sum(l_quantity), avg(l_quantity), avg(l_extendedprice), count(*) GROUP BY rf, ls ORDER BY`
  over SF1 lineitem matches the DuckDB oracle exactly on all sums, counts, and
  `avg(l_quantity)`; two `avg(l_extendedprice)` values differ in the last 1–2 printed digits
  (A|F `…62166` vs `…621674`, N|O `…89083` vs `…890827`, ≤3e-11 abs ≈ 1–2 ulps) — the risk-4
  FP64 order-dependence, expected for a distributed sum. Scalar `avg(l_quantity)` =
  25.507967136654827, oracle-exact.
- Known residual (loud, not silent): a **non-avg** two-phase merge emits no finalize
  projection, so a merged integer `count`/`sum` still leaves the fragment HUGEINT-declared and
  a downstream relay refuses it at C1 (e.g. grouped `count(*)` + ORDER BY without an avg).
  Root fragments are unaffected (top-level results ride the plan-level downcast). Fix when it
  bites: emit the casting projection for every Merge node, not just expanded ones.
  **CLOSED 2026-08-07** — see the two entries below.
- 2026-08-07 **Finalize projection for every Merge node (committed `bb066e90`; closes the residual above)** —
  `node_translator.rs`: the `AggPhase::Merge if expanded` arm became `AggPhase::Merge`, so every
  merge aggregation leaves through `merge_projection`'s throwing casts to the FE-declared
  output-slot types (as the type mapper renders them — a DECIMAL128(38,s) slot is FP64 on both
  ends), not only avg-expanded ones. `merge_aggregation_translates_with_substituted_functions`
  now asserts the projection on a non-avg `sum(decimal)+count` merge: casts to FP64/I64 over
  fields 0/1. **Live repro (the residual was real):** on `cluster2`, `SELECT l_returnflag,
  count(*) c FROM <SF1 lineitem via FILES()> GROUP BY 1 ORDER BY c` plans as scan/partial →
  merge-finalize + SORT (fragment 1) → MERGING-EXCHANGE → result (fragment 0) — the merge
  output crosses a further boundary. Pre-fix binary: refused at the C1 guard ("relay into
  stream 5 … declared VARCHAR but the source sink produces HUGEINT"). Post-fix: the same
  fragment shape with `ORDER BY l_returnflag` returns oracle-exact counts
  (A 1478493 / N 3043852 / R 1478870, Σ = 6001215), and grouped
  `count(*), sum(l_quantity) GROUP BY rf, ls ORDER BY 1,2` matches the DuckDB oracle exactly.
- 2026-08-07 **New finding (distinct, pre-existing, NOT fixed here): sort-tuple column order
  across a fragment boundary.** With the HUGEINT fix in place, `ORDER BY c` (a measure) still
  refuses the hop — now "declared VARCHAR but the source sink produces BIGINT" at column 0.
  The sender's sort materialization projects `sort_tuple_slot_exprs` in FE list order
  (`node_translator.rs:1230-1235`, order-by keys first), while the receiver derives the stream
  schema from the sort tuple's materialized slot order (column_pos/slot-id). When the sort key
  is not the first tuple slot, the two orders diverge and the C1 guard refuses the hop (loud,
  not wrong results). Independent of aggregation phase; any cross-fragment ORDER BY on a
  non-leading column should hit it. Needs its own fix (reorder the projection to tuple-slot
  order, or override the receiver schema positionally like the merge pre-pass).
- 2026-08-07 **Engine-derived conformance gate for the wire-type model (committed `830380f4`; kills
  the duplicated registry risk).** New read-only FFI accessor `Fragment::output_types()`
  (`sirius_ffi.{hpp,cpp}`, cxx bridge + safe wrapper in `rust/crates/sirius{-sys,}`): the built
  fragment's sink column types as DuckDB type-name strings — exactly what the C1 relay guard
  compares. New engine-linked test `wire_type_parity::wire_type_model_matches_the_engine`
  (`experimental/starrocks/src/wire_type_parity.rs`, cn-test only; the translator crate keeps
  zero engine dependency): for all 38 rows instantiating every `wire_columns` arm
  (sum × {decimal64, decimal128, tinyint..bigint, float, double}, count × {*, col, varchar},
  min/max × 12 scalar types incl. DATE/DATETIME/VARCHAR/DECIMAL32/64/128, avg × {bigint,
  double, decimal64}) × {ungrouped, grouped}, it translates the partial and merge fragments
  with the real translator, builds each through the real engine path, and asserts (hop 1)
  partial sink types == the merge fragment's declared stream schema (the model, rendered by
  the receiver-side exchange override) and (hop 2) merge sink types == a downstream consumer
  fragment's declared schema (the finalize projection's contract). Every row built through the
  translator path — no direct-Substrait fallbacks needed; a row that fails to translate or
  build fails the suite with its label. Model drift is now a CI failure, not a refused hop.
- 2026-08-07 **Verification** — translator 13+99; cn-test-no-engine 95+7; cn-test 99 passed
  (2 ignored; +1 = the parity gate, all 76 row-cases green); C++ suite 2189 passed / 1 skipped
  (32,513,712 assertions); Rust GPU harness 19/19 (now also pins `output_types()` before/after
  build beside the relay-guard negative); live gate on `cluster2` as above.
