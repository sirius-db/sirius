# Issue #1010 v2 Design and Delivery Review

**Reviewed documents:**

- [General Dynamic Filters (v2)](issue-1010-dynamic-filter-sip-design-v2.md)
- [Issue #1010 GitHub Delivery Plan (v2)](issue-1010-github-delivery-plan-v2.md)

**Assessment:** the core architecture is sound and should not be redesigned. Retain the direct
pre-partition endpoint, waiter-free publication, per-key scan preference, existing filter/runtime
reuse, and guarded CTE/DELIM/shared-topology exclusions. The findings below are bounded contract gaps
that should be resolved before the design is considered hardened and implementation-ready.

## Required findings

### 1. High: scan target coordinates can be remapped twice

The proposed target metadata stores a `consumer_output_ordinal`
([delivery plan](issue-1010-github-delivery-plan-v2.md#L177)) while retaining scan-boundary remapping
([delivery plan](issue-1010-github-delivery-plan-v2.md#L271)). Existing scan channels do not accept an
output ordinal: `sirius_dynamic_filter_set::push_filter` accepts a coordinate in DuckDB
`column_ids` space and remaps it internally before storing the filter
([contract](../../src/include/op/sirius_dynamic_filter.hpp#L425),
[implementation](../../src/op/sirius_dynamic_filter.cpp#L461)). Both scan implementations install
that remap at plan time
([Parquet](../../src/op/scan/parquet_gpu_ingestible.cpp#L373),
[DuckDB-native](../../src/op/scan/duckdb_native_gpu_ingestible.cpp#L251)).

Passing the documented output ordinal through the retained scan API therefore remaps it twice. The
existing reorder test gives a concrete case: raw columns `[a,b,c]` and output `[c,a]` produce the
mapping `[1, none, 0]`
([test](../../test/cpp/scan/test_duckdb_native_dynamic_filter_remap.cpp#L173)). A filter intended for
output `c` at ordinal `0` would be remapped to output ordinal `1` and applied to `a`. Because both are
`INT32`, a type check does not prevent the error, and valid rows can be removed.

**Required amendment:** distinguish a scan channel's `column_ids`-space push coordinate from a direct
endpoint's output coordinate, or specify an explicitly remap-bypassing push API. Add a reordered,
same-typed scan regression that would fail on double remapping.

### 2. High: target metadata cannot represent per-key route membership

Routing is selected independently for each admitted key
([design](issue-1010-dynamic-filter-sip-design-v2.md#L78)), and R3 requires scan-routed keys not to
receive a direct endpoint
([delivery plan](issue-1010-github-delivery-plan-v2.md#L93)). However, every target's ordinal and type
arrays are described as aligned with the complete `admitted_key[]` array, with no admitted-key index,
active bitmap, optional coordinate, or absent sentinel
([schema](issue-1010-github-delivery-plan-v2.md#L165)).

A composite producer with one scan-routed key and one direct-routed key therefore has no valid target
record. The current publisher demonstrates the natural failure mode: it loops over every key for
every target ([publisher](../../src/op/dynamic_filter_publisher.cpp#L303)). Fabricating an ordinal for
an absent, same-typed key can apply a filter to the wrong column and remove valid rows.

The same target record also does not encode the promised `membership_only` capability for direct
endpoints ([delivery plan](issue-1010-github-delivery-plan-v2.md#L269)). The current channel exposes
only general open/closed acceptance, and the publisher pushes both zone-map and membership filters to
every target.

**Required amendment:** use a sparse target-key binding such as
`{admitted_key_index, target_coordinate}`, or define an explicit absent coordinate. Also encode the
target's accepted filter kinds or route kind so direct endpoints cannot receive scan-only zone maps.

### 3. High: source-filter construction is outside memory admission and measurement

The design requires resource admission to cover all concurrently live application buffers and
replica work ([design](issue-1010-dynamic-filter-sip-design-v2.md#L148)). The delivery plan defines a
concrete change only for the consumer operator's one-input-footprint estimate
([delivery plan](issue-1010-github-delivery-plan-v2.md#L287)); it does not define admission for the
source membership representation.

The current boundary makes this omission observable:

- Source IN-list/Bloom construction uses the source space's default allocator
  ([publisher](../../src/op/dynamic_filter_publisher.cpp#L112),
  [construction](../../src/op/dynamic_filter_publisher.cpp#L241)).
- Publication runs during sink delivery from the completed build batch
  ([hash join](../../src/op/sirius_physical_hash_join.cpp#L1437)).
- Task peak/history is recorded before `publish_output()` invokes the sink
  ([history](../../src/pipeline/gpu_pipeline_task.cpp#L460),
  [sink call](../../src/pipeline/gpu_pipeline_task.cpp#L493)).
- Sink-side OOM is outside the compute OOM-reschedule path and becomes a generic task/query error
  ([executor](../../src/pipeline/gpu_pipeline_executor.cpp#L415)).
- Destination replicas have explicit scoped reservation admission, but it does not cover the source
  representation ([IN-list replica](../../src/cuda/sirius_dynamic_in_list_filter.cu#L266)).

R1/R2 expand publication to no-hint keys that previously had no scan target, so this existing Phase-1
seam becomes part of the #1010 rollout. Peak measurement alone cannot establish the admitted bound
because current history does not include sink publication work.

**Required amendment:** define a source-filter construction estimate and reservation on the
publication stream, including its lifetime overlap with the build batch and destination replication.
Specify denial/OOM behavior and test that admitted peak and recorded peak include this work.

### 4. Medium: computed-key provenance is lost before admission

Computed keys are explicitly outside the initial scope
([design](issue-1010-dynamic-filter-sip-design-v2.md#L58)) and are required negative coverage
([delivery plan](issue-1010-github-delivery-plan-v2.md#L340)). The current comparison-join planner,
however, materializes complex equality expressions and rewrites them to ordinary
`BoundReferenceExpression`s before the conditions are wrapped into the physical join
([materialization](../../src/planner/sirius_plan_comparison_join.cpp#L374),
[rewrite](../../src/planner/sirius_plan_comparison_join.cpp#L440)).

The proposed placement helper runs after physical-plan creation, projection folding, and verification
([delivery plan](issue-1010-github-delivery-plan-v2.md#L119)). At that point a materialized computed key
is indistinguishable from an originally direct reference, so the documented negative scope cannot be
enforced without extra provenance.

**Required amendment:** classify each condition side as direct, cast, or computed before
`materialize_expression_join_keys`, and carry that small immutable classification into the physical
join/admitted-key metadata. This does not require lineage or origin tracing through the probe tree.

### 5. Medium: the explicit-execution error owner graph is incomplete

The delivery plan explicitly discusses freshness and ownership for `gpu_execution`
([delivery plan](issue-1010-github-delivery-plan-v2.md#L293)), but its bind data separately owns both a
`sirius_interface` and the prepared physical plan
([bind data](../../src/sirius_extension.cpp#L218)). During execution the active-query context retains
the engine and prepared plan. On GPU execution failure,
`sirius_execute_pending_query_result()` returns an error result without cleaning that active state
([error path](../../src/sirius_interface.cpp#L187)), and `sirius_execute_query()` returns the error
normally ([caller](../../src/sirius_interface.cpp#L212)). CPU fallback can then replace the result
while the failed active query remains retained
([fallback](../../src/sirius_extension.cpp#L652)).

`SiriusContext::QueryEnd` resets the context-owned query
([QueryEnd](../../src/sirius_context.cpp#L214)), but that alone does not prove that the bind-owned
engine, physical plan, channels, filters, and replicas have been released before pool-baseline checks
or manager teardown.

**Required amendment:** include the bind-data/active-query branch in the owner graph. R3 should cover
explicit-path success, GPU error with fallback, GPU error without fallback, cancellation, and early
drain, proving active-query cleanup and GPU/HOST pool return to baseline. Alternatively, explicitly
exclude this execution surface from the initial rollout.

## Normative contract reconciliation

These are smaller than the findings above, but R0 should resolve them explicitly rather than relying
on a general instruction that the documents agree.

### DuckDB coupling count

The design and delivery plan state that exactly three DuckDB join-filter couplings remain after R1
([design](issue-1010-dynamic-filter-sip-design-v2.md#L118),
[delivery plan](issue-1010-github-delivery-plan-v2.md#L202)). Current planning also reads
`JoinFilterPushdownInfo::build_side_has_filter` to decide whether to wire publication
([planner](../../src/planner/sirius_plan_comparison_join.cpp#L535)). This is producer-policy input, not
scan-target discovery.

R1 simultaneously requires existing hinted scan-filter results and plan counts to remain unchanged
([acceptance](issue-1010-github-delivery-plan-v2.md#L61)). State explicitly whether R1 removes this
gate, reproduces it from Sirius-owned evidence, or acknowledges it as an additional planning-time
dependency.

### Destination failure policy

The delivery plan says destination reservation or copy failure may make a filter unavailable on the
affected device while publication still finishes
([delivery plan](issue-1010-github-delivery-plan-v2.md#L258)). The retained Phase-1 multi-GPU contract
says serious CUDA construction, transfer, or synchronization failures propagate and fail the task or
query ([Phase-1 contract](dynamic-filters-multi-gpu.md#L39),
[transfer contract](dynamic-filters-multi-gpu.md#L255)).

Define reservation denial and intentional target unavailability as safe omissions, then separately
state whether CUDA enqueue, synchronization, and invariant failures fail publication/query. R3's
failure tests and terminal counters need one unambiguous expected policy.

### Superseded documentation and links

[dynamic-filters.md](dynamic-filters.md#L399) still describes a new `sirius_sip_route` and a new hash
join probe code path, both rejected by the v2 design. [README.md](README.md#L45) links the nonexistent
v1 design and delivery filenames, and the v2 design's `Supersedes` link also names a nonexistent v1
file. Update these as part of R0.

## Confirmed decisions that should remain unchanged

No release-blocking scheduling or publication-dominance flaw was found for the declared tree
topology:

- Build publication is synchronous and completes before the build delivery returns
  ([hash join](../../src/op/sirius_physical_hash_join.cpp#L1437)).
- The executor schedules downstream consumers only after the producing task returns
  ([executor](../../src/pipeline/gpu_pipeline_executor.cpp#L448)).
- Once build data exists, P's scheduling hint follows its probe producer
  ([hash join hint](../../src/op/sirius_physical_hash_join.cpp#L488)).
- Pipeline conversion preserves build-first dependencies
  ([converter](../../src/pipeline/sirius_pipeline_converter.cpp#L68)), and query startup schedules
  only the first build-first scan ([scheduler](../../src/pipeline/task_scheduler.cpp#L175)).
- Pre-partition placement matches the current operator-data contract: the dynamic-filter operator
  consumes/returns pipelineable data, while CONCAT requires partitioned data.
- CTE fan-out and DELIM cycle-breaking are exactly the cases covered by the proposed two-stage
  topology guard.

Accordingly, do not add lineage, ordered activation, a route registry, a scan-router replacement, or
a new hash-join probe path to address this review. The required changes are limited to metadata
precision, admission/lifetime proof, and normative reconciliation.

## Disposition (2026-07-20)

Every finding was independently re-verified against the code before amendment; all were accepted,
three with refinements recorded below. Amendments landed on this branch in the same commit as this
section.

| Item | Verification outcome | Amendment landed |
|---|---|---|
| 1 — double remap | Confirmed: `push_filter` remaps at push time; a direct endpoint's channel installs no remap, so its push ordinal is identity | Target schema now carries per-target `channel_push_ordinal` defined in the channel's push-coordinate space (`column_ids` for scan targets, probe-child output ordinal for direct); reordered same-typed regression added to the correctness matrix |
| 2 — dense alignment | Confirmed for routing correctness. Refinement: the membership-only sub-claim is waste and gate hygiene, not correctness — a membership-only apply already ignores AST-only filters at dispatch | Schema rewritten to sparse `key_binding[] {admitted_key_index, channel_push_ordinal}` plus `route_class` and `accepted_filter_kinds`; mixed-route composite row added to the matrix |
| 3 — source admission | Confirmed on all five sub-claims; additionally, unadmitted source allocations count into global usage and can press concurrently admitted tasks into OOM | R2 admits source construction (budget from existing per-kind estimators, scoped reservation on the publication stream); denial = skip + counter, never query failure — the prior "reaches FAILED (or fails the query)" sentence was narrowed to invariant/ordering failures; R3 peak-inclusion evidence and telemetry counters added |
| 4 — computed-key provenance | Confirmed mechanism (materialization erases provenance; projection folding can erase the injected projection; the planner already computes the trichotomy). Refinement: an admitted materialized key would be value-correct — the defect is an unenforceable scope contract, not a runtime hazard | `key_shape` classification captured before materialization and carried alongside `key_casts`; admission consumes the carried record; helper-level computed negatives (probe- and build-side) added to the admitted-key contract and the correctness matrix |
| 5 — explicit-path owner graph | Confirmed. Bounding refinement: plain materialized queries destroy bind data at outer-query end, before `QueryEnd` callbacks — the exposure is held prepared statements, partially drained streaming results, and the CPU-fallback window. Allowlist-based exclusion found incoherent (the shared plan generator has no surface discriminator) | Owner-graph bullet extended with the error branch; R3 lifecycle coverage adds explicit-path success, GPU error with/without fallback, cancellation, early drain, and held-prepared-statement; exclusion re-specified as a surface gate at the plan-generation call site |
| Coupling count | Confirmed: `build_side_has_filter` is a planning-time all-or-nothing wiring gate, reproducible via an `IsFiltering`-equivalent walk over the intact logical build child | Both docs now state four planning-time inputs; hinted scan routes keep the DuckDB gate (behavior-preserving), the direct route uses the Sirius-owned filtered-build-subtree check |
| Destination failure policy | Confirmed conflict. Refinement: the code implements catch-and-omit for ALL filter kinds — `dynamic-filters-multi-gpu.md`'s IN-list/Bloom propagate-on-failure split is false against the code (its own diagram contradicted it) | Docs aligned to implemented behavior: per-destination-target failures omit that replica; source-side construction and pre-publication ordering failures propagate to `FAILED`; `dynamic-filters-multi-gpu.md` corrected |
| Stale docs and links | Confirmed on all three sub-items | `dynamic-filters.md` Phase 2 rewritten to the v2 shape (no `sirius_sip_route`, no probe-path change); README rows point at the v2 files; the v2 `Supersedes` line cites PRs #1179/#1193 instead of the nonexistent v1 file |
