# General Dynamic Filters: Sideways Information Passing at Hash-Join Probe Inputs

> **Status (2026-07-09) — architecture retained, delivery phasing superseded.** Track A shipped
> delete-only in merged #1134 (`1eecaf97`); A1/A2/A3 never merged. Track B is deferred and
> blocks nothing. The [Track C re-evaluation](issue-1010-implementation-plan.md#status-reconciliation-and-track-c-re-evaluation-2026-07-09)
> is authoritative for current ownership, PR boundaries, dependencies, telemetry, execution
> reset, and gates. In particular, former C1c is absorbed into C1b; the implementation vocabulary
> uses `publication_terminal` / `target_visible` and
> `NO_MATERIALIZATION(reason)` where historical sections below say
> `publication_completed` / `channel_filter_visible` and `SEALED_EMPTY`.

**Status:** finalized architecture reference — converged after cross-review (Claude ↔ Codex);
current implementation phasing lives in the Track C re-evaluation linked above

**Baseline:** Phase 1 dynamic table-filter pushdown, merged by
[#794](https://github.com/sirius-db/sirius/pull/794); all `file:line` references below are at
`dev` commit `506a1d9f` unless a pinned DuckDB file is named. Track C now builds on
`dev` `fac81e87`, which already contains #1134; treat old line numbers as approximate.

**Targets:** [#1010](https://github.com/sirius-db/sirius/issues/1010) (extend dynamic filtering to
non-scan probe inputs) and [#1014](https://github.com/sirius-db/sirius/issues/1014) (remove the
unbounded global build-task preference after measurement). v1 intentionally covers only
DuckDB-admitted producer/key candidates; it does not claim every theoretically valid SIP shape.

**Related:** [dynamic-filters.md](dynamic-filters.md),
[dynamic-filters-multi-gpu.md](dynamic-filters-multi-gpu.md).

---

## Decision summary

1. **The new consumer is an intermediate hash join's probe checkpoint.** For a producing join `P`
   and a different hash join `C` in `P`'s probe subtree, membership filters are applied after
   `C`'s probe `CONCAT`/repository pop and immediately before `C` prepares keys and hashes the
   probe batch. This is the Phase 2 consumer described by `dynamic-filters.md`; it is not a unary
   operator inserted in the pre-partition feeder.

2. **Scan and join-probe consumption are layered.** DuckDB-provided scan consumers remain wired.
   Sirius additionally installs a dedicated consumer at every eligible intermediate hash-join
   probe on the admitted branch. The scan is the earliest pruning opportunity; each join-probe
   checkpoint catches batches that passed an earlier checkpoint before publication. Filter
   construction and device replication happen once; immutable filter objects fan out to separate
   consumer channels.

3. **DuckDB is authoritative for candidate admission, not GPU representation.** v1 creates a
   producer/key candidate only when the pinned DuckDB `JOIN_FILTER_PUSHDOWN` pass emitted
   `JoinFilterPushdownInfo` with non-empty `probe_info`, and only for condition indexes named by
   DuckDB. Sirius may narrow that set. Sirius deliberately retains its own GPU materialization
   policy (exact-set/Bloom/none membership plus an independently optional zone map); it does not
   claim to reproduce DuckDB's CPU runtime choice of min/max, optional IN, or optional Bloom.

4. **Discovery, physical resolution, and topology freeze are explicit.** After binding resolution
   but before recursive planning moves logical conditions, a pure pass creates pending target
   values, channels, and logical-node tokens. `create_plan` resolves both endpoints despite the
   child-before-parent order. After pipeline conversion proves branch/runtime identity, one
   finalizer freezes producer and consumer plans together. Runtime sees no DuckDB pointers or
   mutable pending route state.

5. **v1 is opportunistic, without predicting coverage.** Every probe batch applies whatever
   complete filters are visible. No existing hint or scheduler behavior is treated as an ordering
   contract: an intervening join's sibling-build hint drives that join's build, not the outer
   filter producer. C3 measures whether opportunistic join-probe checkpoints see enough filters.
   If not, a narrowly admitted, exactly-once ordered activation protocol lands before default-on.

6. **#1014 is measure → disable → delete, not an asserted memory win.** The existing pass only
   reorders already-created feeder tasks; it is not a bound on live hash tables. Instrument first,
   run the legacy/off comparison, default it off only if coverage, wall time, and resident memory
   pass, retain a rollback release, and then delete it. If the gate fails, land route-local
   ordering and repeat; do not replace it with another global join-aware scheduler policy.

7. **Phase 0 is deferred by decision (2026-07-08): no fork, no backport.** The pinned DuckDB
   v1.5.4 target walk crosses `LIMIT`/`TOP_N`, an upstream wrong-results bug fixed by
   duckdb/duckdb#22963. The exposure is latent at default settings (masked by
   `late_materialization`, `compressed_materialization`, and scheduling races) and already exists
   in merged Phase 1; Sirius's own SIP crossing rules stop at row selectors, so this work does not
   widen it. We accept the latent risk and pick up the fix at the next pin bump to a released
   DuckDB containing it (tracked in sirius-db/sirius#1123). Two rules survive the deferral: an
   unpatched CPU run is never an oracle for LIMIT/TOP-N-shaped tests, and the pin-bump ships the
   sentinel regressions (§ "Phase 0").

---

## Scope and non-goals

### v1 scope

- Producer: a Sirius `BUILD_PROBE` hash join whose DuckDB metadata contains a non-empty dynamic
  filter target and whose producing semantics pass the Sirius gates below.
- Filter kinds: existing Phase 1 membership filters; scan targets may also retain existing zone
  maps. No new filter representation.
- Consumers: existing scan reader/post-decode endpoints plus branch-local probe checkpoints in
  intermediate Sirius equality hash joins running in BUILD_PROBE or STANDARD mode.
- Candidate source: pinned DuckDB metadata only. Sirius traces placement but does not invent a
  producer/key absent from DuckDB's candidate set.
- Runtime: opportunistic per-batch consumption behind a default-off SIP flag, followed by measured
  default-on or a route-local ordered phase.

### Explicit non-goals

- Producing filters from STANDARD/partitioned joins.
- Candidate discovery for keys for which DuckDB found no scan target.
- Mixed-provenance composite keys rejected by DuckDB's all-or-nothing target walk.
- Aggregate, window, set-operation, CTE/materialization, or shared-source crossing.
- Applying a producer's filter at its own authoritative probe.
- Moving filtering into the global scheduler, blocking a worker, or waiting with wall-clock
  timeouts.
- Exact parity with DuckDB's CPU filter representation and cost heuristics.

---

## Verified execution-model facts

These facts determine the design; they are not assumptions delegated to implementation.

**The required point is after the consumer join's probe repository.** Sirius splits every
intermediate join with `PARTITION` + `CONCAT` (`src/pipeline/sirius_pipeline_converter.cpp:459-575`).
A unary operator placed before a join in the pre-split chain becomes part of the feeder before
that pair. It cannot revisit a batch that already passed the unary operator and is queued in the
partition/concat repositories. The Phase 2 catch-up point is instead inside the consuming hash
join, after it obtains its branch-specific probe batch and before the first
`prepare_join_keys(..., is_left_side=true)`/hash operation
(`src/op/sirius_physical_hash_join.cpp:910-1315` — BUILD_PROBE probe keys at `:975`, the STANDARD
probe path from `:1057`; the checkpoint must cover both since STANDARD joins are eligible
consumers, and the original probe batch is re-read at many later sites — `:983,:990,:1021,:1057,
:1094,:1238,:1289` — which is exactly why the `probe_batch_handle` discipline below is mandatory).

**There is no happens-before edge from an outer producer to an intermediate consumer.**
`setup_pipeline_parents` rebuilds dependencies from repository wiring
(`sirius_pipeline_converter.cpp:1140-1173`). The producer build subtree and a lower join's probe
pipeline are normally unordered. Publication may precede a probe batch, land between batches, or
arrive after the consumer drains. Opportunistic consumption is correctness-safe in all three
cases; only its benefit changes.

**The intervening join's build hint is not the filter producer's hint.** A probe `PARTITION` can
delegate to its own build sibling until partition count is known
(`src/op/sirius_physical_partition.cpp:264-312`). In a SIP route, that drives consumer `C`'s build,
not outer producer `P`'s build. The design makes no early-publication prediction from this fact.

**Hints are stateful.** `sirius_physical_hash_join::get_next_task_hint` transitions
`NOT_BUILT → SCHEDULING` while returning `READY`; task-input creation performs
`SCHEDULING → SCHEDULED` (`sirius_physical_hash_join.cpp:486-547`). Any ordered gate must run
before hint resolution, not between the hint and input pop.

**The priority pass only reorders queued feeder tasks.**
`collect_filter_build_pipelines` finds pipelines feeding plan-wired producer build ports, and the
management loop prefers compatible queued tasks from that set
(`src/pipeline/task_scheduler.cpp:164-214,404-416`). It does not create tasks, preempt work, limit
live producers, or directly prioritize the join task that pins the build batch and constructs the
hash table. Its downstream memory effect is empirical.

**Publication is conditional.** Only runtime `BUILD_PROBE` publishes
(`sirius_physical_hash_join.cpp:1319-1428`). STANDARD/MIXED resolution, empty build, downgraded or
unavailable build data, policy suppression, a drained target, failure, and cancellation can all
produce no usable filter. Opportunistic consumers simply pass through; ordered consumers require
a terminal completion outcome for every path.

**The existing channel is N-producer/one-consumer.** A Phase 1 scan's
`DynamicTableFilterSet` may be referenced by several producing joins. Its Sirius channel is closed
by one logical scan consumer and accepts filters from multiple producers
(`src/include/op/sirius_dynamic_filter.hpp:389-417`). A SIP endpoint is different: it gets a
dedicated one-producer/one-consumer channel so its close, gate, telemetry, and possible activation
state cannot affect the scan or another join.

**`BUILD_PROBE` state is a resident floor only after the join task runs.** The build batch is pinned
and the persistent hash table is constructed in `sirius_physical_hash_join::execute`
(`:927-969`), then held until finalize (`:1420-1427`). Filter replicas are likewise query-lived.
Repository batches before that point remain subject to normal downgrade behavior. Telemetry must
measure these lifecycle stages separately.

**Two current planner signals are dead post-resolver.** `build_key_domain_cardinalities` and
`trace_binding_to_get` expect `BOUND_COLUMN_REF`, while `ColumnBindingResolver` has already
rewritten join conditions to `BoundReferenceExpression`. Domain coverage is therefore unknown and
the current publisher gate is inert. Candidate extraction after the resolver is still useful:
`cond.left.index` identifies the producing join's left-child output position, from which the
corresponding `ColumnBinding` can be recovered via the child's bindings.

---

## Issue #1014: measure, disable, then delete the priority pass

The target end state is a filter-agnostic scheduler. The current global pass is a layering
violation—the scheduler `dynamic_cast`s to a hash join and reads dynamic-filter policy—but
correctness-safe deletion does not by itself prove a memory reduction or impose a numerical
concurrency bound.

### Sequence

1. **A1 — instrumentation only.** Add query-level resident high-water, task/build lifecycle
   counts, and channel coverage telemetry without changing dispatch.
2. **A2 — controlled opt-out.** Add
   `dynamic_filter_build_priority={legacy,off}`, default `legacy`, retaining the implementation.
3. **A3 — default-off cutover.** If the acceptance gate passes, default to `off` while preserving
   the rollback switch for one release.
4. **A4 — deletion.** After the default-off release confirms the result, delete
   `collect_filter_build_pipelines`, `_filter_build_pipelines`, and the priority `pop_if` branch.

### Acceptance gate

Run three configurations—dynamic filters disabled, enabled with legacy priority, and enabled with
priority off—on nested/star TPC-H shapes plus a synthetic many-join chain. Record:

- relational/bag-equivalent results (exact row order only for order-guaranteed queries) and wall
  time;
- rows/batches reaching scan and SIP channels before/after publication;
- per-memory-space resident high-water;
- queued/running prioritized feeder tasks;
- build batches delivered to joins;
- live pinned `BUILD_PROBE` batches and live persistent hash tables; and
- resident filter-replica bytes.

The distinction is load-bearing: the pass directly controls only queued-task selection. It may
affect later residency indirectly, but the design does not assume the sign or size of that effect.

Coverage is diagnostic rather than an independent veto: the gate passes only when priority-off has
no material wall-time or resident-peak regression outside measured run variance, while coverage
explains any movement. A lower peak is a hypothesis, not a premise.

If lost join-probe coverage causes a regression, keep the rollback path, land measured route-local
ordering, and repeat. Ordered join-probe activation cannot recover lost Parquet pruning or decode
avoidance. A material scan-I/O regression retains the rollback unless a separately reviewed,
bounded scan-split deferral exists. Track C is not presumed to recover A2 losses.

### Why not CAP or STAGGER by default

CAP and STAGGER retain join/filter topology in the global scheduler and need producer grouping,
in-flight counts, and consumer-liveness state that do not exist today. A hardware-tuned cap has no
principled default, and a consumer-liveness heuristic can still activate many bushy branches.
Own-consumer staggering also fails its stated goal for the case that matters most: a transitive
producer's own consumer sits high in the plan and becomes live late, so staggering delays that
build further and its filter reaches the fact scan later, not earlier.
Route-local ordered activation, if measurement requires it, expresses the actual dependency and
admits at most one ordered producer per runtime consumer pipeline. It is the preferred recovery.

Resolving #1014 means removing the unbounded global preference after this gate. It does not mean
that priority deletion alone bounds the total number of hash tables a query may legitimately use.

---

## Phase 0: repair the pinned DuckDB candidate source

> **Status: deferred (2026-07-08).** No Sirius-owned fork or backport. The fix is picked up at the
> next pin bump to a released DuckDB containing duckdb/duckdb#22963; the sentinel regressions
> below land with that bump. Until then the latent Phase 1 exposure is accepted, Phase 0 blocks
> nothing (C1c/C1e/C3 enablement proceeds), and the only standing obligations are: never use an
> unpatched CPU run as an oracle for LIMIT/TOP-N-shaped tests, and keep Sirius's own crossing
> rules stopping at row selectors (which also lets C3's lineage pass drop a traced target whose
> branch crosses a selector, when SIP is on). Tracked in sirius-db/sirius#1123; the B cluster doc
> is the pin-bump playbook.

Pinned DuckDB v1.5.4 treats `LOGICAL_LIMIT` and `LOGICAL_TOP_N` as transparent while walking a
probe subtree (`duckdb/src/optimizer/join_filter_pushdown_optimizer.cpp:58-72`). Applying a
build-derived predicate below either row-selection operator can change the selected relation.
Upstream duckdb/duckdb#22963 (`4c8c90db44`) makes both terminal.

This exposure already affects merged Phase 1; it is not introduced by SIP. At the pin bump:

1. Backport `4c8c90db44` onto a clean-clone-reachable Sirius-owned DuckDB branch, updating
   `.gitmodules` and the gitlink, or advance to a verified release containing it.
2. Keep the upstream SQL tests and add Sirius variants whose LIMIT/TOP-N input is produced by
   another join.
3. Use explicit expected rows, or join-filter-pushdown-disabled execution, as the oracle. An
   affected unpatched DuckDB CPU run is not a correctness oracle.
4. On the patched pin, verify CPU, GPU filters-off, Phase 1 on, and SIP on against that oracle.
   Keyed by publication-plan/target ID, assert that the outer producer does not cross LIMIT/TOP-N; a join
   wholly inside the selector input may still produce a legal local route.

Do not reimplement this target walk merely to route around the bug. The adapter consumes the
corrected DuckDB contract; the Sirius placement walk only chooses additional consumers for an
already-admitted candidate.

---

## DuckDB candidate contract

### Candidate parity, not materialization parity

The normative invariant is:

```text
Sirius producer/key candidates ⊆ pinned DuckDB optimizer candidates

Sirius runtime filter kinds need not equal DuckDB runtime filter kinds
```

A producing join enters the v1 candidate set iff:

```text
filter_pushdown != nullptr
&& !filter_pushdown->probe_info.empty()
&& at least one filter_pushdown->join_condition entry passes Sirius's gates
```

A non-null object with empty `probe_info` is statistics-only: DuckDB retained aggregates for
perfect-hash planning but found no dynamic-filter target. Sirius creates neither a scan route nor a
SIP route for it.

Conversely, non-empty `probe_info` means DuckDB planned a dynamic table-filter path for the named
key/target (normally at least min/max, subject to empty/NULL runtime outcomes). Sirius therefore
never materializes for a producer/key that DuckDB would not attempt; only the GPU representation
and cost decision may differ.

DuckDB's physical hash join selects min/max, optional IN, and optional Bloom using DuckDB settings,
hash-table state, cardinality estimates, perfect-hash state, and CPU cost assumptions
(`duckdb/src/execution/operator/join/physical_hash_join.cpp:714-905`). Sirius does not execute that
physical path. For an admitted key it independently chooses its GPU representation or suppresses
materialization. Telemetry records candidate admission and the two independently materializable
capabilities:

```text
duckdb_candidate = absent | statistics_only | admitted
membership_materialization = exact_set | bloom | none(reason)
zone_map_materialization = emitted | none(reason)
```

`build_side_has_filter` is snapshotted as `duckdb_build_subtree_has_filter_hint`. It is one input to
DuckDB's Bloom heuristic, not route admission, not proof that the join key was reduced, and not
evidence that DuckDB would actually materialize a Bloom.

### Version-pinned adapter

One `duckdb_join_filter_candidate_adapter` owns every version-sensitive read through two entry
points:

1. **Preservation**, called while the optimized DuckDB logical plan is copied.
2. **Extraction**, called after `ColumnBindingResolver` and before recursive planning moves
   `op.conditions` or children.

Preservation retains exact shared `DynamicTableFilterSet` identity:

```text
copied_join.probe_info[t].dynamic_filters.get()
    == copied_target_get.dynamic_filters.get()
```

This is a Phase 1 pairing invariant. Several producing joins may reference the same scan set; the
adapter must not deep-copy or replace either endpoint independently.

Extraction validates and fails closed on structural invariants it can prove locally:

- out-of-range or duplicate `join_condition` indexes;
- null target channel identities;
- `probe_info[t].columns.size() != join_condition.size()`; and
- a copied logical shape inconsistent with the pinned layout.

The lineage pass later validates that target column ordinal `j` actually corresponds to resolved
condition `join_condition[j]` along the selected branch. A base-table binding cannot be proven
equivalent to a resolved positional condition inside the adapter alone. Failure drops that target
rather than guessing the correspondence.

It then emits Sirius-owned immutable values and preserves three distinct index spaces:

```cpp
struct admitted_dynamic_filter_key {
  std::size_t duckdb_filter_ordinal;  // j in join_condition and probe_info[t].columns
  std::size_t condition_index;        // reordered logical join condition
  std::size_t sirius_key_ordinal;     // compact ordinal after Sirius narrowing
};
```

`join_stats` is not correlated with these values; it uses a different/original condition order.
Runtime publication never dereferences `JoinFilterPushdownInfo`.

### Sirius narrowing gates

Producer eligibility is release-mode fail-closed:

- producing join type is INNER or SEMI;
- producing execution mode is `BUILD_PROBE` at the publication claim;
- routed comparison is `COMPARE_EQUAL`;
- both keys are direct bound references after resolver indexing;
- no key cast;
- GPU membership type is supported; and
- the route contains at least one live target.

`COMPARE_NOT_DISTINCT_FROM` is excluded: NULL probe keys are matchable under null equality, while
current membership masks null-propagate and would drop them. ANTI/MARK/outer-producing joins are
excluded even if runtime mode selection can place some of them in `BUILD_PROBE`.

Implement the join-type predicate as a fresh pure function of `join_type`. Do not derive it from
`prove_unique_columns`' preservation switch (`sirius_plan_comparison_join.cpp:205-222`): that
switch computes data-dependent uniqueness preservation (`left_preserved = right_keys_unique`, even
for INNER), which is a different property from row retention and would mis-gate eligibility.

### Deliberate v1 coverage limit

DuckDB v1.5.4 walks the full candidate column vector as one unit. At a `LogicalGet`, one key whose
binding belongs elsewhere rejects the whole target. Thus a composite key split across different
sides of an intermediate join normally yields empty `probe_info`; Sirius never reaches per-key
fan-out for that shape.

Per-key fan-out remains useful for independently suppressing unsupported components of an already
admitted target. It does not recover mixed-provenance candidates. Those require either an upstream
per-key DuckDB descriptor or a separately reviewed Sirius discovery pass and belong to Track E.

---

## SIP topology and placement

### Normative physical point

For producing join `P` and a different intermediate hash join `C` in `P`'s probe subtree:

```text
C probe feeder
  → PARTITION
  → CONCAT
  → C.default repository
  → [SIP probe checkpoint]
  → prepare_join_keys
  → C hash probe
```

The checkpoint runs after `C` obtains its branch-specific probe batch and before any probe key
cast, hash lookup, or join output allocation. It is a composed part of `C`'s probe execution, not a
new physical source/sink in the converter.

Consequences:

- a batch decoded, partitioned, or queued before publication can still be filtered before `C`'s
  hash probe;
- shared upstream data is not mutated—the checkpoint owns or forwards `C`'s popped batch only;
- partition-count selection, CONCAT folding, and positional sibling wiring remain untouched; and
- the filter cannot save work already paid below `C`, including `C`'s input partition, but it can
  save `C`'s probe and all work above it.

`P` is never its own SIP consumer. Applying its build membership immediately before its own exact
probe would save only a failed lookup while duplicating the same test.

### Layered target model

For each DuckDB-admitted producer/scan branch, retain:

1. the existing scan reader and post-decode endpoint; and
2. one dedicated SIP endpoint at every eligible intermediate hash-join probe on that branch.

For a chain `scan → C1 → C2 → P`, the same filter may therefore be checked at the scan, `C1`'s
probe, and `C2`'s probe. This is intentional:

- the scan has the greatest possible savings when publication is early;
- `C1` catches batches that missed the scan;
- `C2` catches batches that missed both earlier checkpoints; and
- independent gates disable redundant downstream masks after observing that they keep nearly all
  already-filtered rows.

Reaching the base scan suppresses only a duplicate checkpoint in that same scan pipeline. It never
removes join-probe checkpoints already collected above intervening joins.
If the admitted branch contains no intermediate join, Phase 1 remains the only consumer; v1 does
not add a low-value checkpoint at `P`'s own probe.

### Planner lineage pass

Route discovery runs once on the resolved copied logical plan, before physical `create_plan`.
For each admitted key and each `probe_info` target:

1. Recover the starting `ColumnBinding` from the producing join's left-child bindings and the
   resolved `cond.left.index`.
2. Locate the target `LogicalGet` by its preserved `DynamicTableFilterSet` identity. Exactly one
   matching GET is required per `probe_info` target; zero or multiple matches drop that target.
3. Trace the exact branch toward that target, remapping the binding at each supported operator,
   and require the final binding to equal `probe_info[target].columns[duckdb_filter_ordinal]`.
4. Whenever the binding originates in an eligible intermediate join's left/probe child, record
   that join as a consumer before continuing toward the target. The binding must resolve to
   exactly one ordinal in that consumer's probe schema.
5. Group independently traced keys by `(producer logical node, consumer logical node)`, compact
   the supported subset, and deduplicate paths reaching the same pair.

The pure result contains values, not physical pointers. Scan and join-probe ordinals are distinct
strong types because they name different spaces: Phase 1 initially publishes in DuckDB
`column_ids` space and lets the scan channel remap it, while SIP indexes `C`'s runtime probe
schema.

```cpp
struct scan_target_key {
  std::size_t sirius_key_ordinal;
  duckdb_column_ids_index consumer_column;
  std::optional<cudf::data_type> storage_type;
};

struct join_probe_target_key {
  std::size_t sirius_key_ordinal;
  probe_schema_ordinal consumer_column;
};

struct pending_join_probe_target {
  dynamic_filter_publication_plan_id publication_plan_id;
  dynamic_filter_target_id target_id;
  logical_plan_node_id producer;
  logical_plan_node_id consumer;
  std::vector<join_probe_target_key> keys;
  std::shared_ptr<sirius_dynamic_filter_set> channel;
};
```

A query-local registry assigns `logical_plan_node_id` values and owns pending descriptors. During
recursive planning:

- building consumer `C` resolves every pending endpoint registered for `C`;
- building producer `P` resolves the matching producer/key plan; and
- neither side is frozen yet.

The registry survives pipeline conversion. After runtime topology and branch contexts are known, a
finalization pass validates branch uniqueness, the expected probe port/schema ordinal, and the
producer's publication driver. It then freezes the producer publish plan and consumer endpoint
plan together through single-assignment plan slots. Runtime reads only `shared_ptr<const ...>`
snapshots. A rejected target is removed from both sides and its channel is closed as
`PLANNING_REJECTED`; a consumer resolved before its producer later fails does not retain a dangling
endpoint. The converter subsequently asserts this frozen invariant rather than mutating a const
plan.

Only after topology finalization are the temporary logical-node maps and pending builders
destroyed.

This intentionally uses planner lookups and a single freeze boundary. It is simpler and more
testable than threading mutable pending state through every `create_plan` overload, rewriting an
already-built deep physical subtree, or mutating an immutable producer plan during conversion.

### Crossing rules

Producer eligibility and path crossing are separate predicates.

| Logical operator | DuckDB candidate behavior at the fixed pin | Sirius v1 placement behavior |
|---|---|---|
| Filter, ORDER BY without limit, DISTINCT | crosses | cross; binding unchanged |
| Projection | crosses refs and supported integral casts | cross direct pass-through ref only; remap binding |
| Intermediate comparison join | follows left child | for an equality INNER/SEMI join that is not MIXED, record its hash-probe consumer and continue left only when the binding comes from the left child; otherwise stop |
| Aggregate | crosses plain group keys | stop conservatively |
| LIMIT / TOP-N | terminal after Phase 0 | no v1 route below the row selector when it blocks the target |
| WINDOW | unsupported/default terminal | stop |
| UNNEST | conditionally crosses | stop conservatively |
| UNION / INTERSECT / EXCEPT | remaps into selected children | stop at branching/fan-out boundary |
| CTE, recursive, materialization, reusable source | varies | stop |
| Computed expression or cast | candidate-dependent | stop |

For every intermediate comparison join, a right/build-origin binding, an unproved join type, or a
non-hash physical implementation stops descent. An endpoint collected for a logical comparison
join is also dropped if physical planning does not produce the expected Sirius hash join.

The extra Sirius stops are deliberate narrowing, not "mirroring #22963." If DuckDB itself stops
and emits no target, v1 creates no route. If DuckDB admitted a scan target but Sirius stops while
tracing placement, the already-collected lower join-probe endpoints remain valid and the scan
route remains untouched.

### Safety proof

For ordinary equality, each published membership filter is a no-false-negative superset of `P`'s
build keys. At a routed consumer `C`, the filtered column is an unchanged value from `C`'s probe
input to `P`'s probe input. A row rejected at `C` cannot produce a descendant row whose key matches
`P`; `P` would reject that lineage. `P` still rechecks every survivor authoritatively.

v1 restricts intermediate consumers to INNER/SEMI joins and direct left-derived keys. This is
conservative: broader predicate-commutation cases may exist, but they need independent proofs and
do not expand the first production surface.

### Branch and shared-source safety

Discovery stops at logical CTE/materialization/reusable-source fan-out. The checkpoint itself is
also branch-local because it runs after `C` pops from its own probe repository and produces a new
filtered table; it never edits a shared upstream batch. Topology finalization uses the converted
pipelines to require one branch-specific default probe context before either plan slot is frozen.
Failure elides both target ends; subsequent converter/runtime checks only assert the frozen
invariant. Valid Phase 1 scan targets remain.

---

## Publication, target, channel, and filter identity

Identity is explicit and query-relative:

- **`publication_plan_id`:** one producing join plus its admitted key plan; filter construction is
  attempted once per publication plan.
- **`target_id`:** one scan or SIP consumer endpoint receiving that publication.
- **`channel_id`:** one append-only delivery object owned by one logical consumer endpoint. A scan
  channel can serve target edges from multiple publication plans.
- **`filter_id`:** one constructed immutable filter. It is assigned before fan-out and remains the
  same in every target channel receiving that object.

The channel stores `published_filter_entry{filter_id, shared_ptr<const filter>}` rather than
requiring telemetry to reconstruct identity from an address. Mask helpers use the filter payload;
events and offline correlation use the stable ID.

### Scan channels

The existing scan map remains keyed by preserved `DynamicTableFilterSet*`. One scan channel may
have N producing joins and one logical scan consumer. The scan closes it once; producer
publications AND-conjoin in that channel. This preserves Phase 1 exactly.

### SIP channels

Each `(producer P, consumer C)` SIP target gets a dedicated channel with exactly one producer and
one consumer. If the same filter also targets a scan or another join, those endpoints receive
separate channels containing entries with the same `filter_id` and
`shared_ptr<const sirius_dynamic_filter>`.

This is the correct close invariant: **one logical consumer per channel**, not universally one
producer per channel. Gate state, close state, telemetry, and any future activation token are
never shared across consumers.

All four IDs are query-relative monotonic values used for planning, telemetry, and tests. They are
never cached across replans.

---

## Hash-join probe consumer

### Component boundary

`sirius_physical_hash_join` owns zero or more immutable SIP endpoint descriptions. Mutable gate
state lives in a narrow composed `hash_join_probe_filter_consumer`; planner metadata and DuckDB
objects do not enter execution.

For each probe task:

1. Identify the probe batch through the join's port-aware input path.
2. In an initial `BUILD_PROBE` task, construct `C`'s own build hash table as today.
3. Before `prepare_join_keys` for the probe, snapshot each SIP endpoint.
4. Apply available membership filters through the shared gated-mask helper.
5. Produce one `probe_batch_handle` containing stable source/batch identity, the original or owned
   gathered table, its table view, and its memory space.
6. Use that handle for every later probe-side access: key preparation, `left_full`, payload gather,
   distinct/mark/semi helpers, output memory-space selection, and telemetry.

Mixing indices computed against a filtered table with payload rows from the original
`input_batches[0]` would be wrong. After the checkpoint, direct probe-side reads from that original
slot are forbidden; focused tests remove non-prefix rows and project payload columns so index/view
misalignment cannot hide.

The same checkpoint is used for BUILD_PROBE and STANDARD consumers. MIXED consumers are rejected
in v1 rather than adding another predicate/key preparation path. A STANDARD consumer still pays
its already-completed partition/shuffle; SIP saves its hash probe and work above it. This design
makes no claim that post-CONCAT filtering shrinks the shuffle.

STANDARD input creation may pair one probe batch with several build batches. Give the handle a
stable probe-batch ID and record repeated applications. C3 may initially reapply safely, but
STANDARD routes cannot default on unless that duplicated cost is measured acceptable or a
device/generation-qualified filtered-probe cache is added with an explicit memory lifetime.

### Fast paths and semantics

- No channel, no visible filter, unavailable local replica, or a disabled gate forwards the
  original probe batch zero-copy.
- Filtering never mutates the input batch.
- A zero-row filtered probe remains a schema-correct zero-row input and follows ordinary join/task
  completion.
- Replica-unavailable pass-through does not train the gate.
- Each SIP endpoint owns independent combined and per-filter gate statistics.
- A scan and later join probe may apply the same filter; downstream marginal keep-rate disables
  redundant work after its first measurement.

Only membership-capable filters are consumed at a join probe. Zone maps remain scan-reader
capabilities; no empty/sentinel `cudf::data_type` is used to encode that distinction.

### Multi-GPU

- Resolve device identity from the probe batch's memory space, never ambient CUDA context.
- Apply a filter only when `is_available_on_device(device_id)` succeeds.
- Publication constructs and synchronizes each usable replica before pushing the immutable filter
  into any channel.
- Adding SIP targets adds channel fan-out, not another replica-construction pass.
- A target records replica-unavailable pass-through separately from late publication.

---

## Ordering and contingent activation

### C3: opportunistic experiment

Every join-probe batch snapshots its channel and applies the complete filters currently visible.
No task waits. There is deliberately no claim that the producer "usually" runs first: the relevant
outer producer is not driven by the intervening consumer's build hint, and removal of the global
priority pass may change coverage.

Opportunistic C3 is liveness-safe and correctness-safe. Its performance value is an empirical
question answered before default-on:

- how many rows/batches reach each checkpoint before publication;
- how many missed a scan but were caught at C1/C2;
- hash-probe rows and bytes avoided;
- filter/gather overhead and gate disable rate; and
- end-to-end time and resident peak with legacy/off priority.

If a valuable route class sees adequate coverage, it can remain opportunistic. If systematic
misses erase the value, ordered activation for that class is a prerequisite to default-on—not an
assumed future optimization.

### Track D: exactly-once ordered activation

Track D is admitted only after runtime pipelines are finalized.

#### Admission

- At most one ORDERED target per runtime consumer pipeline. Other targets remain opportunistic.
- The selected SIP channel has exactly one registered producer.
- Validate acyclicity over repository/data edges, activation edges, and the synthetic hint path
  used to drive the producer—not activation edges alone.
- Every task-driving node in the consumer-to-producer activation closure is runtime-unique. A
  shared node demotes the target to opportunistic.
- Choose among competing targets deterministically by measured/estimated avoided hash-probe work.
- The activation descriptor stores the resolved producer build-side publication driver/milestone
  pipeline (the build PARTITION/CONCAT path), not merely producing join `P`. Asking `P` for a hint
  can follow its probe back through `C` and manufacture a cycle.

These rules prevent a consumer from waiting for several outer builds and recreating #1014.

#### Publication completion

Each ordered channel has one atomic completion state:

```text
OPEN → PUBLISHING → PUBLISHED | SEALED_EMPTY
OPEN -------------------------> SEALED_EMPTY
PUBLISHING -------------------> SEALED_EMPTY(CONSUMER_CLOSED)
OPEN | PUBLISHING ------------> FAILED
OPEN -------------------------> CANCELLED   // quiescent teardown only
```

`SEALED_EMPTY` records `EMPTY_BUILD`, `UNSUPPORTED_MODE`, `POLICY_SKIPPED`, or
`SOURCE_UNAVAILABLE`, plus `CONSUMER_CLOSED` when this target stops accepting filters. Channel
close and completion are one atomic/locked operation, so a detached target is never later reported
as `PUBLISHED`. All terminal transitions go through one compare/exchange
`complete_once(...)` API. Mode/build-finish fallbacks may seal only `OPEN`; they never overwrite
`PUBLISHING`. Success reaches `PUBLISHED` only after construction, all usable replicas, and all
channel pushes complete.

#### Exactly-once waiter

Each `(channel, consumer runtime pipeline)` has one query-owned activation token:

```text
IDLE → ARMED → QUEUED → CLAIMED
  \       \       \----> DETACHED
   \-------\------------> DETACHED
```

Before resolving any operator hint:

1. A completed `PUBLISHED`/`SEALED_EMPTY` target channel proceeds normally.
2. `FAILED`/`CANCELLED` creates no work.
3. Otherwise CAS `IDLE → ARMED`, register the token, and re-read completion.
4. Terminal success/empty calls `release_once`; only `ARMED → QUEUED` enqueues the consumer.
5. Repeated scheduling requests observing `ARMED` wait without registering another token.
6. The task creator must CAS `QUEUED → CLAIMED` before dereferencing the request's pipeline. A
   zero-input close or teardown may win `IDLE/ARMED/QUEUED → DETACHED`; a queued request carries
   the shared token and becomes a no-op if detached. A claimed request is quiesced before query
   destruction.

The activation check precedes hint resolution, repository pop, GPU memory reservation, and task
accounting. Synthetic `WAITING_FOR_INPUT_DATA{publication_driver}` drives the exact producing
build/milestone path through existing recursion. That edge is included in cycle validation.
Wakeups are queued only after channel, join, partition, and pipeline-status locks are released;
completion callbacks never synchronously recurse into hints. If bounded-pool slot acquisition
remains before hint resolution, an armed waiter may occupy only that control-plane slot—never a GPU
memory reservation or worker executing query data.

#### Milestones and teardown

- STANDARD/MIXED decision seals `UNSUPPORTED_MODE` outside join/partition locks.
- Build-CONCAT completion performs `complete_if_open(SEALED_EMPTY(...))`, covering empty and
  no-batch paths without waiting for a probe. Join finalize is never the seal point.
- Every downgraded/unavailable/policy early return completes the publication attempt and each
  affected target channel.
- A consumer finishing through zero input or early close detaches its token before destruction.

Error teardown order is:

1. stop task creation and reject new activation requests;
2. detach `IDLE`/`ARMED`/`QUEUED` tokens without enqueueing and invalidate queued activation
   requests;
3. drain the task-creation queue/claimed requests, scan/GPU executors, and in-flight publishers;
4. cancel still-open channels;
5. remove activation edges/tokens;
6. destroy query pipelines/operators; and
7. restart task creation for the next query only after both normal and activation request queues
   are empty.

Activation requests retain pipeline-qualified query-owned identity; they do not add another raw
operator pointer whose lifetime exceeds the query.

Rejected alternatives remain: blocking inside `execute`, gating the deepest scan, a FULL data
barrier, wall-clock timeout, or coordinated fan-in.

---

## Producer and materialization changes

### Waiter-free publication lifecycle (A1/C1)

Coverage telemetry cannot wait for Track D. Every `publication_plan_id` therefore gets a
waiter-free attempt state in A1/C1:

```text
OPEN → PUBLISHING → PUBLISHED | NO_MATERIALIZATION(reason) | FAILED
OPEN -------------> NO_MATERIALIZATION(reason) | FAILED
OPEN | PUBLISHING -> CANCELLED   // query teardown, no wake
```

Reasons include `EMPTY_BUILD`, `UNSUPPORTED_MODE`, `POLICY_SKIPPED`, and `SOURCE_UNAVAILABLE`.
Every transition records one monotonic timestamp and is exactly once, but it has no waiter and
schedules no task. Each target edge separately records whether its channel accepted the filters,
was already `CONSUMER_CLOSED`, or was rejected during planning. Thus STANDARD, empty, unavailable,
policy-skip, and closed-consumer paths are observable even when ordered activation is disabled.

Track D reuses these outcomes to complete per-channel activation; it does not introduce the first
notion of producer completion.

### Immutable Sirius plan

Publication claim and execution move entirely to an immutable Sirius value:

```cpp
struct dynamic_filter_key_plan {
  std::size_t condition_index;
  cudf::size_type build_column_index;
  cudf::data_type build_type;
  std::optional<std::size_t> build_key_domain_cardinality;
};

struct scan_publish_target {
  dynamic_filter_target_id target_id;
  std::shared_ptr<sirius_dynamic_filter_set> channel;
  std::vector<scan_target_key> keys;
};

struct join_probe_publish_target {
  dynamic_filter_target_id target_id;
  std::shared_ptr<sirius_dynamic_filter_set> channel;
  std::vector<join_probe_target_key> keys;
};

using dynamic_filter_publish_target =
  std::variant<scan_publish_target, join_probe_publish_target>;

struct dynamic_filter_publication_plan {
  dynamic_filter_publication_plan_id id;
  std::vector<dynamic_filter_key_plan> keys;
  std::vector<dynamic_filter_publish_target> targets;
};
```

Runtime claim checks `_dynamic_filter_plan.enabled()` and release-mode producer eligibility; it
does not re-read `filter_pushdown`, `join_condition`, `key_casts`, or DuckDB target columns.

### Per-key fan-out

Replace the current target-arity all-or-nothing push with independent target-key entries. Each
published equality component is a necessary condition, so independently admitted subsets are
safe. This hardening applies only after DuckDB admission; it does not create mixed-provenance
candidates DuckDB rejected.

### Materialization policy

For an admitted equality key, Sirius chooses `exact_set`, `bloom`, or `none` using explicit GPU
policy. `duckdb_build_subtree_has_filter_hint` may be recorded or used as a cost hint, but it is
not a correctness gate. Keep the policy change separate from the adapter refactor:

- first preserve Phase 1's observable route/materialization behavior under fixed configuration;
- revive or replace the dead domain/selectivity signal in shadow mode;
- instrument membership exact/Bloom/none, independent zone-map emission, and usefulness; and
- separately A/B-enable the repaired selectivity gate and remove the current blanket
  `build_side_has_filter` gate.

This avoids describing a broader producer policy as "no behavior change."

### Selectivity signal

Repair the dead domain signal either with a pre-resolver snapshot keyed to logical bindings or a
post-resolver source whose semantics are proved. The first PR records the would-suppress decision
only; feeding a formerly dead value into the existing publisher gate would itself change behavior.
A later policy PR enables enforcement behind A/B. Until a value is proved, unknown remains
`std::nullopt`—not sentinel zero—and the runtime consumer gate is the only selectivity backstop.
The default-on gate must include build cost, resident bytes, first-mask cost, and keep-rate, not
just result correctness.

### Publication streams and target liveness

Keep durable pooled publication streams and the existing per-stream replica reservation. Build
each filter once on the producer, replicate once per active device, synchronize, then fan the same
immutable objects into accepting targets. A closed target can be skipped; closure of one SIP
channel never closes the scan or another SIP target.

---

## Memory model

### Resident floors

The query has two non-spillable dynamic-filter-related floors:

- pinned `BUILD_PROBE` build batches plus persistent hash tables; and
- device-local dynamic-filter replicas retained for query/channel lifetime.

The count of simultaneously resident joins, not only the 500 MB per-build admission cap, controls
the first. Filter bytes scale with admitted keys × devices, while channels share filter objects and
do not duplicate replicas.

Add a per-query aggregate filter-replica budget before any Track E candidate expansion. Its
downgrade order is exact set → Bloom → none, and every choice is recorded. v1 publication-plan
count remains bounded by DuckDB-admitted joins; adding targets does not duplicate replicas, but
telemetry still measures aggregate resident bytes.

### Transient reservation

Remove the scan operator's current optimistic `stats.bytes` override. Do not replace it with a
fixed `2.1 × input`: during a multi-filter cascade the caller can retain the original batch while
the previous gathered table, the next gathered table, the BOOL8 mask, optional gather map, and
cuDF scratch overlap (`src/op/scan/dynamic_filter_merge.cpp:81-87`). Narrow schemas make the mask
and gather map proportionally large, and a hash-join consumer may retain the filtered probe while
allocating keys and join output.

Extend `input_stats` with an optional row count and use a shared, saturating estimate of
simultaneous **new allocations**:

- previous cascade result retained while another filter is applied;
- next output, bounded by the input data footprint plus table metadata;
- `rows × sizeof(bool)` mask;
- any `rows × sizeof(cudf::size_type)` gather map used by the cuDF path; and
- measured backend scratch allowance.

Existing resident input ownership is recorded in query-level peak telemetry, not charged again as
new operator allocation; nonresident input materialization remains the separate
`bytes_to_materialize_input` term. If exact decoded rows are unavailable, use a conservative
schema/byte-derived row bound. Hash-join tasks use probe-batch rows/bytes, not aggregate
probe+build statistics, for mask sizing.

The no-history estimate is never below the generic fallback. `sirius_physical_hash_join` currently
has no mode-aware override, so C2 adds one rather than referring to a nonexistent existing
key/hash estimate. It models allocations that overlap inside the composed checkpoint and join:

- the first BUILD_PROBE task includes persistent hash-table/build-key allocation plus filtering,
  probe-key, index, output, and scratch peaks;
- later BUILD_PROBE probe tasks exclude the already-resident hash table from **new** reservation
  demand while resident high-water still observes it; and
- STANDARD tasks estimate their probe/build pair, repeated probe filtering, local hash/index
  allocations, output, and scratch.

The pipeline-level `max()` cannot combine allocations inside this composed join path, so the join
override performs that overlap calculation itself. Execution history may tighten the estimate
only after successful runs.

Focused tests cover one-column INT32/nullable inputs, wide rows, unknown row counts, multiple
filters, keep ratios near 0/0.5/1, repeated STANDARD probe IDs, multi-batch tasks, and OOM
rescheduling.

---

## C++ structure and design guidance

The structure follows the guidance in Iglberger, *C++ Software Design*:

| Decision | Guideline / consequence |
|---|---|
| One adapter owns every pinned DuckDB metadata read and copy invariant | G24 Adapter; third-party layout does not leak into runtime |
| Candidate extraction, lineage tracing, eligibility, and grouping are pure functions over values | G4 testability; G19/G23 value-based strategies |
| Key, publication, target, and endpoint descriptors use value semantics, strong index types, and `std::optional` for absence | G22; invalid/cross-space sentinel states are avoided |
| `std::variant<scan_publish_target, join_probe_publish_target>` closes the two target-specific index spaces without a target hierarchy | G17; exhaustive cold-path dispatch |
| `hash_join_probe_filter_consumer` is composed into the join rather than adding planner/routing responsibilities to the join class | G2 SRP; G3 interface segregation |
| Scan and join consumers reuse a free mask-application operation over `sirius_mask_applicable`; no filter-kind switch | G5 OCP; G15 operation axis over a closed type set |
| Capability discovery remains outside the per-row path | G18; dynamic dispatch cost is per filter/batch |
| Query-local registry owns temporary logical pairing; runtime owns only Sirius values/channels | G9 dependency inversion and explicit lifetime |
| Scheduler loses join/filter knowledge after A4 | G2/G9; policy dependency is removed at the correct layer |
| Activation uses one DAG edge and an exactly-once token, not an Observer callback graph | G25; teardown and duplicate notification are explicit |
| Sirius-side admission, shared activation, STANDARD producers, and aggregate producers wait for a measured payer | G2/G5 YAGNI |

No type erasure is added to the filter zoo: several named capabilities make its tradeoff worse than
the existing small polymorphic hierarchy (G32).

---

## Phasing

Per-PR implementation plans (adversarially reviewed against the baseline commit) live in
[issue-1010-implementation-plan.md](issue-1010-implementation-plan.md) and
[issue-1010-plans/](issue-1010-plans/).

Phase 0 is deferred (see § "Phase 0") and blocks nothing; the B1 row becomes the pin-bump
playbook. While on the unpatched pin, LIMIT/TOP-N-shaped tests must use explicit expected rows or
a filters-disabled reference run, never an unpatched CPU run.

The table below records the re-evaluated delivery boundaries. C1d/C1e are independent policy
experiments rather than C3 prerequisites. To preserve C1b's scan hooks while C2 moves the shared
mask code, the implementation order is C1a-2 → C1b → C2a → C2b. C3a depends on C1b, and C3b
depends on both C3a and C2b.

| PR | Track | Content | Gate |
|---|---|---|---|
| A1 | #1014 | Stable publication/target/channel/filter IDs, waiter-free outcomes, resident high-water, lifecycle counts, channel-level scan coverage (SIP targets reuse this later) | instrumentation only |
| A2 | #1014 | `dynamic_filter_build_priority={legacy,off}` comparison switch | legacy default; no deletion |
| A3 | #1014 | Default priority off if measured gate passes | one-release rollback |
| A4 | #1014 | Delete pass and scheduler filter knowledge | after default-off release |
| B1 | Phase 0 | **Deferred** — executes at the next pin bump to a released DuckDB containing #22963; ships the explicit-oracle sentinel regressions then | pin-bump playbook; blocks nothing (sirius-db/sirius#1123) |
| C1a-1 | foundation | Version-pinned adapter and pointer-identity preservation | CPU parity; no runtime consumer |
| C1a-2 | foundation | One candidate cache/identity allocator, Sirius snapshots, claim decoupling, prepared/noexcept runtime-plan commit, waiter-free lifecycle, and strong-generation execution reset | fixed-config behavior parity |
| C1b | producer | Canonical target types; former C1c compaction/parity; ID-carrying channel entries; scan/materialization telemetry; shadow selectivity | policy observable, behavior preserved |
| C1d | producer | Enforce membership selectivity after stable-ID audit; zone maps stay shadow-only | explicit suppression policy change |
| C1e | producer | Remove blanket `build_side_has_filter` gate behind default-off A/B | explicit candidate expansion; widens the accepted latent LIMIT/TOP-N exposure marginally (#1123) |
| C2a | consumer | After C1b, extract the reusable mask operation/probe handle and add an immutable capability/proof-token endpoint with preallocated local state | behavior parity; no planned routes |
| C2b | memory | Add strong-generation/join-state-aware full coexistence floor and mode-specific allocation model | history transition, OOM, multiplicity tests |
| C3a | SIP | Discovery/resolution, frozen planning-topology descriptor, and planning telemetry behind `enable_dynamic_filter_sip=false` | no channel or runtime endpoint |
| C3b | SIP | Grouped route preparation, noexcept-only runtime-topology commit, layered scan + opportunistic join-probe targets, and coverage experiment | first value experiment |
| C4 | SIP | Default-on only if coverage/value gates pass; otherwise depends on D | results + wall time + memory |
| D | conditional | Exactly-once ordered activation for selected non-shared, acyclic, one-producer pipelines | required only for route classes that miss |
| E | vNext | Sirius-side/no-scan admission, mixed provenance, broader crossing, STANDARD producers, aggregate producers, replica narrowing | aggregate filter budget first |

Track A is complete and supplies no runtime flag or instrumentation. C1d, C1e, and C3 experiments
record one another's snapshotted settings but remain independently attributable and revertible.

---

## Testing and observability

### Unit and adapter contracts

- Copied join/get `DynamicTableFilterSet` pointer identity, including two producers sharing one
  scan channel.
- Zero and duplicate logical GET matches for one target identity fail that target closed.
- Null/statistics-only/admitted metadata; malformed condition indexes and target arity; exact
  DuckDB ordinal → reordered condition → compact Sirius key alignment.
- Producing join-type, equality/INDF, cast, NULL, GPU type, and runtime-mode gates. INDF/cast
  rejection is per condition: an eligible ordinary-equality sibling remains routable.
- Lineage remapping through pass-through projection/filter/order/distinct; stop cases for every
  unsupported operator and fan-out boundary.
- Composite admitted-key compaction; mixed-provenance composite rejected as
  `duckdb_not_admitted`.
- Deterministic grouping/deduplication into `(producer, consumer)` targets.
- Per-key publisher fan-out and independent consumer gate state.

### Planner and topology

- `scan → C1 → P` retains the scan channel and installs a distinct endpoint in `C1`.
- `scan → C1 → C2 → P` installs endpoints in both intermediate joins without inserting a unary
  feeder operator.
- The checkpoint runs after the consumer's probe repository pop and before
  `prepare_join_keys`/hash probe.
- PARTITION/CONCAT shapes, partition negotiation, and sibling positional contracts are unchanged.
- CTE/shared-source branches either stop discovery or prove branch-local post-pop filtering; an
  unrelated branch observes unfiltered input.
- Topology finalization freezes producer/consumer plan slots exactly once.
- Failure to resolve one SIP consumer drops both target ends, closes its pending channel, and does
  not disconnect valid scan targets.

### Correctness end to end

- ON/OFF relational/bag equivalence for nested INNER/SEMI chains, multiple intermediate joins,
  composite admitted keys, NULL keys, empty build, downgraded build, STANDARD consumers, and
  multi-GPU consumers on non-producer devices. Compare exact order only where SQL guarantees it.
- No producing publication for LEFT/ANTI/MARK/FULL joins; INDF/cast conditions are rejected per
  key while an eligible equality sibling remains routable.
- LIMIT/TOP-N explicit expected results on the patched pin, including join-sourced inputs; keyed by
  publication-plan/target ID, assert that the outer producer does not cross the selector.
  Independent joins wholly inside the selector input may still have legal routes.
- Zero-row filtered probes complete normally.
- Shared-source isolation and query teardown under failure.

### Race and coverage

- Publication before scan, between scan and C1, between C1 and C2, and after all consumers.
- A batch already queued behind `PARTITION`/`CONCAT` before publication is still caught at the join
  probe checkpoint; a filtered non-prefix payload proves every later probe access uses the handle.
- Stable probe-batch IDs expose repeated STANDARD applications across build-batch pairings.
- Partial multi-key fan-out, consumer closure racing publication, unavailable device replica, and
  gate re-arm after new filters.
- TSan over publish/snapshot/close; opportunistic mode has no waiter and cannot hang.

### Memory and scheduling

- Historical #1014 results come from #1124; Track C has no priority dimension.
- Separate counts for feeder tasks, delivered build batches, pinned build batches, hash tables,
  and filter replicas.
- Component reservation tests cover narrow/wide/multi-filter inputs, no-filter-history→visible
  filter transitions, channel growth, duplicate-heavy INNER output, widening casts, and OOM re-entry.
- Query aggregate replica budget/downgrade tests before Track E.

### Track D fault injection

- simultaneous register/seal, duplicate schedule notifications, completion during re-read;
- STANDARD, empty, policy skip, unavailable source, CONSUMER_CLOSED, FAILED, and CANCELLED
  outcomes;
- cancellation during publication and zero-input consumer detachment;
- several candidate targets on one runtime pipeline (only one ordered);
- the synthetic wait drives the resolved build publication driver, never the producing join's
  probe hint;
- cycles visible only in the union of data, activation, and synthetic wait edges; and
- no wake while holding pipeline/join/channel locks.

### Telemetry

Object addresses are never event identity. Use a query-relative monotonic clock and emit:

```text
publication_plan_created(publication_plan_id, producer, admitted_keys)
candidate_rejected(reason)
target_planned(target_id, publication_plan_id, channel_id, kind, consumer, mode)
publication_started(publication_plan_id)
publication_completed(publication_plan_id, outcome, filter_ids, replica_bytes, devices)
channel_filter_visible(channel_id, target_id, generation, filter_ids, accepted_at)
target_publication_terminal(target_id, channel_id, outcome)
consume_batch(channel_id, rows_in, rows_out,
              visible_filter_ids, masks_applied, masks_skipped,
              replica_unavailable)
channel_closed(channel_id, reason)
activation_armed/released/cancelled(target_id, runtime_pipeline_id)  // Track D
```

Scan coverage is channel-level in A1 because one scan channel may have several producers. Planned
publication/target IDs, per-target terminal events, channel generations, and stable filter IDs are
correlated offline to distinguish pre-publication batches and sequential fan-out. The consumer does
not pretend to attribute a combined scan mask's row reduction to one producer. Dedicated SIP
channels support direct per-target coverage.

Aggregate into:

- no DuckDB candidate vs Sirius rejection vs runtime no-materialization;
- rows before publication, with an applicable filter, and after terminal empty outcome;
- late miss vs replica unavailable vs gate skip vs closed consumer;
- scan-caught, C1-caught, C2-caught, and redundant downstream keep-rate;
- hash-probe rows/bytes avoided and mask/gather cost; and
- resident filter bytes plus live join-state counts.

---

## Risks and retiring evidence

1. **Opportunistic coverage is too late or zero.** C3 per-layer timing and avoided-probe telemetry;
   ordered activation is the recovery before default-on.
2. **#1014 priority removal regresses transitive coverage.** A1-A3 paired comparison; retain
   rollback, add route-local ordering, and repeat rather than guessing a global cap.
3. **DuckDB pin/layout drift.** Adapter contract tests and Phase 0 sentinels on every pin bump.
4. **GPU materialization diverges unprofitably from DuckDB's policy.** Record candidate,
   membership exact/Bloom/none, and independent zone-map decisions with build/apply time and
   keep-rate.
5. **Layered consumers add redundant masks.** Independent marginal gates plus per-layer overhead;
   reduce target depth only with measured evidence.
6. **Filter replica bytes grow with admitted producer/key count.** Shared filter objects,
   resident-byte telemetry, and aggregate budget before candidate expansion.
7. **Lineage/projection mistakes cause wrong-column filtering.** Pure binding-based tests,
   branch-anchor validation, and release-mode fail-closed endpoint installation.
8. **Track D introduces wake/lifetime races.** Exactly-once token, combined-edge cycle validation,
   lock-free wake boundary, and fault-injection suite before it can ship.

---

## Expected file changes

| Concern | Files |
|---|---|
| #1014 instrumentation/switch/removal | `src/{include/,}pipeline/task_scheduler.*`, memory/executor telemetry |
| DuckDB correctness pin | `duckdb/src/optimizer/join_filter_pushdown_optimizer.cpp`, upstream regression SQL, `.gitmodules`, gitlink |
| Adapter preservation/extraction | new `src/{include/,}planner/duckdb_join_filter_candidate_adapter.*`, `src/transparent/sirius_optimizer_extension.cpp` |
| Route registry, lineage, topology freeze | new planner values/registry; `sirius_physical_plan_generator.*`, `sirius_plan_comparison_join.cpp`, `sirius_plan_get.cpp`, converter/query finalization invariant |
| Sirius producer plan/materialization/lifecycle | `sirius_dynamic_filter.*`, `dynamic_filter_publish_plan.*`, `dynamic_filter_publisher.*`, `sirius_physical_hash_join.*` |
| Reusable mask operation and probe consumer | move/generalize `dynamic_filter_merge.*` and gate interfaces under `src/{include/,}op/`; new hash-join probe consumer component; retain scan operator |
| Reservation inputs/estimator | `src/include/op/sirius_physical_operator.hpp`, `src/pipeline/gpu_pipeline_task.cpp`, focused memory tests |
| Documentation | this file and `dynamic-filters.md` Phase 2/ordering sections |
| Tests | adapter, planner lineage, publisher, hash-join probe, pipeline topology, race, memory, multi-GPU suites |

Track D additionally touches `task_creator.*`, `sirius_pipeline.*`, query-owned activation
registry/tokens, join mode/build completion hooks, and teardown ordering. It does not add a
pre-partition unary consumer.

---

## Open questions resolved by measurement

1. Whether every eligible intermediate join probe pays for its first mask, or target depth should
   be capped after C3 telemetry.
2. Whether a join-probe gate needs a different keep threshold because it saves hash work but not
   scan/partition work.
3. Which route class, if any, merits ordered activation and which single route wins when several
   target the same runtime pipeline.
4. Whether the current Sirius materialization policy should incorporate DuckDB's build-filter and
   probe/build-ratio hints after their independent predictive value is measured.

Questions intentionally deferred to Track E—no-scan candidate discovery, mixed provenance,
aggregate crossing, STANDARD producers, shared activation, and device-narrowed replica sets—are
not v1 implementation choices.

---

## References

- `docs/super-sirius/dynamic-filters.md`, `dynamic-filters-multi-gpu.md` — Phase 1 architecture,
  N-producer/one-consumer scan channels, replica model, and Phase 2 hash-join-probe goal.
- `src/pipeline/task_scheduler.cpp:164-214,404-416` — global queued-task preference addressed by
  #1014.
- `src/op/sirius_physical_hash_join.cpp:486-547,910-1315,1319-1428` — stateful hints, probe
  execution (BUILD_PROBE and STANDARD paths), and publication lifecycle.
- `src/include/op/sirius_dynamic_filter.hpp`, `src/op/dynamic_filter_publisher.cpp` — append-only
  channel and GPU materialization policy.
- `src/op/scan/dynamic_filter_merge.cpp`, `dynamic_filter_gate.hpp` — mask operation and adaptive
  gate reused at the hash-join probe.
- `duckdb/src/optimizer/join_filter_pushdown_optimizer.cpp` and
  `duckdb/src/execution/operator/join/physical_hash_join.cpp` — pinned candidate and runtime
  materialization contracts.
- duckdb/duckdb#22963 (`4c8c90db44`) — LIMIT/TOP-N correctness fix.
- Klaus Iglberger, *C++ Software Design* — guidelines cited in the structure table.
