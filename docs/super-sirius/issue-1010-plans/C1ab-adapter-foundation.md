# Track C foundation: C1a adapter/runtime boundary and C1b compact targets, lifecycle telemetry, and shadow selectivity

Companion to [the governing SIP design](../issue-1010-dynamic-filter-sip-design.md).

**Implementation baseline:** dev fac81e87. This baseline already contains merged PR #1134,
commit 1eecaf97; do not construct a synthetic "dev + #1134" base or start from #1134's old feature
branch. Symbol names in this document are normative. Re-grep exact line numbers immediately before
each patch.

**Working-tree audit status (2026-07-10):** C1a-1 is mostly present. C1a-2 has the cache, builder,
key-resolution, freeze, and publisher-decoupling foundation, but it does not yet have the required
reasoned publication lifecycle or fresh-execution reset. Therefore this document remains the
target contract, not a claim that C1a-2 is complete. See the
[current-code audit](C1-current-code-audit-2026-07-10.md).

PRs covered, in strict order:

1. **C1a-1 — adapter and preservation boundary.** No runtime consumer or log-shape change.
2. **C1a-2 — Sirius values, one identity domain, immutable candidate cache, legal topology freeze,
   publisher decoupling, and waiter-free publication completion.**
3. **C1b — compact strong target entries, shadow domain evidence, lifecycle/materialization
   telemetry, and analyzer support.** C1b absorbs the former behavior-neutral C1c target
   compaction/parity work. There is no separate C1c fan-out flag or PR.

C1a-1 remains independently mergeable and revertible. C1a-2 and C1b each have their own build,
test, compatibility, and rollback gate.

---

## Authoritative decisions

1. **One executable-plan identity domain.** One allocator owned by
   **sirius_physical_plan_generator** mints planning IDs. A producer logical join receives exactly
   one publication-plan ID, reused by every scan and future SIP target. C3 may request target and
   channel IDs from this allocator; it must not create another allocator or mint another
   publication ID per producer/consumer route. Runtime filter IDs use an execution-scoped counter
   in the same identity state, reset once per strong execution generation; the event clock epoch
   is a separate timestamp.
2. **Canonical strong types land once.** C1a-2 defines all four entity IDs, the execution
   generation, and every ordinal-space wrapper. C1b and C3 include those definitions. Neither
   redeclares target types nor introduces the alternate spelling dynamic_filter_filter_id; the
   canonical type is dynamic_filter_id.
3. **Candidate extraction is immutable and repeatable.** A generator-local cache owns one
   immutable extracted candidate per logical comparison join. C3 discovery and
   **plan_comparison_join** read the same value non-destructively. There is no pointer-keyed
   destructive take operation.
4. **Pre-resolver evidence covers every entry path.** Domain evidence is captured before the
   generator's sole ColumnBindingResolver pass. Explicit execution, FFI, transparent execution,
   validation, and transparent replan all use that entry point. Callers that currently resolve
   first must stop or invoke capture before resolution; known evidence does not become unknown
   merely because of entry path.
5. **There is one legal freeze boundary.** The hash-join constructor resolves keys into a
   Sirius-only builder but exposes no runtime plan. After pipeline conversion and before
   create_query/task creation, fallible preparation builds every immutable value/assignment and an
   allocation-free/noexcept commit publishes them through single-assignment slots. C1a-2 freezes
   scan-only plans there; C3a reads only the sanctioned planning view, and C3b supplies validated,
   grouped SIP targets to the generic producer seam. Runtime never mutates a committed plan or
   observes an intermediate commit.
6. **C1 owns publication completion.** Every enabled plan has an exactly-once waiter-free outcome
   and query-relative monotonic timestamp. C1 covers EMPTY_BUILD, UNSUPPORTED_MODE, POLICY_SKIPPED,
   SOURCE_UNAVAILABLE, CONSUMER_CLOSED, PUBLISHED, FAILED, and quiescent-teardown CANCELLED, plus
   one terminal acceptance outcome per target. Track D reuses this state; it does not introduce it.
7. **Prepared execution starts fresh.** Committed immutable slots live with the cached physical
   plan; C3b's canonical routing descriptor/fingerprint lives in a separate cached prepared-topology
   record. A newly created, query-owned `dynamic_filter_execution_plan` borrows that state and is
   the sole begin/end coordinator for one execution; it owns no commit flag or fingerprint.
   Immutable plan IDs may repeat because logs are segmented by execution. Mutable attempts,
   filter-ID counters, channel contents/generation, acceptance state, and target statistics are
   reset exactly once after the previous execution is quiescent. Every channel receives the same
   exact strong generation; generation preflight and allocation-free local begin finish before the
   next task can run.
8. **Planning logs commit only on success.** Every generator has explicit validation or execution
   purpose. It buffers planning events and flushes them only after top-level create/fold/verify
   success. Validation and failed/fallback generators emit no executable-plan events. Freeze and
   runtime events come only from the accepted execution.
9. **Actual and shadow policy are separate facts.** Materialization reports what happened. A
   separate shadow decision reports what enforcing selectivity would have done. Shadow mode never
   labels an emitted filter as none_shadow_would_suppress.
10. **C1b absorbs behavior-neutral target compaction.** Structurally valid full-arity DuckDB
    targets become compact typed entries aligned to admitted Sirius keys. The whole-target arity
    fence remains fail-closed, pushed filters stay identical, and no flag is added.

---

## Scope and compatibility promise

### In scope

- Preserve copied join/GET DynamicTableFilterSet pointer identity.
- Centralize every non-legacy JoinFilterPushdownInfo, join filter_pushdown, and scan
  dynamic_filters metadata read behind the adapter during planning.
- Preserve three key spaces: DuckDB filter ordinal, reordered condition index, and compact Sirius
  key ordinal.
- Remove DuckDB comparison enums and JoinFilterPushdownInfo from operator/runtime headers.
- Provide one identity allocator and immutable candidate cache for C1 and C3.
- Provide the one-shot producer-plan freeze seam consumed by C3.
- Provide waiter-free publication/target completion and a fresh execution-state boundary.
- Compact targets per admitted key in C1b without changing pushed filters.
- Repair the dead domain signal in shadow mode and make policy/lifecycle observable.

### Compatibility promise

Under fixed configuration C1a-2 and C1b preserve:

- DuckDB-admitted producers and condition ordinals;
- scan target/channel routing and register_producer side effects;
- materialization choices and pushed-filter multiset;
- empty-build, unavailable-source, drained-target, failure, and relational results;
- the existing human-readable terminal INFO line; and
- behavior with dynamic-filter pushdown disabled.

The new lifecycle state and telemetry observe existing paths; they do not wait, schedule work, or
change materialization policy. C1b compaction expresses the existing per-key behavior as typed
entries. Structural target corruption still rejects the whole target.

### Out of scope

- selectivity enforcement (later C1d);
- removal of the build_side_has_filter route gate (later C1e);
- SIP route discovery and enablement (C3);
- probe consumption and mask memory modeling (C2);
- ordered activation (Track D); and
- candidates not emitted by pinned DuckDB.

---

## Historical baseline surfaces audited before implementation

This inventory describes the `fac81e87` baseline that was inspected before implementation. It is
not an inventory of the 2026-07-10 working tree. Re-run the searches against the actual delivery
commit before using the line-level guidance.

### Producer metadata

- **src/transparent/sirius_optimizer_extension.cpp**: clone/preservation and copy_logical_plan.
- **src/planner/sirius_plan_comparison_join.cpp**: dead domain walk, route wiring, target-column
  conversion, and ownership transfer.
- **src/include/op/sirius_physical_hash_join.hpp** and its implementation: constructor/member,
  claim, and publisher construction.
- **src/include/op/dynamic_filter_publisher.hpp** and its implementation: runtime reads of
  join_condition, casts, and build-key indexes.
- The nested-loop-join pushdown member/overload is dead and is deleted in C1a-2.

### Channel identity

- The plan generator's dynamic_filter_channels map is keyed by the preserved
  DynamicTableFilterSet pointer during planning.
- **sirius_plan_get.cpp** and **sirius_plan_comparison_join.cpp** must reach the same pointer.
- The adapter candidate retains shared ownership while the pointer is a planning-map key. Runtime
  scans do not need a DuckDB lifetime anchor after planning; delete the unused table/parquet/DuckDB
  scan anchor members when the new owner is installed.

### Index facts

Pinned DuckDB reorders conditions before recording join_condition. Equality and NOT DISTINCT
conditions form a stable prefix. DuckDB records ordinary equality and range comparisons but skips
COMPARE_NOT_DISTINCT_FROM. Sirius applies the same stable equality-first reorder. Therefore:

- every recorded condition index addresses the Sirius-reordered condition vector;
- key_casts and right_key_col_indices are addressable by condition index only in the equality
  prefix;
- range candidates are valid and receive non_equality, not malformed; and
- an admitted equality after a skipped NOT DISTINCT condition keeps its condition index and must
  not be confused with its compact Sirius key ordinal.

The current post-resolver build-key-domain walk is dead because it expects bound column bindings
after the resolver has produced positional references. C1a-2 deletes it; C1b restores useful
capture before resolution.

---

# C1a-1 — adapter and preservation boundary

## Goal

Move pinned-DuckDB copying and extraction behind one adapter without changing runtime code, logs,
or routing.

## Files and API

Add:

- **src/include/planner/duckdb_join_filter_candidate_adapter.hpp**
- **src/planner/duckdb_join_filter_candidate_adapter.cpp**

Register the source in CMake's planner block. The adapter owns:

- absent, statistics_only, admitted, and malformed classification;
- a probe target containing shared DynamicTableFilterSet identity plus copied column/type values;
- clone_filter_pushdown_info;
- preserve_dynamic_filter_metadata;
- extract for a logical comparison join; and
- scan_channel_identity for a logical GET.

The shared identity is planning-only and opaque. Sirius never dereferences it.

## Steps

1. Move clone and parallel preservation from the transparent module. Preserve the exact shared
   DynamicTableFilterSet object on join and GET copies. Continue omitting min_max_aggregates: the
   Sirius copy never executes DuckDB's physical join and the untouched original remains the CPU
   fallback plan.
2. Move preserved_counts to the planner namespace. Update the public copy_logical_plan declaration,
   definition, diagnostic variable, and call site.
3. Classify null pushdown as absent; non-null with empty probe_info as statistics_only;
   out-of-range/duplicate condition indexes or target-column arity mismatch as malformed; and
   non-empty structurally valid metadata as admitted. statistics_only is reserved for DuckDB's
   deliberate zero-target state — its telemetry count sizes the Track E opportunity — so an
   anomalous candidate is never classified there.
4. Keep all DuckDB-recorded ordinals. Range comparisons are narrowed later. Drop an individual
   null-channel target while live siblings remain (their positional pairing is provably intact);
   a non-empty probe_info left with zero live targets is malformed, not statistics_only. A
   malformed result carries only its classification; no other field is meaningful.
5. Copy values and shared ownership only. No operator/runtime header stores a DuckDB comparison
   enum or JoinFilterPushdownInfo.

## Tests and gate

- Keep all seven existing preservation assertions (namespace requalification only).
- Add absent/statistics-only/admitted/malformed cases, equality+range full arity,
  duplicate/out-of-range indexes, target-arity corruption, and null-channel rejection.
- Add join-to-GET identity and two-producers/one-GET identity cases.
- Register all new tests in TEST_SOURCES.
- Gate with pixi make, make test, and pre-commit.
- Follow-up (a later PR, not C1a-1): relocate `test_preserve_dynamic_filter_metadata.cpp` from
  `test/cpp/transparent/` to `test/cpp/planner/` and retag, so a pin-bump audit grepping the
  adapter's tests finds it; C1a-1 keeps the path/tags to stay assertion-identical.

No GPU, log, or performance gate is required for C1a-1.

---

# C1a-2 — identity, immutable candidates, freeze seam, and runtime decoupling

## Canonical strong types and allocator

Add **src/include/op/dynamic_filter_identity.hpp**. Use one strong-value template but distinct tags
for:

- dynamic_filter_publication_plan_id;
- dynamic_filter_target_id;
- dynamic_filter_channel_id;
- dynamic_filter_id;
- dynamic_filter_execution_generation;
- duckdb_filter_ordinal;
- join_condition_index;
- sirius_key_ordinal;
- duckdb_column_ids_index; and
- probe_schema_ordinal.

Entity ID and execution-generation zero are invalid; counters/generations begin at one. Ordinal
zero is valid and must not share the entity-ID validity API. Add equality/ordering; log sites
format the public `.value` member directly (no `format_as`/formatter specialization — the
environment's spdlog 1.8.x bundles fmt 7, which predates `format_as`, and op headers must not
include spdlog/fmt; see resolution 20).
`dynamic_filter_execution_generation` is the reset generation, not a timestamp: central begin
sets every channel to the exact generation derived from the query execution ID. The separate
query-relative event epoch is a monotonic-clock time point.

The generator owns one dynamic_filter_identity_allocator and a producer-node-to-publication-ID
memo. A producer receives one ID no matter how many targets are added. Scan targets receive unique
target IDs. Producers sharing one scan channel reuse its channel ID but retain distinct target IDs.
C3 receives this allocator through the generator/route registry handoff and never creates another
counter set. The accepted executable plan retains an execution identity state with an atomic
filter-ID counter.

## Immutable candidate cache

Add a generator-local dynamic_filter_candidate_cache with three operations:

1. capture_pre_resolver(root, context);
2. extract_post_resolver(root); and
3. find(join), returning a shared pointer to const candidate data.

C1a-2 records node identity pre-resolver and stores extracted routing/comparison values
post-resolver. C1b enriches the same entries with domains. Both C3 discovery and physical join
planning call find; neither re-extracts or consumes the entry.

The authoritative generator sequence is:

1. capture pre-resolver evidence;
2. resolve operator types;
3. run the sole ColumnBindingResolver pass;
4. populate immutable candidates;
5. run optional C3 discovery;
6. recursively create the physical plan;
7. fold and verify; and
8. commit buffered events only for execution purpose.

Remove redundant pre-generator resolution in explicit and FFI call sites, or move capture ahead of
it in the same PR. Debug builds reject an already-positionally-resolved plan without a completed
snapshot.

## Builder, key resolution, and final validation

In `dynamic_filter_publish_plan.hpp` define the canonical decision enum:

    enum class dynamic_filter_key_decision : std::uint8_t {
      admitted, non_equality, cast, unresolved
    };

The planner creates a Sirius-only dynamic_filter_publish_plan_builder containing:

- publication ID;
- the preserved Phase 1 `wired` decision;
- full-arity scan target drafts with target/channel IDs;
- policy and replica spaces;
- DuckDB key count;
- build input column count;
- key candidates;
- decisions;
- resolved keys; and
- optional domain evidence, null in C1a-2.

The hash-join constructor resolves candidates after normal equality-key extraction. It records one
`dynamic_filter_key_decision` per candidate. It retains only the builder
and drops the JoinFilterPushdownInfo constructor argument/member.

Final construction validates:

- decision count equals candidate count and identities match one-to-one;
- every admitted decision has one ordinal and exactly one matching key;
- rejected decisions have no Sirius ordinal;
- key count equals admitted-decision count;
- Sirius ordinals are unique and contiguous;
- DuckDB ordinals are unique and below DuckDB key count;
- condition indexes match candidates and resolved conditions;
- each build column is below the captured build input width;
- full-arity scan column/type vectors equal DuckDB key count;
- enabled plan/target/channel IDs are nonzero, target IDs are unique, and channels are non-null;
- replica spaces retain GPU/HOST-tier and unique-device validation; and
- disabled plans contain no live target but are still installed before runtime.

## Sanctioned pre-freeze planning view

After the hash-join constructor has resolved keys and before the runtime slot is frozen, planner
code may inspect exactly one read-only value surface:

    struct dynamic_filter_planning_ordinal_view {
      duckdb_filter_ordinal duckdb_ordinal;
      dynamic_filter_key_decision decision;
      std::optional<dynamic_filter_key_plan> admitted_key;
      std::optional<cudf::data_type> build_type;
    };

    struct dynamic_filter_planning_view {
      dynamic_filter_publication_plan_id publication_plan_id;
      bool wired;
      bool enabled;
      std::span<const dynamic_filter_planning_ordinal_view> by_duckdb_ordinal;
    };

`sirius_physical_hash_join::planning_view()` returns this value without exposing the mutable
builder. `wired` records whether the preserved Phase 1 wiring predicate admitted the producer;
`enabled` records whether the resolved builder can produce an enabled publication plan. The
ordinal array has exactly DuckDB key count entries in DuckDB ordinal order. `admitted_key` and
`build_type` are engaged if and only if `decision == admitted`; rejected decisions expose neither
a Sirius key nor a build type.

This is the only sanctioned C3 bind-time view. C3 copies the publication ID, wired/enabled state,
decision, admitted key, and build type from it while resolving physical endpoints. It never reads
the mutable builder, calls the runtime `dynamic_filter_plan()` accessor before freeze, or
reconstructs these values from protected hash-join members. The view is immutable for the rest of
planning and must agree byte-for-byte with the values consumed by final validation.

## One-shot freeze seam

The constructor does not assign a runtime plan. The join owns:

    single_assignment<shared_ptr<const dynamic_filter_publish_plan>> runtime_plan

The freeze primitive has an explicit two-phase API: `prepare_assignment(value)` performs
all validation/allocation and returns a move-only token; `commit_assignment(token) noexcept`
publishes the already-built pointer. Calling the ordinary checked assignment twice remains an
internal error. This makes the commit path allocation-free and non-throwing.

C1 owns the generic producer boundary in `dynamic_filter_publish_plan.hpp`; C1b completes
the target variant before C3 consumes it:

    struct dynamic_filter_target_addition {
      sirius_physical_hash_join* producer;
      std::vector<join_probe_publish_target> targets;
    };
    prepared_dynamic_filter_plans prepare_dynamic_filter_plans(
      std::span<dynamic_filter_target_addition const> grouped_additions); // fallible
    void commit_dynamic_filter_plans(prepared_dynamic_filter_plans&&) noexcept;

`prepared_dynamic_filter_plans` owns one prebuilt immutable plan and prepared assignment per
builder. C1a-2 calls preparation with no additions; C1b generalizes the target variant; C3b
supplies only its already validated, **grouped-by-producer** target additions. C1 knows nothing
about C2 consumer tokens or a C3 route-registry bundle.

The engine invokes this generic preparation/commit boundary unconditionally and enumerates every
retained C1 builder.
Disabled, scan-only, zero-admitted, no-registry, and all-C3-rejected joins receive a frozen plan
with zero additions; registry presence is never the condition for assigning the runtime slot.

On first preparation, all builder, slot, identity, and value checks finish before the no-throw
commit. On cached-plan re-execution, C1a-2 recomputes canonical values from the builders and the
already frozen plans and requires full equality without invoking assignment again. It does not
need a separately stored C1a-2 fingerprint. C3b may add a persistent routing descriptor beside
these producer values. A digest may be used only as a fast rejection; success still requires full
canonical value comparison. An incompatible descriptor or a direct second commit is an internal
error.

Runtime access requires a frozen slot. Planner tests either inspect the builder through a narrow
test seam or invoke the same finalizer. Test runtime-before-freeze, direct double-assignment,
idempotent matching verification, incompatible re-finalization, and preparation failure before any
slot changes.

## Planner and publisher rewiring

1. plan_comparison_join retrieves the cached candidate before planning children. Preserve the
   existing wiring predicate and INFO lines, including the statistics-only "build side is
   unfiltered" case.
2. Preserve registration for every currently wired shape, including RIGHT and equality+range
   joins, because scan insertion depends on register_producer even when runtime cannot publish.
3. Convert comparisons to Sirius is_equality at the planner boundary.
4. Build the plan builder with full DuckDB arity, one producer ID, target/channel IDs, replica
   spaces, and captured build width.
5. Rewrite dynamic_filter_publisher to borrow only the frozen plan. It iterates admitted keys and
   resolved build columns. Runtime column type remains authoritative; plan/type drift is WARN-only.
6. Keep target mismatch WARN+skip in release. C1b replaces full-arity runtime vectors with compact
   validation.
7. Keep the existing terminal INFO line byte-compatible, including DuckDB key count and the
   "Pushed 0" case.
8. Use the adapter on the scan side and remove unused runtime DuckDB identity anchors.

## Waiter-free lifecycle and fresh execution state

Add a composed publication-attempt component:

    OPEN -> PUBLISHING -> PUBLISHED | NO_MATERIALIZATION(reason) | FAILED
    OPEN ---------------------------> NO_MATERIALIZATION(reason) | FAILED
    OPEN | PUBLISHING --------------> CANCELLED

Every terminal transition is exactly once and records time relative to the execution's monotonic
epoch. The publisher returns a structured result with actual materialization decisions, filter
IDs/bytes/devices, and one target outcome: ACCEPTED, CONSUMER_CLOSED, or PLANNING_REJECTED.

Required completion sites:

- mode resolution/build completion seals UNSUPPORTED_MODE;
- empty build seals EMPTY_BUILD;
- non-GPU or downgraded build delivery seals SOURCE_UNAVAILABLE;
- zero constructed filters seals POLICY_SKIPPED with detailed reasons;
- all accepting targets closed records target outcomes and seals CONSUMER_CLOSED;
- successful fan-out seals PUBLISHED only after synchronization and pushes;
- exceptions seal FAILED before propagation; and
- quiescent query teardown seals remaining state CANCELLED.

Committed single-assignment slots remain on the cached physical plan. Cached prepared execution
data separately owns the canonical frozen descriptor/fingerprint used by no-allocation
verification (C3b adds its routing descriptor beside C1's producer values). That prepared record,
not an execution object, exposes `topology_committed()` and persists until the prepared plan
is destroyed.

Add a query-owned `dynamic_filter_execution_plan` as the sole begin/end coordinator for one
execution. It borrows the already committed physical graph/prepared record, enumerates its unique
channels, attempts, and registered local hooks, and owns only the active begin/end interval. It is
neither serialized nor reused across cached executions and retains no descriptor, fingerprint, or
commit bit.

The prepared topology also owns a single-execution lease. `begin_execution` may start only after
the previous borrower has completed teardown; overlapping use of the same cached topology is
rejected before reset (or the caller must clone the topology and its mutable channels). No begin
may reset channels that a concurrent execution can still observe.

`begin_execution(execution_id)` derives one nonzero
`dynamic_filter_execution_generation` from that execution ID and captures a separate
`steady_clock::time_point` event epoch. After the previous execution is quiescent, it
deduplicates by strong IDs, resets the filter-ID counter and every unique publication attempt once,
and clears/reopens every unique channel **to that exact generation**; channels do not independently
`++`. Empty topology still establishes the generation, filter-ID domain, and event epoch.

Before any local hook runs, the coordinator checks that every C2 endpoint reports the exact
generation. A mismatch aborts initialization and enters the canonical abort teardown below. C2
allocates all endpoint/gate/tracker storage during topology preparation, so each registered
`begin_sip_execution(generation) noexcept` hook only resets storage in place. It must not clear
channels, reset attempts, allocate, create a second filter-ID domain, change generation, or
establish another epoch. All of this finishes before hints or tasks can run.

`end_execution(status) noexcept` is the one success/abort algorithm:

1. stop task creation and quiesce publishers/tasks;
2. emit exactly one normal or `partial=1 reason=...` scan summary per unique scan channel
   and one SIP summary per installed SIP endpoint;
3. normal-close or abort-close each unique channel exactly once;
4. invoke every started C2 local end hook, which resets local storage in place;
5. transition residual OPEN/PUBLISHING attempts to CANCELLED exactly once and force-close only
   idempotent residue; and
6. assert every enabled attempt is terminal and every channel is closed.

An exception before consumer finalization still follows these steps. Operators and C2 components do
not independently perform central reset or cancellation. Executing one cached prepared plan twice
must therefore republish from clean state without double-resetting a shared scan/SIP channel.

## C1a-2 tests and gate

Add tests for:

- candidate-cache contract (from the cache-piece review): single-use throws (double capture,
  extract-before-capture, double extract); the resolver fence (debug throw on a resolved
  pushdown-bearing plan; release `saw_resolved_plan()` true with extraction still positionally
  correct; no fence for resolved plans without pushdown, BOUND_CAST-wrapped keys, or
  constants-only conditions); `find` tri-state (unknown join → null, capture-only → null,
  pushdown-free join post-extraction → non-null `kind == absent`); repeated `find` returns the
  same shared entry with the adapter run exactly once per join; DELIM joins share the
  `plan_comparison_join` entry; and a pre-resolving helper fails loudly in debug builds
  (the silently-vacated-coverage regression class);
- repeated candidate lookup and C3-discovery-to-physical-planning reuse;
- the pre-freeze planning view's publication ID, wired/enabled state, and exact per-DuckDB-ordinal
  decision/key/build-type parity with the frozen plan;
- rejection of pre-freeze runtime-plan access and any mutable-builder escape;
- one publication ID per producer and unique target/channel identity;
- equality+range and equality+NOT-DISTINCT index alignment;
- all decision kinds and every final-plan invariant;
- build-schema bounds and candidate/decision/key bijection failures;
- filtered/unfiltered/statistics-only/RIGHT/pushdown-disabled planning;
- supported publication, zero admitted keys, drained/sibling targets, zone-map compatibility, and
  runtime type authority;
- every lifecycle outcome and exactly-once transition under publish/close/failure races;
- central begin deduplicating shared channels/attempts and ordering C2 local begin after the one
  filter-ID reset, exact generation assignment, separate event epoch, and generation preflight;
- allocation-free/noexcept C2 begin after all fallible preparation;
- fault injection at each producer-plan/assignment preparation allocation with zero slot changes,
  plus compile-time `noexcept` assertions for commit operations;
- success and failure-before-consumer-finalize end ordering, including one partial summary, one
  close, C2 local end, and residual-attempt cancellation;
- freeze once, early runtime read, direct double-assignment failure, and matching-fingerprint
  verification reuse; and
- two executions of one prepared plan with no stale filters, outcome, generation, or filter ID.

Gate with build/test/pre-commit, TPC-H SF1 results and timed INFO runs, paired existing terminal-line
multisets, a separate DEBUG audit, and ON/OFF TPC-H plus clickbench parity for results, targets,
keys, push counts, and scan-apply trajectories.

---

# C1b — compact targets, shadow selectivity, and structured telemetry

## Canonical target contract

C1b defines the target types C3 consumes:

- scan_target_key: admitted key, duckdb_column_ids_index, optional storage type;
- join_probe_target_key: admitted key, probe_schema_ordinal, key type;
- scan_publish_target: target/channel IDs, channel, compact keys;
- join_probe_publish_target: target/channel IDs, channel, compact keys; and
- one variant over the scan and join-probe target alternatives.

C3 includes these definitions; it does not redeclare them.

At freeze C1b compacts each structurally valid scan draft from full DuckDB arity to exactly one
entry per admitted key. It retains the DuckDB ordinal in each key and validates each entry against
the plan key vector. Structural arity corruption rejects the whole target. Independently rejected
non-equality/cast/unresolved keys are absent. This is the existing observable behavior, so it lands
without a config flag. A SIP alternative without C3 capability is rejected at planning/freeze,
never by a runtime logic_error after tasks start.

## Scan-channel entries and coverage

C1b replaces each channel's bare filter pointer with an immutable, provenance-carrying entry:

    struct dynamic_filter_channel_entry {
      dynamic_filter_publication_plan_id publication_plan_id;
      dynamic_filter_target_id target_id;
      dynamic_filter_channel_id channel_id;
      dynamic_filter_id filter_id;
      std::shared_ptr<const sirius_dynamic_filter> filter;
    };

As part of C1b, declare `sirius_dynamic_filter_set::register_producer() noexcept`. Its
implementation remains the existing allocation-free atomic increment. This gives later C3
transaction preparation a concrete, statically provable no-throw registration operation.

The publisher allocates one filter ID before fan-out. Every target receives the same immutable
filter pointer and filter ID, while its entry retains that target's publication, target, and
channel IDs. Validate all IDs as nonzero and require the entry's channel ID to match the receiving
channel. Object addresses and `(probe_col_idx, filter_kind)` are never correlation identities.

For every accepted scan-target insertion, emit one DEBUG `target_visible` record keyed by
`publication_plan_id`, `target_id`, `channel_id`, and `filter_id`, plus key ordinal, filter kind,
`target_kind=scan`, and `consumer_column`. On the first measurement of every inserted
membership entry, emit exactly one DEBUG `membership_measured` record with the same four
IDs, `target_kind=scan`, `consumer_column`, input/kept rows, kept ratio, and
`decision=KEEP|SKIP`. KEEP is recorded as deliberately as SKIP; later batches reuse the stored
gate decision and do not emit another first-measurement record.

At scan-consumer finalization emit exactly one INFO `scan_consume_summary` per channel per
execution, including `channel_id`, `targets_visible`, `filters_visible`, `membership_measured`,
`keep`, `skip`, `batches`, `rows_in`, `rows_out`, `batches_before_first_filter`, and
`rows_before_first_filter`, plus `partial=0|1` and `reason`. Emit the summary even when
every count is zero or abort precedes normal finalization. This is the INFO-level
scan coverage consumed by C3; DEBUG records provide stable-ID attribution in the separate audit
pass. Normal summaries use `partial=0 reason=NONE`; abort summaries use
`partial=1` and a strict teardown-reason enum.

## Shadow domain evidence

Extend capture_pre_resolver to trace every candidate build binding to its source GET. Store one
optional size per DuckDB ordinal:

- proved nonzero source cardinality becomes a value;
- zero, missing callback, exception, computed expression, or untraceable source becomes null.

Evidence lives in the generator-local immutable cache. No raw JoinFilterPushdownInfo pointer
escapes the adapter, no take operation removes evidence, and no SiriusContext query-lifecycle
registry is needed.

At publication compute shadow policy without skipping. Keep actual and hypothetical values
separate:

- membership_materialization: exact_set, bloom, none_unsupported_type;
- zone_map_materialization: emitted, none_disabled, none_invalid_minmax; and
- shadow_selectivity_decision: unknown, would_publish, would_suppress.

The later enforcement PR consumes this optional evidence and decision; it adds no snapshot path.

## Generator purpose and event commit

Add explicit sirius_plan_generation_purpose values validation and execution. Every production
construction site chooses one; there is no production default. Tests use named helpers.

Candidate and rejection records remain buffered in the generator. Execution records flush only
after top-level create/fold/verify success. Validation and exceptions/fallback discard them.
Topology-dependent finalized events and `PLANNING_REJECTED` target terminals emit during
accepted-plan preparation/commit; other lifecycle and target events emit from runtime.

Stable event vocabulary includes:

- INFO candidate with `publication_plan_id`, producer, and DuckDB key count;
- INFO finalized with `publication_plan_id` and admitted/non-equality/cast/unresolved counts;
- INFO publication_terminal with `publication_plan_id`, outcome, filter IDs, bytes, and devices;
- INFO scan_consume_summary with the required scan-channel coverage fields above;
- DEBUG per-key materialized with actual and shadow decisions;
- DEBUG target_visible and membership_measured with publication/target/channel/filter IDs; and
- DEBUG target_publication_terminal with publication-plan/target/channel IDs, outcome, and
  `filter_ids`.

Keep the existing human terminal INFO line unchanged. Events use IDs rather than addresses and
include query-relative timestamps where ordering matters.

## Analyzer

In C1b add the dynamic-filter metric module, parser wiring, strict patterns and enums, per-query
CSV/JSON counts, a shape-version bump, and executable fixtures for INFO-only, DEBUG, malformed
fields, unknown enums, duplicate terminals, and incomplete queries.

The strict scan patterns require every `scan_consume_summary` field named above and the full
four-ID key, `target_kind`, and `consumer_column` on `target_visible` and
`membership_measured`; `target_visible` also requires `filter_kind`,
`membership_measured` requires input/kept rows, and target terminal records require
`filter_ids`. Validate
`keep + skip == membership_measured`; with DEBUG capability, reject duplicate first measurements
and references to an unknown target-visible tuple. INFO-only input omits the attribution table by
capability—it does not fail merely because DEBUG records are absent.

The analyzer must accept INFO-only timing logs. Remove the unconditional requirement for both TRACE
and DEBUG tags. Metrics absent at the captured level produce empty detail tables plus a capability
note, not process exit. DEBUG/TRACE audits remain separate from timed INFO runs. Register the parser
test command in the merge gate.

## C1b tests and gate

Extend C1a tests with:

- compact alignment after partial key rejection;
- structural corruption remaining whole-target fail-closed;
- repeated cache reads returning identical evidence;
- signal parity across transparent, explicit, FFI, validation, and transparent-replan paths, while
  retaining validation's no-executable-summary rule;
- null, small-domain, and high-coverage shadow cases with identical pushes;
- actual materialization independent of would_suppress;
- one immutable filter pointer/filter ID fanning into multiple scan targets with distinct target
  provenance;
- two producers sharing one scan channel remaining distinguishable by publication, target, and
  filter IDs;
- exactly one first membership measurement for each KEEP and SKIP case, with later batches
  producing no duplicate;
- zero/nonzero INFO scan summaries whose totals reconcile with DEBUG target and measurement
  records;
- compile-time non-interchangeability of scan and probe ordinals;
- validation and failed generators emitting no executable summaries;
- successful execution records flushing exactly once;
- INFO-only and DEBUG analyzer fixtures with no warnings; and
- prepared execution twice with independent mutable state.

Repeat the C1a-2 compatibility protocol. Additionally require:

- actual and shadow records for every admitted key in the DEBUG pass;
- a deterministic synthetic case, not an accidental TPC-H threshold crossing, proving
  would_suppress while push counts remain unchanged;
- one planning/freeze/publication summary set per accepted executable producer;
- one INFO scan summary per channel/execution and complete target-visible/first-measurement
  attribution in the DEBUG audit;
- no executable summaries for validation/fallback; and
- a successful INFO-only analyzer run with no format drift.

Rollback is one C1b revert that leaves the C1a adapter, identity, cache, lifecycle, and freeze seam
intact.

---

## Dependencies and handoff

- C1a-1 to C1a-2 to C1b is strict.
- C2 may develop independently, but C2a merges after C1b so its consumer-local hooks preserve
  C1b's ID-carrying channel entries and scan-coverage hooks.
- C3 is blocked on C1b. It consumes the canonical identity/ordinal/target types, immutable
  candidate cache, sanctioned planning view, one producer publication ID, scan-coverage summaries,
  execution coordinator, and generic prepared-plan/noexcept-commit seam. It must not redeclare
  types, destructively extract candidates, mint publication IDs, or mutate a committed plan.
- C1d is enforcement-only and consumes C1b's optional domain/shadow result.
- C1e is candidate expansion and remains separate.

Do not squash the three PRs: the adapter-only rollback boundary is valuable, and lifecycle/freeze
review should not be hidden inside telemetry changes.

---

## Risks and retiring evidence

1. **Ordinal misalignment:** equality+range, equality+NOT-DISTINCT, cast, and partial-admission tests
   across candidate, decision, plan, and target values.
2. **Identity collision:** one producer/many targets, several producers/one scan channel, and C3
   fixtures using one allocator.
3. **Evidence consumed twice:** repeated immutable reads and the discovery-to-planning sequence.
4. **Unfinished plan observed:** early-read/direct-double-assignment failures, prepared no-throw
   commit, matching verification reuse, and TSan around start.
5. **Prepared execution leaks state:** two executions of one cached plan with exact distinct strong
   generations and separate clock epochs.
6. **Lifecycle remains open:** exhaustive transitions plus success/abort end assertions that every
   endpoint has one summary/close and every enabled plan is terminal.
7. **Telemetry lies about shadow:** assert actual pushes independently from would_suppress.
8. **Phantom planning events:** validation, unsupported-parent fallback, and successful execution
   tests.
9. **Hardware policy tests flap:** exact-set/Bloom boundaries live in a pure policy seam.

---

## Review-resolution appendix

### Earlier audit resolutions retained

1. Range comparisons remain valid full-arity candidates and narrow as non_equality.
2. Zero admitted keys retain the existing terminal "Pushed 0" line.
3. Runtime target mismatch remains WARN plus skip-target.
4. Runtime build type remains authoritative; plan type is a WARN-only detector.
5. The seven preservation test cases remain and gain topology coverage.
6. Planner tests require success and use the explicit freeze/test seam.
7. Statistics-only metadata preserves the unfiltered-build INFO behavior.
8. Timed INFO and non-timed DEBUG passes remain separate.

### 2026-07-09 re-evaluation resolutions

1. **BLOCKER — publication completion was assigned to removed scaffolding.** C1a-2 now owns the
   waiter-free FSM, reasons, timestamps, per-target outcomes, and execution reset; C1b owns logs and
   analyzer support.
2. **BLOCKER — C1a and C3 allocated colliding IDs and C3 minted one plan ID per route.** One
   generator/executable-plan allocator owns IDs; producer IDs are memoized and reused.
3. **BLOCKER — destructive snapshot take let C3 consume evidence before physical planning.** One
   generator-local immutable cache is read non-destructively.
4. **BLOCKER — C1a finalized in the constructor while C3 needed post-conversion targets.** C1a-2
   now defines generic fallible preparation plus allocation-free/noexcept commit after conversion.
5. **MAJOR — raw scan/probe indexes were called strong and C3 redeclared incompatible types.**
   Canonical ID, ordinal, and target types now land once.
6. **MAJOR — build-index validation lacked build width and candidate/decision/key bijection.** The
   builder carries width and the validation contract is explicit.
7. **MAJOR — shadow evidence was path-dependent.** Capture precedes the sole resolver for every
   entry path.
8. **MAJOR — accepted-plan telemetry could include validation or fallback fragments.** Purpose is
   explicit and records commit only after top-level success.
9. **MAJOR — the analyzer rejected INFO-only timing logs.** C1b removes that requirement and tests
   INFO-only parsing.
10. **MAJOR — actual materialization used a shadow-only none reason.** Actual and hypothetical
    decisions are independent.
11. **MAJOR — cached prepared execution could retain filters and terminal state.** C1a-2 adds exact
    strong-generation reset, separate event epoch, generation preflight, canonical success/abort
    end, and repeated-execution tests.
12. **SCOPE — former C1c duplicated C1b target compaction.** It is folded into C1b without a flag.
13. **BASELINE — the old plan described a synthetic base.** It now targets dev fac81e87, which
    already contains merged #1134 commit 1eecaf97.
14. **MAJOR — C3 depended on scan coverage that C1b did not specify.** C1b now owns immutable
    ID-carrying channel entries, target-visible and first-measurement attribution, INFO scan
    summaries, strict analyzer fields, and reconciliation tests.
15. **BLOCKER — C3 bind needed producer facts before the runtime plan existed.** C1a-2 now exposes
    one immutable `dynamic_filter_planning_view`; runtime-plan access and builder inspection remain
    forbidden before freeze.
16. **BLOCKER — channel reset had multiple owners.** The query-owned execution plan now performs
    the single deduplicated channel/attempt/filter-ID reset, exact generation assignment, and
    separate clock-epoch capture; preflight precedes allocation-free C2 hooks, and canonical
    success/abort end orders summaries/close/local teardown/residual cancellation.
17. **MAJOR — the planning view referenced an undefined decision/wired source.** C1 owns
    `dynamic_filter_key_decision` and stores the preserved `wired` decision in the builder.
18. **BLOCKER — a throwing multi-slot freeze could expose partial topology.** C1 now prepares every
    immutable plan/assignment before a statically-noexcept commit; C3 wraps only the generic grouped
    producer seam.
19. **MAJOR — cached re-finalization conflicted with one-shot assignment.** Matching reuse performs
    full descriptor verification without assignment; direct second commit remains an error.

### 2026-07-09 implementation resolutions (C1a-2 in flight)

20. **fmt support on strong identity types is deliberately omitted.** The environment's spdlog
    1.8.x bundles fmt 7 (predates `format_as`), op headers must not include spdlog/fmt, and nvcc
    cannot compile fmt's chrono headers. Log sites format the public `.value` member directly.
    C1b's telemetry work must not "restore" a formatter hook.
21. **Internal-error guards throw `sirius::internal_exception`.** The candidate cache's single-use
    protocol guards and the debug resolver fence use the project's own
    `sirius::internal_exception` (project convention in op/planner code), not
    `duckdb::InternalException`. Contract tests assert that type. Consequence accepted: on the
    transparent path a fence trip is caught and logged rather than invalidating the session, so
    the "pre-resolving harness fails loudly in debug" contract test is the guard against silently
    vacated coverage.
22. **`prepare_dynamic_filter_plans` takes the producer enumeration explicitly.** The engine
    enumerates every retained builder and passes
    `std::span<sirius_physical_hash_join* const>` as the first argument; the seam stays a pure
    function with no tree-walking knowledge. C3b's grouped additions remain the second argument
    and every addition's producer must be one of the enumerated joins.
23. **The builder does not store a separate DuckDB key count.** It is
    `key_candidates.size()` by construction (the adapter's whole-target arity fence already
    guarantees ordinal/column agreement), exposed as `duckdb_key_count()` and used verbatim for
    the terminal INFO line's key count once the `JoinFilterPushdownInfo` member is dropped.
24. **`dynamic_filter_planning_ordinal_view` carries `condition_index`, and the frozen plan
    stores the ordinal records.** The runtime cast-skip DEBUG line prints the condition index of
    a REJECTED key, which has no `admitted_key` to read it from, so the view/record type carries
    it top-level (same argument as `duckdb_ordinal`: rejected entries need self-carried
    identity). The frozen `dynamic_filter_publish_plan` stores the builder's ordinal records
    verbatim — view/plan parity holds by construction, and the publisher replays plan-time
    decisions instead of re-deriving them.
25. **The frozen plan's `probe_target` carries `target_id`/`channel_id`, and verification is
    builder-vs-frozen.** With the IDs in the frozen targets, the cached-re-execution check needs
    no separately stored fingerprint in C1a-2: `freeze_or_verify_dynamic_filter_plans` (the
    engine's one entry point) recomputes the descriptor from the builders and from the frozen
    plans and requires exact equality. C3b's cached routing descriptor slots into the same
    `verify_frozen_dynamic_filter_topology(cached, current)` value comparison later. A
    builder-less join (tests, non-producer shapes) freezes to the canonical disabled plan with
    an invalid (zero) publication ID — the nonzero-ID rule applies to builder-produced plans.

### 2026-07-10 implementation-audit clarifications

26. **C1a-2 is not complete at the builder/freeze boundary.** The working tree's builder, key
    decisions, prepare/commit seam, and publisher decoupling are foundation work. Completion still
    requires the reasoned publication result, exactly-once attempt state, prepared-topology lease,
    execution generation, channel reset, attempt reset, filter-ID reset, central begin/end, and a
    repeated-execution proof.
27. **The adapter remains the only allowed reader of pinned DuckDB metadata.** The candidate cache
    must ask the adapter whether metadata is present; it must not read `join.filter_pushdown`
    directly. In adapter terminology, `admitted` means that DuckDB metadata is structurally valid
    and has at least one live target. It does not mean that Sirius has proved lineage or can execute
    a route. The preserved `build_subtree_has_filter_hint` is the legacy Phase-1 wiring gate, not
    candidate admission and not lineage proof.
28. **Cache capture and extraction must cover the exact same node set.** Extraction must reject an
    unseen join, reject a captured join that disappears, and fill each captured entry exactly once.
    Before C1b coding starts, choose and document whether pre-resolver evidence is captured only for
    adapter candidates or for all condition ordinals and then selected during extraction.
29. **The planning view must be valid before C3 can read it.** It must guarantee strict vector
    position equals DuckDB filter ordinal, `admitted_key` exists if and only if the decision is
    admitted, and only equality candidates are admitted. Waiting until `finalize()` to discover
    these errors is too late for a sanctioned pre-freeze view.
30. **Freeze validates identities across the whole plan, not only within one builder.** Publication
    IDs and target IDs must be globally unique. Channel ID and channel object must form a two-way
    one-to-one mapping. This validation happens before any slot changes.
31. **Cached verification compares every runtime-relevant value.** That includes all ordinals,
    decisions, columns, types, channel-object associations, policy values, domain evidence, replica
    placement, and the preserved wiring decision. Comparing only IDs and decision bytes is not
    exact topology verification.
32. **C1b changes evidence representation and behavior together.** Unknown domain evidence becomes
    `std::optional<size_t>{}`; it is not sentinel zero. C1b computes and logs shadow decisions
    only. It must not populate the current numeric vector while leaving the existing publisher
    suppression branches active, because that would silently deliver C1d enforcement.
33. **A stable channel ID is created with the channel.** It must be available even when a scan sees
    zero filters and regardless of whether the producer or scan is planned first. One channel
    object and one channel ID name each other for the lifetime of the prepared topology.
34. **Filter-ID phase ownership is explicit.** C1a-2 owns the execution-scoped counter, reset, and
    one ID per materialized immutable filter before fan-out. C1b owns ID-carrying channel entries,
    telemetry fields, and reconciliation. C1b does not introduce a second counter.
35. **The single-assignment token contract is checked, not assumed.** Committing a foreign token or
    an already consumed token is an internal error. A failed check must not consume the token or
    leave its original slot permanently pending. Tests cover both cases.
36. **Delivery units remain independent.** A branch that combines C1a-1 with later C1 work does not
    satisfy the planned review and rollback boundary even when the combined code passes tests.
