# PR C3 implementation plan — SIP discovery/resolution, topology validation, runtime routes, and experiment telemetry

> **Status (2026-07-09):** Track A is merged delete-only (#1134, `1eecaf97` on
> `dev`); A1/A2/A3 never merged. Track B is deferred and blocks nothing. This revision
> incorporates the Track C implementation audit summarized in the
> [program plan](../issue-1010-implementation-plan.md#status-reconciliation-and-track-c-re-evaluation-2026-07-09).

Companion to [issue-1010-dynamic-filter-sip-design.md](../issue-1010-dynamic-filter-sip-design.md);
baseline `dev` `fac81e87` (contains #1134). File:line anchors were captured at `506a1d9f`
and must be re-grepped.

PR IDs covered: **C3a** (planner discovery/resolution, a frozen planning-topology descriptor,
flag, and planning telemetry), **C3b** (transactional runtime-topology installation, publisher
fan-out, execution-scoped channels, consumer wiring, and coverage tooling), and the **C4 gate
definition**. The Track D sketch remains contingent.

## 1. Goal + non-goals

C3 delivers the design's "first value experiment" row (design:968): a query-local route registry
that (a) discovers SIP lineage on the resolved logical plan before physical planning, (b) resolves
pending DuckDB ordinals through C1's constructor-resolved planning view, (c) freezes a
planning-only route descriptor in C3a, and then, in C3b, (d) validates physical pipeline topology
and C2 endpoint compatibility before transactionally adding producer targets and consumer
endpoints at C1's unconditional freeze seam. Runtime fans membership filters into dedicated
one-producer/one-consumer SIP channels consumed opportunistically at the C2 hash-join probe
checkpoint. Everything remains behind `enable_dynamic_filter_sip=false`, with telemetry needed to
answer the C3 coverage questions and gate C4. C3a creates no live channel, validates or installs no
runtime endpoint, and cannot affect publication or consumption. With the flag off,
planner/publisher/runtime behavior is byte-identical to Phase 1. Non-goals: ordered activation
(Track D — sketch only, §"Track D"), any producer-policy change (C1b-C1e), STANDARD-route
default-on (design:608-612), mixed-provenance/no-scan candidates (Track E, design:361-371),
MIXED-mode consumers (design:604-606), pre-partition unary consumers (design:100-110), scheduler
changes (Track A).

## 2. Deliverables — public types and APIs

**Reuse, do not redeclare, the C1 contracts.** `dynamic_filter_identity.hpp` owns all four
strong entity-ID types, the strong execution generation, and the executable-plan allocator.
`dynamic_filter_publish_plan.hpp` owns
`dynamic_filter_key_plan`, `probe_schema_ordinal`, `join_probe_target_key`, and
`join_probe_publish_target`. In particular, C3 does not introduce another publication/target
counter or a second type with the same name.

**Reuse C2's consumer endpoint; do not add a C3 copy.**
`hash_join_probe_filter_consumer.hpp` already owns `sip_endpoint_desc` and the immutable
endpoint vector. C3 populates its canonical publication-plan/target/channel IDs, dedicated channel,
and gate threshold, validates it through C2, and moves it through
`install_validated_sip_topology`. Admitted producer key/build-type data remains exclusively
in C1's `join_probe_publish_target`; the C2 endpoint needs no duplicate key vector. Discovery
does not pretend that the pre-constructor candidate cache contains admitted Sirius keys, and
freeze never indexes protected, equality-compacted hash-join members.

**`src/include/utils/single_assignment.hpp`** is the C1a-owned freeze primitive reused here:

```cpp
namespace sirius {
template <class T> class single_assignment {   // assign exactly once before runtime; lock-free reads
 public:
  prepared_assignment<T> prepare_assignment(T v); // all checks/allocation; may throw
  void commit_assignment(prepared_assignment<T>&&) noexcept;
  void assign(T v);                             // checked convenience; second assign throws
  [[nodiscard]] T const* get() const noexcept;  // nullptr while unassigned
  [[nodiscard]] bool assigned() const noexcept;
};
}
```

**`src/include/planner/dynamic_filter_lineage.hpp` / `src/planner/dynamic_filter_lineage.cpp`** — the pure pass (design:427-447, 494-518; G4 testability, design:929):

```cpp
namespace sirius::planner {
struct logical_plan_node_id { std::uint32_t value; };

enum class sip_crossing {
  CROSS,               // Filter / ORDER BY (no limit) / DISTINCT — binding unchanged (design:501)
  CROSS_REMAP,         // Projection: direct BoundReferenceExpression pass-through only (design:502)
  CONSUMER_THEN_LEFT,  // eligible intermediate comparison join (design:503)
  STOP                 // everything else: aggregate, LIMIT/TOP-N, WINDOW, UNNEST, set ops,
                       // CTE/materialization/CTE_REF, DELIM joins, computed expr (design:504-510)
};
[[nodiscard]] sip_crossing classify_sip_crossing(duckdb::LogicalOperator const& node);

// Fresh pure predicate of join_type — deliberately NOT derived from prove_unique_columns'
// preservation switch (sirius_plan_comparison_join.cpp:205-222), per design:355-359.
[[nodiscard]] bool sip_consumer_join_type_eligible(duckdb::JoinType t);        // INNER || SEMI
[[nodiscard]] bool sip_consumer_join_shape_eligible(duckdb::LogicalComparisonJoin const& j);
  // = type eligible && has COMPARE_EQUAL condition && !(equality+inequality mix that becomes
  //   MIXED_JOIN — mirrors the plan-time-constant ctor decision at sirius_physical_hash_join.cpp:342-347)

// Producer-shape predicate: the producing node itself must be a plain comparison join.
// LOGICAL_DELIM_JOIN / LOGICAL_ASOF_JOIN are LogicalComparisonJoin subclasses that Phase 1 DOES
// wire as producers today (delim routes through plan_comparison_join via
// sirius_plan_delim_join.cpp:73-76; the build_side_has_filter comment at
// sirius_plan_comparison_join.cpp:425 explicitly covers the delim case), but their probe subtrees
// contain DuckDB delim machinery and their pipeline shapes hit the RIGHT_DELIM_JOIN build-wrapping
// special case (sirius_pipeline_converter.cpp:1182-1197) — excluded from SIP in C3.
[[nodiscard]] bool sip_producer_shape_eligible(duckdb::LogicalOperator const& node);
  // = node.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN (exactly; not DELIM/ASOF)

// Remap a binding downward through one crossed node; nullopt == cannot remap (→ STOP).
[[nodiscard]] std::optional<duckdb::ColumnBinding>
remap_binding_through(duckdb::LogicalOperator const& node, duckdb::ColumnBinding b);

enum class sip_reject_reason : std::uint8_t {
  FLAG_DISABLED, PRODUCER_SHAPE, CROSSING_STOP, RIGHT_ORIGIN_BINDING, NO_GET_MATCH,
  DUPLICATE_GET_MATCH, FINAL_BINDING_MISMATCH, CONSUMER_INELIGIBLE, PRODUCER_NOT_WIRED,
  CONSUMER_NOT_PLANNED, CONSUMER_MIXED_MODE, NOT_HASH_JOIN_PHYSICAL, BRANCH_AMBIGUOUS,
  KEY_NOT_ADMITTED, NO_ADMITTED_KEYS, ORDINAL_OUT_OF_RANGE, TYPE_MISMATCH,
  NO_INTERMEDIATE_JOIN, TOPOLOGY_INSTALL_REJECTED
};

struct pending_join_probe_key {
  op::duckdb_filter_ordinal duckdb_ordinal;
  op::probe_schema_ordinal consumer_column;
  cudf::data_type consumer_type;
};

struct pending_join_probe_target {          // design:461-469
  std::optional<op::dynamic_filter_publication_plan_id> publication_plan_id; // filled from C1 view at bind
  op::dynamic_filter_target_id           target_id;
  logical_plan_node_id producer, consumer;
  std::vector<pending_join_probe_key> pending_keys; // discovery output; no admitted/build data yet
  std::vector<op::join_probe_target_key> keys;      // populated only at producer bind
  op::dynamic_filter_channel_id channel_id;
};
struct sip_candidate_rejection { logical_plan_node_id producer; op::duckdb_filter_ordinal duckdb_ordinal;
                                 sip_reject_reason reason; };
struct sip_discovery_result { std::vector<pending_join_probe_target> targets;
                              std::vector<sip_candidate_rejection> rejections; };

struct staged_consumer_addition {
  op::sirius_physical_hash_join* consumer;
  op::validated_sip_topology topology;
};
struct runtime_topology_additions {
  std::vector<op::dynamic_filter_target_addition> producers; // C1-owned generic value
  std::vector<staged_consumer_addition> consumers;
}; // move-only; exactly one entry per physical producer/consumer

struct topology_fingerprint { std::array<std::byte, 32> digest; }; // deterministic value encoding
struct canonical_sip_topology; // full retained descriptor; digest is only a fast reject
class prepared_sip_topology_state { // cached prepared-execution data, assigned exactly once
 public:
  [[nodiscard]] bool topology_committed() const noexcept;
 private:
  single_assignment<std::shared_ptr<canonical_sip_topology const>> retained;
};
struct prepared_channel_registration {
  std::shared_ptr<op::sirius_dynamic_filter_set> channel;
}; // one dedicated SIP channel, already checked; commit performs one noexcept atomic increment
struct prepared_sip_topology_commit { // opaque, move-only
  op::prepared_dynamic_filter_plans producer_plans;
  runtime_topology_additions additions; // consumer proof tokens
  std::vector<prepared_channel_registration> registrations;
  prepared_assignment<std::shared_ptr<canonical_sip_topology const>> retained_state;
};
prepared_sip_topology_commit prepare_sip_topology(
  prepared_sip_topology_state& destination,
  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> const& scheduled,
  runtime_topology_additions&&); // all fallible work
void commit_sip_topology(prepared_sip_topology_commit&&) noexcept;
void verify_sip_topology(
  prepared_sip_topology_state const& committed,
  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> const& scheduled);

// One producing join with DuckDB filter candidates in C1a's generator-local candidate cache.
// Pure over the resolved logical tree; the shared allocator mints only target/channel IDs.
sip_discovery_result trace_sip_routes(duckdb::LogicalOperator& resolved_root,
                                      duckdb::LogicalComparisonJoin& producer,
                                      duckdb_join_filter_candidate const& cached_candidate,
                                      dynamic_filter_identity_allocator& ids);
}
```

**`src/include/planner/dynamic_filter_route_registry.hpp` / `src/planner/dynamic_filter_route_registry.cpp`** — the query-local registry (design:471-492):

```cpp
namespace sirius::planner {
class dynamic_filter_route_registry {
 public:
  // --- discovery (pre-create_plan) ---
  logical_plan_node_id node_id_for(duckdb::LogicalOperator const& node);      // assign-on-first-use
  void add_discovery(sip_discovery_result r);

  // --- endpoint resolution (during create_plan; both roles resolved here, design:471-475) ---
  void bind_physical_join(duckdb::LogicalOperator const& logical_node,
                          op::sirius_physical_hash_join& physical,
                          op::dynamic_filter_planning_view const& producer_view);

  // --- planning-descriptor freeze (after create/fold/verify; no pipeline/channel/runtime work) ---
  // A repeated call for the same physical plan verifies/reuses the identical descriptor.
  struct finalize_stats { std::size_t frozen_targets, rejected_targets; };
  finalize_stats freeze_planning_descriptor();
  [[nodiscard]] bool planning_descriptor_frozen() const noexcept;

  // C3b only: validate converted pipelines and C2 compatibility, then stage both immutable ends.
  // This may allocate channels but installs nothing; every rejection drops both route ends.
  runtime_topology_additions stage_runtime_additions(
      duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> const& scheduled) const;

 private:
  // pending state: node-id map (LogicalOperator const* -> id), pending targets,
  // logical-node -> physical-join binds, rejections buffer; ALL cleared inside the first
  // planning freeze so runtime never sees DuckDB pointers or mutable route state (design:46-47,
  // 487-489). The frozen descriptor contains values and physical endpoint identities only.
};
}
```

**C1/C2 seams consumed by C3** (`src/include/op/sirius_physical_hash_join.hpp`):

```cpp
// C1a-2: a read-only view of the resolved builder, legal only during planning/finalization.
// It contains publication ID, wired/enabled policy state, and admitted ordinal/key/build-type
// decisions. It is not the runtime plan.
[[nodiscard]] dynamic_filter_planning_view planning_view() const;

// C1a-2/C1b: C1's generic preparation invokes this for every join, with zero or more grouped,
// topology-validated additions. Commit only publishes the returned prebuilt assignment.
prepared_dynamic_filter_plan_assignment
prepare_dynamic_filter_runtime_plan(std::vector<join_probe_publish_target>);
void commit_dynamic_filter_runtime_plan(prepared_dynamic_filter_plan_assignment&&) noexcept;

// C2a: state-pure, fallible preparation returns a proof token before C3's no-throw commit.
[[nodiscard]] sip_consumer_shape sip_consumer_capability() const noexcept;
static sip_topology_validation_result validate_sip_topology(
    sip_consumer_shape, std::vector<sip_endpoint_desc> endpoints);

// C2a/C3b: immutable consumer topology is installed in the same commit as producer additions.
void install_validated_sip_topology(validated_sip_topology&&) noexcept;
[[nodiscard]] bool sip_generation_matches(dynamic_filter_execution_generation) const noexcept;
void begin_sip_execution(dynamic_filter_execution_generation) noexcept; // C2-local state only
void end_sip_execution() noexcept; // after quiescent summary/channel close
```

C1a-2's engine-owned execution-boundary pass is the sole mutable-state coordinator. At execution
begin it visits every unique scan/SIP channel and publication attempt exactly once, clears filters
and outcomes, reopens acceptance, sets every channel to the exact strong execution generation,
resets the filter-ID counter, and establishes a separate monotonic event epoch. It then validates
every C2 endpoint generation before invoking allocation-free/noexcept
`begin_sip_execution(generation)` hooks. On mismatch it enters the same abort teardown used
for task failure. At query end the engine quiesces tasks/publishers, emits normal-or-partial
summaries, closes each channel, invokes `end_sip_execution()`, cancels residual C1 attempts,
and asserts terminal/closed state. C2 never resets a channel, changes a generation/attempt, or
removes an immutable endpoint.

**Config**: `bool enable_dynamic_filter_sip = false;` in `operator_params` + YAML + `SET enable_dynamic_filter_sip` (recipe in §3.9).

**Finalizer boundary.** C1 owns generic
`prepare_dynamic_filter_plans(grouped_producer_additions)` /
`commit_dynamic_filter_plans(prepared) noexcept` and knows no C2/C3 bundle. C3b owns the
`prepare_sip_topology` / `commit_sip_topology` wrapper (§3.4); that wrapper always
invokes the C1 boundary, including empty/no-registry cases. C3a contributes only a planning
descriptor.

## 3. Step-by-step changes

### 3.1 `src/planner/sirius_physical_plan_generator.{hpp,cpp}` — discovery hook + registry ownership

- Header: next to the scan-channel map (`sirius_physical_plan_generator.hpp:80-85`), add
  `std::shared_ptr<dynamic_filter_route_registry> sip_registry; // null unless SIP enabled and routes pending`.
- Cpp: add helper `dynamic_filter_sip_enabled(duckdb::ClientContext&)` cloning `dynamic_filter_pushdown_enabled` (`sirius_physical_plan_generator.cpp:38-45`), returning false with no `SiriusContext`.
- In `create_plan(duckdb::unique_ptr<duckdb::LogicalOperator>)`: after `resolver.VisitOperator(*op)` (`sirius_physical_plan_generator.cpp:126-128`) and before the recursion `create_plan(*op)` (`:132`), run discovery when `dynamic_filter_sip_enabled(context) && dynamic_filter_pushdown_enabled(context)`:
  first populate C1a's generator-local candidate cache exactly once for every comparison join,
  attaching any C1b shadow-domain snapshot. Then walk the tree; for each node with
  **`node.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN` exactly**
  (`LOGICAL_DELIM_JOIN`/`LOGICAL_ASOF_JOIN` reject as `PRODUCER_SHAPE`) whose cached value has
  DuckDB candidate probe targets, call `trace_sip_routes(..., cached_candidate, identity_allocator)` and
  `sip_registry->add_discovery(...)`. `plan_comparison_join` later reads the same cache entry;
  it never invokes a destructive second adapter extraction. This is the required post-resolver,
  pre-condition-drain window.
- Note `fold_adjacent_projections` (`:135`) runs on the physical plan and does not touch joins — physical pointers recorded by `bind_physical_join` stay valid.

**Lineage-pass internals** (`dynamic_filter_lineage.cpp`), per DuckDB filter ordinal and
`probe_info` target (design:430-443):
1. Starting binding = `producer.children[0]->GetColumnBindings()[cond_left_index]` — post-resolver left keys are `BoundReferenceExpression` positions into the left child's flattened output (design:155-160; the existing extraction precedent for right keys is `sirius_physical_hash_join.cpp:300-338`).
2. Locate the target `LogicalGet` by preserved `DynamicTableFilterSet` identity — walk `resolved_root` for `LogicalGet` with `get.dynamic_filters.get() == pi.dynamic_filters.get()` (same pairing key used today at `sirius_plan_get.cpp:264-266` and `sirius_plan_comparison_join.cpp:444`); zero or ≥2 matches → reject `NO_GET_MATCH`/`DUPLICATE_GET_MATCH` (design:433-435).
3. Descend from the producer's left child toward that GET, applying `classify_sip_crossing`/`remap_binding_through` (the projection remap mirrors the value-preserving pass-through logic used in `trace_binding_to_get`, `sirius_plan_comparison_join.cpp:288-300`, but over `BOUND_REF` post-resolver expressions and failing closed on anything computed). At each `CONSUMER_THEN_LEFT` join: if the binding originates in its left child (check via child `GetColumnBindings()` ownership), record `(consumer node, probe ordinal = position in left-child bindings, consumer probe-column logical type → cudf::data_type)` and continue left; right/build-origin → `RIGHT_ORIGIN_BINDING` stop (design:438-443, 512-513). The producer itself is never recorded (trace starts below it — design:402-404).
4. Require the final remapped binding to equal `pi.columns[duckdb_filter_ordinal].probe_column_index` (the field consumed today at `sirius_plan_comparison_join.cpp:451`); mismatch → `FINAL_BINDING_MISMATCH` (design:321-325, 437).
5. Group per `(producer node, consumer node)`, deduplicate paths, and stage only
   `(duckdb_filter_ordinal, consumer probe ordinal, consumer probe type)`. Use C1's shared allocator
   to mint the future target/channel IDs, but allocate no channel and construct no
   `join_probe_target_key`. Leave `publication_plan_id` unset until producer bind. Admission, the
   compact Sirius key ordinal, and authoritative producer build type do not exist in this pass;
   they come only from C1's resolved-builder planning view. A route with no intermediate join
   yields no target.

### 3.2 `src/planner/sirius_plan_comparison_join.cpp` — endpoint resolution

In `plan_comparison_join`, after the join is constructed and configured (`sirius_plan_comparison_join.cpp:504-519`), add:

```cpp
if (sip_registry) {
  sip_registry->bind_physical_join(op, hj, hj.planning_view());
}
```

`dynamic_filter_planning_view` is C1's sanctioned read-only view of the constructor-resolved
builder, not the runtime plan. It supplies the producer's existing `publication_plan_id`, wired
state, and one decision per DuckDB ordinal; an admitted decision carries the compact Sirius key
and authoritative build type. Producer bind maps each pending ordinal through that view, drops
nonadmitted ordinals with `KEY_NOT_ADMITTED`, and constructs `join_probe_target_key` values only
for admitted decisions. A non-wired producer rejects `PRODUCER_NOT_WIRED`; a target left with no
keys rejects `NO_ADMITTED_KEYS`. The same call records consumer physical identity and probe schema
width, but deliberately does not use a C2 type or call `sip_consumer_capability()`; C3b reads
that immutable value from the still-live bound physical consumer during runtime-topology staging.
It never calls
`dynamic_filter_plan()` before freeze and never indexes protected,
equality-compacted join members. Children are planned first, so consumer joins bind before their
producer. Delim/asof binds remain lookup-miss-tolerant no-ops because discovery rejected them.

Timing note: the SIP flag is snapshotted into the executable plan. `SET enable_dynamic_filter_sip`
affects a newly prepared plan, not a cached prepared plan.

### 3.3 Registry lifetime — plan-generator → engine handoff

The generator is stack-local at all four instantiation sites and dies immediately after
`create_plan` returns. The executable registry/frozen C3a planning descriptor must survive until
post-conversion validation; validation-purpose generators discard it and emit no accepted-plan
events. C1's builder/finalizer and execution-boundary state remain C1-owned operator/engine data.
Carrier for the C3 descriptor: `sirius_prepared_statement_data` (keeps operators free of planner
types):

- `src/include/sirius_interface.hpp:26-38`: add the registry/descriptor handle (null for flag-off
  or validation-only plans).
- Assign it at the three construction sites (all grep-verified):
  - `src/sirius_extension.cpp:601-605` — change `SiriusGeneratePhysicalPlan` (`:531-538`) to also out-return `physical_planner.sip_registry`, assign onto `gpu_prepared`.
  - `src/sirius_ffi.cpp:170-173` — hoist the generator into a named local; assign `gpu_prepared->sip_registry`.
  - `src/transparent/physical_sirius_execution.cpp:148-152` — same.
- `src/include/sirius_engine.hpp` (class at `:54`): add `std::shared_ptr<planner::dynamic_filter_route_registry> sip_registry;` in the **public** member section (the engine's data members are public and its only friends are pipeline classes, `sirius_engine.hpp:55-58` — a private trailing-underscore member would not be assignable from `sirius_interface`). Set it in `sirius_interface::sirius_pending_statement_internal` (`src/sirius_interface.cpp:150-177`) — the engine is created at `:160-163` and initialized at `:177` (the only `engine.initialize` call site, grep-verified; `sirius_execute_query` at `:214-240` only delegates via `sirius_pending_statement_or_prepared_statement`): `engine.sip_registry = statement_p->sip_registry;` before `engine.initialize(std::move(sirius_collector))`.
- **Do not clear the engine's registry in `sirius_engine::reset()`.** `initialize` calls
  `reset()` before `initialize_internal`; clearing there makes flag-on SIP a silent no-op.
  Retain the C3a descriptor through C3b staging, then drop only the engine's working registry
  reference. The prepared statement retains immutable topology for later execution preparation.
- The validation-only path discards plan+generator+registry together. Transparent replan builds a
  fresh executable plan each time. Cached prepared plans may reuse topology/IDs across distinct
  query log segments, but never reuse runtime channel/filter/gate/outcome contents.

### 3.4 C3b static validation and the unconditional C1 preparation/commit

At the end of successful top-level create/fold/verify, C3a calls
`freeze_planning_descriptor()`. It resolves no pipeline branch, allocates no channel, touches no
operator slot, and emits only buffered planning telemetry. Validation/fallback generators discard
the descriptor and telemetry. A C3a-only build therefore passes zero C3 additions to C1 and has
provably no runtime effect.

In `initialize_internal`, after `auto result = converter.convert(*root_pipeline);`
(`src/sirius_engine.cpp:353-355`) and before `materialize_repository_wiring`
(`:358-359`), C3b stages additions if a descriptor exists and the engine always invokes
the C3 wrapper, which in turn invokes C1's generic preparation/commit even for empty additions:

```cpp
if (!prepared_sip_topology_state->topology_committed()) {
  runtime_topology_additions additions; // empty for flag-off, C3a-only, and no-route plans
  if (sip_registry) {
    additions = sip_registry->stage_runtime_additions(result.scheduled_pipelines);
  }
  auto prepared = prepare_sip_topology(
      *prepared_sip_topology_state, result.scheduled_pipelines, std::move(additions));
  commit_sip_topology(std::move(prepared)); // noexcept; always commits C1 base plans
} else {
  verify_sip_topology(
      *prepared_sip_topology_state, result.scheduled_pipelines); // full value comparison; no allocate/reinstall
}
dynamic_filter_execution_plan = std::make_unique<dynamic_filter_execution_plan>(
    *prepared_sip_topology_state); // per-execution borrower; no commit/fingerprint ownership
dynamic_filter_execution_plan->begin_execution(next_execution_id()); // generation preflight, then C2
// Only now may create_query/start_query observe frozen topology and fresh state.
```

`prepare_sip_topology` invokes C1's generic preparation over every builder in the converted
graph, including disabled plans, scan-only producers, producers with every C3 route rejected, and
plans built without a SIP registry. No code path may use registry presence as the condition for
preparing/freezing C1.

Before the commit, `stage_runtime_additions` performs all fallible work over the C3a descriptor:

1. **Physical existence:** both endpoints were bound as `sirius_physical_hash_join`; a logical
   comparison join planned as NLJ rejects `NOT_HASH_JOIN_PHYSICAL`.
2. **Producer viability:** the C1 planning-view bind already supplied publication ID, wired state,
   admitted key, and build type. No runtime-plan accessor is legal here. Non-wired producers and
   targets with no admitted key were already rejected.
3. **Branch uniqueness:** require exactly one scheduled producer pipeline and one scheduled
   consumer pipeline with the expected build/probe dependency shape. Missing or duplicated
   branches reject `BRANCH_AMBIGUOUS`.
4. **Key validation:** check the captured consumer ordinal against the bound probe-schema width and
   compare the consumer type captured at discovery with the producer build type supplied at bind.
   This never indexes `right_key_col_indices`, `key_casts`, or another protected,
   equality-compacted member.
5. **Build and group both value ends:** allocate each dedicated channel into a temporary,
   unregistered staged value and construct its matching `join_probe_publish_target` and C2
   `sip_endpoint_desc`. Group all targets exactly once per physical producer and all endpoints
   exactly once per physical consumer before validation.
6. **C2 proof tokens:** read `auto shape = consumer->sip_consumer_capability()` now, after
   C2a is present and while the bound physical consumer is alive, then call
   `validate_sip_topology(shape, std::move(endpoints))` once
   per physical consumer. It verifies INNER/SEMI/non-MIXED shape, IDs/channels, and preallocates all
   local state. Its rejection granularity is deliberately the whole consumer group: if any endpoint
   fails, discard every endpoint in that vector, remove every matching target from all producer
   groups, and record one rejection plus one `PLANNING_REJECTED` target terminal per discarded
   route. No valid sibling remains attached to a consumer whose single proof token failed. A
   survivor contributes one move-only `validated_sip_topology`.
7. **Global preparation:** validate a bijection across producer targets and consumer tokens on
   `(publication_plan_id, target_id, channel_id, channel pointer)`; validate every registration
   and destination slot; build one `prepared_channel_registration` per dedicated SIP channel,
   every immutable C1 plan/prepared assignment, and the full descriptor/fingerprint. A rejected
   route leaves scan targets untouched.

All allocation, hashing, vector growth, slot checks, and failure injection happen before commit.
C1b marks `sirius_dynamic_filter_set::register_producer() noexcept`; its implementation remains
the existing single atomic increment. `commit_sip_topology(prepared) noexcept` consumes the
prepared-registration vector and consists only of those noexcept atomic increments,
C1 prepared-assignment publication, and one aggregate-pointer move into each C2 consumer. It
installs exactly once per physical producer and consumer. Tasks cannot observe topology until the
whole no-throw commit returns, so the transaction is observationally atomic. Add
`static_assert(noexcept(...))` for every commit operation and fault injection at each
preparation allocation; every injected failure must leave all slots/registrations unchanged.
C1's canonical `target_publication_terminal` records planning rejection/terminal outcomes; C3
does not invent another target state machine.

**Prepared-plan re-execution.** Immutable topology, channel objects, preallocated C2 storage, and
IDs may be reused; mutable execution contents may not. The first commit retains the full canonical
descriptor, digest, and commit bit in a cached `prepared_sip_topology_state` owned by prepared
execution data. The per-execution `dynamic_filter_execution_plan` only borrows that record and
is destroyed after its canonical end. A repeated initialization performs a full value
comparison (the digest is only a shortcut), emits `reused=true`, and neither allocates a new
channel/token nor invokes an install/assignment. Once the prior execution is quiescent, the single
C1 engine pass clears/reopens each unique channel and attempt, resets filter IDs and the separate
event epoch, and sets every channel to the exact next strong generation.
It preflights all endpoint generations and then calls allocation-free C2
`begin_sip_execution(generation)` to reset only local gates/counters/tracking. Canonical
success/abort end emits normal/partial summaries and closes channels before C2 local end and
residual-attempt cancellation. Tests change parameters or source data between executions so stale
filters, outcomes, IDs, or gates cannot hide behind identical results.

**Reachability of the verify/`reused=true` branch (overlaid review note).** On baseline
`fac81e87` every supported entry path rebuilds fresh prepared data: the extension path
`std::move`s `gpu_prepared` out (`sirius_interface.cpp:181`) so a second `EXECUTE` re-binds a
fresh generator/registry, the transparent path re-plans per execute
(`physical_sirius_execution.cpp:148-152`), and FFI is fresh per call. `topology_committed()` is
therefore observed `true` only if a *cached* `sirius_prepared_statement_data` is re-initialized
without a rebuild — which no current path does. Land the `verify_sip_topology`/`reused=true`
branch as a **forward-looking guard** (for a future Sirius physical-plan cache), not as protection
against a currently-live double-finalize hazard; either identify and pin the path that genuinely
re-initializes the same `prepared_sip_topology_state`, or mark the branch defensive and cover the
fresh-rebuild path with the same reset invariants. The e2e prepared-re-execution test (§4 test 6)
asserts fresh generation/filter-IDs/counters and **must not require** `reused=true` to be
observed — a run that reuses cached state and one that rebuilds fresh are both correct, and only
the reset invariants are load-bearing.

### 3.5 Producer fan-out — C1b frozen publication-plan seam

C3b depends on C1b's canonical target variant and C1's prepared-plan commit seam. Runtime
claim/publication reads only the frozen Sirius value; it never re-reads DuckDB metadata.

- C3's preparation wrapper passes the **grouped producer additions only** to C1. C1 combines each
  builder's validated scan base with that producer's staged SIP targets and prebuilds one immutable
  plan before `create_query`, including builders with no SIP additions. The unified target
  iteration keeps a drained-scan/live-SIP target publishable.
- The runtime claim gate is unchanged: C3 keeps SIP producers a subset of C1-wired producers, so
  publication consults only the already-frozen plan's `enabled()` value. Planning and topology
  validation never call the runtime accessor.
- The variant visitor pushes **membership filters only** into `join_probe_publish_target`
  channels—one filter per `join_probe_target_key` at
  `key.consumer_column.value`—and never zone maps. The same immutable filter object/ID fans
  out; channels co-own it without replica duplication.
- Key alignment is by construction at producer bind: each SIP key carries the admitted C1 key for
  its DuckDB ordinal. Freeze validates but does not rediscover that mapping.
- Reuse C1b's `target_visible` for each accepted SIP insertion and
  `target_publication_terminal` once per SIP target after fan-out. Extend the canonical INFO
  `publication_terminal` event with SIP target counts; do not add another visibility event.
- **Hard contract:** C3b rebases to the merged C1b target/freeze API. It never redeclares the
  target types or re-implements fan-out against pre-C1 internals.

### 3.6 Consumer wiring — `src/include/op/sirius_physical_hash_join.hpp`, `src/op/sirius_physical_hash_join.cpp`

- Reuse C1's producer runtime-plan `single_assignment` slot and C2's already-defined
  `hash_join_probe_filter_consumer` immutable topology/install seam. C3 adds no parallel
  consumer-plan type, endpoint type, or second slot; op headers include no planner type.
- Move each C2 `validated_sip_topology` proof token through
  `install_validated_sip_topology` in C3's no-throw commit. The token owns the exact endpoint
  vector and all preallocated gate/tracker state. After C1 resets channels/attempts/filter IDs and
  sets the exact generation/clock epoch, the coordinator preflights every
  `sip_generation_matches(generation)`. Only then does
  `begin_sip_execution(generation) noexcept` reset local state in place. It cannot allocate,
  reject, or remove one end of a route.
- Fast paths per design:615-620: null plan / `!channel->has_filters()` (`sirius_dynamic_filter.hpp:498-501`) / replica unavailable / disabled gate → zero-copy forward; replica-unavailable pass-through does not train the gate.
- The execution coordinator owns both success and abort teardown. After quiescence it emits exactly
  one normal or `partial=1 reason=...` INFO summary per endpoint, normal- or abort-closes
  each channel once, calls C2 `end_sip_execution()` to reset local state, then cancels
  residual C1 attempts and force-closes idempotent residue. Consumer code never resets a channel.

### 3.7 STANDARD-route probe-batch identity

Give the C2 `probe_batch_handle` its stable probe-batch ID from repository identity so repeated
STANDARD applications are visible. Tracking is execution-scoped and bounded: retain only the
aggregate count plus C2's capped FIFO diagnostic tracker (overflow counter emitted), then clear it at the
next `begin_sip_execution`. C3 reapplies safely and does not cache filtered tables.

### 3.8 Telemetry — log lines + analyzer (recon R1 recommendation: log lines primary; quent later)

Events below extend C1a-2's publication lifecycle through C1b's stable-ID event vocabulary,
per-target outcomes, query event sequence, and scan-channel coverage. C3 does not create a second
telemetry module. Machine-parsed
per-query summaries use `[dynf_summary]` at INFO; per-batch detail remains TRACE.

| line (k=v style per `MEM_HISTORY_RE` precedent) | level | emitting file |
|---|---|---|
| `[dynf_summary] sip_descriptor_frozen candidates= rejected= reused=` | INFO | C3a registry after accepted create/fold/verify |
| `[dynf_summary] topology_frozen sip_targets= rejected= reused=` | INFO | C3b preparation wrapper after accepted descriptor commit |
| `[dynf_summary] target_planned publication_plan_id= target_id= channel_id= kind=join_probe producer_op= consumer_op= keys=` | INFO | C3b wrapper after the complete topology commit; never for a C3a-only candidate |
| `[dynf_summary] publication_terminal publication_plan_id= outcome= ... sip_targets=` | INFO | C1 publisher finalization |
| `[dynf_summary] sip_consume_summary publication_plan_id= target_id= channel_id= partial= reason= batches= rows_in= rows_out= masks_applied= masks_skipped= replica_unavailable= apply_us= batches_before_first_filter= rows_before_first_filter=` (exactly once per endpoint after quiescence/abort) | INFO | execution coordinator + hash join |
| `[sip_registry] [dynf] candidate_rejected producer_node_id= publication_plan_id={id|none} ordinal= reason=` (pre-bind discovery rejection uses `none`; buffered until the accepted planning descriptor freezes) | DEBUG | `dynamic_filter_route_registry.cpp` |
| `[sirius_physical_hash_join] [dynf] target_visible publication_plan_id= target_id= channel_id= filter_id= key_ordinal= target_kind=join_probe filter_kind=membership consumer_column=` | DEBUG | `dynamic_filter_publisher.cpp` |
| `[apply_dynamic_filters] [dynf] membership_measured publication_plan_id= target_id= channel_id= filter_id= target_kind=join_probe consumer_column= input_rows= kept_rows= keep_ratio= decision=` | DEBUG | C2 consumer component .cpp |
| `[sirius_physical_hash_join] [dynf] target_publication_terminal publication_plan_id= target_id= channel_id= outcome= filter_ids=` | DEBUG | C3 preparation for `PLANNING_REJECTED`; publisher finalization otherwise |
| `[sip_consume] consume_batch channel_id= probe_batch_id= rows_in= rows_out= filters_visible= masks_applied= masks_skipped= replica_unavailable= apply_us=` | TRACE | C2 consumer component .cpp |
| `[sirius_physical_hash_join] [dynf] channel_closed channel_id= reason={CONSUMER_FINALIZED|ABORT}` | DEBUG | execution coordinator/hash join |

When C3 preparation records `PLANNING_REJECTED` it calls a C1/hash-join-owned telemetry
helper; it does not emit a hash-join-prefixed line from the registry file. The execution coordinator
likewise asks the owning hash join to emit `channel_closed`. This preserves the rule that a
component prefix names the component that actually emits the record.

Rules: IDs are in-line, never addresses. Publication and per-target events carry
`publication_plan_id`. C1b's INFO `scan_consume_summary` remains per shared scan channel
and therefore does **not** require one singular publication-plan ID; stable publication/target/filter
attribution comes from DEBUG `target_visible` and `membership_measured`. C3 reuses those
names and `target_publication_terminal` rather than adding parallel visibility or terminal
events. Per-batch events stay TRACE; one-shot events are INFO/DEBUG. Normal consume summaries use
`partial=0 reason=NONE`; abort summaries use `partial=1` plus the shared strict teardown
reason enum. The coordinator deduplicates scan summaries by channel ID (one per unique shared scan
channel) and emits SIP summaries by target ID (one per installed SIP endpoint).

`topology_frozen` is emitted only in C3b when an accepted C3 descriptor reaches runtime
preparation; the unconditional empty C1 base-plan commit emits no SIP line. Extend C1b's
dynamic-filter analyzer with SIP tables/patterns and one `SHAPE_VERSION` bump per landed
log-shape change. The parser accepts INFO-only timing logs without requiring DEBUG/TRACE families.
It warns only when a C3b run has `topology_frozen sip_targets>0` but zero INFO consume
summaries; C3a-only descriptor runs are not warned.

**Coverage report:** routes planned/rejected; per-channel rows/batches before first filter and after;
scan-caught versus C1/C2-caught using C1b `scan_consume_summary` plus shared filter IDs;
hash-probe rows and
estimated bytes avoided; mask cost/gate decisions; publication outcomes/replica bytes; and wall
time plus valid query-scoped memory evidence. Scan attribution is a C1b hard dependency, not
something inferred from publisher events alone.

**Runbook** (ships as a section in the dynamic-filters.md update, §3.10) — **log level is stated per pass and is part of the protocol**:
- **Timing passes: `SIRIUS_LOG_LEVEL=info`.** N≥5 paired runs; all wall-time numbers come only
  from these passes. Coverage is derived from canonical `publication_terminal` and
  `scan_consume_summary` plus C3's `sip_consume_summary` in the same timed runs — no
  TRACE required.
- **Coverage-detail passes: `SIRIUS_LOG_LEVEL=trace`, separate and non-timed.** Needed only for item (c) filter-level attribution and per-batch `consume_batch` drill-down; their timings are excluded from every statistic. (CI already exports `SIRIUS_LOG_LEVEL: trace`, `test.yml:116` — fine for tests, never for timing.)
- Functional/timing matrix: {pushdown off reference} ∪ {pushdown on} × {sip off,on}; no priority
  dimension. The run manifest records the plan-snapshotted
  `dynamic_filter_selectivity_gate` and `enable_dynamic_filter_unfiltered_build` values
  for every pass; the core C3 comparison pins them to `shadow` and `false` respectively
  (or records `not_available` if that orthogonal sibling has not landed). Re-prepare after each
  `SET` because flags are plan-snapshotted. Use all TPC-H queries at SF10 plus the synthetic
  chain; report which shapes actually admitted routes. Timing legs may share a warmed process.
  **Memory acceptance legs run one query/configuration per fresh process**
  unless comprehensive query-scoped samplers have landed; process-lifetime peaks from serialized
  timing runs are not memory-gate evidence.

### 3.9 Config flag — exactly the R3 recipe

- `src/include/sirius_config.hpp`: `bool enable_dynamic_filter_sip = false;` in `operator_params` (fields block `:102-118`).
- `src/sirius_config.cpp`: `r.optional("enable_dynamic_filter_sip", opt.enable_dynamic_filter_sip);` in `from_yaml(..., operator_params&)` (`:160-180`, before `reject_unknown` `:179`).
- `src/sirius_extension.cpp`: `SetEnableDynamicFilterSip` cloning `SetEnableDynamicFilterPushdown` (`:1604-1611`); registration next to `:1841-1848` with default from a fresh `sirius::operator_params{}` (leak-rationale convention at `:1734-1736`).
- Consumption: only `dynamic_filter_sip_enabled` in the plan generator (§3.1). No runtime read.

### 3.10 `docs/super-sirius/dynamic-filters.md` — Phase 2 section update (ships with C3; design:1122)

Rewrite `:389-408`:
- Consumer is the **in-join probe checkpoint** after the consumer's CONCAT/repository pop, before `prepare_join_keys` — supersede the current text's implicit shape and state explicitly it is not a pre-partition unary operator (design:100-110, 376-404).
- **Supersede** `:395` ("Routing (new): Sirius-owned `sirius_sip_route` route key") and `:405` ("`dynamic_filter_channels` becomes keyed by a variant"): the finalized design keeps the scan map keyed by `DynamicTableFilterSet*` unchanged and gives each `(P,C)` SIP target a dedicated one-producer/one-consumer channel minted by the query-local route registry (design:558-577). Update `:128` ("route key generalizes to a variant") with a pointer to the superseding design.
- **Supersede** `:396` coordination sentence: v1 is opportunistic (C3); ordered activation is contingent Track D, not "implicit upstream meta-pipeline" ordering (design:49-53, 642-660).
- Add: flag `enable_dynamic_filter_sip` (default false), layered target model summary (design:406-425), identity model (four IDs), telemetry event names, the C3 runbook (incl. per-pass log levels), and a link to `issue-1010-dynamic-filter-sip-design.md`. Extend References (`:438-446`) with the new registry/lineage/sip-plan files.

### 3.11 Build wiring

New prod files added to the extension source list in `CMakeLists.txt`; new test files appended to `TEST_SOURCES` (`CMakeLists.txt:562`, executable at `:693`) — mandatory, enforced by the orphan-test hook (`.pre-commit-config.yaml:81-83`, `scripts/check_orphan_tests.py`).

## 4. Tests

All Catch2 in the single `sirius_unittest` binary; CI runs everything on `gpu-2xl4` (`.github/workflows/test.yml:115,133-136`). SQLLogic is not CI-run (recon R4 §4) — e2e coverage is Catch2.

1. **`test/cpp/planner/test_dynamic_filter_lineage.cpp`** — pure crossing/remap/eligibility
   truth tables, including DELIM/ASOF/UNION stops, plus SQL-through-planner cases for left-derived
   versus right-origin bindings, missing/duplicate GET identity, final mismatch, grouping/dedup,
   and no-intermediate-join. Assert discovery output contains only DuckDB ordinal, consumer
   ordinal/type, future target/channel IDs, and node identity: no admitted Sirius key, producer
   build type, channel object, or runtime endpoint exists.
2. **`test/cpp/planner/test_sip_planning_descriptor.cpp`** — C3a bind/freeze cases. A
   sanctioned C1 `dynamic_filter_planning_view` supplies one publication ID and decisions;
   admitted ordinals become compact keys with the view's build type, rejected ordinals record
   `KEY_NOT_ADMITTED`, and an empty survivor records `NO_ADMITTED_KEYS`. Cover non-wired
   producers, consumer-first bind order, duplicate bind, and descriptor reuse. Assert C3a neither
   includes the C2 endpoint/shape contract nor calls `sip_consumer_capability()`. With C3a alone,
   enabling the flag allocates no channel, installs no endpoint, never calls the runtime-plan
   accessor, emits `sip_descriptor_frozen` only after accepted create/fold/verify, and
   produces the same runtime plan/publications as flag off.
3. **`test/cpp/planner/test_sip_topology_freeze.cpp`** — C3b transaction cases. Validate
   physical existence, branch uniqueness, ordinal/type parity, and C2 static topology before any
   mutation. MIXED/non-INNER-or-SEMI or a bad destination drops both ends. Verify C3's wrapper still
   invokes generic C1 preparation/commit for disabled, scan-only, zero-admitted, all-C3-rejected,
   and no-registry builders, each with zero additions. A two-producers→one-consumer case groups one
   install per operator and validates the four-field producer/consumer bijection; a valid+invalid
   sibling case proves one failed C2 group rejects both routes and removes both matching producer
   targets. The C2 proof token owns the exact endpoint vector and preallocated local storage. Inject
   failure at every preparation
   allocation/slot/fingerprint step and assert no registration or slot changes. The final commit
   has compile-time `noexcept` assertions for `register_producer` and every other operation,
   and only prepared atomic registrations/moves/publications.
   Re-finalization verifies an identical fingerprint without assigning again and rejects an
   incompatible descriptor.
4. **Execution-boundary ownership:** extend C1/C2 lifecycle tests to count resets. One engine pass
   visits each shared scan/SIP channel and attempt once, resets filter IDs, sets the exact strong
   generation, and separately establishes the event epoch. Generation preflight precedes
   allocation-free/noexcept C2 begin. Test normal end and failure before consumer finalization:
   tasks/publishers quiesce, one normal/partial summary and close emit, C2 end resets local state,
   then C1 cancels/asserts residual attempts and force-closes residue. Re-execute a prepared plan
   with changed parameters/data to expose any stale filter, outcome, ID, or gate.
5. **`test/cpp/operator/test_dynamic_filter_sip_publish.cpp`** — membership-only fan-out;
   zone maps never enter SIP channels; one filter object/ID reaches scan and SIP targets; a closed
   SIP target does not affect a scan sibling; drained-scan/live-SIP still publishes. Assert
   canonical `target_visible`, `target_publication_terminal`, and
   `publication_terminal` records with `publication_plan_id` and no alternate
   visibility event.
6. **`test/cpp/integration/test_dynamic_filter_sip_e2e.cpp`** — nested INNER/SEMI chains,
   composite keys, NULLs, empty build, **downgraded (CPU-fallback) build** (design:1037-1039),
   zero-row probe, STANDARD consumers, and non-prefix payloads.
   Assert flag-on C3b emits `topology_frozen sip_targets>=1` plus INFO consume summaries, and on
   one deterministic route-bearing case assert a **positive masking result**
   (`masks_applied>=1`, `rows_out<rows_in`) so a route that freezes topology but silently masks
   nothing is caught (SIP never changes results, so no correctness assertion can). The
   downgraded-build case asserts ON/OFF bag-equivalence and that the producer seals
   `NO_MATERIALIZATION(SOURCE_UNAVAILABLE)` and pushes no membership when its build downgrades.
   Prepared re-execution changes a parameter or source table and observes a new generation, filter
   IDs restarting at one, and fresh counters (it does not require `reused=true`; see §3.4).
   LIMIT/TOP-N variants use explicit expected rows or a
   filters-disabled reference; B1 is not a gate.
7. **Multi-GPU:** extend `test/cpp/operator/test_sirius_dynamic_filter_mgpu.cpp` — a SIP
   endpoint on a non-producer device applies the local replica; replica unavailable passes through
   and does not train the gate.
8. **Flag-off/analyzer invariance:** no registry and no SIP lines with the flag off; C3a-on/C3b-absent
   has planning telemetry but zero runtime route effect. INFO-only fixtures include
   `publication_terminal`, shared-channel `scan_consume_summary` without a required
   singular publication ID, and per-endpoint `sip_consume_summary`. DEBUG fixtures require
   four-ID `target_visible`/`membership_measured` and
   `target_publication_terminal`.

## 5. Gate & rollback

**Merge gate for C3:** C3a with the flag on produces only the frozen planning descriptor/telemetry
and is runtime-equivalent to flag off: zero live channels, endpoints, or publication changes. C3b
keeps `enable_dynamic_filter_sip=false` by default; flag-on e2e includes common-finalizer
liveness, transactional installation, and split C1/C2 state-reset assertions.
Publish/snapshot/close is TSan-clean; INFO-only and TRACE analyzer fixtures parse without format
warnings. B1 is not required for merge, experiment, or enablement. At least one deterministic
route-bearing e2e case additionally asserts a **positive runtime-masking result**
(`masks_applied>=1` and `rows_out<rows_in` on its `sip_consume_summary`): because SIP never
changes query results, the correctness, liveness, and `topology_frozen sip_targets>=1`
assertions all pass even if C3b fan-out or the C2 checkpoint is wired but silently masks nothing,
and the zero-consume analyzer warning catches only a consumer that never ran, not one that ran and
filtered zero rows.

**C4 gate definition (the deliverable — measured after the C3 experiment, decides default-on; design:969, 642-660, 199-201):**
1. **Correctness:** ON/OFF bag-equivalence across the full e2e matrix incl. multi-GPU, STANDARD consumers, and downgraded (CPU-fallback) build (design:1037-1039); exact order only where SQL guarantees it (design:1013-1015). Zero diffs. Hard veto.
2. **Prerequisites:** C1a-2 lifecycle/reset, C1b scan telemetry, and C2b's history-aware reservation floor are
   merged; the explicit-or-filters-disabled oracle is used on the current DuckDB pin. Neither B1
   nor an A2 flag is a prerequisite.
3. **Value:** on route-admitting queries, wall-time improvement outside measured run variance attributable (via consume-summary/rows-avoided) to SIP checkpoints; full-suite geomean not regressed beyond variance. All timing from INFO-level passes (§3.8 runbook).
4. **Coverage explains value:** for every route class kept opportunistic, pre-publication miss rate low enough that measured value persists; a class with systematic misses that erases its value makes C4 **depend on Track D** for that class rather than defaulting on (design:53, 657-660, 1092-1093).
5. **Memory:** per-space query high-water and filter-replica bytes are unchanged beyond variance.
   Evidence comes from comprehensive query-scoped samplers or fresh-process memory legs, never a
   process-lifetime peak from serialized timing runs.
6. **Overhead bound:** on non-benefiting queries, first-mask cost is bounded by gate disable behavior; redundant-downstream keep-rate shows layered masks disabling as designed (design:417-421, 1100-1101).
7. **STANDARD routes:** default-on only if repeated-application cost measured acceptable; otherwise STANDARD consumers stay flag-gated (design:608-612).
8. **Priority independence:** moot — the build-priority pass was deleted in #1134, so there is no `build_priority=legacy` mode for SIP value to depend on (was design:973-975).

**Rollback:** `SET enable_dynamic_filter_sip=false` followed by re-prepare, or YAML; C3a/C3b are
separately revertible and introduce no persistent schema. Retain the flag for one release after C4.

## 6. Dependencies & ordering

- **Hard before C3a:** C1a-2's cache, canonical types/allocator,
  `dynamic_filter_planning_view`/`planning_view()`, lifecycle, and unconditional common
  freeze seam, plus C1b's compact targets, executable-plan telemetry, and scan coverage.
- **Hard before C3b:** C1's prepared-plan commit and per-execution
  `dynamic_filter_execution_plan`, plus
  C2a's immutable consumer capability, proof-token/preallocation seam, generation preflight, and
  noexcept local begin/end hooks; C2b's full reservation floor/memory-order contract also lands first.
- **A/B:** #1134 is already on `dev`; no A flag exists. B1 is deferred and blocks nothing.
- **Producer policies:** former C1c is in C1b. C1d/C1e are orthogonal; record their snapshotted
  settings in every C3 run.
- **Internal order:** C3a freezes and reports a planning descriptor but installs/observes no runtime
  route. C3b adds post-conversion validation, transactionally committed fan-out/consumer
  endpoints, registration with C1's execution coordinator, e2e/mgpu tests, analyzer, and runbook.
- **C4** is a measurement/config PR (default flip + doc), not new machinery. **Track D** only if C4 criterion 4 fails for a valuable class.

## 7. Size estimate

Prod: C3a contains lineage, planning-view bind, value-only descriptor, config, and planning
telemetry. C3b contains converted-pipeline/C2 static validation, staged channels, preparation/commit wrapper
additions, C2 install/hook integration, publisher fan-out, runtime telemetry/analyzer, and docs. C1-owned
IDs, prepared-plan commit seam, and execution coordinator are reused rather than
copied. Recommendation: keep the C3a/C3b split in §6—C3a is mergeable with proven zero runtime
effect; C3b carries all runtime risk behind the default-off flag.

## 8. Risks (implementation-level) & mitigations

1. **Registry lifecycle vs `sirius_engine::initialize`'s internal `reset()`.** Clearing the
   C3a descriptor before C3b stages additions makes SIP a silent no-op. Mitigation: prepared data
   retains the value descriptor through post-conversion staging; the engine then drops only its
   working registry reference. The C3 wrapper still invokes generic C1 preparation/commit when no
   registry exists.
2. **Dangling operator pointers in the registry.** Validation-purpose generators never export a
   registry; freezing the C3a descriptor destroys logical-pointer maps, and C3b drops physical
   working references after its prepared commit.
3. **Logical-node address instability.** `plan_comparison_join` moves `op.conditions`/`filter_pushdown` (`sirius_plan_comparison_join.cpp:408-409,514`) but node objects stay alive through `create_plan`; the node-id map is used only during discovery+bind and destroyed at freeze. Guard: discovery captures everything it needs (bindings, types, ordinals) as values before recursion.
4. **Ordinal-space confusion** (`duckdb_filter_ordinal` vs `condition_index` vs
   `sirius_key_ordinal` vs `probe_schema_ordinal`). Mitigation: discovery stores only the
   DuckDB ordinal plus consumer ordinal/type; producer bind maps through C1's ordered planning view
   to the admitted key/build type. Freeze never indexes protected equality-compacted members.
5. **Freeze/converter ordering drift.** C1's prepared assignment and C2's proof token reject
   preparation/installation after `create_query` or any task; runtime access requires committed
   state.
6. **Publisher pushing zone maps into SIP channels.** Mitigation: the variant visitor pushes
   membership filters only for the join-probe alternative; publisher test 5 asserts.
7. **`register_producer` double-count / scan-elision interference.** SIP registration is
   validated before and performed once in the statically-noexcept commit; channels are never consulted by
   scan elision. Transaction test 3 covers.
8. **Delim/ASOF producers minting SIP routes.** Phase 1 *does* wire delim producers today (`sirius_plan_delim_join.cpp:73-76` delegates to `plan_comparison_join`; the `build_side_has_filter` comment at `sirius_plan_comparison_join.cpp:425` covers the delim case), so this is reachable, not "impossible by construction": without the §3.1 producer-shape check, routes would trace through DuckDB's delim machinery and freeze step 4 would meet the `RIGHT_DELIM_JOIN` build-wrapping pipeline shape (`sirius_pipeline_converter.cpp:1182-1197`). Mitigation: `sip_producer_shape_eligible` rejects `PRODUCER_SHAPE` at discovery; crossing STOP covers the consumer side; `bind_physical_join` stays lookup-miss-tolerant; lineage test pins both (test 1).
9. **C1 seam drift.** C3a/b are blocked on C1a-2's canonical types/allocator/cache/lifecycle/
   execution/prepared-commit contracts and C1b's target/event/scan-coverage contracts; C3 never
   carries a private compatibility copy.
10. **Log-format drift breaking the experiment tooling.** Anchors + strict regexes land in the same PR as the emitters, `SHAPE_VERSION` bumped, and format-warning counters surface drift in `_summary.json`. The zero-consume warning is scoped to C3b `topology_frozen sip_targets>0`, so C3a planning-only runs do not false-positive.
11. **Flag-on plan-time cost** (extra full-tree walks per candidate producer). Discovery does not
    know admission yet; bound the walk to joins with non-empty DuckDB candidate targets, measure it,
    and memoize the GET-identity index if needed.
12. **Identity collision or one-producer/many-plan IDs.** Only C1's allocator mints IDs; producer
    bind copies its existing publication ID into every SIP target. Tests mix scan and SIP targets.
13. **Destructive candidate extraction.** Discovery and physical planning share one immutable
    generator cache; an extraction-count assertion proves one adapter call per logical join.
14. **Prepared-execution state leak or double reset.** C1
    `dynamic_filter_execution_plan::begin_execution` alone clears/reopens channels/attempts
    and resets filter IDs, assigns one exact strong generation, and captures a separate clock epoch.
    Generation preflight precedes allocation-free C2 local reset. Strong-ID dedup counts and a
    parameter-changing prepared test are merge blockers.
15. **Filter visibility invalidates memory history.** C3b cannot merge before C2b's generation- and
    join-state-aware reservation floor is evaluated even on history-backed tasks.
16. **Conditional C1 freeze leaves disabled/no-route joins half-built.** C3's wrapper invokes C1
    preparation/commit unconditionally and tests no-registry, disabled, scan-only, and
    all-C3-rejected builders.
17. **Late C2 rejection strands a producer endpoint.** C2 preparation returns a reason or the only
    installable proof token before any mutation; execution begin assumes the proof and cannot
    remove a route.
18. **Telemetry forks the C1b schema.** C3 reuses `publication_terminal`,
    `target_visible`, `membership_measured`, and
    `target_publication_terminal` with `publication_plan_id`. Shared-channel
    `scan_consume_summary` has no singular publication ID.
19. **Consumer eligibility inputs disappear or create a hidden C3a→C2a dependency.** C3a retains
    physical consumer identity only. C3b reads C2's immutable `sip_consumer_shape` through the
    public capability accessor during staging, while the physical operator is alive; it never
    rereads logical or protected join state.
20. **Validation token is detached from installed bytes.** C2 validation consumes the grouped
    endpoint vector and returns the only installable move-only token, which also owns preallocated
    local state.
21. **Multi-operator commit throws halfway.** All allocation/slot checks occur in preparation;
    commit operations are statically `noexcept` and tasks cannot start until the entire commit
    returns. Fault injection proves preparation changes no slot/registration.
22. **One operator is installed twice or route ends diverge.** Group once per physical operator and
    validate the four-field producer/consumer bijection before commit; include
    two-producers→one-consumer coverage.
23. **Generation, clock, or abort lifecycle drifts.** Use C1's strong exact generation and separate
    clock epoch, preflight before allocation-free begin, and one normal/abort end algorithm with
    partial summaries and channel closure.

---

## Track D sketch (contingent — NOT planned in detail; design:662-757)

**Admission** (post-freeze pass in the same registry, new `select_ordered_targets(...)`): among frozen SIP targets, admit at most one ORDERED target per runtime consumer pipeline; require the channel's single registered producer; validate acyclicity over the union of repository/data edges (`repository_wirings_`/`setup_pipeline_parents`, `sirius_pipeline_converter.cpp:1133-1147`), activation edges, and the synthetic hint path used to drive the producer; require every task-driving node in the consumer→producer activation closure runtime-unique, else demote to opportunistic; tie-break deterministically by estimated avoided probe work. The activation descriptor stores the producer's **build-side publication driver pipeline** (build PARTITION/CONCAT path — `dependencies[0]` chain, `sirius_pipeline_converter.cpp:1170-1171`), never producing join `P` itself, because `P`'s hint recursion can re-enter through `C` and manufacture a cycle (design:666-679).

**Token**: one query-owned `sip_activation_token` per `(channel, consumer runtime pipeline)` with CAS FSM `IDLE→ARMED→QUEUED→CLAIMED | →DETACHED`; the check runs in the task creator **before hint resolution** (hints are stateful: `NOT_BUILT→SCHEDULING` on read, `sirius_physical_hash_join.cpp:486-547`), i.e., ahead of `get_next_task_hint`/input pop/reservation in `src/creator/task_creator.cpp`. Ordered activation reuses C1's publication-attempt FSM
(`OPEN→PUBLISHING→PUBLISHED|NO_MATERIALIZATION(reason)|FAILED|CANCELLED`) and adds no
second completion taxonomy. Completion at STANDARD/MIXED decision or
build-CONCAT calls `release_once`; only `ARMED→QUEUED` enqueues; task creator CAS
`QUEUED→CLAIMED` before dereferencing; wakeups are queued only after locks are released.

**Teardown** (error path, order fixed by design:740-750): stop task creation & reject activations → detach `IDLE/ARMED/QUEUED` tokens and invalidate queued requests → drain creator queue/claimed requests/executors/publishers → cancel open channels → remove edges/tokens → destroy pipelines/operators → restart creation only when both queues are empty. Files touched: `src/creator/task_creator.*`, `src/pipeline/sirius_pipeline.*`, query-owned activation registry, join mode/build-completion hooks; fault-injection suite per design:1042-1052 is a ship blocker for D.

---

## Review resolution

| # | Finding | Resolution |
|---|---|---|
| 1 | BLOCKER — clearing the registry in `reset()` makes SIP a silent no-op | Applied: C3a freezes a value-only planning descriptor retained through C3b post-conversion staging; the engine drops only its working reference afterward. C3's wrapper invokes generic C1 preparation/commit regardless of registry presence (§3.3-3.4, risk 1). |
| 2 | MAJOR — "DELIM plans reaching bind is impossible" false; delim producers are Phase-1-wired and reach `bind_physical_join` via `sirius_plan_delim_join.cpp:73-76` | Applied: new `sip_producer_shape_eligible` requires `node.type == LOGICAL_COMPARISON_JOIN` exactly; `DELIM_JOIN`/`ASOF_JOIN` rejected with new reason `PRODUCER_SHAPE` at discovery (§2, §3.1); lineage test pins both types (§4 test 1); risk claim rewritten honestly (risk 8). |
| 3 | MAJOR — freeze TYPE_MISMATCH read protected, equality-ordinal-compacted join members | Applied: discovery stages only DuckDB ordinal plus consumer column/type. Producer bind maps the ordinal through C1's sanctioned `planning_view()` to the admitted key/build type; freeze validates copied values and never indexes protected join members (§2, §3.1-3.4, risk 4). |
| 4 | MAJOR — C1's plan was immutable in the ctor while C3 appended after conversion | Applied: C1 exposes generic prepare/noexcept-commit for grouped producer additions; C3 wraps it with C2 proof tokens and commits only prebuilt values. Disabled/no-registry/scan-only builders receive zero additions (§3.4-3.5). |
| 5 | MINOR — lazy gate construction on first probe task is a data race | Applied: C2 proof-token preparation preallocates gates before commit; `begin_sip_execution(generation) noexcept` only resets them after C1's central reset/preflight (§3.4, §3.6). |
| 6 | MINOR — §3.3 misattributed the assignment site; member visibility | Applied: function corrected to `sirius_interface::sirius_pending_statement_internal` (`sirius_interface.cpp:150-177`; `:177` engine.initialize; `sirius_execute_query` `:214-240` delegates); `sip_registry` declared public without trailing underscore to match the engine's public-member style (friends are pipeline classes only, `sirius_engine.hpp:55-58`) (§3.3). |
| 7 | MINOR — `require_two_gpus` citation | Applied: `mgpu_test_utils.hpp:144` (`mgpu_env_params` at `:62`) (§4 test 7). |
| 8 | BLOCKER — prepared topology reuse leaked closed channels, filters, gates, outcomes, and counters | Applied: topology/preallocated local storage may be reused, runtime contents may not. C1 sets exact strong generation and separate clock epoch; generation preflight precedes allocation-free C2 begin. Canonical success/abort end handles summaries/close/local end/residual cancellation (§2, §3.4, §4). |
| 9 | MINOR — two loose citations | Applied: producer-subset invariant now cited as an inference from design:964-966 (C1e), not design:72-79 (§3.2); `key_casts`/`right_key_col_indices` cited at `hpp:212`/`:225` (§2, §3.4). |
| 10 | MINOR — runbook/analyzer log-level mismatch | Applied: timing is INFO-only and parseable without DEBUG/TRACE; trace passes are separate; memory legs use fresh processes unless exact samplers land (§3.8). |
| 11 | BLOCKER — C1/C3 independently minted IDs and redeclared target types | Applied: C3 reuses C1's canonical types/allocator, mints only new target/channel IDs, and copies the producer publication ID at bind (§2, §3.1-3.2). |
| 12 | BLOCKER — C3 discovery consumed C1b's destructive snapshot before physical planning | Applied: one generator-local immutable candidate cache feeds both consumers (§3.1). |
| 13 | MAJOR — deferred B1 was still required for experiment/C4 | Applied: B1 blocks nothing; current-pin selector tests use explicit/filter-disabled oracles (§4-6). |
| 14 | BLOCKER — C2 history could under-reserve after a filter becomes visible | Applied as dependency: C3b waits for C2b's generation/state-aware reservation floor (§6, risk 15). |
| 15 | BLOCKER — discovery treated candidate ordinals as admitted keys/build types | Applied: C3a stages pending DuckDB ordinals and consumer column/type only; producer bind consumes C1's `dynamic_filter_planning_view`, drops nonadmitted ordinals, and rejects `NO_ADMITTED_KEYS` (§2, §3.1-3.2). |
| 16 | BLOCKER — bind/freeze called `dynamic_filter_plan()` before the runtime slot existed | Applied: planning uses only `planning_view()`; runtime-plan access starts after C1's prepared commit (§2, §3.2-3.5). |
| 17 | BLOCKER — registry-conditional freeze left base C1 builders unfrozen | Applied: C3's wrapper invokes generic C1 preparation/commit unconditionally for every builder with zero or more grouped producer additions (§3.4, test 3, risk 16). |
| 18 | MAJOR — C3a both claimed zero runtime effect and installed live topology | Applied: C3a ends at a value-only descriptor/planning telemetry; C3b alone validates converted topology, allocates channels, and installs endpoints (§1-3, §6). |
| 19 | MAJOR — C2 could reject during begin after producer commit | Applied: C2 preparation returns a reason or move-only token owning exact endpoints/preallocated state; begin is allocation-free/noexcept and assumes the installed proof (§3.4, §3.6, risk 17). |
| 20 | MAJOR — reset and telemetry had competing schemas/owners | Applied: C1 owns central begin/end and C3 reuses canonical `publication_terminal`, `target_visible`, `membership_measured`, and `target_publication_terminal`. Shared `scan_consume_summary` has no singular publication ID (§2, §3.8, risk 18). |
| 21 | MAJOR — Track D introduced an alternate empty-terminal outcome | Applied: Track D reuses C1's `NO_MATERIALIZATION(reason)` publication outcome and adds no completion taxonomy. |
| 22 | BLOCKER — static validation lacked join type/MIXED inputs without creating a hidden C3a→C2a dependency | Applied: C3a stores only physical consumer identity; C3b reads C2's immutable `sip_consumer_shape` through the public capability accessor during staging and passes it directly to validation (§2-3, risk 19). |
| 23 | BLOCKER — validation did not prove the installed endpoint vector | Applied: validation consumes the grouped vector and returns the only installable move-only proof token (§2, §3.4, risk 20). |
| 24 | BLOCKER — claimed atomic commit still had throwing allocations/assignments | Applied: two-phase C1 prepared assignments, C2 preallocated tokens, C3 wrapper preparation, noexcept-only commit, static assertions, and fault injection (§2, §3.4, test 3, risk 21). |
| 25 | MAJOR — one-install grouping/bijection was unspecified | Applied: additions group once per physical producer/consumer and validate publication/target/channel/pointer bijection, including two-producers→one-consumer (§2, §3.4, test 3, risk 22). |
| 26 | MAJOR — generation and clock epoch were conflated and begin could allocate | Applied: C1 exact strong generation and separate clock, all-generation preflight, preallocated reset-only/noexcept C2 begin (§2, §3.4, test 4, risk 23). |
| 27 | MAJOR — abnormal teardown could bypass summaries/closure | Applied: canonical success/abort end emits normal/partial summaries, closes channels, runs C2 end, then cancels residual attempts; failure-before-finalize test is mandatory (§2, §3.6, test 4). |
