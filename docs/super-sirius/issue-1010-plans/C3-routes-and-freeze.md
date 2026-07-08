# PR C3 implementation plan — SIP discovery/resolution/topology-freeze registry, layered targets, SIP channels, experiment telemetry

Companion to [issue-1010-dynamic-filter-sip-design.md](../issue-1010-dynamic-filter-sip-design.md); baseline dev 506a1d9f.

PR IDs covered: **C3** (recommended split **C3a**: planner/registry/freeze/flag/telemetry, **C3b**: publisher fan-out + consumer wiring + experiment tooling), plus the **C4 gate definition** (deliverable of the C3 experiment) and the contingent **Track D sketch**. All `file:line` at `dev` 506a1d9f. Design = the companion doc (cited as `design:N`).

## 1. Goal + non-goals

C3 delivers the design's "first value experiment" row (design:968): a query-local route registry that (a) discovers SIP routes on the resolved logical plan before physical planning (design:42-47, 427-447), (b) resolves both physical endpoints during `create_plan` despite child-before-parent order, (c) freezes producer publish targets and consumer endpoint plans together after pipeline conversion proves branch/runtime identity (design:479-492), and (d) fans membership filters into dedicated one-producer/one-consumer SIP channels consumed opportunistically at the C2 hash-join probe checkpoint — all behind `enable_dynamic_filter_sip=false`, with the telemetry needed to answer the C3 coverage questions (design:642-660) and gate C4. With the flag off, planner/publisher/runtime behavior is byte-identical to Phase 1. Non-goals: ordered activation (Track D — sketch only, §"Track D"), any producer-policy change (C1b-C1e), STANDARD-route default-on (design:608-612), mixed-provenance/no-scan candidates (Track E, design:361-371), MIXED-mode consumers (design:604-606), pre-partition unary consumers (design:100-110), scheduler changes (Track A).

## 2. Deliverables — public types and APIs

**`src/include/op/dynamic_filter_ids.hpp`** (A1-owned per design:957; C3 lands it if A1 has not — coordinate, see §6). All four IDs are query-relative monotonic values (design "Publication, target, channel, and filter identity"):

```cpp
namespace sirius::op {
template <class Tag> struct df_strong_id {
  std::uint32_t value{0};
  friend bool operator==(df_strong_id, df_strong_id) = default;
};
using dynamic_filter_publication_plan_id = df_strong_id<struct publication_plan_tag>;
using dynamic_filter_target_id           = df_strong_id<struct target_tag>;
using dynamic_filter_channel_id          = df_strong_id<struct channel_tag>;
using dynamic_filter_filter_id           = df_strong_id<struct filter_tag>;
}  // + fmt formatters
```

**`src/include/op/dynamic_filter_sip_plan.hpp`** — runtime-facing immutable values (design:449-469, 794-814; SIP variant leg only):

```cpp
namespace sirius::op {
struct probe_schema_ordinal { cudf::size_type value; };  // consumer C's runtime probe schema space

struct join_probe_target_key {
  std::size_t duckdb_filter_ordinal;      // k in filter_pushdown.join_condition — indexes the
                                          // publisher's per-key arrays (dynamic_filter_publisher.cpp:120-125)
  std::size_t sirius_key_ordinal;         // compact post-narrowing ordinal (aligns with C1b's
                                          // admitted_dynamic_filter_key, design:330-335)
  probe_schema_ordinal consumer_column;
  cudf::data_type key_type;               // producer build key type, CAPTURED AT DISCOVERY from
                                          // C1a's extraction value (dynamic_filter_key_plan
                                          // .build_type, design:784-791) — never read back from
                                          // the physical join at freeze (its right_key_col_indices
                                          // / key_casts are protected, sirius_physical_hash_join
                                          // .hpp:212/:225, and equality-ordinal-compacted so
                                          // duckdb_filter_ordinal cannot index them; §3.4 step 5)
};

struct join_probe_publish_target {        // producer side, frozen; same shape as C1a's variant
  dynamic_filter_target_id  target_id;    // alternative in dynamic_filter_publication_plan
  dynamic_filter_channel_id channel_id;   // (design:800-814)
  std::shared_ptr<sirius_dynamic_filter_set> channel;   // dedicated 1-producer/1-consumer (design:564-573)
  std::vector<join_probe_target_key> keys;
};

struct sip_consumer_endpoint {            // consumer side, frozen
  dynamic_filter_publication_plan_id publication_plan_id;
  dynamic_filter_target_id  target_id;
  dynamic_filter_channel_id channel_id;
  std::shared_ptr<sirius_dynamic_filter_set> channel;
  std::vector<join_probe_target_key> keys;
};
struct sip_consumer_plan { std::vector<sip_consumer_endpoint> endpoints; };
}
```

**`src/include/utils/single_assignment.hpp`**:

```cpp
namespace sirius {
template <class T> class single_assignment {   // assign exactly once before runtime; lock-free reads
 public:
  void assign(T v);                             // throws internal_exception on second assign
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
  ORDINAL_OUT_OF_RANGE, TYPE_MISMATCH, NO_INTERMEDIATE_JOIN
};

struct pending_join_probe_target {          // design:461-469
  op::dynamic_filter_publication_plan_id publication_plan_id;
  op::dynamic_filter_target_id           target_id;
  logical_plan_node_id producer, consumer;
  std::vector<op::join_probe_target_key> keys;       // key_type populated here (discovery)
  std::vector<cudf::data_type> consumer_probe_types; // per key: consumer probe-column type,
                                                     // captured at discovery from the consumer's
                                                     // left-child logical types (§3.4 step 5)
  op::dynamic_filter_channel_id channel_id;
  std::shared_ptr<op::sirius_dynamic_filter_set> channel;
};
struct sip_candidate_rejection { logical_plan_node_id producer; std::size_t duckdb_filter_ordinal;
                                 sip_reject_reason reason; };
struct sip_discovery_result { std::vector<pending_join_probe_target> targets;
                              std::vector<sip_candidate_rejection> rejections; };

// One producing join with non-empty probe_info (candidates via the C1a adapter, design:296-338).
// Pure over the resolved logical tree; registry only mints ids/channels.
sip_discovery_result trace_sip_routes(duckdb::LogicalOperator& resolved_root,
                                      duckdb::LogicalComparisonJoin& producer,
                                      /* C1a extraction value: admitted keys (condition_index,
                                         build_column_index, build_type) + probe_info identities */,
                                      dynamic_filter_route_registry& registry);
}
```

**`src/include/planner/dynamic_filter_route_registry.hpp` / `src/planner/dynamic_filter_route_registry.cpp`** — the query-local registry (design:471-492):

```cpp
namespace sirius::planner {
class dynamic_filter_route_registry {
 public:
  // --- discovery (pre-create_plan) ---
  logical_plan_node_id node_id_for(duckdb::LogicalOperator const& node);      // assign-on-first-use
  op::dynamic_filter_publication_plan_id mint_publication_plan_id();
  op::dynamic_filter_target_id mint_target_id();
  std::pair<op::dynamic_filter_channel_id, std::shared_ptr<op::sirius_dynamic_filter_set>>
      mint_sip_channel();                                                     // dedicated, no remap
  void add_discovery(sip_discovery_result r);

  // --- endpoint resolution (during create_plan; both roles resolved here, design:471-475) ---
  void bind_physical_join(duckdb::LogicalOperator const& logical_node,
                          op::sirius_physical_hash_join& physical,
                          bool phase1_producer_wired);

  // --- freeze (after conversion; single-assignment of slots) ---
  // First call freezes; LATER CALLS ARE DEFINED NO-OPS that return the first call's stats and
  // re-emit topology_frozen with reused=true — required because the extension prepared path
  // caches the physical plan and re-executes it (§3.4 "Prepared-plan re-execution").
  struct finalize_stats { std::size_t frozen_targets, rejected_targets; };
  finalize_stats finalize(
      duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> const& scheduled);
  [[nodiscard]] bool finalized() const noexcept;

 private:
  // pending state: node-id map (LogicalOperator const* -> id), pending targets,
  // logical-node -> physical-join binds, rejections buffer; ALL cleared inside the first
  // finalize() so runtime never sees DuckDB pointers or mutable route state (design:46-47,
  // 487-489). Only finalize_stats + the spent flag survive for the no-op re-call.
};
}
```

**`sirius_physical_hash_join` additions** (`src/include/op/sirius_physical_hash_join.hpp`):

```cpp
// Frozen SIP slots — assigned at most once by dynamic_filter_route_registry::finalize, strictly
// before create_query/start_query; runtime reads snapshots only (design:483-485).
// install_sip_consumer_plan also constructs the C2 checkpoint's per-endpoint gate state here,
// single-threaded (§3.6) — probe tasks never construct gates.
void install_sip_publish_targets(std::shared_ptr<std::vector<op::join_probe_publish_target> const>);
void install_sip_consumer_plan(std::shared_ptr<op::sip_consumer_plan const>);
[[nodiscard]] std::shared_ptr<std::vector<op::join_probe_publish_target> const>
    sip_publish_targets() const noexcept;      // nullptr == none
[[nodiscard]] std::shared_ptr<op::sip_consumer_plan const> sip_consumer_plan() const noexcept;
[[nodiscard]] bool is_mixed_join() const noexcept;   // _join_mode==MIXED_JOIN; plan-time constant (cpp:342-347)
```

**Config**: `bool enable_dynamic_filter_sip = false;` in `operator_params` + YAML + `SET enable_dynamic_filter_sip` (recipe in §3.9).

**Free function** `finalize_sip_topology` is `registry.finalize(...)` invoked from the engine (§3.4).

## 3. Step-by-step changes

### 3.1 `src/planner/sirius_physical_plan_generator.{hpp,cpp}` — discovery hook + registry ownership

- Header: next to the scan-channel map (`sirius_physical_plan_generator.hpp:80-85`), add
  `std::shared_ptr<dynamic_filter_route_registry> sip_registry; // null unless SIP enabled and routes pending`.
- Cpp: add helper `dynamic_filter_sip_enabled(duckdb::ClientContext&)` cloning `dynamic_filter_pushdown_enabled` (`sirius_physical_plan_generator.cpp:38-45`), returning false with no `SiriusContext`.
- In `create_plan(duckdb::unique_ptr<duckdb::LogicalOperator>)`: after `resolver.VisitOperator(*op)` (`sirius_physical_plan_generator.cpp:126-128`) and before the recursion `create_plan(*op)` (`:132`), run discovery when `dynamic_filter_sip_enabled(context) && dynamic_filter_pushdown_enabled(context)`:
  walk the logical tree; for each node with **`node.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN` exactly** (`sip_producer_shape_eligible`; `LOGICAL_DELIM_JOIN`/`LOGICAL_ASOF_JOIN` are rejected with `PRODUCER_SHAPE` — Phase 1 wires delim producers today via `sirius_plan_delim_join.cpp:73-76` → `plan_comparison_join`, so without this check delim producers would mint SIP routes over DuckDB's delim machinery and perturb the freeze step-4 pipeline-shape assumptions via `sirius_pipeline_converter.cpp:1182-1197`) whose C1a extraction reports non-empty admitted `probe_info` candidates (design:262-268), call `trace_sip_routes(...)` and `sip_registry->add_discovery(...)` (create `sip_registry` lazily on first admitted producer). This is exactly the required window: post-resolver, pre-condition-drain (design:302-304, 429) — `plan_comparison_join` later drains `op.conditions` at `sirius_plan_comparison_join.cpp:408-409` and `filter_pushdown` at `:514`.
- Note `fold_adjacent_projections` (`:135`) runs on the physical plan and does not touch joins — physical pointers recorded by `bind_physical_join` stay valid.

**Lineage-pass internals** (`dynamic_filter_lineage.cpp`), per admitted key `k` and `probe_info` target `t` (design:430-443):
1. Starting binding = `producer.children[0]->GetColumnBindings()[cond_left_index]` — post-resolver left keys are `BoundReferenceExpression` positions into the left child's flattened output (design:155-160; the existing extraction precedent for right keys is `sirius_physical_hash_join.cpp:300-338`).
2. Locate the target `LogicalGet` by preserved `DynamicTableFilterSet` identity — walk `resolved_root` for `LogicalGet` with `get.dynamic_filters.get() == pi.dynamic_filters.get()` (same pairing key used today at `sirius_plan_get.cpp:264-266` and `sirius_plan_comparison_join.cpp:444`); zero or ≥2 matches → reject `NO_GET_MATCH`/`DUPLICATE_GET_MATCH` (design:433-435).
3. Descend from the producer's left child toward that GET, applying `classify_sip_crossing`/`remap_binding_through` (the projection remap mirrors the value-preserving pass-through logic used in `trace_binding_to_get`, `sirius_plan_comparison_join.cpp:288-300`, but over `BOUND_REF` post-resolver expressions and failing closed on anything computed). At each `CONSUMER_THEN_LEFT` join: if the binding originates in its left child (check via child `GetColumnBindings()` ownership), record `(consumer node, probe ordinal = position in left-child bindings, consumer probe-column logical type → cudf::data_type)` and continue left; right/build-origin → `RIGHT_ORIGIN_BINDING` stop (design:438-443, 512-513). The producer itself is never recorded (trace starts below it — design:402-404).
4. Require the final remapped binding to equal `pi.columns[duckdb_filter_ordinal].probe_column_index` (the field consumed today at `sirius_plan_comparison_join.cpp:451`); mismatch → `FINAL_BINDING_MISMATCH` (design:321-325, 437).
5. Group per `(producer node, consumer node)`, compact keys, dedup paths, mint `publication_plan_id`/`target_id`/`channel_id` + channel per group (design:441-445, 461-469). Each key carries `key_type` = the producer build-key `build_type` from C1a's extraction value (`dynamic_filter_key_plan`, design:784-791) plus the captured consumer probe-column type from step 3 — freeze validates types purely from these captured values (§3.4 step 5). A route with no intermediate join yields no target — Phase 1 stays the only consumer (design:424-425).

### 3.2 `src/planner/sirius_plan_comparison_join.cpp` — endpoint resolution

In `plan_comparison_join`, after the join is constructed and configured (`sirius_plan_comparison_join.cpp:504-519`), add:

```cpp
if (sip_registry) {
  sip_registry->bind_physical_join(op, hj, /*phase1_producer_wired=*/phase1_wired);
}
```

where `bool phase1_wired = !filter_targets.empty();` is snapshotted just before `filter_targets` is moved into `filter_plan` (`:497-502`). `bind_physical_join` resolves **both roles** for this logical node (design:471-475): records `&hj` as physical consumer for every pending target whose `consumer` node-id maps to `&op`, and as physical producer for pending targets whose `producer` maps to `&op` — remembering `phase1_producer_wired`. This deliberately reuses the Phase 1 producer gates as the single source of truth: if the join was not wired (unfiltered build `:426-431`, missing GPU/HOST spaces `:437-440`, no channel from the central gate `:444-445`), the pending SIP targets are later rejected `PRODUCER_NOT_WIRED` — preserving the invariant SIP producers ⊆ Phase 1 producers. That invariant is an inference from keeping the blanket `build_side_has_filter` gate until C1e removes it (design:964-966), not a design bullet. Children are planned at `:372-373` before this point, so consumer joins in the probe subtree are always bound before their producer — no post-hoc patching. `plan_delim_join` needs no change: it delegates to `plan_comparison_join` (`sirius_plan_delim_join.cpp:73-76`), so `bind_physical_join` **does** execute for delim-join logical nodes — but discovery excludes delim producers (`PRODUCER_SHAPE`, §3.1) and crossing STOPs at delim consumers, so no pending target can name them; `bind_physical_join` is lookup-miss-tolerant and the bind is a no-op there.

Timing note: because the SIP flag is read once here (plan time) and never at runtime, `SET enable_dynamic_filter_sip` is effective at the next query (recon R3 §4; same contract as `enable_dynamic_filter_pushdown`).

### 3.3 Registry lifetime — plan-generator → engine handoff

The generator is stack-local at all four instantiation sites and dies immediately after `create_plan` returns: `src/sirius_extension.cpp:534-536` (inside `SiriusGeneratePhysicalPlan` `:531-538`), `src/sirius_ffi.cpp:171`, `src/transparent/physical_sirius_execution.cpp:148-149`, and the validation-only path `src/sirius_context.cpp:843-863`. The registry must survive until conversion. Carrier: `sirius_prepared_statement_data` (keeps ops free of planner types — G9, design:935):

- `src/include/sirius_interface.hpp:26-38`: add public member `std::shared_ptr<planner::dynamic_filter_route_registry> sip_registry;` (forward-declared; default null).
- Assign it at the three construction sites (all grep-verified):
  - `src/sirius_extension.cpp:601-605` — change `SiriusGeneratePhysicalPlan` (`:531-538`) to also out-return `physical_planner.sip_registry`, assign onto `gpu_prepared`.
  - `src/sirius_ffi.cpp:170-173` — hoist the generator into a named local; assign `gpu_prepared->sip_registry`.
  - `src/transparent/physical_sirius_execution.cpp:148-152` — same.
- `src/include/sirius_engine.hpp` (class at `:54`): add `std::shared_ptr<planner::dynamic_filter_route_registry> sip_registry;` in the **public** member section (the engine's data members are public and its only friends are pipeline classes, `sirius_engine.hpp:55-58` — a private trailing-underscore member would not be assignable from `sirius_interface`). Set it in `sirius_interface::sirius_pending_statement_internal` (`src/sirius_interface.cpp:150-177`) — the engine is created at `:160-163` and initialized at `:177` (the only `engine.initialize` call site, grep-verified; `sirius_execute_query` at `:214-240` only delegates via `sirius_pending_statement_or_prepared_statement`): `engine.sip_registry = statement_p->sip_registry;` before `engine.initialize(std::move(sirius_collector))`.
- **Do NOT clear the registry in `sirius_engine::reset()`.** `sirius_engine::initialize` calls `reset()` at `src/sirius_engine.cpp:141` *before* `initialize_internal` at `:149` — clearing there would null the registry before the freeze ever runs, making SIP a silent no-op on every path with the flag on (no `register_producer`, no slot installs, no telemetry) while all correctness tests stay green. Instead the engine releases it inside `initialize_internal` immediately after `finalize` (§3.4), where it is spent.
- The validation-only path (`sirius_context.cpp:843-863`) discards plan+generator+registry together — pending state and channels are all plan-local shared_ptrs, so nothing leaks or dangles. The transparent replan path builds a fresh generator/registry per execute (`physical_sirius_execution.cpp:110-149`) — IDs are query-relative and never cached across replans (design:575-577).

### 3.4 Topology freeze — `src/sirius_engine.cpp` + `dynamic_filter_route_registry::finalize`

In `initialize_internal`, after `auto result = converter.convert(*root_pipeline);` (`src/sirius_engine.cpp:353-355`) and before `materialize_repository_wiring` (`:358-359`):

```cpp
if (sip_registry) {
  sip_registry->finalize(result.scheduled_pipelines);  // emits [dynf_summary] topology_frozen (§3.8)
  sip_registry.reset();  // spent; MUST NOT be cleared earlier (reset() runs pre-initialize_internal,
                         // sirius_engine.cpp:141 — see §3.3)
}
```

`finalize` logs its own `topology_frozen` line from `dynamic_filter_route_registry.cpp` (component prefix `[dynf_summary]`, §3.8), so the engine call site adds no logging. Per pending target:
1. **Physical existence**: both producer and consumer were bound and are `sirius_physical_hash_join` (a logical comparison join that planned to NLJ at `sirius_plan_comparison_join.cpp:597-608` was never bound → `NOT_HASH_JOIN_PHYSICAL`, design:513-514).
2. **Producer viability**: `phase1_producer_wired` recorded at bind, and `producer.publishes_dynamic_filters()` (`sirius_physical_hash_join.hpp:167-170`) → else `PRODUCER_NOT_WIRED`.
3. **Consumer mode**: `!consumer.is_mixed_join()` (plan-time constant, `sirius_physical_hash_join.cpp:342-347`) → else `CONSUMER_MIXED_MODE` (design:604-606). BUILD_PROBE vs STANDARD is runtime-decided (`sirius_physical_hash_join.cpp:434-461`) and both are eligible — no check.
4. **Branch uniqueness** (design:483, 536-539): exactly one pipeline in `scheduled` whose `source == &consumer` and exactly one whose `source == &producer`. The converter guarantees join pipelines have `dependencies[0]=build concat, dependencies[1]=probe concat` (`sirius_pipeline_converter.cpp:1164-1204`, probe at `:1172-1173`); require `dependencies.size() >= 2` on the consumer pipeline (the branch-specific default probe context). CTE-duplicated or missing → `BRANCH_AMBIGUOUS`. (Delim producers, whose build wrapping at `:1182-1197` would perturb this shape, never reach freeze — excluded at discovery, §3.1.)
5. **Key validation** — purely over values captured at discovery/bind; freeze reads **no** join key metadata (`right_key_col_indices`/`key_casts` are `protected`, `sirius_physical_hash_join.hpp:212/:225`, and equality-ordinal-compacted — the ctor loop skips inequality conditions, `sirius_physical_hash_join.cpp:290-341` — so `duckdb_filter_ordinal` cannot index them even with an accessor): every `consumer_column.value < consumer probe schema width` (consumer's `children[0]` output arity captured at bind time) → else `ORDINAL_OUT_OF_RANGE`; `key_type` (producer `build_type` from C1a's extraction value, captured at discovery — §3.1 step 5) equals the captured consumer probe-column type (both sides no-cast by the narrowing gates, design:349-351) → else `TYPE_MISMATCH`.
6. **Survivor**: `channel->register_producer()` (`sirius_dynamic_filter.hpp:475`); append `join_probe_publish_target` to the producer's batch and `sip_consumer_endpoint` to the consumer's batch. **Rejected**: `channel->close_for_new_filters()` (`:484`), emit `channel_closed reason=PLANNING_REJECTED`, drop both ends; scan targets are untouched (design:485-486, 516-518, 1007-1009). A consumer resolved before its producer failed retains no dangling endpoint because endpoints are only installed here, never at bind.
7. Assign each participating join's slots exactly once via `install_sip_publish_targets` / `install_sip_consumer_plan` (skip empty; the consumer install also constructs gate state, §3.6), emit `target_planned` per frozen target and buffered `candidate_rejected` lines, then **destroy all pending state** — node-id map, logical pointers, bind records (design:487-489); only `finalize_stats` and the spent flag survive. Freeze precedes `execute()` → `create_query` → `start_query` (`src/sirius_engine.cpp:151-167`), so runtime only ever observes assigned-or-null slots; no converter change is needed (the converter never reads SIP state; its scan elision at `sirius_pipeline_converter.cpp:277-278` concerns scan channels only, whose `register_producer` happened at plan time `sirius_plan_comparison_join.cpp:446`).

**Prepared-plan re-execution.** The extension path caches `gpu_prepared` on the bind result (`sirius_extension.cpp:601-605`) and `sirius_interface` supports prepared statements (`sirius_pending_statement_or_prepared_statement`, `sirius_interface.cpp:132-142, 221-222`), so the same physical plan — and therefore the same registry held by `statement_p->sip_registry` — can be initialized more than once. **The second and later `finalize` calls are defined per-plan-instance no-ops**: the registry keeps a spent flag + the first call's `finalize_stats`, returns them, re-emits `topology_frozen ... reused=true`, and installs nothing (so `single_assignment` never double-throws). Channel-content reuse across executions of the same prepared plan matches Phase 1's existing exposure exactly — scan channels are likewise minted at plan time and live in the physical plan (`sirius_plan_comparison_join.cpp:444-446`) — so C3 adds no new hazard class; an e2e test executes the same prepared statement twice with the flag on (§4 test 4). The transparent path re-plans per execute (`physical_sirius_execution.cpp:110-152`) and never hits this.

### 3.5 Producer fan-out — C1a publication-plan seam, `src/op/sirius_physical_hash_join.cpp`

C3's hard dependency C1a replaces the current publisher internals with an immutable `dynamic_filter_publication_plan` whose `targets` are `std::vector<dynamic_filter_publish_target>` = `variant<scan_publish_target, join_probe_publish_target>`, and whose claim path "does not re-read `filter_pushdown`, `join_condition`, `key_casts`" (design:794-830, 962). C3 integrates **against that seam**, not against the dev-line publisher (`dynamic_filter_publisher.hpp:43-53` ctor taking `JoinFilterPushdownInfo`/`key_casts`/`right_key_col_indices`, per-key arrays sized from `_filter_pushdown.join_condition` at `dynamic_filter_publisher.cpp:120-125`, scan loop `:305-329`, drained early-out `:83-91` — those cites document the code C1a rewrites and will not survive it):

- `install_sip_publish_targets` is the freeze→producer hand-off: before `create_query`, the join folds the frozen `join_probe_publish_target` entries into its `dynamic_filter_publication_plan.targets` as the variant's join-probe alternative (design:800-814); the plan is immutable from then on. The publisher's target loop, per-target accepting check, arity handling, and all-targets-drained early-out then cover SIP targets **for free** — a drained-scans-but-live-SIP plan still publishes because the early-out iterates the unified vector.
- The claim gate is unchanged: C3 keeps SIP producers ⊆ Phase 1 wired producers (§3.2), so the existing `_dynamic_filter_plan.enabled()` claim condition (dev-line `sirius_physical_hash_join.cpp:1364`; C1a's equivalent) needs no edit.
- The variant visitor pushes **membership filters only** into `join_probe_publish_target` channels — one `per_key_membership`-equivalent filter per `join_probe_target_key`, at `key.consumer_column.value`; **never zone maps** (design:626-627; zone maps stay scan-reader capabilities and remain `scan_publish_target`-only). The same `shared_ptr<sirius_dynamic_filter const>` object fans out — channels co-own filters, no replica duplication (design:565-569, 633-634; channel contract `sirius_dynamic_filter.hpp:437-440`). Key alignment is by construction: SIP keys were built at freeze from the same C1a-admitted key set (`duckdb_filter_ordinal` ↔ key-plan index), enforce with `assert`.
- Emit `channel_filter_visible` per push and `target_publication_terminal` per SIP target after fan-out (§3.8); extend the existing per-publication summary line with `sip_targets={}` count.
- **Coordination requirement (merge-blocking for C3b):** if C1a lands with a different variant/plan shape, C3b rebases the fold-in on it; if C1a is delayed, C3b either amends C1a's structs in the same stack or is blocked — C3b must not re-implement fan-out against the pre-C1a publisher internals.

### 3.6 Consumer wiring — `src/include/op/sirius_physical_hash_join.hpp`, `src/op/sirius_physical_hash_join.cpp`

- Add the two `single_assignment` slot members + accessors + `is_mixed_join()` (§2). Include only `dynamic_filter_sip_plan.hpp` (op-layer header; no planner include — the registry is forward-declared nowhere in op headers).
- Connect the frozen `sip_consumer_plan` to the **C2** checkpoint component (`hash_join_probe_filter_consumer`, design:582-612). **Gate state is constructed inside `install_sip_consumer_plan` at freeze** — one `dynamic_filter_gate` per endpoint (independent combined + per-filter stats, design:622-624; gate type at `src/include/op/scan/dynamic_filter_gate.hpp:46-121`, which C2 relocates under `src/include/op/`). Freeze is single-threaded and strictly precedes `create_query`, so no synchronization is needed for construction; probe tasks only *mutate per-batch stats* under the gate's own accounting. Lazy construction on first probe task is explicitly ruled out: probe tasks for one join run concurrently (STANDARD per-partition tasks; BUILD_PROBE probe tasks after `BUILT`, `sirius_physical_hash_join.cpp:510-517`), so first-task construction is a data race.
- Fast paths per design:615-620: null plan / `!channel->has_filters()` (`sirius_dynamic_filter.hpp:498-501`) / replica unavailable / disabled gate → zero-copy forward; replica-unavailable pass-through does not train the gate.
- `on_finalize_operator` (`sirius_physical_hash_join.cpp:1407-1428`): after the existing publication-window close (`:1412-1418`), close each consumer endpoint channel via `close_for_new_filters()` + emit `channel_closed reason=CONSUMER_FINALIZED` — mirroring the scan consumer's close (`src/op/scan/sirius_physical_dynamic_filter.cpp:43-46`). One logical consumer per SIP channel makes this the unique closer (design:571-573).

### 3.7 STANDARD-route probe-batch identity

Give the C2 `probe_batch_handle` its stable probe-batch ID from the batch's repository identity so repeated STANDARD applications (one probe × several build batches) are visible in `consume_batch` telemetry (design:608-612, 1029). C3 reapplies safely and only measures; no cache.

### 3.8 Telemetry — log lines + analyzer (recon R1 recommendation: log lines primary; quent later)

Events (subset of design:1058-1070 that is C3's to emit; A1 owns `publication_plan_created/started/completed`, the waiter-free outcomes design:762-780, the query-relative clock, and resident high-water). Prefix rules: machine-parsed per-query summary lines use the `[dynf_summary]` prefix at INFO; other new lines keep the existing bracketed component prefix of the file they live in (`dynamic_filter_publisher.cpp` emits under `[sirius_physical_hash_join]` today — e.g. `:83-91` — so its new lines do too); new files get their own prefix.

| line (k=v style per `MEM_HISTORY_RE` precedent) | level | emitting file |
|---|---|---|
| `[dynf_summary] topology_frozen targets= rejected= reused=` | INFO | `dynamic_filter_route_registry.cpp` (inside `finalize`) |
| `[dynf_summary] target_planned publication_plan_id= target_id= channel_id= kind=join_probe producer_op= consumer_op= keys=` | INFO | same |
| `[dynf_summary] sip_consume_summary channel_id= batches= rows_in= rows_out= masks_applied= masks_skipped= replica_unavailable= apply_us= batches_before_first_filter= rows_before_first_filter=` (per endpoint, at consumer finalize — makes coverage (b),(d),(e) INFO-derivable) | INFO | hash join .cpp |
| `[sip_registry] candidate_rejected publication_plan_id= ordinal= reason=` (discovery rejections buffered, logged at freeze so one file emits) | DEBUG | `dynamic_filter_route_registry.cpp` |
| `[sip_registry] channel_closed channel_id= reason=PLANNING_REJECTED` | DEBUG | same |
| `[sirius_physical_hash_join] channel_filter_visible channel_id= target_id= filter_id= generation=` | DEBUG | `dynamic_filter_publisher.cpp` |
| `[sirius_physical_hash_join] target_publication_terminal target_id= channel_id= outcome=` | DEBUG | same |
| `[sip_consume] consume_batch channel_id= probe_batch_id= rows_in= rows_out= filters_visible= masks_applied= masks_skipped= replica_unavailable= apply_us=` | TRACE | C2 consumer component .cpp |
| `[sirius_physical_hash_join] channel_closed channel_id= reason=CONSUMER_FINALIZED` | DEBUG | hash join .cpp |

Rules: IDs in-line (never addresses, design:1055-1056); `channel_closed` has two emitters with different prefixes, so its strict regex must pin neither filename nor prefix (log-analyzer contract; `tools/log_analyzer/patterns.py:102` precedent pins filenames — deliberately omit here). Per-batch events TRACE, one-shot events INFO/DEBUG (matches existing split, `patterns.py:184-195` / the publisher summary line).

Analyzer: new `tools/log_analyzer/metrics/dynamic_filter_sip.py` exposing `COLUMNS` + `parse(lines, warnings)` (pattern: `metrics/memory_reservation.py:22-50`), wired in `parse_logs.py` `process_query` (`:142`); anchors + strict regexes in `patterns.py` and `SHAPE_VERSION` bump `"1.6"` → `"1.7"` (`patterns.py:11-17`). Segmentation is automatic via existing QueryBegin/QueryEnd anchors (`patterns.py:46-51`). The module warns "flag-on run but zero `consume_batch`/`sip_consume_summary` lines seen" so a mis-leveled coverage pass cannot silently produce empty CSVs.

**Coverage report** (computed by the analyzer module + a small aggregation notebook/script in `tools/log_analyzer/`): per query × config, (a) routes planned/rejected by reason (INFO); (b) per SIP channel: rows/batches consumed before first filter visible vs after (pre-publication miss rate — INFO via `sip_consume_summary.{batches,rows}_before_first_filter`); (c) layer attribution scan-caught vs C1-caught vs C2-caught via distinct channel_ids sharing filter_ids (design:1074-1077, 1083-1084 — needs the DEBUG `channel_filter_visible` legs); (d) hash-probe rows avoided = Σ(rows_in − rows_out) and estimated bytes (INFO); (e) mask overhead Σapply_us and gate disable/skip rates (INFO); (f) joined with A1's wall-time + per-space resident high-water lines (design:644-649, 1079-1086).

**Runbook** (ships as a section in the dynamic-filters.md update, §3.10) — **log level is stated per pass and is part of the protocol**:
- **Timing passes: `SIRIUS_LOG_LEVEL=info`.** N≥5 paired runs; all wall-time numbers come only from these passes. Coverage items (a),(b),(d),(e),(f) are derived from the INFO `[dynf_summary]` aggregates of the same timed runs — no TRACE required.
- **Coverage-detail passes: `SIRIUS_LOG_LEVEL=trace`, separate and non-timed.** Needed only for item (c) filter-level attribution and per-batch `consume_batch` drill-down; their timings are excluded from every statistic. (CI already exports `SIRIUS_LOG_LEVEL: trace`, `test.yml:116` — fine for tests, never for timing.)
- Serialize configs in one process (`SET` is process-global per DB instance — `src/sirius_context.cpp:919`; LOAD before SET — recon R3 gotcha 1); 2×2 matrix + reference: {pushdown off} ∪ {pushdown on} × {sip off,on} × {`dynamic_filter_build_priority` legacy,off (A2 flag)} (priority×SIP matrix, design:1036); all 22 TPC-H queries at SF10 on the dev box (SF1 sanity in CI) plus the synthetic many-join chain from the A-track gate (design:184-185); expected route-bearing shapes to inspect first: Q2, Q5, Q7, Q8, Q9, Q10, Q17, Q18, Q20, Q21 (Q21's orders self-join is the documented Phase-2 case, `docs/super-sirius/dynamic-filters.md:348`) — but the report's first table is "which queries actually admitted routes", since DuckDB owns candidate admission (design:35-41). Bag-equivalence checks per design:1013-1015.

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

1. **`test/cpp/planner/test_dynamic_filter_lineage.cpp`** — tag `[dynamic_filter][sip][lineage]`. Pure/no-GPU-execution. Two styles: no-DB logical construction (pattern of `test/cpp/transparent/test_preserve_dynamic_filter_metadata.cpp:116-183`) for `classify_sip_crossing`/`remap_binding_through`/`sip_consumer_join_type_eligible`/`sip_producer_shape_eligible` truth tables (every row of the crossing table design:498-510, incl. DELIM_JOIN and UNION stops; **producer-shape table pins `LOGICAL_DELIM_JOIN` and `LOGICAL_ASOF_JOIN` → `PRODUCER_SHAPE` rejection** — reachable today because delim routes through `plan_comparison_join`, `sirius_plan_delim_join.cpp:73-76`); SQL-through-planner (`generate_sirius_plan`, `test/cpp/planner/test_distinct_hash_join_detection.cpp:41-95`, fixture with `SIRIUS_CONFIG_FILE`+`SIRIUS_DISABLE` `:112-158`) for: consumer recorded on left-derived binding / stop on right-origin; zero+duplicate GET match fail-closed; final-binding-mismatch drop; grouping/dedup into `(producer, consumer)`; per-key INDF/cast rejection keeps the equality sibling routable (design:988-994); no target when no intermediate join exists; a correlated-subquery plan whose delim join admits Phase 1 filters mints **zero** SIP targets.
2. **`test/cpp/planner/test_sip_topology_freeze.cpp`** — tag `[dynamic_filter][sip][topology]`. Registry-level: hand-built `sirius_pipeline` objects with stub hash joins as sources (stub-child pattern of `test/cpp/operator/test_no_history_peak_memory_estimate.cpp:40-80`; hand-built `sirius_pipeline(pipeline_build_context{})` proven feasible by `test/cpp/pipeline/test_get_next_ports_after_sink.cpp:61`, `dependencies` public at `sirius_pipeline.hpp:157`); cases: freeze assigns slots exactly once; **second `finalize` is a defined no-op returning first-call stats (and emits `reused=true`), not an error**; `PRODUCER_NOT_WIRED` and `CONSUMER_MIXED_MODE` rejection closes channel (`accepting_filters()==false`) and installs nothing on either side while a sibling scan target survives; `BRANCH_AMBIGUOUS` (two pipelines, same source op); `ORDINAL_OUT_OF_RANGE`/`TYPE_MISMATCH` (validated against captured discovery values — no join internals); consumer-resolved-before-producer-failure leaves no endpoint (design:997-1009). Plus one plan-through case asserting `scan → C1 → P` yields scan channel + one C1 endpoint, and `scan → C1 → C2 → P` yields endpoints in both (design:999-1003).
3. **`test/cpp/operator/test_dynamic_filter_sip_publish.cpp`** — tag `[dynamic_filter][sip]`, needs a CUDA device (constructs filters like `test/cpp/operator/test_sirius_dynamic_filter.cpp:42-58`). Publisher fan-out through the unified publication-plan variant vector (§3.5): membership-only into SIP channels (zone map never pushed); identical `shared_ptr` filter object (same `filter_id`) lands in scan and SIP channels; closed SIP target skipped without affecting the scan target; drained-scan/live-SIP still publishes (unified early-out); channel close racing push loses no correctness (push after close is a no-op, `sirius_dynamic_filter.hpp:440`).
4. **`test/cpp/integration/test_dynamic_filter_sip_e2e.cpp`** — tag `[integration][dynamic_filter][sip]`, GPU. `GpuExecutionFixture::compare_gpu_vs_cpu` (`test/cpp/utils/gpu_execution_fixture.hpp:144-186`) with `SET enable_dynamic_filter_sip=true` via the registered `SiriusContext` (router-test style, `test/cpp/planner/test_dynamic_filter_router.cpp:46-57`): nested INNER/SEMI chains (2- and 3-deep), composite admitted keys, NULL probe keys, empty build, zero-row filtered probe completes normally, STANDARD consumer shape, non-prefix-payload query that would expose index/view misalignment if any post-checkpoint access bypassed the handle (design:598-603, 1027-1029). **Liveness assertion (the silent-no-op trap, §3.3):** one flag-on route-bearing case registers a temporary log-capture sink and `REQUIRE`s the `[dynf_summary] topology_frozen` line with `targets >= 1` (correct results alone would pass even if the registry were dropped before freeze). **Prepared re-execution:** execute the same prepared statement twice with the flag on — both runs correct, second run's `topology_frozen` has `reused=true`, no throw (§3.4). LIMIT/TOP-N explicit-oracle variants gated on B1 (design:1018-1020).
5. **Multi-GPU**: extend `test/cpp/operator/test_sirius_dynamic_filter_mgpu.cpp` (skip helper `require_two_gpus`, `mgpu_test_utils.hpp:144`; env params `mgpu_env_params`, `:62`) — SIP endpoint on non-producer device applies the local replica; a device without a replica passes through and increments `replica_unavailable` without training the gate (design:629-637).
6. **Flag-off invariance**: one case in file 2 asserting `sip_registry == nullptr` and zero SIP lines with the flag off; plus the existing TPC-H CI snapshot (`test.yml:171-176`) serving as the no-change regression under default config.

## 5. Gate & rollback

**Merge gate for C3 (the PR itself):** default `enable_dynamic_filter_sip=false`; with flag off, zero planner/publisher behavior change (existing `[dynamic_filter]` suites + TPC-H snapshot green, no new log anchors emitted); with flag on, e2e suite green **including the `topology_frozen` liveness assertion** (test 4 — result-correctness alone cannot catch a registry lifecycle regression, §3.3), TSan-clean publish/snapshot/close (design:1032), analyzer parses all new lines with zero format warnings; B1 **not** required to merge (flag stays off) but **required before running the C3 experiment or enabling the flag anywhere shared** (design:947-949, 961).

**C4 gate definition (the deliverable — measured after the C3 experiment, decides default-on; design:969, 642-660, 199-201):**
1. **Correctness:** ON/OFF bag-equivalence across the full e2e matrix incl. multi-GPU and STANDARD consumers; exact order only where SQL guarantees it (design:1013-1015). Zero diffs. Hard veto.
2. **Prerequisites:** B1 merged and Phase-0 sentinels green on the pin (design:961); A2 flag available so the priority×SIP matrix is measurable (design:1036).
3. **Value:** on route-admitting queries, wall-time improvement outside measured run variance attributable (via consume-summary/rows-avoided) to SIP checkpoints; full-suite geomean not regressed beyond variance. All timing from INFO-level passes (§3.8 runbook).
4. **Coverage explains value:** for every route class kept opportunistic, pre-publication miss rate low enough that measured value persists; a class with systematic misses that erases its value makes C4 **depend on Track D** for that class rather than defaulting on (design:53, 657-660, 1092-1093).
5. **Memory:** per-space resident high-water and filter-replica bytes unchanged beyond variance (fan-out adds channels, not replicas — design:633-634, 875).
6. **Overhead bound:** on non-benefiting queries, first-mask cost is bounded by gate disable behavior; redundant-downstream keep-rate shows layered masks disabling as designed (design:417-421, 1100-1101).
7. **STANDARD routes:** default-on only if repeated-application cost measured acceptable; otherwise STANDARD consumers stay flag-gated (design:608-612).
8. **Priority independence:** SIP value must not require `build_priority=legacy` (A/C matrix independence, design:973-975).

**Rollback:** runtime — `SET enable_dynamic_filter_sip=false` (next query) or YAML; code — C3 is a single revertible PR (plus C3b if split, §7): no persistent state, no schema, all channels/registry/IDs query-local; analyzer `SHAPE_VERSION` reverts with it. Retain the flag for one release after any C4 default-on (mirrors A3, design:177-178).

## 6. Dependencies & ordering

- **Hard, before C3 merges:** C1a (version-pinned adapter + extraction values + **the unified `dynamic_filter_publication_plan` variant target vector that §3.5's fan-out folds into** — discovery consumes its extraction output including per-key `build_type`; design:296-338, 784-814, 962) and C2 (probe checkpoint component + `probe_batch_handle` + relocated mask/gate helpers + join memory-model override; design:967 "no planned routes yet" — C3 supplies the routes). C1b's strong key types shared with `join_probe_target_key` — coordinate the `sirius_key_ordinal` field.
- **Soft/coordination:** A1 (ID header, waiter-free publication outcomes, query-relative clock, high-water lines) — C3 reuses; if A1 is unlanded, C3 lands `dynamic_filter_ids.hpp` and the clock helper itself and A1 adopts them. A2's `dynamic_filter_build_priority` needed only to *run* the experiment matrix, not to merge.
- **B1:** required before flag-on experiments and before C4 (design:947-949, 961); not a merge blocker for flag-off C3.
- **C1c/C1d/C1e:** orthogonal producer-policy A/Bs; the C3 experiment runs against whatever producer policy is current — record config in every run (candidate-parity invariant, design:254-259).
- **Internal order if split (§7):** C3a (planner: lineage+registry+freeze+flag+`target_planned`/`topology_frozen` telemetry; publisher/consumer untouched → planned routes visible, zero runtime effect) → C3b (publisher fan-out via C1a's publication plan + consumer wiring + `consume_batch`/`sip_consume_summary` telemetry + e2e/mgpu tests + runbook + doc update).
- **C4** is a measurement/config PR (default flip + doc), not new machinery. **Track D** only if C4 criterion 4 fails for a valuable class.

## 7. Size estimate

Prod: lineage ~360 (incl. producer-shape predicate); registry+freeze ~420 (incl. spent/no-op re-finalize); sip-plan/ids/single_assignment ~180; generator/join-planner/engine/interface/ffi/extension wiring ~140; hash-join slots + gate construction at install + finalize-close + consumer hookup ~160; publisher fold-in ~70 (thinner than a parallel loop — rides C1a's variant vector); config/flag ~60; telemetry lines ~90; analyzer (py) ~180; docs ~180. **≈ 1.65k prod + 180 py + 180 doc.** Tests: lineage ~480; topology ~420; publish ~250; e2e ~450; mgpu ~120. **≈ 1.7k test.** Recommendation: **split into C3a/C3b as in §6** — C3a is mergeable with provably zero runtime effect (strongest reviewability), C3b carries all runtime risk; both stay behind the one flag.

## 8. Risks (implementation-level) & mitigations

1. **Registry lifecycle vs `sirius_engine::initialize`'s internal `reset()`** (`sirius_engine.cpp:141` runs *before* `initialize_internal` `:149`). If the registry were cleared in `reset()` — the natural-looking place — SIP becomes a **silent no-op on every path** with correct results, and a correctness-only gate never notices. Mitigation: `reset()` never touches `sip_registry`; release happens in `initialize_internal` immediately after `finalize` (§3.4); the e2e liveness test asserts the `topology_frozen` line (test 4); the merge gate names that assertion explicitly (§5).
2. **Dangling operator pointers in the registry** (validation-only plans `sirius_context.cpp:843-863`, transparent replans, conversion failure). Mitigation: registry never dereferences physical pointers outside `finalize`; the first `finalize` clears all pending maps; the engine releases the registry right after freeze; plan-local ownership means discarded plans discard everything.
3. **Logical-node address instability.** `plan_comparison_join` moves `op.conditions`/`filter_pushdown` (`sirius_plan_comparison_join.cpp:408-409,514`) but node objects stay alive through `create_plan`; the node-id map is used only during discovery+bind and destroyed at freeze. Guard: discovery captures everything it needs (bindings, types, ordinals) as values before recursion.
4. **Ordinal-space confusion** (`duckdb_filter_ordinal` vs `condition_index` vs `sirius_key_ordinal` vs `probe_schema_ordinal`; `join_stats` is original-order — design:336-338 and MEMORY note; `right_key_col_indices` is additionally equality-ordinal-compacted, `sirius_physical_hash_join.cpp:290-341`, which is why freeze never indexes join internals at all — §3.4 step 5). Mitigation: strong types, all key metadata captured as values at discovery from C1a's extraction, and the exact alignment test (design:986-987).
5. **Freeze/converter ordering drift** (someone moves the finalize call after `create_query`). Mitigation: `install_*` slots throw on assignment after the join has executed a task (cheap `finalized`/first-task flag assert), and `sip_consumer_plan()`/`sip_publish_targets()` are snapshot reads.
6. **Publisher pushing zone maps into SIP channels** (would violate design:626-627 and the consumer's mask-only contract). Mitigation: the variant visitor pushes membership filters only for the join-probe alternative; test 3 asserts.
7. **`register_producer` double-count / scan-elision interference.** SIP channels are registered only at freeze and never consulted by the converter's scan elision (`sirius_pipeline_converter.cpp:277-278`); scan channels untouched. Test 2 covers.
8. **Delim/ASOF producers minting SIP routes.** Phase 1 *does* wire delim producers today (`sirius_plan_delim_join.cpp:73-76` delegates to `plan_comparison_join`; the `build_side_has_filter` comment at `sirius_plan_comparison_join.cpp:425` covers the delim case), so this is reachable, not "impossible by construction": without the §3.1 producer-shape check, routes would trace through DuckDB's delim machinery and freeze step 4 would meet the `RIGHT_DELIM_JOIN` build-wrapping pipeline shape (`sirius_pipeline_converter.cpp:1182-1197`). Mitigation: `sip_producer_shape_eligible` rejects `PRODUCER_SHAPE` at discovery; crossing STOP covers the consumer side; `bind_physical_join` stays lookup-miss-tolerant; lineage test pins both (test 1).
9. **C1a seam drift.** §3.5 integrates with C1a's publication-plan variant vector; if C1a's landed shape differs, C3b rebases (coordination requirement in §3.5/§6) — C3b never re-targets the pre-C1a publisher internals, whose cited lines will not survive C1a.
10. **Log-format drift breaking the experiment tooling.** Anchors + strict regexes land in the same PR as the emitters, `SHAPE_VERSION` bumped, format-warning counters (validators) surface drift in `_summary.json` (patterns contract, `tools/log_analyzer/patterns.py:1-17`); the analyzer's "flag-on but zero consume lines" warning catches mis-leveled runs (§3.8).
11. **Flag-on plan-time cost** (extra full-tree walks per admitted producer). Bounded: discovery only runs with the flag on and only for joins with non-empty `probe_info`; walks are O(plan size × admitted producers); acceptable for an experiment flag — measure in C3 runs, memoize the GET-identity index per query if it shows up.

---

## Track D sketch (contingent — NOT planned in detail; design:662-757)

**Admission** (post-freeze pass in the same registry, new `select_ordered_targets(...)`): among frozen SIP targets, admit at most one ORDERED target per runtime consumer pipeline; require the channel's single registered producer; validate acyclicity over the union of repository/data edges (`repository_wirings_`/`setup_pipeline_parents`, `sirius_pipeline_converter.cpp:1133-1147`), activation edges, and the synthetic hint path used to drive the producer; require every task-driving node in the consumer→producer activation closure runtime-unique, else demote to opportunistic; tie-break deterministically by estimated avoided probe work. The activation descriptor stores the producer's **build-side publication driver pipeline** (build PARTITION/CONCAT path — `dependencies[0]` chain, `sirius_pipeline_converter.cpp:1170-1171`), never producing join `P` itself, because `P`'s hint recursion can re-enter through `C` and manufacture a cycle (design:666-679).

**Token**: one query-owned `sip_activation_token` per `(channel, consumer runtime pipeline)` with CAS FSM `IDLE→ARMED→QUEUED→CLAIMED | →DETACHED`; the check runs in the task creator **before hint resolution** (hints are stateful: `NOT_BUILT→SCHEDULING` on read, `sirius_physical_hash_join.cpp:486-547`), i.e., ahead of `get_next_task_hint`/input pop/reservation in `src/creator/task_creator.cpp`. Channel completion (`complete_once`: `OPEN→PUBLISHING→PUBLISHED|SEALED_EMPTY(reason)|FAILED|CANCELLED`, sealed at STANDARD/MIXED decision and at build-CONCAT completion — never join finalize) calls `release_once`; only `ARMED→QUEUED` enqueues; task creator CAS `QUEUED→CLAIMED` before dereferencing; wakeups queued only after channel/join/partition/pipeline locks are released (design:681-729).

**Teardown** (error path, order fixed by design:740-750): stop task creation & reject activations → detach `IDLE/ARMED/QUEUED` tokens and invalidate queued requests → drain creator queue/claimed requests/executors/publishers → cancel open channels → remove edges/tokens → destroy pipelines/operators → restart creation only when both queues are empty. Files touched: `src/creator/task_creator.*`, `src/pipeline/sirius_pipeline.*`, query-owned activation registry, join mode/build-completion hooks; fault-injection suite per design:1042-1052 is a ship blocker for D.

---

## Review resolution

| # | Finding | Resolution |
|---|---|---|
| 1 | BLOCKER — clearing the registry in `reset()` makes SIP a silent no-op (`reset()` at `sirius_engine.cpp:141` runs before `initialize_internal` `:149`) | Applied: `reset()` never touches `sip_registry`; the engine releases it in `initialize_internal` immediately after `finalize` (§3.3, §3.4); e2e liveness test asserts the `topology_frozen` line and the merge gate names it (§4 test 4, §5, risk 1). |
| 2 | MAJOR — "DELIM plans reaching bind is impossible" false; delim producers are Phase-1-wired and reach `bind_physical_join` via `sirius_plan_delim_join.cpp:73-76` | Applied: new `sip_producer_shape_eligible` requires `node.type == LOGICAL_COMPARISON_JOIN` exactly; `DELIM_JOIN`/`ASOF_JOIN` rejected with new reason `PRODUCER_SHAPE` at discovery (§2, §3.1); lineage test pins both types (§4 test 1); risk claim rewritten honestly (risk 8). |
| 3 | MAJOR — freeze TYPE_MISMATCH read protected, equality-ordinal-compacted join members | Applied: `key_type` (producer `build_type`) is captured at discovery from C1a's `dynamic_filter_key_plan` extraction value, and the consumer probe-column type is captured during the lineage trace; freeze validates types purely from captured values, no join accessor added (§2, §3.1 steps 3/5, §3.4 step 5, risk 4). The review's alternative (public `build_key_type` accessor) rejected as unnecessary given value capture. |
| 4 | MAJOR — §3.5 wired fan-out into the pre-C1a publisher C1a replaces | Applied: §3.5 restated against C1a's unified `dynamic_filter_publication_plan` variant target vector (design:794-814) — freeze's `install_sip_publish_targets` entries are folded in as `join_probe_publish_target` variant alternatives before `create_query`; early-out/arity/per-target push come from the unified loop; dev-line ctor-span recipe and `:83-91`/`:121-126`/`:305-337` line surgery dropped (kept only as "code C1a rewrites" documentation); explicit C3b↔C1a coordination requirement added (§3.5, §6, risk 9). |
| 5 | MINOR — lazy gate construction on first probe task is a data race (concurrent probe tasks, `sirius_physical_hash_join.cpp:510-517`) | Applied: gate state constructed inside `install_sip_consumer_plan` at freeze, single-threaded, pre-`create_query`; runtime only mutates per-batch stats (§2, §3.6). |
| 6 | MINOR — §3.3 misattributed the assignment site; member visibility | Applied: function corrected to `sirius_interface::sirius_pending_statement_internal` (`sirius_interface.cpp:150-177`; `:177` engine.initialize; `sirius_execute_query` `:214-240` delegates); `sip_registry` declared public without trailing underscore to match the engine's public-member style (friends are pipeline classes only, `sirius_engine.hpp:55-58`) (§3.3). |
| 7 | MINOR — `require_two_gpus` citation | Applied: `mgpu_test_utils.hpp:144` (`mgpu_env_params` at `:62`) (§4 test 5). |
| 8 | MINOR — prepared/cached plan re-execution vs finalize-once | Applied: chose **defined per-plan-instance no-op** (not single-use verification): registry keeps spent flag + first-call stats, re-emits `topology_frozen reused=true`, installs nothing; rationale: `gpu_prepared` is cached (`sirius_extension.cpp:601-605`) and re-executable, and channel-content reuse matches Phase 1's plan-time scan-channel lifetime, so no new hazard class; topology test covers no-op re-finalize and e2e executes a prepared statement twice (§2, §3.4, §4 tests 2/4). |
| 9 | MINOR — two loose citations | Applied: producer-subset invariant now cited as an inference from design:964-966 (C1e), not design:72-79 (§3.2); `key_casts`/`right_key_col_indices` cited at `hpp:212`/`:225` (§2, §3.4). |
| 10 | MINOR — runbook omitted the log level coverage depends on | Applied: per-pass log levels are protocol (timing passes `SIRIUS_LOG_LEVEL=info`, coverage-detail passes `trace`, separate and non-timed); added an INFO `[dynf_summary] sip_consume_summary` per-endpoint aggregate (with before-first-filter counters) so coverage (b),(d),(e) derive from INFO timing runs; analyzer warns on flag-on runs with zero consume lines (§3.8). Prefixes also aligned to conventions: `[dynf_summary]` for machine-parsed per-query INFO lines, file-native `[sirius_physical_hash_join]` for publisher/consumer lines. |
