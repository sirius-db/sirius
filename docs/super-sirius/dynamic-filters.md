# Dynamic Filters

A **dynamic filter** is a predicate that is computed at query runtime by one operator (the *producer*) and consumed by another (the *consumer*) to prune data. The producer sees data, learns something about it, emits a filter; the consumer uses that filter to do less work — ideally before paying the cost of materialization.

This is a category, not a single feature. It spans:

- **Dynamic table-filter pushdown** — an eligible `BUILD_PROBE` hash-join build pushes runtime membership filters into a downstream GPU scan (parquet or duckdb-native); an optional zone-map can additionally prune parquet row groups against the actual build-side key range. It is a pure optimization — redundant with the join, so it never changes results. Membership pushdown is **on by default**; zone maps are off by default.
- **Sideways information passing (SIP)** — a hash-join build applies its membership filter inside its own probe subtree, as deep as the key stays a faithful pass-through, so intervening operators also skip rows the join would discard. Implemented behind a temporary default-off flag.
- **Aggregation-driven pushdown** — a `GROUP BY` or `DISTINCT` exposes its distinct-value set to downstream consumers.
- **Sort- or top-N-driven pruning** — a post-sort min/max is exact and free; a top-N's current threshold tightens upstream filters.
- **Adaptive runtime predicates** — operators that observe data and refine filters over the lifetime of a pipeline.

This document describes the implemented framework and the design-only directions that could generalize it further. Phase 1 has a `BUILD_PROBE` hash-join-build producer, a GPU-scan consumer (parquet and duckdb-native), and three filter kinds (zone map, IN-list, and Bloom). Membership pushdown is enabled by default; the workload-specific zone-map path remains opt-in. Phase 2 is implemented behind the temporary default-off `enable_dynamic_filter_sip` flag; phases 3 and 4 are design only.

## How the phases generalize

The framework has four axes of generality. Each phase opens one axis:

| Axis | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|------|---------|---------|---------|---------|
| Filter kind | zone map + Bloom + IN-list | (reuses Phase 1's filter zoo) | (reuses) | (reuses) |
| Consumer kind | parquet reader + post-decode scan operator (parquet + duckdb-native) | + membership endpoint spliced into the producer's own probe subtree | + any operator with a column input | (unchanged) |
| Producer kind | `BUILD_PROBE` hash-join build only | (unchanged) | + agg, sort, filter | (unchanged) |
| Coordination | single-shot build-port publication; direct probes ordered, transitive scan targets opportunistic | deterministic under the active task strategy; opportunistic under lookahead | (unchanged) | streaming / incremental refinement |

Anything implemented in Phase 1 is reused unchanged by later phases. We do not replace DuckDB's static table-filter pushdown — static filters continue to flow through the existing translator path and are AND-merged with dynamic filters at the consumer.

## Generalized architecture

The framework has four pieces, all designed to be filter-kind, producer-kind, and consumer-kind agnostic:

1. **`sirius_dynamic_filter`** — polymorphic base class for runtime-computed filters. Each subclass knows how to lower itself to a cuDF AST fragment, to a runtime apply pass, or both.
2. **`sirius_dynamic_filter_set`** — thread-safe append-only channel that connects producers and consumers. Push, storage, and lookup share one coordinate: the consumer operator's output ordinal, supplied by the discovery walk.
3. **Target discovery** — the producing join walks its own built probe subtree per admitted key (`dynamic_filter_target_discovery`): a trace that bottoms out at a GPU scan binds the key into that scan (the channel is attached to the scan node itself, which is the pairing point when several producers reach one scan), and any other terminal is a join-edge endpoint site when the SIP flag allows it. No route key and no registry exist.
4. **Producer / consumer roles** — concrete operators that push filters into channels (producers) or read them out (consumers). An operator can be both for different channels.

```mermaid
flowchart LR
    subgraph PLAN["Plan construction"]
        EVIDENCE["build_filter_evidence<br/>IsFiltering mirror over the logical build child"]
        DISCOVERY["target discovery walk<br/>one trace per admitted key over the built probe subtree"]
        PUBPLAN["dynamic_filter_publish_plan<br/>target channels + key bindings<br/>GPU/HOST replica spaces + policy"]

        EVIDENCE -->|"arms the scan route"| DISCOVERY
        DISCOVERY -->|"attach or mint"| CHANNEL
        DISCOVERY -->|"freeze producer configuration"| PUBPLAN
    end

    subgraph RUN["Runtime"]
        JOIN["sirius_physical_hash_join<br/>dynamic_filter_publisher"]
        CHANNEL["shared sirius_dynamic_filter_set<br/>append-only publication channel"]
        READER["parquet_gpu_ingestible<br/>zone-map AST at read (parquet only)"]
        APPLY["sirius_physical_dynamic_filter<br/>IN-list / Bloom post-decode<br/>+ zone-map row masks on duckdb-native"]
        DOWN["PARTITION → CONCAT → authoritative hash-join probe"]

        PUBPLAN --> JOIN
        JOIN -->|"construct + replicate, then push_filter"| CHANNEL
        CHANNEL --> READER
        CHANNEL --> APPLY
        READER -->|"decoded GPU table"| APPLY
        APPLY --> DOWN
    end
```

### Filter polymorphism (filter kind axis)

`sirius_dynamic_filter` is the base for every dynamic filter kind. The base carries the kind tag and the cheap device-availability query common to device-backed filters. Concrete consumer-side capabilities (AST lowering and runtime mask application) live on separate **capability mixin** interfaces that filters inherit alongside the base. Producer-side replica construction is likewise isolated in `sirius_device_replicable`. This avoids forcing a filter to implement a path it cannot satisfy: a bloom filter, which has no meaningful AST representation, simply does not inherit from `sirius_ast_lowerable`.

```cpp
class sirius_dynamic_filter {                  // base — kind + device availability
 public:
  virtual ~sirius_dynamic_filter() = default;
  [[nodiscard]] virtual sirius_dynamic_filter_kind kind() const = 0;
  [[nodiscard]] virtual bool is_available_on_device(int device_id) const noexcept;
};

class sirius_ast_lowerable {                   // capability: AST lowering
 public:
  virtual ~sirius_ast_lowerable() = default;
  [[nodiscard]] virtual cudf::ast::expression const& to_ast(
    cudf::ast::tree& tree,
    cudf::ast::expression const& column_ref,
    int device_id = -1) const = 0;
};

class sirius_dynamic_zone_map_filter
  : public sirius_dynamic_filter,
    public sirius_ast_lowerable,
    public sirius_device_replicable { /* AST consumer + producer replication */ };
```

**Consumer-side dispatch** is by `dynamic_cast` from a `sirius_dynamic_filter` pointer to the capability the consumer needs. The `merge_ast_dynamic_filters_into_tree` helper performs the AST-capability cast internally and silently skips filters that lack it, so consumers building an AST tree don't need any per-call-site logic. `sirius_dynamic_filter::kind()` remains for cases where the consumer needs filter-kind-specific behavior beyond what a capability mixin describes (e.g., a parquet reader that natively understands bloom filters via a path other than `apply`).

**AST lowering** — for consumers that build a `cudf::ast::tree` (parquet reader `set_filter`, expression evaluator). Tree nodes are owned by `tree`; device scalars referenced by literals are owned by the filter instance.

**Runtime apply lowering** — `sirius_mask_applicable` is the sibling capability for consumers that evaluate a filter against a materialized `cudf::column_view`. IN-list and Bloom implement it; zone maps use the AST path.

### Channel — `sirius_dynamic_filter_set` (decoupling axis)

A `sirius_dynamic_filter_set` is the append-only publication channel between producers and a consumer. It owns the filters and is co-owned via `shared_ptr` by every operator that holds either end; it is not a readiness barrier.

Properties:

- **Append-only.** `push_filter(col_idx, f)` adds; nothing removes.
- **Thread-safe.** A mutex guards the underlying map.
- **Push, storage, and lookup share one coordinate** — the consumer operator's output ordinal. The discovery walk supplies it as the trace's exit ordinal: the bound scan's output position for a scan route, the sited operator's output position for a join-edge endpoint. No translation happens inside the channel. Multiple producers targeting the same output column AND-conjoin at the consumer.
- **N producers, one logical consumer endpoint per channel.** Multiple joins may publish into the
  same channel. Multiple consumers use separate channels; the producer can fan the same immutable
  filter object into each one.
- **Filters are co-owned via `shared_ptr<filter const>`.** Producers may push the same filter object into multiple channels. Phase 1 uses this for scan-target fan-out, and a Phase 2 producer with both a scan-routed and a join-edge key fans distinct filters into distinct channels the same way.

Consumer access is via `filters_for_column(col_idx)` and `filtered_columns()`. The free helper `merge_ast_dynamic_filters_into_tree(tree, existing_root, set, resolver)` walks the channel, lowers every AST-capable filter, AND-conjoins the per-column and cross-column fragments, and returns `AND(existing_root, dynamic_root)` — or `existing_root` unchanged if no filter contributed.

### Target discovery — the producing join's own walk (routing axis)

The producing join owns its whole built probe subtree before its constructor runs, so it discovers every consumer itself: per admitted key, `trace_probe_key` follows the shared descent rules from the key's probe-child ordinal. A trace that bottoms out at a GPU table scan binds the key **into** that scan — the channel is stored on the physical scan node, which is the pairing point (an inner producer's scan already carries its channel when an outer producer's walk reaches it, so N producers share one channel by reaching one node). Any other terminal is a join-edge endpoint site, spliced by `place_endpoint` when the SIP flag allows it. There is no route key, no channel registry, and no consumer remap anywhere; the same walk also fans out through physical set operations (one terminal per UNION branch), though no planner constructs one yet.

Sirius reads no DuckDB dynamic-filter metadata in production: the metadata DuckDB's optimizer still computes is consumed only by DuckDB's own CPU fallback and by the test-only parity oracle (`duckdb_join_filter_candidate_adapter`, linked into the test target), which pins Sirius discovery against DuckDB's public `GetPushdownFilterTargets` per key. The scan route is gated on Sirius-owned build-filter evidence (`build_filter_evidence`, a mirror of DuckDB's `IsFiltering` over the logical build child) and on the producer join type (INNER, RIGHT, or SEMI — the mirror of DuckDB's `GenerateJoinFilters` gate; other types preserve or negate unmatched probe rows, so a probe-side filter would change results). A future Phase 3 producer that must pair with an operator it does not own would need new pairing machinery; nothing does today.

### Producer / consumer wiring

*Introduced in Phase 1.1.* The **producer-key admission boundary** (called the producer-key
admission *seam* in some issue-planning terminology) is the plan-time boundary where DuckDB join
conditions become Sirius-owned publication metadata. The runtime publisher receives only the
immutable result; it does not reinterpret DuckDB metadata.

Admission is Sirius-owned and reads the conditions alone: it admits every condition that passes its
legality rules (equality, no cast on either carried shape, bound references on both sides, a
cuDF-representable build type). No DuckDB hints exist anywhere in production -- where each admitted
key lands is decided afterwards by the discovery walk, which supplies the scan or endpoint push
ordinal. Publication then constructs a filter only for a key some target binds, so an unbound key
is a recorded legality fact that costs no GPU work.

Three inputs are gathered before physical planning recurses into the join's children:
per-condition domain evidence, the build subtree's uniqueness proof, and the build-filter evidence
that arms the scan route. All three read the logical children, and `create_plan` moves data out of
them, so computing any afterwards would read emptied nodes. If discovery binds no target, the plan
is disabled even if admission found legal keys -- `enabled()` is target-based, not key-based.

The planner-side components and the order they run in:

```mermaid
flowchart TB
    LOGICAL["DuckDB optimized logical comparison join"]
    EVIDENCE["build-filter evidence<br/>IsFiltering mirror"]
    DOMAIN["build-key domain walk<br/>native row upper bounds"]
    UNIQUE["build-subtree uniqueness proof"]
    ADMIT["key admission<br/>dense admitted keys"]
    DISCOVERY["target discovery<br/>one trace per admitted key"]
    PLAN["immutable dynamic_filter_publish_plan"]
    JOIN["BUILD_PROBE hash join"]
    POLICY["source policy<br/>gate + representation"]
    PUB["publisher<br/>construct + replicate + fan out"]
    CHANNEL["append-only filter channel"]
    RDR["parquet reader zone-map AST"]
    PQPOST["parquet post-decode membership"]
    NATIVE["native post-decode AST + membership"]
    EDGE["join-edge endpoint"]
    AUTH["authoritative hash join"]

    LOGICAL --> EVIDENCE
    LOGICAL --> DOMAIN
    LOGICAL --> UNIQUE
    DOMAIN --> ADMIT
    UNIQUE --> ADMIT
    EVIDENCE --> DISCOVERY
    ADMIT --> DISCOVERY
    ADMIT --> PLAN
    DISCOVERY --> PLAN
    PLAN --> JOIN --> POLICY --> PUB --> CHANNEL
    CHANNEL --> RDR --> AUTH
    CHANNEL --> PQPOST --> AUTH
    CHANNEL --> NATIVE --> AUTH
    CHANNEL --> EDGE --> AUTH
```

A **producer** receives an immutable publication plan. It uses a dense admitted-key array for
filter construction and sparse per-target bindings for fan-out:

```cpp
class dynamic_filter_publish_plan {
 public:
  struct admitted_key {
    std::size_t planner_condition_index;       // provenance, in original planner order
    cudf::size_type build_key_ordinal;         // runtime build-table column
    cudf::size_type probe_key_ordinal;         // probe-child output column; a direct route's
                                               // descent entry ordinal
    cudf::data_type storage_type;              // build side
    cudf::data_type probe_storage_type;        // probe side; EMPTY when untranslatable
    dynamic_filter_condition_shape key_shape;  // carried pre-materialization classification
    std::size_t build_key_domain_cardinality;  // 0 = unknown, coverage gates off for this key
    bool build_key_proven_unique;              // arms the membership coverage gate
  };

  struct key_binding {
    std::size_t admitted_key_index;    // dense admitted-key array
    std::size_t channel_push_ordinal;  // this target channel's push space
    cudf::data_type probe_storage_type;
  };

  struct probe_target {
    std::shared_ptr<sirius_dynamic_filter_set> filter_set;
    dynamic_filter_route_class route_class;
    bool accepts_zone_map_filters;
    std::vector<key_binding> key_bindings;
  };

 private:
  std::vector<admitted_key> _admitted_keys;
  std::vector<probe_target> _probe_targets;
  dynamic_filter_publication_policy _policy;  // config-transported, ingress-validated
  std::vector<dynamic_filter_replica_space> _replica_spaces;
};
```

Admission keeps five persisted coordinates distinct:

| Coordinate | Meaning |
|---|---|
| Original condition index | Provenance in the pre-wrap, pre-reorder join-condition vector; also indexes carried pre-materialization shapes and per-condition domain evidence |
| Admitted-key index | Dense position after statically illegal keys are removed |
| Build-key ordinal | Column in the materialized runtime build table |
| Probe-key ordinal | Column in the producing join's probe-child output; the discovery walk's ENTRY ordinal |
| Channel push ordinal | The discovery walk's EXIT ordinal, in the bound consumer's output space: the bound scan's output position for a scan route, or the output space of the operator `place_endpoint` sited the endpoint on for a direct route. Push, store, and lookup are this one coordinate; it equals the probe-key ordinal only when the walk accepted no hop |

The entry and exit ordinals relate only through the walk. Every other index (a target index
selecting matching target entries) aligns plan-construction vectors only and persists nowhere.

The planner freezes routing, placement, and policy into the hash join's `const dynamic_filter_publish_plan`. Each `dynamic_filter_replica_space` pairs one target GPU space with its selected HOST staging space. The build-port hook claims publication as soon as the complete build batch arrives, holding the batch's read-only accessor from before it is routed so the GPU representation cannot be downgraded underneath publication. The producer builds and replicates each filter, then fans it into the accepting channels. Finalization only closes an unclaimed publication window.

A **consumer** holds the channel directly via `std::shared_ptr<sirius_dynamic_filter_set> sirius_dynamic_filters` and reads from it during execution.

## Lifetimes and ordering

**Channel lifetime.** The `sirius_dynamic_filter_set` is co-owned via `shared_ptr` by every operator that references it.

**Scalar lifetime.** Scalars referenced by AST literals or apply kernels are owned by the filter instance, which is owned by the set. The set must outlive any AST tree or apply invocation built from filters it contains.

### Immediate-probe ordering

For a scan that directly supplies a `BUILD_PROBE` join's probe input, the primary publication attempt completes before the scan is activated. Build-side `CONCAT` waits for the build partition pipeline, folds the complete build side into one GPU batch, and synchronously calls `sirius_physical_hash_join::push_data_batch_partitioned("build", ...)`. That call waits for the batch writer event, constructs the filters, completes device replication, fans the ready filters into the channels, and reaches `FINISHED` before returning. Only after the CONCAT task returns does downstream task creation ask that same join for another hint; with build data present and probe data absent, the hint follows the immediate probe producer.

This is an ordering property of the producing join's **immediate** probe edge, not a universal barrier in front of every scan to which DuckDB routes the filter. The following sequence is deliberately scoped to that direct shape:

```mermaid
sequenceDiagram
    participant B as Build GPU_SCAN / PARTITION
    participant C as Build CONCAT (concat_all)
    participant J as BUILD_PROBE hash join
    participant R as Device replica spaces
    participant F as Dynamic-filter channel(s)
    participant T as task_creator
    participant P as Immediate probe GPU_SCAN

    B->>C: Complete all build partitions
    C->>C: Fold the complete build side to one GPU batch
    C->>J: push_data_batch_partitioned("build", batch)
    activate J
    J->>J: Wait for the batch writer event
    J->>J: OPEN → PUBLISHING; construct filters
    J->>R: Materialize device-local replicas
    R-->>J: Replica completion
    J->>F: push_filter(...) fan-out (possibly none by policy)
    J->>J: PUBLISHING → FINISHED
    J-->>C: Publication attempt finished
    deactivate J
    C-->>T: CONCAT task completes; schedule downstream join
    T->>J: get_next_task_hint()
    J-->>T: WAITING_FOR_INPUT_DATA(probe producer)
    T->>P: Create and enqueue probe data-scan tasks
    P->>F: has_filters() and device-local snapshot (never waits)
    P->>P: read_parquet, then membership post-filter
```

### Transitive scan targets and publication timing

The discovery walk traces a key through operators on the probe side, including intervening comparison joins, until it reaches a base scan. Such a scan is a **transitive probe target**: it contributes to the producing join's probe subtree, but it does not directly feed that join's probe port. The producing join's build-port ordering therefore does not gate the scan.

For example, consider this simplified Q8-shaped join:

```sql
FROM lineitem AS l
JOIN supplier AS s ON l.l_suppkey = s.s_suppkey
JOIN part AS p ON l.l_partkey = p.p_partkey
WHERE p.p_type = 'PROMO BRUSHED COPPER'
```

One valid physical shape makes the `supplier` join part of the `part` join's probe subtree:

```mermaid
flowchart LR
    PART["scan part<br/>selective build"] --> BC["build CONCAT<br/>for J_part"] -->|"build"| JP["J_part"]
    LINE["scan lineitem<br/>dynamic-filter target"] -->|"probe"| JS["J_supplier"] -->|"probe"| JP
    SUP["scan supplier"] -->|"build"| JS
    BC -. "publish l.l_partkey membership" .-> LINE
```

The filter is still redundant and safe: every `lineitem` row that survives `J_part` must have a key in the filtered `part` build. This is still Phase 1 scan pushdown, not Phase 2 SIP: the discovery walk traverses `J_supplier`'s probe block while finding the target, but `J_supplier` neither consumes nor applies the filter; the `lineitem` parquet scan remains the consumer. Scheduling is different from the direct case, however:

1. `J_supplier` finishes its own build and activates the `lineitem` scan.
2. The task creator can enqueue many `lineitem` GPU scan tasks. Enqueuing does not snapshot the dynamic-filter channel.
3. Probe `PARTITION` and non-`concat_all` `CONCAT` may stream enough `J_supplier` output toward `J_part` while more `lineitem` tasks remain queued or in flight.
4. Once task creation reaches `J_part`, its missing build causes the `part` build subtree to be created.
5. The scheduler dispatches the resulting tasks through its normal device-affinity and queue-order policy. There is no filter-specific preference or readiness barrier.

The transitive consumer remains deliberately opportunistic. It does not create a missing build
task, preempt a running scan, or wait for filter readiness. Splits already past a consumer
checkpoint are not revisited. The discovery walk follows the producing join's probe subtree
through intervening operators to attach the channel to a base scan; the scheduler does not
perform a second filter-specific topology walk.

Issue [#1124](https://github.com/sirius-db/sirius/issues/1124) compared filters disabled, the former
build-subtree preference, and normal scheduling at SF300. Every measured scan consumed zero rows
before publication in both filter-enabled configurations, while normal scheduling improved wall
time by 9–25% and substantially reduced variance with bit-identical results and unchanged peak
memory. The preference was therefore deleted. This result motivates the policy but does not create
an ordering guarantee for other workloads; any future consumer that requires ordering must express
that dependency explicitly outside the global scheduler.

Metadata preparation is independent of consumer routing and task scheduling. Footer parsing, split coalescing, and background prefetch preparation may happen before publication; those activities do not decode or materialize table rows and do not snapshot the channel. A queued GPU scan task also has not necessarily consumed the filter yet. A parquet task selects zone maps immediately before `read_parquet` and membership filters in the following fused post-decode operator; publication during decode may therefore still be observed by the later membership stage. A duckdb-native task instead has one post-decode checkpoint that selects both AST-lowerable zone maps and membership filters. Publication after a format's relevant checkpoint cannot make that checkpoint run again.

The channel therefore remains opportunistic and never waits on readiness. A direct target normally observes the completed fan-out. A transitive target racing publication may observe no filter, a subset while `push_filter` appends the individually complete filters, or the full set after publication reaches `FINISHED`; every filter becomes visible only after its own device replicas are ready. An intentionally empty publication, an unsupported or policy-gated key, an unavailable local replica, or a filter disabled by the selectivity gate is always a safe pass-through because the join remains authoritative.

## Static + dynamic filter at the consumer

The static and dynamic filters live on **separate, non-interfering paths**, which is what lets the dynamic side be a pure optimization. `parquet_gpu_ingestible` owns the coalesced DuckDB static expression and translates it for reader pushdown; device-local dynamic AST fragments are conditionally merged during per-split materialization:

- **Static filter** — a `shared_ptr<duckdb::Expression>` (`parquet_gpu_ingestible::_duckdb_filter_expression`), lowered by `gpu_expression_translator::translate_expression_with_names` to a cuDF AST and installed via `reader_options::set_filter` (or, if translation fails, evaluated post-decode by `expression_evaluator`). It is authoritative for correctness and is never touched by the dynamic side.
- **Dynamic filter** — the zone-map is AST-lowerable, so `merge_dynamic_filters_into_ast` (column-name references) AND-merges it onto the reader's filter root, installed via `reader_options::set_filter` (in `parquet_gpu_ingestible::materialize_metadata_to_table`). cuDF's parquet reader uses a `set_filter` predicate both to prune row groups by their statistics *and* to evaluate it row-wise during decode. Membership filters (IN-list / bloom) are not AST-lowerable, so the merge skips them and they are applied **post-decode** by the `sirius_physical_dynamic_filter` operator. Either way the dynamic predicate is a conjunctive *superset* of the join condition, so it never changes which rows survive — the join is authoritative.

The dynamic zone-map AST is merged onto the static filter root when the static filter translated, or built standalone when it did not, so the earlier "translation failed ⇒ dynamic pushdown skipped" restriction no longer applies: the dynamic side contributes regardless of the static filter's translation outcome, and the static filter keeps its own path unchanged.

The duckdb-native GPU scan is post-decode only. It has no reader-side filter hook — decode is Sirius's own native path and static filters are evaluated in `post_filter_and_project` — so its `sirius_physical_dynamic_filter` runs in `include_ast_row_masks` mode: an opted-in zone map is evaluated row-wise via `cudf::compute_column` alongside the membership masks, behind the same gate. Row-group stat pruning in the native metadata walk remains static-only.

This mixing rule is a property of the consumer scan format, not the framework. A Phase 2 join-edge endpoint has no scan format and no reader hook, so it applies membership masks only and merges nothing.

```mermaid
flowchart TB
    STATIC["DuckDB static Expression"]
    TRANSLATE["gpu_expression_translator<br/>translate_expression_with_names"]
    STATICROOT["Static AST root<br/>or no root if translation fails"]
    CHANNEL["sirius_dynamic_filter_set"]
    MERGE["merge_dynamic_filters_into_ast<br/>AND local zone map with static root"]
    OPTIONS["reader_options::set_filter<br/>when an AST root exists"]
    READ["cudf::io::read_parquet<br/>row-group pruning + decode"]
    FALLBACK["expression_evaluator<br/>static fallback when needed"]
    APPLY["sirius_physical_dynamic_filter<br/>selectivity gate"]
    MASK["IN-list / Bloom compute_mask<br/>apply_boolean_mask"]
    DOWN["PARTITION → CONCAT → authoritative hash-join probe"]

    STATIC --> TRANSLATE --> STATICROOT --> MERGE
    CHANNEL -->|"AST-capable zone map<br/>local scalar replica"| MERGE
    MERGE --> OPTIONS --> READ --> APPLY
    READ -.->|"if static AST translation failed"| FALLBACK
    FALLBACK --> APPLY
    CHANNEL -->|"mask-capable IN-list / Bloom<br/>local membership replica"| APPLY
    APPLY --> MASK --> DOWN
```

---

## Phase 1 — Dynamic table-filter pushdown

**Goal:** establish the framework end-to-end against a single (producer, consumer) pair — `BUILD_PROBE` hash-join build → GPU scan (parquet and duckdb-native) — and exercise filter-kind polymorphism via progressively richer filters.

**Producer:** `BUILD_PROBE` hash-join build side.
**Consumer:** GPU scan — parquet (`parquet_gpu_ingestible`: reader zone-map + post-decode operator) and duckdb-native (post-decode operator only).
**Routing:** Sirius-owned discovery (a walk over the built probe subtree binds each admitted key into the scan its trace bottoms out at).
**Coordination:** synchronous build-side CONCAT publication strictly precedes the producing join's immediate probe data scan; transitive scan targets remain nonblocking and race publication under normal scheduler order.

### 1.1 Foundational wiring

Wires the scaffolding into operators end-to-end against a degenerate single-zone (N=1) zone-map filter — equivalent to a global min/max bound. Validates the channel + discovery + consumer-merge plumbing in a real query.

Plan-gen / type plumbing:

- `sirius_physical_table_scan`'s `sirius_dynamic_filters` field (consumer endpoint, attached by the producing join's discovery walk and propagated through `parquet_ingestible_table_info` to `parquet_gpu_ingestible` and through `duckdb_native_ingestible_table_info` to the native ingestible)
- `dynamic_filter_publish_plan::probe_target` entries plus non-owning paired GPU/HOST replica spaces (producer endpoint, held privately by the hash join)
- Plan-gen wiring in `sirius_plan_comparison_join.cpp`, gated by `enable_dynamic_filter_pushdown`; the join attaches or reuses one channel per bound scan and merges its keys into one target per scan

#### Ordered build-port publication

Publishing from `finalize_operator()` or from the hash-table build would be too late because the `BUILD_PROBE` hash table is constructed only after the first probe batch arrives. The implemented producer instead uses the complete build batch delivered by build-side `CONCAT`; it requires no probe batch and does not depend on the hash-table state machine.

The normal path is deliberately ordered:

1. Pipeline construction places the build child before the probe child. The remainder of this sequence applies when partitioning selects `BUILD_PROBE`.
2. Build `PARTITION` selects `BUILD_PROBE` only when a build-side CONCAT can fold the input, then sets that CONCAT to `concat_all`.
3. Build CONCAT waits for its source pipeline, folds the complete build side to one GPU batch, and synchronously calls `push_data_batch_partitioned("build", batch)`.
4. The hook acquires the batch's read-only accessor before routing it — once deposited into a repository the batch becomes a downgrade candidate, and the shared lock pins its GPU representation until publication completes. It then waits for the representation's writer event, claims `OPEN -> PUBLISHING`, constructs the selected filters, completes device replication, pushes the immutable filters into every accepting channel, and stores `FINISHED` before returning.
5. Only after the CONCAT task returns does downstream task creation ask the join for its next hint and follow `WAITING_FOR_INPUT_DATA` into the immediate probe producer. A scan on that edge therefore cannot run while normal build-port publication is in progress.

The publish gate admits two `BUILD_PROBE` shapes where one partition's build batch is the complete build side: a **single partition** (`_partition_build_states.size() == 1`), and a **broadcast** join (`_broadcast`), where the small build table is replicated to every GPU so each partition's `concat_all`-folded batch is the full build. Under broadcast there is one build CONCAT per GPU, each racing the build-port hook; the `OPEN -> PUBLISHING` compare-exchange in `publish_dynamic_filters` selects exactly one publisher (the first to arrive), while the rest return at the CAS before constructing anything. A genuinely hash-partitioned (non-broadcast) multi-partition build keeps pushdown disabled, because each partition holds only a slice of the build keys and no single batch could emit a complete filter (cross-partition aggregation is a future extension). The gate is `(_partition_build_states.size() == 1 || _broadcast)` in `sirius_physical_hash_join::push_data_batch_partitioned`.

That sequence does not gate a scan reached transitively through an intervening join. Those targets
run under normal locality-aware dispatch and may observe no filter, a partial fan-out, or the
complete publication at their checkpoints. See
[Transitive scan targets and publication timing](#transitive-scan-targets-and-publication-timing).

The publication attempt may intentionally emit no filter—for example, for an empty build, a cast or unsupported key, a domain-covering key, or a non-selective zone range. That successful no-op is still `FINISHED`. Allocation pressure may also leave an optional replica unavailable. Consumers need no readiness protocol: they test the channel and local device availability and pass the batch through when nothing useful can apply. Replica materialization for every filter kind treats a per-target failure (reservation denial, cloning, copy, or completion synchronize) as best-effort: it is logged and that target's replica is omitted.

A batch that arrives already non-GPU-resident (possible only when it was shared with an earlier consumer, e.g. CTE fan-out, and downgraded before delivery) skips publication — filters are optional. `on_finalize_operator` never publishes; it only changes an unclaimed `OPEN` state to `CLOSED`. GPU construction and replication run without holding `op_state_mutex`.

Replica bytes use direct peer DMA where empirically verified, otherwise they borrow chunked pre-pinned storage from the planned Sirius/CuCascade HOST memory space. The dynamic-filter code performs no direct pinned allocation and does not modify CuCascade. See [dynamic-filters-multi-gpu.md](dynamic-filters-multi-gpu.md) for the replica design and validation.

**Application in the target scan.** A dynamic join filter is *redundant with the join that produced it*: the join discards every non-matching probe row, so the filter is a conjunctive superset and never has to be exact. The zone map rides the parquet reader's filter: `merge_dynamic_filters_into_ast` AND-merges it onto the reader root and `materialize_metadata_to_table` installs it via `reader_options::set_filter`, so cuDF can prune row groups by statistics and evaluates the predicate during decode. Membership filters (IN-list / Bloom) are not AST-lowerable and ride the **post-decode** path (`sirius_physical_dynamic_filter` → `apply_dynamic_filters_to_view`, `membership_masks_only` mode). On the duckdb-native scan, which has no reader filter, the operator instead runs in `include_ast_row_masks` mode so zone maps apply row-wise there too. The shared `dynamic_filter_gate` measures scan-level and per-filter usefulness before allowing that post-decode cost to repeat. The static filter is untouched and remains authoritative.
>
> **Follow-up — zone-map row-GROUP-only.** Merging the zone-map into `set_filter` keeps cuDF's row-group stats-pruning win but also pays a row-level evaluation cost during decode. A row-group-*only* path — evaluate the zone-map against `filter_row_groups_with_stats` and feed only `reader_options::set_row_groups`, never `set_filter` — would drop that row-level cost but is **not yet implemented**. The cost is bounded today: the zone-map is off by default (`enable_dynamic_zone_map_filter`) and TPC-H's scattered keys prune nothing.

> **Redesign note (nsys-driven, SF50).** An earlier implementation applied dynamic predicates redundantly through both `reader_options::set_filter` and an unconditional post-decode `compute_column` + `apply_boolean_mask`. Profiling showed that row-level work regressed scattered-key workloads where it dropped almost no rows. The current design gives each filter kind one path per consumer format: on a parquet scan, zone maps use the reader's `set_filter` and membership filters use the gated post-decode operator; a duckdb-native scan has no read-time filter hook, so its zone maps ride that same post-decode operator as an AST row mask (`include_ast_row_masks`), under the same scan-level gate. Each filter is evaluated exactly once, never redundantly through both paths. Membership filters and duckdb-native zone maps run only while the gate keeps the post-decode operator active; parquet zone maps remain on the reader path outside that gate. The filter never affects results because the join is authoritative.

#### Measured

- **SF50, full TPC-H (22 queries), regression guard — OFF vs ON:** aggregate **+0.2 %** (136.17 s → 136.39 s), i.e. net-zero within the per-query noise floor (~±9 % on the small queries; the large queries are all ≤ ±3 %). **All 22 results bit-identical** OFF vs ON. `pruned=0` on every query because TPC-H's scattered keys span the key domain; the gated/default-off paths bound that cost. Full unit suite passes.
- **SF30, `lineitem ⋈ keyset` on clustered `l_orderkey`** (the case the zone-map is *for*): the zone-map excludes the non-overlapping row groups, ≈1.6× — preserved by the row-group path. This requires the keyset's narrow range be *runtime*-determined: for a literal/range-derivable build, DuckDB's static transitive pushdown already prunes (measured feature-OFF == ON on SF50), so the dynamic zone-map adds nothing — hence it is gated off by default (`enable_dynamic_zone_map_filter`).

Zone-map pruning only pays off when the build-side join key is range-restricted on a *clustered* fact (e.g. `l_orderkey`). An immediate probe read is ordered after publication. A transitive target may materialize early splits first and observes filters opportunistically; for every split that does observe the zone map, key distribution determines whether row groups can be rejected. TPC-H joins on scattered keys (`l_partkey` in Q14/Q17/Q19) prune nothing — those are handled by the membership filters below.

### 1.3–1.4 Membership filters (raw/hash IN-list + Bloom)

The zone-map captures only the build keys' `[min,max]` range, which is useless for *scattered* keys — even a 0.24%-selective `part` build spans the whole partkey domain (measured). Set **membership** is what distinguishes those keys. Membership filters use the `sirius_mask_applicable` capability (a `compute_mask(probe) → BOOL` mask, distinct from `sirius_ast_lowerable`) and have three concrete representations:

- **`sirius_dynamic_small_in_list_filter`** — exact membership for 1–12 null-free INT32/INT64 build rows, counting duplicates. Each successfully materialized device-local replica owns a raw snapshot of the build values (the *needles*), and one CUB bulk kernel compares every probe value with every needle. It has no hash build, slot array, or reserved empty-key sentinel.
- **`sirius_dynamic_in_list_filter`** — hash-based membership via a persistent `cuco::static_set`, with a device kernel probing its read-only set reference. It is exact for representable set keys and conservatively passes cuCO's reserved empty-key value, so it remains a safe join-filter superset.
- **`sirius_dynamic_bloom_filter`** — a `cuco::bloom_filter` (PIMPL'd in a `.cu`; INT32/INT64 keys), a few bits/key, for large builds where the hash IN-list is too big. False positives only let extra rows through — harmless, because the join is authoritative; no false negatives means a true match is never dropped.

**Producer policy** (`dynamic_filter_publisher::publish`). The master switch `enable_dynamic_filter_pushdown` emits at most one **membership filter** per key, in this order:

1. Use the raw-needle exact IN-list when `sirius_dynamic_small_in_list_filter::supports(col)` sees 1–12 null-free INT32/INT64 build rows. The gate uses the view's row count, including duplicates; it does not compute distinct cardinality.
2. Otherwise use the hash IN-list when the column is supported and its estimated `cuco::static_set` footprint fits the minimum L2 size across the active probe GPUs.
3. Otherwise use the Bloom filter whenever its key type is supported; its bitset may itself exceed L2 for a sufficiently large build.

After the earlier empty-build, cast, and domain-coverage gates, **none** is selected only when the key type has no supported membership representation. The hash-set and Bloom estimates use the build row count (an upper bound on distinct keys) and the planned devices' minimum `cudaDevAttrL2CacheSize`. The small-list decision precedes that L2 comparison and stores exactly the raw INT32/INT64 input bytes; duplicates are harmless because membership is existential.

**Domain-coverage gate.** Before paying to build a membership structure, publication skips a key whose build covers at least `dynamic_filter_domain_coverage_threshold` (default 0.9) of its key domain -- such a filter keeps nearly every probe row, and the consumer-side keep-ratio gate remains the runtime backstop for everything this gate does not catch. The domain (`admitted_key::build_key_domain_cardinality`) is derived at plan time by a positional lineage walk (`planner/dynamic_filter/build_key_domain.hpp`): the build key's output ordinal is followed down through operators whose rows are an injective image of the traced child's rows -- projection/filter/order pass-throughs, LIMIT/TOP_N/DISTINCT, single-grouping-set aggregate groups, SEMI/ANTI/RIGHT_SEMI/RIGHT_ANTI joins on their only emitted block, and MARK/SINGLE joins on their left block -- to a base scan. Every other shape (INNER joins above all: they multiply the traced side's rows) refuses and records 0, which disables the gate for that key. Evidence comes solely from DuckDB-native table scans, whose `NodeStatistics::max_cardinality` (committed rows plus transaction-local inserts) is a true upper bound; Parquet and every other table function are refused as declared scope. Uniqueness matters because only for a unique key is `build_rows / domain` the coverage fraction it claims to be; for duplicate keys the same ratio measures row retention, which can be near 1.0 while the filter is highly selective. Before this mechanism landed the recorded domain was always 0 and both coverage gates were inert; the membership gate alone is live now, and deterministic gate decisions are observable through `SiriusContext::get_dynamic_filter_stats_snapshot()`.

| Condition | Coverage-gate result |
|---|---|
| `threshold > 1.0` | Disabled unconditionally; this is the rollback lever |
| Domain is `0` (evidence refused or absent) | Disabled for that key |
| Key is not proven unique in its base relation | Disabled for that key |
| `build_rows / domain >= threshold` | Fires -- the key is skipped before either filter is built |
| Otherwise | Does not fire |

Exactly `1.0` is an active threshold and fires only at full coverage. The check precedes both constructions, so a firing gate suppresses an opted-in zone map along with the membership filter.

A second switch, `enable_dynamic_zone_map_filter` (default **off**, requires the master), *additionally* emits a **zone map** (build-key min/max) per key — a complementary consumer path. Parquet scans use it for read-time row-group pruning; duckdb-native scans evaluate it row-wise post-decode. It is off by default because on TPC-H-shaped joins DuckDB's static transitive-predicate pushdown already prunes range-derivable builds, while scattered keys span the domain and prune nothing. Its publication range-coverage gate is inactive: the only domain evidence is a row count, and a value span divided by a row count over-fires on sparse integer keys, so the gate receives a domain of 0 until base-column value-range evidence exists. On a parquet scan the zone map rides the reader's `set_filter` and is therefore outside the post-decode `dynamic_filter_gate`; on a duckdb-native scan it is evaluated row-wise inside the gated post-decode operator, so it does sit behind that scan-level gate. It is never *per-filter* skipped either way — only membership filters record a marginal keep ratio.

**Consumer**: membership rides the post-decode `apply_dynamic_filters_to_view` path on both scan formats, but in different modes — parquet scans wrap it in `membership_masks_only` (the reader already evaluated the AST-lowerable filters), duckdb-native scans in `include_ast_row_masks`. The zone-map rides the parquet reader's `set_filter` (cuDF prunes row groups by stats and evaluates it during decode); a duckdb-native scan has no reader hook, so that same post-decode pass evaluates the (opted-in) zone map row-wise as an AST row mask alongside the membership masks. Because membership is applied *post-decode*, it never saves the scan I/O/decode — only the *downstream* work on dropped rows; so it wins on selective builds feeding expensive downstream and is neutral on scan-dominated single joins.

The **hash-IN-list-to-Bloom** cutover is hardware-adaptive — it replaced an earlier fixed `2M`-row-count threshold — landing at ≈1.5M INT64 keys on a 24 MB L2, ~2.5M at 40 MB, ~6M at 96 MB. A historical TPC-H SF50 sweep showed that cutover to be **wall-clock-neutral** (every query whose kind flipped was flat across `[500K, 8M]`; forcing the affected builds all the way to *none* moved them ≤1.5%, within run-to-run noise). The L2 policy keeps the hash IN-list while it fits L2 and otherwise uses the smaller Bloom. All result sets stayed bit-identical across hash IN-list / Bloom / none.

That sweep and the SF50/SF300 measurements below predate the raw-needle representation. They characterize the surrounding membership system, not the new small-list cutoff or a raw-scan-versus-hash performance comparison; no such benchmark is recorded here.

**Adaptive selectivity gate** (`sirius_physical_dynamic_filter::_gate`): the first applicable non-empty batch records the combined keep ratio of everything the operator applied in its cascade — the membership masks, plus (in `include_ast_row_masks` mode, i.e. duckdb-native scans) the zone map's row mask; if it keeps more than `dynamic_filter_keep_threshold` (default 90%) of its rows, the scan-level gate becomes `DISABLED`, otherwise it becomes permanently `ACTIVE`. Within an active cascade, each *membership* filter also records its marginal keep ratio; a membership filter keeping more than 50% of the rows reaching it is skipped on later splits. The AST/zone-map step is cross-column, so it has no per-column ratio to gate on and its drop is deliberately not recorded per-filter — only the scan-level gate can turn a native scan's zone map off. A GPU without a local replica does not train the shared gate. Applicability is lock-free, while rare decision updates are serialized. The gate is filter-count-aware because a direct target normally starts after the complete fan-out, while a transitive target may observe additional filters on later splits.

**Measured (SF50, full TPC-H, ON vs OFF, robust medians):** real wins appeared where a selective build feeds expensive downstream — **Q21 −10.6%** and **Q2 −7.9%** in that historical sweep. Net suite ≈ **−1.5%**. All 22 results were bit-identical OFF vs ON; membership pushdown is enabled by default. The Q21 attribution was later found not to represent the Phase 1 scan path (see the caveat below), so it must not be read as evidence for a publication-timing effect.

> **Cold vs warm.** The numbers above are **warm** (page-cache-resident) medians. A cold/deployment-representative sweep (drop OS cache before each run) shows the wins **largely evaporate** — membership applies *post-decode*, so it cannot cut scan I/O, and the big nominal wins are on I/O-bound queries: full-suite net ≈ **−0.5 % cold** vs ≈ −1.5 / −2.2 % warm. The durable benefit is on small-selective-build, *compute-bound* queries (Q2-shaped); benefit ∝ 1/(how I/O-bound the query is). *Q21 caveat:* a per-query SF50 diagnostic found Q21 wires only a ~20K-key IN-list on the Phase-1 scan-pushdown path (not the ~37M-key `orders` bloom the −10.6 % was attributed to), and its measured swing is within run-to-run noise — a bloom over the F-status `orders` self-join would be a Phase 2 join-edge endpoint, which was not wired when that measurement was taken.

**Measured (SF300, two GPUs, pinned-host compute regime):** with physical GPUs 1 and 2, grouped execution, five iterations per query, and iteration 0 discarded, the sum of Q1-Q22 warm medians improves from **13.7013505 s to 8.2939465 s: 39.466212% (1.651970x)**. All 22 ON/OFF result files are byte-identical. This is a fixed-two-GPU feature A/B, not a one-to-two-GPU scaling or cold-I/O claim; the exact protocol and per-query table are in [dynamic-filters-multi-gpu.md](dynamic-filters-multi-gpu.md#performance).

*Follow-ups:* wider and variable-width membership keys. INT32 and INT64 are implemented.

#### Configuration

- `enable_dynamic_filter_pushdown` (bool, default **true**) — master switch; when off, discovery never runs, so no channel exists anywhere and there is zero overhead. Enabled by default to wire the membership (raw/hash IN-list or Bloom) filters.
- `enable_dynamic_zone_map_filter` (bool, default **false**, requires the master) — additionally emit a zone map. Parquet scans use it for read-time row-group pruning; duckdb-native scans apply it row-wise post-decode. Off by default: static pushdown already handles range-derivable builds and scattered keys prune nothing, so it is reserved for clustered-keyset workloads whose narrow range is runtime-determined. Its publication range-coverage gate is inactive pending base-column value-range evidence.
- `dynamic_filter_domain_coverage_threshold` (double, default **0.9**) — skip publishing a key's membership filter when the build covers at least this fraction of the key's domain; fires only for proven-unique build keys with DuckDB-native scan evidence, and the zone-map range gate stays inactive. Above 1.0 the gate is disabled outright (the rollback lever); exactly 1.0 fires only at full coverage.
- `dynamic_filter_keep_threshold` (double, default **0.9**) — consumer-side scan gate: disable a scan's post-decode filtering once a measured split keeps more than this fraction of its rows; in [0, 1], 1.0 keeps filtering always on.
- `enable_dynamic_filter_sip` (bool, default **false**, requires the master) — **[TEMPORARY]** place join-edge endpoints inside the producing join's own probe subtree (Phase 2). Collapses into the master switch once the rollout decision is recorded.

**Rollback levers, in order of blast radius.** `SET enable_dynamic_filter_sip = false` restores the exact scan-route-only topology. `SET dynamic_filter_domain_coverage_threshold = 2.0` restores pre-gate publication behavior without disabling dynamic filtering. `SET enable_dynamic_filter_pushdown = false` disables the whole subsystem: discovery never runs, so neither producer nor consumer wires anything. Each is a setting that already exists; reversing this subsystem introduces no new knob.

#### Observability

`SiriusContext` owns connection-lifetime cumulative counters, read through `get_dynamic_filter_stats_snapshot()`; tests take before/after snapshots around a query. The counters split into three families and the split is a contract. `producers_enabled` stands alone as a **plan-time fact**: the hash-join constructor increments it on receiving an enabled plan, before execution begins, so nothing races it and it is the one counter that may anchor an exact equality. It is also an honest capability signal: discovery creates a target only when a key actually binds, so an enabled producer always has at least one bound key and a publication attempt that can push. The **policy-decision** family (`keys_considered`, `keys_with_known_domain`, `keys_skipped_domain_gate`, `keys_skipped_type_mismatch`, `keys_build_exceeded_domain`, filters built) is deterministic for attempts that reach per-key processing, and is the anchor for gate regressions. The **delivery** family (attempts, finished/failed, source-not-resident, targets-drained, `filters_pushed`) races probe-side draining and target liveness: assert it as deltas or directions, never as an equality anchor. `keys_with_known_domain` means only that nonzero row evidence exists -- uniqueness is separate, so the counter alone does not mean the gate was armed.

#### Ready replicas and per-split snapshots

The publisher constructs every selected zone-map and membership structure and completes all usable device replicas before fan-out begins. It then appends filters to target channels one `push_filter` call at a time. Each filter is therefore immutable and fully ready on every successful replica as soon as it becomes visible, but the complete multi-filter, multi-target fan-out is not an atomic channel operation.

An immediate probe target is activated only after publication reaches `FINISHED`, so it normally sees the complete fan-out. A transitive scan can race fan-out and see none, a safe subset, or the full set. The consumer takes a fresh channel snapshot at each application checkpoint for that reason. An intentionally empty publication, a missing local replica, or applying only a visible subset is always correct because the authoritative join performs the exact match.

### 1.2 Multi-zone zone maps (design only)

The `sirius_dynamic_zone_map_filter` representation supports N zones and lowers them as an OR of bounded ranges, but the current publisher emits one global `(min,max)` zone per key. A future producer could retain per-build-partition bounds; its AST would grow as `O(partitions)` and would need a fallback-to-global budget.

Zone maps currently implement `sirius_ast_lowerable` only. They do not implement the post-decode membership-mask path.

### 1.3 Bloom filter

`sirius_dynamic_bloom_filter` builds a GPU Bloom filter over INT32/INT64 build keys. The consumer applies it post-decode through `sirius_mask_applicable::compute_mask`.

Bloom filters are runtime-only — there is no AST node that evaluates "is this hash in this bitset" without a custom kernel.

A channel may carry both a zone map for the reader and a Bloom for the post-decode operator. Each becomes visible only after its device replicas are ready. A direct target normally sees both; a transitive target can safely observe either one or both at its per-split checkpoints because the filters are optional conjuncts and the join remains authoritative.

**Build heuristic: raw exact list, then L2-sized hash set, then Bloom (implemented).** The producer first handles 1–12 null-free INT32/INT64 build rows with a linear scan over raw needles. For larger supported columns, it queries the active GPUs' L2 sizes (`cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)`) and selects the hash IN-list only if its set fits the smallest L2; otherwise it uses the smaller Bloom, even if that bitset also spills L2. This L2 decision subsumed the original fixed row-count hash/Bloom cutover.

### 1.4 IN-list representations (raw needles + hash set)

Both IN-list representations require INT32/INT64 keys **with no nulls** — each `supports()` takes the build `column_view` and rejects `null_count() > 0` — and expose the post-decode `sirius_mask_applicable` runtime path; neither expands keys into an AST. The Bloom's `supports()` tests the key type alone, so a *nullable* INT32/INT64 build key skips both IN-lists and is published as a Bloom. `sirius_dynamic_small_in_list_filter` eagerly copies its input view into an owned raw source-device needle buffer. Its mask kernel performs a linear equality scan for every probe value, so every bit pattern—including the value reserved as the hash set's empty-slot sentinel—is representable. `sirius_dynamic_in_list_filter` instead builds and owns a persistent `cuco::static_set`; its reserved sentinel is conservatively treated as a match to preserve the no-false-negative contract.

For both representations, the constructor enqueues creation of an owned source representation on the source GPU. The caller must keep the build-key backing storage valid until that construction stream completes; the publisher satisfies this precondition by pinning the build representation and synchronizing the stream before replication. After completion, the filter retains no build-key column. Before publication, raw needle bytes or finalized hash-set slots are copied to each additional probe GPU whose target reservation and transfer complete. A replica enters the ready set only after its target stream synchronizes. Every consumer probes only its local read-only representation and passes through if that optional replica is unavailable; teardown selects each replica's owning CUDA device before releasing its storage.

---

## Phase 2 — Sideways information passing (join-edge endpoints)

**Goal:** generalize the *consumer* axis to reach keys no scan can filter. Phase 1 can only prune at a GPU scan the probe-spine trace bottoms out at. Phase 2 places a membership endpoint **inside the producing join's own probe subtree**, at the deepest operator where the probe key is still the same value-preserving column.

**Producer:** `BUILD_PROBE` hash-join build (unchanged).
**Consumer (new placement):** `sirius_physical_dynamic_filter` in `membership_masks_only` mode, spliced into the producer's probe subtree — the same operator Phase 1 puts above a scan, in a new position. No new operator, no new filter kind, no new code path inside the hash-join probe.
**Routing:** the same discovery walk as the scan route -- scan binds and join-edge endpoints are the two terminal actions of one trace per admitted key, and a scan bind wins. Under SIP, a trace that crosses a build block and bottoms out at a GPU scan therefore takes the **scan** route (zone-map capable, the Phase 1 consumer wrap above that scan) when the build-filter evidence gate is armed; the membership endpoint is the outcome when the trace bottoms out anywhere else, or when the producing join's build is unfiltered (the scan gate is disarmed). An endpoint's channel is minted at placement, registered to the producer, and handed to the endpoint directly.
**Flag:** `enable_dynamic_filter_sip`, default off and temporary — it collapses into the master switch when the rollout decision is recorded on [#1010](https://github.com/sirius-db/sirius/issues/1010) against a benchmark protocol registered before any measurement was examined.

### What it reaches that Phase 1 cannot

The canonical shape is `(A join B) join C` where the outer join's key comes from **B**, the inner join's *build* side. At a comparison join, DuckDB's `GetPushdownFilterTargets` descends `children[0]` only -- the probe spine -- so it abandons the push the moment the tracked column comes from a build side, and attaches nothing at all. (Set operations are the one place it fans out into every child, remapping bindings per child; no build side is involved there, so the conclusion is unaffected.) Pushing C's value set onto B wins twice: the inner join builds a smaller hash table, and the key set B later hands to A is drawn from an already-pruned B, hence tighter. Neither win requires new machinery — both fall out of the build-before-probe read order.

### Placement

`planner::place_endpoint` descends from the producing join's probe child, remapping the traced ordinal at each hop, and splices the endpoint above the deepest operator that accepts. The hop rules are a closed, default-free set shared with the scan route's trace: a projection whose output is a plain reference to its input; a `FILTER` (a row predicate over unchanged columns, through its passthrough or gather output); a single-grouping-set `GROUP BY` on a grouping key; a hash join's probe block for `INNER`/`LEFT`/`SEMI`/`ANTI`/`MARK`; a hash join's **build** block for `INNER`/`LEFT`, allowed only under the SIP policy bit; and another endpoint, which is a pure row mask and therefore transparent. Everything else refuses, and the endpoint lands at the floor — the producing join's immediate probe child — which is always available.

Refusals are correctness rules, not conservatism. Value preservation alone is insufficient: a cardinality-selecting operator (`LIMIT`, `TOP_N`) preserves values yet can *add* a result row if rows are removed beneath it, so it refuses. `RIGHT`/`FULL OUTER` refuse because they null-pad the traced block; `MARK`'s build-block ordinal is a synthetic boolean; `SEMI`/`ANTI` emit no build block; `SINGLE` is unimplemented by the GPU join.

Descending into a join's **build** input is sound only because the producing join compares this key with `equal` and never null-equal. Under `LEFT`, pruning a build row can turn a matched row into a NULL-padded one; that row is dropped at the producing join precisely because a NULL key matches nothing under `equal`. Admission enforces the equality rule as the sole guard, and build-side placement depends on it.

### Coordinates and route exclusivity

The endpoint's push ordinal is the descent's **exit** ordinal, in the sited operator's output space — not the probe-key ordinal it started from, which the two share only when no hop was accepted. Push space, store space, and lookup space are all that one ordinal, on both routes.

One route per key, structurally: both routes are terminal actions of the same per-key trace, so a key's terminal is either a scan bind or an endpoint site and no second walk can disagree. A scan bind wins (it applies earlier and can prune row groups); the join-edge route takes only the keys no scan bound. No key is filtered twice.

### Ordering

Under the default `active` task strategy, publication completes on build-batch arrival before any operator in the probe subtree is activated, so an endpoint observes the filter. Under the opt-in `lookahead` strategy that ordering does not hold and an endpoint may run first, observe an empty channel, and pass rows through. The loss is pruning, never correctness: the channel never waits and the producing join stays authoritative.

## Phase 3 — Beyond hash-join producers

**Goal:** generalize the *producer* axis. Operators other than hash-join build can produce dynamic filters when they expose useful properties of their output.

**Candidate producer kinds:**

- **Aggregation / DISTINCT.** A `GROUP BY` exposes its distinct-key set; this is an exact IN-list (or large hash-set) for downstream consumers.
- **Sort.** Post-sort, the min/max of the sort key is exact and available for free — strictly cheaper than the `cudf::reduce` path used by hash-join build.
- **Filter (narrowed).** A filter operator that has been further narrowed at runtime (e.g., by an upstream agg pushing an exact set) can republish the narrowed predicate downstream.

Each new producer type carries its own `probe_target`-shaped struct and its own way of pairing with consumers it does not own. The filter zoo, the channel, and the consumer side are unchanged.

This phase is opportunistic — its value depends on where bottlenecks land after Phases 1 and 2. It is in the design so the producer side is not silently locked to "hash-join only" by Phase 1's choices.

## Phase 4 — Dynamic refinement (speculative)

**Goal:** generalize the *coordination* axis — producers update filters incrementally as they observe more data; consumers wait for finalization or apply progressively-tightening filters as they arrive. Use cases: streaming refinement, cross-pipeline channels with no implicit edge, adaptive runtime predicates.

An incrementally refined producer/consumer pair, or a consumer that requires a filter for correctness, would need an explicit lifecycle protocol (for example, versioned snapshots plus readiness/finalization). Nothing implemented needs one: publication is single-shot, immediate probes and join-edge endpoints under the default task strategy are externally ordered after it, and transitive scan targets treat whatever immutable filters are visible as optional pruning. No Phase 4 work is planned until a concrete use case justifies it.

---

## Open questions

1. ~~**Build/probe publication ordering.**~~ **Resolved with two cases.** For the normal GPU-resident `BUILD_PROBE` path, build-side CONCAT calls the publication hook synchronously and the attempt reaches `FINISHED` before downstream scheduling follows that join into its immediate probe producer. A scan reached transitively through an intervening join is not covered by that edge ordering and observes publication opportunistically under normal scheduler order. Metadata preparation may occur before either case and does not snapshot the channel.
2. **AST-size threshold for multi-zone maps.** Phase 1.2's fallback-to-global threshold should be tuned with a microbenchmark on TPC-H Q14 / Q19 once 1.2 lands.
3. **Bloom build-cost gate.** Phase 1.3 uses Bloom when the hash IN-list estimate exceeds the minimum probe-GPU L2, but a separate lower bound on build cardinality (below which Bloom is dominated by zone-map / IN-list) is unspecified. Candidate: skip Bloom when build cardinality < 10k.

## References

- `src/include/op/dynamic_filter/sirius_dynamic_filter.hpp`, `src/op/dynamic_filter/sirius_dynamic_filter.cpp`, `test/cpp/operator/test_sirius_dynamic_filter.cpp` — framework API, zone-map implementation, channel, and focused tests
- `src/cuda/sirius_dynamic_small_in_list_filter.cu`, `src/cuda/sirius_dynamic_in_list_filter.cu`, `src/cuda/sirius_dynamic_bloom_filter.cu` — raw-needle, hash-set, and Bloom membership filters plus replica construction
- `src/include/op/scan/dynamic_filter_merge.hpp`, `src/op/scan/dynamic_filter_merge.cpp`, `test/cpp/scan/test_dynamic_filter_merge.cpp` — consumer-side merge/apply helpers (`merge_dynamic_filters_into_ast`, `apply_dynamic_filters_to_view`, `apply_dynamic_filters_gated_view`)
- `src/planner/sirius_plan_comparison_join.cpp`, `src/planner/dynamic_filter/dynamic_filter_target_discovery.cpp`, `src/planner/dynamic_filter/build_filter_evidence.cpp` — producer plan-gen wiring, the discovery walk, and the scan-route evidence gate
- `src/planner/dynamic_filter/duckdb_join_filter_candidate_adapter.cpp` — TEST-ONLY parity oracle over DuckDB's join-filter metadata (linked into the test target, not production)
- `src/op/sirius_physical_concat.cpp`, `src/op/dynamic_filter/dynamic_filter_publisher.cpp`, `src/op/sirius_physical_hash_join.cpp` — synchronous build-port publication, filter selection/replication, and fan-out
- `src/include/expression_evaluator/gpu_expression_translator_internal.hpp` — existing AST construction patterns (`cudf::ast::tree::emplace`, scalar lifetime)
- `duckdb/src/optimizer/join_filter_pushdown_optimizer.cpp` — `GetPushdownFilterTargets` / `IsFiltering`, the walks Sirius discovery mirrors and the parity oracle compares against

### Validation map

Which test pins which contract, so a change to one knows where its guard lives:

| Test | Contract it pins |
|---|---|
| `test/cpp/planner/test_build_key_domain.cpp` | The lineage walk admits only shapes whose rows are an injective image of the traced child's, and refuses everything else with domain 0 |
| `test/cpp/planner/test_build_filter_evidence.cpp` | The scan-route evidence gate mirrors DuckDB's `IsFiltering`: GET-with-filters, FILTER, and TOP_N fire, and evidence propagates up through any subtree |
| `test/cpp/planner/test_dynamic_filter_key_admission.cpp` | Admission is Sirius-owned and reads the conditions alone; the coordinate spaces stay distinct; only `equal` with a probe-side reference is admitted |
| `test/cpp/planner/test_dynamic_filter_target_discovery.cpp` | The discovery rules: which hop each operator kind accepts (FILTER and UNION fan-out included), how the traced ordinal is remapped, the SIP policy bit, the producer join-type gate, and that trace and splice agree |
| `test/cpp/planner/test_dynamic_filter_discovery_parity.cpp` | Per-key parity with DuckDB's own `GetPushdownFilterTargets`, with every conservative divergence (LIMIT, TOP_N, cast crossing, joint-bail) asserted on BOTH sides |
| `test/cpp/operator/test_dynamic_filter_source_policy.cpp` | Membership-representation selection and both publication gates, as pure functions with no device |
| `test/cpp/operator/test_dynamic_filter_publisher.cpp` | Publication builds filters only for bound keys, fans out sparsely along bindings, and keeps zone maps out of membership-only targets |
| `test/cpp/planner/test_plan_tree_shape.cpp` | Where the endpoint sits in the finished plan tree, including on a join's build input, and that no endpoint appears when a guard rejects the key |
| `test/cpp/pipeline/test_pipeline_dynamic_filter_native_shape.cpp` | Every endpoint is fed pipelineable data, never a PARTITION's output, on both routes |
| `test/cpp/integration/test_gpu_execution_dynamic_filter_native.cpp` | Scan-route results match CPU exactly, and the coverage gate fires and stays quiet where it should |
| `test/cpp/integration/test_gpu_execution_dynamic_filter_sip.cpp` | Join-edge placement changes no result row, and the publication counters show it is not inert |
