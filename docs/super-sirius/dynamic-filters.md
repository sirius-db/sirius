# Dynamic Filters

A **dynamic filter** is a predicate that is computed at query runtime by one operator (the *producer*) and consumed by another (the *consumer*) to prune data. The producer sees data, learns something about it, emits a filter; the consumer uses that filter to do less work — ideally before paying the cost of materialization.

This is a category, not a single feature. It spans:

- **Dynamic table-filter pushdown** — a hash-join build pushes runtime membership filters into a downstream parquet scan; an optional zone-map can additionally prune row groups against the actual build-side key range. It is a pure optimization — redundant with the join, so it never changes results. Membership pushdown is **on by default**; zone maps are off by default.
- **Sideways information passing (SIP)** — a hash-join build pushes a filter into another join's probe input, so the second join can reject probe-side rows that can't possibly match.
- **Aggregation-driven pushdown** — a `GROUP BY` or `DISTINCT` exposes its distinct-value set to downstream consumers.
- **Sort- or top-N-driven pruning** — a post-sort min/max is exact and free; a top-N's current threshold tightens upstream filters.
- **Adaptive runtime predicates** — operators that observe data and refine filters over the lifetime of a pipeline.

This document describes the general dynamic-filter framework in Sirius and the phased plan for delivering it. **The framework scaffolding — the `sirius_dynamic_filter` base and the `sirius_dynamic_filter_set` channel — landed in an earlier PR. This PR delivers Phase 1: the first concrete use case (dynamic table-filter pushdown), with a hash-join-build producer, a parquet-scan consumer, and three filter kinds (zone-map, IN-list, bloom).** Membership pushdown is enabled by default and has concentrated wins (Q2 — see *Measured* below); the workload-specific zone-map path remains opt-in. Phases 2–4 below remain design-only.

## How the phases generalize

The framework has four axes of generality. Each phase opens one axis:

| Axis | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|------|---------|---------|---------|---------|
| Filter kind | zone map → bloom → IN-list | (reuses Phase 1's filter zoo) | (reuses) | (reuses) |
| Consumer kind | parquet metadata scan only | + hash-join probe | + any operator with a column input | (unchanged) |
| Producer kind | hash-join build only | (unchanged) | + agg, sort, filter | (unchanged) |
| Coordination | opportunistic (build-side publish) | explicit readiness for non-implicit pairs | (unchanged) | streaming / incremental refinement |

Anything implemented in Phase 1 is reused unchanged by later phases. We do not replace DuckDB's static table-filter pushdown — static filters continue to flow through the existing translator path and are AND-merged with dynamic filters at the consumer.

## Generalized architecture

The framework has four pieces, all designed to be filter-kind, producer-kind, and consumer-kind agnostic:

1. **`sirius_dynamic_filter`** — polymorphic base class for runtime-computed filters. Each subclass knows how to lower itself to a cuDF AST fragment, to a runtime apply pass, or both.
2. **`sirius_dynamic_filter_set`** — thread-safe append-only channel that connects producers and consumers. Keyed by the consumer's column index in its output schema.
3. **Filter router** *(Phase 1.1)* — keeps a map of channels keyed by a *route key*, so multiple operators can attach to the same channel during plan construction.
4. **Producer / consumer roles** *(Phase 1.1)* — concrete operators that push filters into channels (producers) or read them out (consumers). An operator can be both for different channels.

Pieces 1 and 2 (and the `merge_ast_dynamic_filters_into_tree` helper) were introduced in the earlier scaffolding PR. Pieces 3 and 4 are added by this PR, alongside the first producer and consumer (Phase 1.1).

```
   PRODUCERS                     CHANNEL                          CONSUMERS
  ─────────────              ───────────────                     ─────────────
  hash join build            sirius_dynamic_                     parquet metadata
  agg / distinct               filter_set                          scan       (AST)
  sort                                                           hash join probe
  filter (narrowed)          push_filter(col, f)                   (apply)
  …                          filters_for_column(col)             expression exec
                             [ready signal — Phase 4]              (AST)
                                                                 post-decode filter
                                                                   (apply)

  Filter zoo (subclasses of sirius_dynamic_filter):
    sirius_dynamic_zone_map_filter       to_ast Y    apply N
    sirius_dynamic_bloom_filter          to_ast N    apply Y
    sirius_dynamic_in_list_filter        to_ast N    apply Y
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
    cudf::ast::tree& tree, cudf::ast::expression const& column_ref) const = 0;
};

class sirius_dynamic_zone_map_filter
  : public sirius_dynamic_filter,
    public sirius_ast_lowerable,
    public sirius_device_replicable { /* AST consumer + producer replication */ };
```

**Consumer-side dispatch** is by `dynamic_cast` from a `sirius_dynamic_filter` pointer to the capability the consumer needs. The `merge_ast_dynamic_filters_into_tree` helper performs the AST-capability cast internally and silently skips filters that lack it, so consumers building an AST tree don't need any per-call-site logic. `sirius_dynamic_filter::kind()` remains for cases where the consumer needs filter-kind-specific behavior beyond what a capability mixin describes (e.g., a parquet reader that natively understands bloom filters via a path other than `apply`).

**AST lowering** — for consumers that build a `cudf::ast::tree` (parquet reader `set_filter`, expression executor). Tree nodes are owned by `tree`; device scalars referenced by literals are owned by the filter instance.

**Runtime apply lowering** — `sirius_mask_applicable` is the sibling capability for consumers that evaluate a filter against a materialized `cudf::column_view`. IN-list and Bloom implement it; zone maps use the AST path.

### Channel — `sirius_dynamic_filter_set` (decoupling axis)

A `sirius_dynamic_filter_set` is the rendezvous between producers and a consumer. It owns the filters and is co-owned via `shared_ptr` by every operator that holds either end.

Properties:

- **Append-only.** `push_filter(col_idx, f)` adds; nothing removes.
- **Thread-safe.** A mutex guards the underlying map.
- **Keyed by consumer column index.** Producers push for a column index in the *consumer's* output schema. Multiple producers targeting the same column AND-conjoin at the consumer.
- **N producers, M consumers.** The channel does not distinguish between them.
- **Filters are co-owned via `shared_ptr<filter const>`.** Producers may push the same filter object into multiple channels (fan-out — a single bloom filter from a hash-join build can prune both a parquet scan and a downstream join probe without cloning device scalars).

Consumer access is via `filters_for_column(col_idx)` and `filtered_columns()`. The free helper `merge_ast_dynamic_filters_into_tree(tree, existing_root, set, resolver)` walks the channel, lowers every AST-capable filter, AND-conjoins the per-column and cross-column fragments, and returns `AND(existing_root, dynamic_root)` — or `existing_root` unchanged if no filter contributed.

### Filter router — plan-gen channel map (routing axis)

*Introduced in Phase 1.1.* During plan construction, multiple operators must find each other and agree on a shared channel. The router lives on `sirius_physical_plan_generator` and maps a *route key* to a channel:

```cpp
// Phase 1.1
std::unordered_map<
  const duckdb::DynamicTableFilterSet*,
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set>
> dynamic_filter_channels;
```

The route key in Phase 1.1 is `const duckdb::DynamicTableFilterSet*` — DuckDB's optimizer creates a `DynamicTableFilterSet` and references it from both the join's `JoinFilterPushdownInfo::probe_info` and the target `LogicalGet`'s `dynamic_filters`. The pointer is the identity that pairs them.

For Phase 2 (SIP) and Phase 3 (other producers), there is no DuckDB-supplied pointer — Sirius creates the pairing itself, and the route key generalizes to a variant covering Sirius-owned producer/consumer ID pairs. The router's logic is unchanged: find or create a channel for a route key, attach to producer, attach to consumer. Only the key set grows.

### Producer / consumer wiring

*Introduced in Phase 1.1.* A **producer** receives an immutable publication plan. Its per-target entries hold the channel and the column-index translation from the build keys to the consumer:

```cpp
class dynamic_filter_publish_plan {
 public:
  struct probe_target {
    std::shared_ptr<sirius_dynamic_filter_set> filter_set;
    std::vector<std::size_t> probe_col_idx;   // build-key idx -> consumer col idx
  };

 private:
  std::vector<probe_target> _probe_targets;
  std::vector<dynamic_filter_replica_space> _replica_spaces;
};
```

The planner freezes routing, placement, and policy into the hash join's `const dynamic_filter_publish_plan`. The build-port hook normally claims publication as soon as the complete build batch arrives; the hash-table `BUILT` transition is the data-bearing fallback. The producer builds and replicates each filter, then fans it into the accepting channels. Finalization only closes an unclaimed publication window.

A **consumer** holds the channel directly via `std::shared_ptr<sirius_dynamic_filter_set> sirius_dynamic_filters` and reads from it during execution.

## Lifetimes and ordering

**Channel lifetime.** The `sirius_dynamic_filter_set` is co-owned via `shared_ptr` by every operator that references it.

**Scalar lifetime.** Scalars referenced by AST literals or apply kernels are owned by the filter instance, which is owned by the set. The set must outlive any AST tree or apply invocation built from filters it contains.

**Producer-consumer ordering.** The consumer cannot rely on reading *after* the producer pushed: Sirius injects `PARTITION`+`CONCAT` pipelines between the probe scan and the join probe, so the probe scan runs in its own pipeline *concurrently* with the build and is not ordered after the build's finalize (see *Producer/consumer timing* below). Consumption is therefore **purely opportunistic** — each split applies whatever filters are present in the channel when it runs (`has_filters()`, lock-free), and a filter that publishes after a split was read simply does not prune it. Correctness is never at stake because the join is authoritative; only pruning is lost. The build-side producer publish (below) is what makes the filter available early enough to prune most splits without any consumer-side waiting.

## Static + dynamic filter at the consumer

The static and dynamic filters live on **separate, non-interfering paths**, which is what lets the dynamic side be a pure optimization. `parquet_gpu_ingestible` owns the coalesced DuckDB static expression and translates it for reader pushdown; dynamic AST fragments are opportunistically merged during per-split materialization:

- **Static filter** — a `shared_ptr<duckdb::Expression>` (`parquet_gpu_ingestible::_duckdb_filter_expression`), lowered by `gpu_expression_translator::translate_expression_with_names` to a cuDF AST and installed via `reader_options::set_filter` (or, if translation fails, evaluated post-decode by `expression_evaluator`). It is authoritative for correctness and is never touched by the dynamic side.
- **Dynamic filter** — the zone-map is AST-lowerable, so `merge_dynamic_filters_into_ast` (column-name references) AND-merges it onto the reader's filter root, installed via `reader_options::set_filter` (in `parquet_gpu_ingestible::materialize_metadata_to_table`). cuDF's parquet reader uses a `set_filter` predicate both to prune row groups by their statistics *and* to evaluate it row-wise during decode. Membership filters (IN-list / bloom) are not AST-lowerable, so the merge skips them and they are applied **post-decode** by the `sirius_physical_dynamic_filter` operator. Either way the dynamic predicate is a conjunctive *superset* of the join condition, so it never changes which rows survive — the join is authoritative.

The dynamic zone-map AST is merged onto the static filter root when the static filter translated, or built standalone when it did not, so the earlier "translation failed ⇒ dynamic pushdown skipped" restriction no longer applies: the dynamic side contributes regardless of the static filter's translation outcome, and the static filter keeps its own path unchanged.

This mixing rule is a property of the consumer (parquet GPU scan), not the framework. Other consumers (hash-join probe in Phase 2) have their own merge rules.

---

## Phase 1 — Dynamic table-filter pushdown

**Goal:** establish the framework end-to-end against a single (producer, consumer) pair — hash-join build → parquet scan — and exercise filter-kind polymorphism via progressively richer filters.

**Producer:** hash-join build side.
**Consumer:** parquet GPU scan (`parquet_gpu_ingestible`).
**Routing:** DuckDB-paired (`DynamicTableFilterSet*` route key).
**Coordination:** opportunistic (build-side publish; no consumer-side wait).

### 1.1 Foundational wiring

Wires the scaffolding into operators end-to-end against a degenerate single-zone (N=1) zone-map filter — equivalent to a global min/max bound. Validates the channel + router + consumer-merge plumbing in a real query.

Plan-gen / type plumbing:

- `sirius_physical_plan_generator::dynamic_filter_channels` map (the router), gated by `enable_dynamic_filter_pushdown`
- `sirius_physical_table_scan` / `sirius_physical_parquet_scan` `sirius_dynamic_filters` field (consumer endpoint, propagated through the scan binding to `parquet_gpu_ingestible`)
- `dynamic_filter_publish_plan::probe_target` entries plus non-owning GPU replica spaces (producer endpoint, held privately by the hash join)
- Plan-gen wiring in `sirius_plan_get.cpp` and `sirius_plan_comparison_join.cpp`; the join attaches a channel per target

#### Producer/consumer timing — the problem this phase actually had to solve

The naïve producer (push the filter at the hash join's `finalize_operator`) and naïve consumer (read the channel opportunistically at parquet-read time) are **correct but deliver no pruning**, because the producer publishes far too late relative to the concurrently-running probe scan. Two facts, both measured on SF30:

1. The probe scan's footer-parse + split generation runs eagerly at `prepare_for_query`, and its read tasks run in a pipeline that is **not** ordered after the build (injected `PARTITION`/`CONCAT`). With opportunistic consumption, `merged@read = 0` — every split is read before any filter exists.
2. The probe side's `CONCAT` withholds probe batches from the join until the *entire* probe scan finishes, so a build/probe-coupled producer (which needs a probe batch to build its hash table) cannot publish the filter until the scan is already done. Even throttling the probe scan to wait for the filter then **deadlocks the filter on the probe batch it is waiting for** — measured: a wait made the query 12× *slower* with `merged@read` still 0.

Phase 1.1 resolves this with two changes, both reused unchanged by later phases:

**(a) Opportunistic consumption (no consumer-side wait).** The channel exposes only a lock-free `has_filters()`; each split applies whatever filters are present when it runs. An earlier design added a producer-counted readiness signal (`register_producer` / `mark_producer_ready` / `ready`) and a bounded `wait_until_filters` so a scan could throttle until the build published — but that wait was evaluated *per split*, compounded across a large fact scan while a nested-join build had not yet published, and measured catastrophic (+74 % at 150 ms, +154 % at 300 ms on full TPC-H). Since the build-side publish (b) already makes the filter available early enough to prune most splits, the wait bought nothing; it, the `dynamic_filter_wait_ms` knob, and the entire readiness protocol were removed. Consumption is opportunistic only — a late filter costs pruning on the splits already read, never correctness.

**(b) Build-side producer publish — independent of any probe batch.** The fundamental obstacle (fact 2) is that in `BUILD_PROBE` the hash table — and so any filter computed in `execute()` — is built from the *first probe batch*, which the probe-side `CONCAT` withholds until the whole scan finishes. The fix is to compute and publish the filter the moment the **build batch is delivered to the join's build port**, before any probe batch and entirely outside the join's task state machine: `sirius_physical_hash_join::push_data_batch_partitioned` overrides the build-port delivery, and when the (single, concat-folded) build batch arrives for a wired `BUILD_PROBE` join it reduces per-key `(min,max)` on a stream borrowed from the build GPU's memory space and calls `publish_dynamic_filters`. **Publish contract — immutable plan, order, replicate, then publish.** Plan construction freezes the target channels, column mappings, GPU-memory-space placements, NUMA-local HOST staging spaces, and policy in the hash join's `const dynamic_filter_publish_plan`. The pooled stream first waits on the build representation's writer event; the `BUILT` fallback also switches from its worker stream to the build memory space's durable pool. Filter construction is then drained; the producer-only `sirius_device_replicable` capability receives the planned spaces and materializes device-local representations with each space's pooled stream and allocator; and only then does any `push_filter` make the immutable filter visible. The spaces are non-owning: the Sirius memory manager, its allocators, stream pools, and fixed-block HOST resources must outlive the plan, replica-copy operations, and all published replicas. This makes min/max scalars and IN-list/Bloom structures fully computed, cross-stream-visible, and device-local before any consumer can read them; without it a consumer can observe a stale bound or a foreign-device pointer. The drain must cover **zone-map-only** keys too. This is safe because the build side completes independently of the probe scan (the build PARTITION sets `BUILD_PROBE` + enables build-concat-all *before* the build CONCAT folds and delivers). An attempt to instead make the join's *first task* build-only (no probe batch) was reverted — it deadlocks the `task_creator`: the join's `NOT_BUILT` hint returning `WAITING(probe producer)` is what drives probe production via `get_operator_for_next_task`, so short-circuiting it to `READY` parks the creator in `sem_wait`. The build-port hook avoids the scheduler entirely. Publication uses an atomic `OPEN -> PUBLISHING -> PUBLISHED/FAILED` state machine; the `BUILT` transition in `execute()` is the data-bearing fallback, while `on_finalize_operator` changes only an unclaimed `OPEN` window to `CLOSED`. The GPU work never runs under `op_state_mutex`. Replica bytes use direct peer DMA where empirically verified, otherwise they borrow chunked pre-pinned storage from the planned Sirius/CuCascade HOST memory space; the dynamic-filter code performs no direct pinned allocation and does not modify CuCascade. See [dynamic-filters-multi-gpu.md](dynamic-filters-multi-gpu.md) for the replica design and validation.

**(c) Application in the probe scan.** A dynamic join filter is *redundant with the join that produced it*: the join discards every non-matching probe row, so the filter is a conjunctive superset and never has to be exact — its worthwhile effect is to **skip whole row groups** that cannot contain a match. The zone-map rides the parquet reader's filter: `merge_dynamic_filters_into_ast` AND-merges it onto the reader root and `materialize_metadata_to_table` installs it via `reader_options::set_filter`, so cuDF prunes row groups by statistics from that predicate *and* evaluates it row-wise during decode. Membership filters (IN-list / bloom) are not AST-lowerable and ride the **post-decode** apply path (`sirius_physical_dynamic_filter` → `apply_dynamic_filters_to_view`, `membership_masks_only` mode) behind a selectivity gate that disables a non-selective filter after its first applicable batch. The static filter is untouched and remains authoritative.
>
> **Follow-up — zone-map row-GROUP-only.** Merging the zone-map into `set_filter` keeps cuDF's row-group stats-pruning win but also pays a row-level evaluation cost during decode. A row-group-*only* path — evaluate the zone-map against `filter_row_groups_with_stats` and feed only `reader_options::set_row_groups`, never `set_filter` — would drop that row-level cost but is **not yet implemented**. The cost is bounded today: the zone-map is off by default (`enable_dynamic_zone_map_filter`) and TPC-H's scattered keys prune nothing.

> **Redesign note (nsys-driven, SF50).** The original 1.1 applied the dynamic filter at the **row level** — AND-merged into `reader_options::set_filter` *and* a post-decode `compute_column`+`apply_boolean_mask`. nsys + NVTX profiling showed *that* was the TPC-H regression source (not the publish, which is ~1.5 ms): on scattered keys the reader evaluated a predicate that pruned nothing, and the post-decode pass filtered the whole table for ~0 rows. (The earlier "`merged@read=0` ⇒ never applies" reading was an **artifact of grepping `SIRIUS_LOG_DEBUG` apply logs that are suppressed at `info`** — the NVTX `dynfilter::apply_output` range is ground truth: the filter *was* applying and costing time.) The fix moved the membership filters to a *gated* post-decode apply (a non-selective build disables itself after one applicable batch instead of filtering the whole scan) and dropped the unconditional `compute_column`+`apply_boolean_mask`. The zone-map rides `set_filter`, where cuDF's stats-based row-group pruning keeps the prune-at-read win when keys cluster; its residual row-level eval cost is bounded by the off-by-default gate (see the row-GROUP-only follow-up above). The filter never affects results because the join is authoritative.

#### Measured

- **SF50, full TPC-H (22 queries), regression guard — OFF vs ON:** aggregate **+0.2 %** (136.17 s → 136.39 s), i.e. net-zero within the per-query noise floor (~±9 % on the small queries; the large queries are all ≤ ±3 %). **All 22 results bit-identical** OFF vs ON. `pruned=0` on every query — TPC-H's scattered keys (and the leaf-scan-vs-late-build timing) prune nothing, but now that costs nothing. Full unit suite passes.
- **SF30, `lineitem ⋈ keyset` on clustered `l_orderkey`** (the case the zone-map is *for*): the zone-map excludes the non-overlapping row groups, ≈1.6× — preserved by the row-group path. This requires the keyset's narrow range be *runtime*-determined: for a literal/range-derivable build, DuckDB's static transitive pushdown already prunes (measured feature-OFF == ON on SF50), so the dynamic zone-map adds nothing — hence it is gated off by default (`enable_dynamic_zone_map_filter`).

Zone-map pruning only pays off when the build-side join key is range-restricted on a *clustered* fact (e.g. `l_orderkey`) **and** the filter publishes before the probe reads. TPC-H joins on scattered keys (`l_partkey` in Q14/Q17/Q19) prune nothing — those are handled by the membership filters below.

### 1.3–1.4 Membership filters (IN-list + Bloom)

The zone-map captures only the build keys' `[min,max]` range, which is useless for *scattered* keys — even a 0.24%-selective `part` build spans the whole partkey domain (measured). Set **membership** is what distinguishes those keys, so two kinds were added through the `sirius_mask_applicable` capability (a `compute_mask(probe) → BOOL` mask, distinct from `sirius_ast_lowerable`):

- **`sirius_dynamic_in_list_filter`** — exact membership via a persistent `cuco::static_set`, with a device kernel probing its read-only set reference. For *small* selective INT32/INT64 builds.
- **`sirius_dynamic_bloom_filter`** — a `cuco::bloom_filter` (PIMPL'd in a `.cu`; INT32/INT64 keys), a few bits/key, for *large* builds (millions of keys) where an exact IN-list is too big. False positives only let a few extra rows through — harmless, the join is authoritative; no false negatives, so a true match is never dropped.

**Producer policy** (`dynamic_filter_publisher::publish`). The master switch `enable_dynamic_filter_pushdown` emits a **membership filter** per key, chosen by **L2-cache fit**: build the structure whose device footprint fits the GPU's L2 cache, so its random per-row probe runs at L2 bandwidth instead of thrashing HBM. Prefer the exact **IN-list** (`cuco::static_set`, ≈ `16·N` bytes for INT64 and `8·N` for INT32) if it fits L2; else the smaller **Bloom** (≈ `2·N` bytes) whenever the key type supports it — built even if the bitset spills L2, since once the exact set overflows L2 the Bloom is the only viable membership structure; **none** only for key types other than INT32/INT64. Sizes come from `sirius_dynamic_in_list_filter::estimated_set_bytes` / `sirius_dynamic_bloom_filter::estimated_bytes` against the minimum `cudaDevAttrL2CacheSize` of the active probe GPUs, computed from the build row count (an upper bound on distinct keys; exact for the Bloom).

A second switch, `enable_dynamic_zone_map_filter` (default **off**, requires the master), *additionally* emits a **zone-map** (build-key min/max) per key for **read-time row-group pruning** — a complementary consumer path. It is off by default because on TPC-H-shaped joins DuckDB's static transitive-predicate pushdown already prunes range-derivable builds (measured: feature OFF == ON), and scattered keys prune nothing; the zone-map pays off only on clustered-keyset joins whose narrow key range is runtime-determined. The consumer **gates** it (`_zonemap_gate`, mirroring `_membership_gate`): if the first split with row groups prunes none, the per-split stats check is disabled for the rest of the scan.

**Consumer**: membership rides the post-decode `apply_dynamic_filters_to_view` path (`membership_masks_only` mode); the zone-map rides the parquet reader's `set_filter` (cuDF prunes row groups by stats and evaluates it during decode). Because membership is applied *post-decode*, it never saves the scan I/O/decode — only the *downstream* work on dropped rows; so it wins on selective builds feeding expensive downstream and is neutral on scan-dominated single joins.

The cutover is **hardware-adaptive** — it replaced an earlier fixed `2M`-row-count threshold — landing at ≈1.5M INT64 keys on a 24 MB L2, ~2.5M at 40 MB, ~6M at 96 MB. A TPC-H SF50 sweep showed the IN-list↔bloom choice is **wall-clock-neutral** (every query whose kind flips is flat across `[500K, 8M]`; forcing the affected builds all the way to *none* moved them ≤1.5%, within run-to-run noise) — membership is post-decode behind the adaptive gate, so the structure, and even whether one exists, barely moves the clock. The L2 policy was adopted not for an SF50 win but because it is principled: the exact IN-list is used only while it fits L2, otherwise the ~8×-smaller Bloom is built (even when its bitset spills L2, as it stays the only viable membership structure for very large builds). All result sets stay bit-identical across IN-list / Bloom / none.

**Adaptive selectivity gate** (`_membership_gate`): the first applicable batch applies the filter and records the keep ratio (free — `apply_boolean_mask` already sized the output); if it keeps `> 25%` of its rows the gate flips to DISABLED and the rest of that scan skips the membership apply. A GPU without a local best-effort replica does not train the shared gate. A newly published filter re-arms it for one measurement. The applicability check remains lock-free, while the rare post-mask decision update is serialized and re-reads the state under its lock. ACTIVE is terminal, so a stale unselective task on one GPU cannot overwrite a selective decision committed by another.

**Measured (SF50, full TPC-H, ON vs OFF, robust medians):** real wins where a selective build feeds expensive downstream — **Q21 −10.6%** (bloom on F-status `orders`, ~37M keys, drops half of lineitem before its 3-way self-join), **Q2 −7.9%** (IN-list on the selective `part` build). Net suite ≈ **−1.5%** (concentrated on Q21/Q2; the wins are percentage-large but partly on cheap queries). All 22 results bit-identical OFF vs ON; the gate keeps Q21/Q2 ACTIVE and protects non-selective builds. Q21's win is timing-dependent (the bloom must publish before lineitem's self-joins — see the producer/consumer timing problem above). Membership pushdown is now on by default.

> **Cold vs warm.** The numbers above are **warm** (page-cache-resident) medians. A cold/deployment-representative sweep (drop OS cache before each run) shows the wins **largely evaporate** — membership applies *post-decode*, so it cannot cut scan I/O, and the big nominal wins are on I/O-bound queries: full-suite net ≈ **−0.5 % cold** vs ≈ −1.5 / −2.2 % warm. The durable benefit is on small-selective-build, *compute-bound* queries (Q2-shaped); benefit ∝ 1/(how I/O-bound the query is). *Q21 caveat:* a per-query SF50 diagnostic found Q21 wires only a ~20K-key IN-list on the Phase-1 scan-pushdown path (not the ~37M-key `orders` bloom the −10.6 % was attributed to), and its measured swing is within run-to-run noise — a bloom over the F-status `orders` self-join would be a Phase-2 SIP consumer, which does not exist yet.

**Measured (SF300, two GPUs, pinned-host compute regime):** with physical GPUs 1 and 2, grouped execution, five iterations per query, and iteration 0 discarded, the sum of Q1-Q22 warm medians improves from **13.7013505 s to 8.2939465 s: 39.466212% (1.651970x)**. All 22 ON/OFF result files are byte-identical. This is a fixed-two-GPU feature A/B, not a one-to-two-GPU scaling or cold-I/O claim; the exact protocol and per-query table are in [dynamic-filters-multi-gpu.md](dynamic-filters-multi-gpu.md#performance).

*Follow-ups:* wider and variable-width membership keys. (INT32 and INT64 are implemented. The RMM-backed bloom allocator landed in `88a9326`. The per-split consumer wait was removed as a net regression; a *one-shot* readiness wait could be revisited only if Q21's timing-dependent win is ever shown worth making reliable.)

> **Residual cost / tracked optimization.** `filter_row_groups_with_stats` is ~46 ms/call (host work, O(file row groups)) and is invoked per split; the SF50 net-zero shows it is fully absorbed by scan-thread parallelism (the scan is not the TPC-H critical path), but it is still redundant per-split work. A per-file pruning cache (compute once per file, intersect each split host-side — identical results) would remove it; deferred since it adds thread-safety for no measured gain on an off-by-default feature.

#### Configuration

- `enable_dynamic_filter_pushdown` (bool, default **true**) — master switch; when off, the router hands out no channels so neither side wires anything and there is zero overhead. Enabled by default to wire the membership (IN-list / Bloom) filters.
- `enable_dynamic_zone_map_filter` (bool, default **false**, requires the master) — additionally emit a read-time zone-map for row-group pruning. Off by default: measured net-neutral / inert on TPC-H (DuckDB's static pushdown already prunes range-derivable builds and scattered keys prune nothing — feature OFF == ON), so it is reserved for clustered-keyset workloads whose narrow key range is runtime-determined. The consumer gate bounds its cost to one split when it can't prune.
#### Filter availability & late filters (read before Phase 1.3)

Consumption is opportunistic — there is no consumer-side wait (see §1.1 (a)). Two properties keep "a zone-map is present before the bloom arrives" from being a problem once a channel carries **more than one filter kind** (e.g. a zone-map *and* a bloom, Phase 1.3):

1. **Correctness is never affected.** Dynamic filters are conjunctive *superset* predicates — applying a subset (or none) keeps a superset of the rows that can join, and the join does the exact match. A missing or late filter costs *pruning*, never correctness.
2. **The consumer re-snapshots the channel per split, not once.** `merge_dynamic_filters_into_ast` / `apply_dynamic_filters_to_view` run on every split's `execute()` and AND-conjoin *all* filters currently present, so a filter that lands late (a bloom built after the zone-map) is automatically applied to every split read after it arrives — an early filter does **not** lock out a later one. (Because bloom is not `sirius_ast_lowerable`, it rides the post-decode `apply` path while the zone-map rides `set_filter` — they compose, they don't compete.)

### 1.2 Multi-zone zone maps

Producer keeps per-build-partition bounds rather than reducing to a global min/max. Same filter class (`sirius_dynamic_zone_map_filter`), now with N>1 zones. AST grows to `O(partitions)` per filter; the consumer adds a fallback-to-global threshold when AST size exceeds a budget.

This sub-stage adds the runtime apply path (`sirius_dynamic_filter::apply`) so the next sub-stage (bloom) can plug in. No new filter kind here — the same zone-map filter gains a trivial `apply` implementation.

### 1.3 Bloom filter

`sirius_dynamic_bloom_filter`. Producer builds a GPU bloom over its build keys (sized by build cardinality and a target false-positive rate). Consumer applies post-decode via the 1.2 path.

Bloom filters are runtime-only — there is no AST node that evaluates "is this hash in this bitset" without a custom kernel.

This is the first time a channel carries two filter kinds (zone-map at read + bloom post-decode). Because consumption is opportunistic and the consumer re-snapshots the channel per split (see §1.1 *Filter availability & late filters*), the two kinds compose without consumer-side coordination. A producer publishes its zone-map and membership snapshot only after their device replicas are ready; filters from other joins may still arrive later and are picked up by subsequent splits.

**Build heuristic: prefer an exact set only while it fits L2 (implemented).** A membership probe is bandwidth-bound — every probed row issues random loads against the set/bitset. The producer queries the active GPUs' L2 sizes (`cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)`) and applies the policy described above: exact IN-list if its set fits the smallest L2, otherwise the much smaller Bloom (even if its bitset also spills L2), and no filter only for types other than INT32/INT64. This subsumed the original fixed row-count cutover.

### 1.4 IN-list / hash set

`sirius_dynamic_in_list_filter` owns a persistent `cuco::static_set` for INT32/INT64 keys and exposes the `sirius_mask_applicable` runtime path. It is post-decode only: the current implementation deliberately does not expand keys into a large AST. The set's finalized slots are copied to each probe GPU before publication, and each consumer probes only its local read-only replica.

---

## Phase 2 — Sideways information passing

**Goal:** generalize the *consumer* axis. The hash-join probe becomes a consumer, allowing a build-side filter to prune the probe input of a *different* join — essential for star-schema queries where filters from dimension joins should reach the fact-table scan via intermediate joins.

**Producer:** hash-join build (same as Phase 1).
**Consumer (new):** hash-join probe input.
**Routing (new):** Sirius-owned `sirius_sip_route` route key.
**Coordination:** implicit, where the producer's meta-pipeline is upstream of the consumer's; explicit readiness (Phase 4) otherwise.

What changes:

- **Route key variant.** `dynamic_filter_channels` becomes keyed by a variant including a Sirius-owned `sirius_sip_route`. Plan-gen in `sirius_plan_comparison_join.cpp` extends: a join can register itself as a *consumer* of filters from upstream join builds in addition to its existing producer role. Producer/consumer pairing is Sirius's responsibility — no DuckDB pointer to lean on.
- **New consumer code path in hash-join probe.** Before hashing, the probe applies each filter from its incoming channel via `sirius_dynamic_filter::apply`. Reuses Phase 1.2's apply path; no new filter kinds, no AST path (AST is parquet-reader-only).

The filter zoo, the channel type, and the producer side are unchanged.

## Phase 3 — Beyond hash-join producers

**Goal:** generalize the *producer* axis. Operators other than hash-join build can produce dynamic filters when they expose useful properties of their output.

**Candidate producer kinds:**

- **Aggregation / DISTINCT.** A `GROUP BY` exposes its distinct-key set; this is an exact IN-list (or large hash-set) for downstream consumers.
- **Sort.** Post-sort, the min/max of the sort key is exact and available for free — strictly cheaper than the `cudf::reduce` path used by hash-join build.
- **Filter (narrowed).** A filter operator that has been further narrowed at runtime (e.g., by an upstream agg pushing an exact set) can republish the narrowed predicate downstream.

Each new producer type carries its own `probe_target`-shaped struct, registers via plan-gen, and adds a new route-key variant. The filter zoo, the channel, and the consumer side are unchanged.

This phase is opportunistic — its value depends on where bottlenecks land after Phases 1 and 2. It is in the design so the producer side is not silently locked to "hash-join only" by Phase 1's choices.

## Phase 4 — Dynamic refinement (speculative)

**Goal:** generalize the *coordination* axis — producers update filters incrementally as they observe more data; consumers wait for finalization or apply progressively-tightening filters as they arrive. Use cases: streaming refinement, cross-pipeline channels with no implicit edge, adaptive runtime predicates.

This adds explicit lifecycle methods on the channel (`signal_ready` / `wait_ready` / `ready`) and may version filters so consumers can opt into "latest" semantics. Phases 1-3 do not call any of this. The phase is speculative and included so the channel's lifecycle is not silently locked to "push once at finalization" by earlier phases. No work is planned until a concrete use case justifies it. (Phase 1.1 briefly implemented a concrete version — `register_producer` / `mark_producer_ready` / `ready` plus a bounded `wait_until_filters` — to let the probe scan wait for the build; it was removed when the per-split wait proved a net regression and opportunistic consumption proved sufficient. A future refinement would use a *one-shot*, not per-split, wait.)

---

## Open questions

1. ~~**Meta-pipeline ordering verification.**~~ **Resolved (negatively), then closed.** The implicit ordering does *not* hold (probe scan runs concurrently with the build; the probe-side `CONCAT` withholds probe batches until the scan finishes). Phase 1.1 therefore does not rely on ordering: it uses runtime apply on materialized data (cached + post-decode) **and** a build-side producer publish (`push_data_batch_partitioned` build-port hook) that publishes the filter the moment the build side completes — independent of any probe batch — so consumption point 1 (skip a split's read I/O) is reached without any consumer-side wait.
2. **AST-size threshold for multi-zone maps.** Phase 1.2's fallback-to-global threshold should be tuned with a microbenchmark on TPC-H Q14 / Q19 once 1.2 lands.
3. **Bloom build-cost gate.** Phase 1.3 sizes bloom against L2, but a separate lower bound on build cardinality (below which bloom is dominated by zone-map / IN-list) is unspecified. Candidate: skip bloom when build cardinality < 10k.

## References

- `src/include/op/sirius_dynamic_filter.hpp`, `src/op/sirius_dynamic_filter.cpp`, `test/cpp/operator/test_sirius_dynamic_filter.cpp` — framework API, implementation, and tests; the base + channel landed in the scaffolding PR and are extended here with the IN-list/zone-map filter kinds and the `sirius_mask_applicable` capability
- `src/cuda/sirius_dynamic_bloom_filter.cu` — `sirius_dynamic_bloom_filter` (PIMPL'd `cuco::bloom_filter`, INT32/INT64 keys), added by this PR
- `src/include/op/scan/dynamic_filter_merge.hpp`, `src/op/scan/dynamic_filter_merge.cpp`, `test/cpp/scan/test_dynamic_filter_merge.cpp` — consumer-side merge/apply helpers (`merge_dynamic_filters_into_ast`, `apply_dynamic_filters_to_view`, `apply_dynamic_filters_gated_view`), added by this PR
- `src/planner/sirius_plan_comparison_join.cpp`, `src/planner/sirius_plan_get.cpp`, `test/cpp/planner/test_dynamic_filter_router.cpp` — producer/consumer plan-gen wiring and the router, added by this PR
- `src/include/expression_evaluator/gpu_expression_translator_internal.hpp` — existing AST construction patterns (`cudf::ast::tree::emplace`, scalar lifetime)
- `duckdb/src/include/duckdb/execution/operator/join/join_filter_pushdown.hpp` — `JoinFilterPushdownInfo`, `JoinFilterPushdownFilter` (consumed by Phase 1.1)
