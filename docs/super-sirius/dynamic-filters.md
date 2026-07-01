# Dynamic Filters

A **dynamic filter** is a predicate that is computed at query runtime by one operator (the *producer*) and consumed by another (the *consumer*) to prune data. The producer sees data, learns something about it, emits a filter; the consumer uses that filter to do less work — ideally before paying the cost of materialization.

This is a category, not a single feature. It spans:

- **Dynamic table-filter pushdown** — a hash-join build pushes a filter into a downstream parquet scan, so row-group pruning and post-decode filtering can happen against the actual build-side keys instead of the static predicates.
- **Sideways information passing (SIP)** — a hash-join build pushes a filter into another join's probe input, so the second join can reject probe-side rows that can't possibly match.
- **Aggregation-driven pushdown** — a `GROUP BY` or `DISTINCT` exposes its distinct-value set to downstream consumers.
- **Sort- or top-N-driven pruning** — a post-sort min/max is exact and free; a top-N's current threshold tightens upstream filters.
- **Adaptive runtime predicates** — operators that observe data and refine filters over the lifetime of a pipeline.

This document describes the general dynamic-filter framework in Sirius and the phased plan for delivering it. **The current PR adds the framework scaffolding only — no producers, no consumers, no behavior change.** Phase 1 is the first concrete use case (dynamic table-filter pushdown) and lands in follow-up PRs that plug into the scaffolding introduced here.

## How the phases generalize

The framework has four axes of generality. Each phase opens one axis:

| Axis | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|------|---------|---------|---------|---------|
| Filter kind | zone map → bloom → IN-list | (reuses Phase 1's filter zoo) | (reuses) | (reuses) |
| Consumer kind | parquet metadata scan only | + hash-join probe | + any operator with a column input | (unchanged) |
| Producer kind | hash-join build only | (unchanged) | + agg, sort, filter | (unchanged) |
| Coordination | implicit (meta-pipeline ordering) | explicit readiness for non-implicit pairs | (unchanged) | streaming / incremental refinement |

Anything implemented in Phase 1 is reused unchanged by later phases. We do not replace DuckDB's static table-filter pushdown — static filters continue to flow through the existing translator path and are AND-merged with dynamic filters at the consumer.

## Generalized architecture

The framework has four pieces, all designed to be filter-kind, producer-kind, and consumer-kind agnostic:

1. **`sirius_dynamic_filter`** — polymorphic base class for runtime-computed filters. Each subclass knows how to lower itself to a cuDF AST fragment, to a runtime apply pass, or both.
2. **`sirius_dynamic_filter_set`** — thread-safe append-only channel that connects producers and consumers. Keyed by the consumer's column index in its output schema.
3. **Filter router** *(future)* — keeps a map of channels keyed by a *route key*, so multiple operators can attach to the same channel during plan construction.
4. **Producer / consumer roles** *(future)* — concrete operators that push filters into channels (producers) or read them out (consumers). An operator can be both for different channels.

Pieces 1 and 2 (and the `merge_ast_dynamic_filters_into_tree` helper) are introduced in the scaffolding PR. Pieces 3 and 4 land with the first producer and consumer (Phase 1.1).

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
    sirius_dynamic_zone_map_filter       to_ast Y    apply Y (Phase 1.2 add)
    sirius_dynamic_bloom_filter          to_ast N    apply Y (Phase 1.3 add)
    sirius_dynamic_in_list_filter        to_ast Y    apply Y (Phase 1.4 add)
```

### Filter polymorphism (filter kind axis)

`sirius_dynamic_filter` is the base for every dynamic filter kind. The base carries only the kind tag — concrete consumer-side capabilities (AST lowering today, runtime apply in Phase 1.2) live on separate **capability mixin** interfaces that filters inherit alongside the base. This avoids forcing a filter to implement a path it cannot satisfy: a bloom filter, which has no meaningful AST representation, simply does not inherit from `sirius_ast_lowerable`.

```cpp
class sirius_dynamic_filter {                  // base — kind tag only
 public:
  virtual ~sirius_dynamic_filter() = default;
  [[nodiscard]] virtual sirius_dynamic_filter_kind kind() const = 0;
};

class sirius_ast_lowerable {                   // capability: AST lowering
 public:
  virtual ~sirius_ast_lowerable() = default;
  [[nodiscard]] virtual cudf::ast::expression const& to_ast(
    cudf::ast::tree& tree, cudf::ast::expression const& column_ref) const = 0;
};

class sirius_dynamic_zone_map_filter
  : public sirius_dynamic_filter, public sirius_ast_lowerable { /* both */ };
```

**Consumer-side dispatch** is by `dynamic_cast` from a `sirius_dynamic_filter` pointer to the capability the consumer needs. The `merge_ast_dynamic_filters_into_tree` helper performs the AST-capability cast internally and silently skips filters that lack it, so consumers building an AST tree don't need any per-call-site logic. `sirius_dynamic_filter::kind()` remains for cases where the consumer needs filter-kind-specific behavior beyond what a capability mixin describes (e.g., a parquet reader that natively understands bloom filters via a path other than `apply`).

**AST lowering** — for consumers that build a `cudf::ast::tree` (parquet reader `set_filter`, expression executor). Tree nodes are owned by `tree`; device scalars referenced by literals are owned by the filter instance.

**Runtime apply lowering** *(Phase 1.2)* — for consumers that evaluate a filter against a materialized `cudf::column_view`. Lands as a sibling capability mixin with the same shape.

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

*Introduced in Phase 1.1.* A **producer** holds the channel via a per-target struct that also carries the column-index translation from its own schema to the consumer's:

```cpp
// Phase 1.1, on sirius_physical_hash_join
struct probe_target {
  std::shared_ptr<sirius_dynamic_filter_set> filter_set;
  std::vector<std::size_t> probe_col_idx;   // build-key idx -> consumer col idx
};
std::vector<probe_target> probe_targets;
```

At finalization, the producer iterates over its targets, computes a filter for each join key, and pushes into the corresponding channel.

A **consumer** holds the channel directly via `std::shared_ptr<sirius_dynamic_filter_set> sirius_dynamic_filters` and reads from it during execution.

## Lifetimes and ordering

**Channel lifetime.** The `sirius_dynamic_filter_set` is co-owned via `shared_ptr` by every operator that references it.

**Scalar lifetime.** Scalars referenced by AST literals or apply kernels are owned by the filter instance, which is owned by the set. The set must outlive any AST tree or apply invocation built from filters it contains.

**Producer-consumer ordering.** The consumer must not read filters until every producer has pushed. In Phases 1-3 this is guaranteed implicitly by meta-pipeline structure: the consumer is downstream of the producer in the engine's pipeline graph, so the producer's finalization runs before the consumer's task executes. This is an invariant of the engine, not of the framework. Phase 4 adds explicit readiness signals (`signal_ready` / `wait_ready`) for patterns where this implicit ordering does not hold; Phases 1-3 do not call them.

## Static + dynamic filter mixing at the consumer

A parquet scan's static filter is a DuckDB `Expression` stored on the `parquet_gpu_ingestible`. During the per-file metadata task it is lowered to a cuDF AST via `gpu_expression_translator::translate_expression_with_names`. The translation outcome is what the dynamic-filter merge has to interleave with:

- **No static filter.** The consumer builds an AST tree containing only the dynamic-filter fragments and installs it via `reader_options::set_filter`.
- **Static filter, translation succeeds.** The consumer takes the translated AST, calls `merge_ast_dynamic_filters_into_tree(tree, static_root, set, resolver)` to AND-conjoin every dynamic fragment into it, and installs the resulting root via `set_filter`. Parquet row-group pruning sees the combined predicate.
- **Static filter, translation fails.** The static filter falls through to post-decode evaluation by `gpu_expression_executor`. **AST-path dynamic pushdown is skipped** here — we cannot mix an AST fragment with a DuckDB expression in `reader_options::set_filter`. Phase 1.2's runtime apply path lifts this restriction for filter kinds that support `apply`.

This mixing rule is a property of the consumer (parquet metadata scan), not the framework. Other consumers (hash-join probe in Phase 2) have their own merge rules.

---

## Phase 1 — Dynamic table-filter pushdown

**Goal:** establish the framework end-to-end against a single (producer, consumer) pair — hash-join build → parquet scan — and exercise filter-kind polymorphism via progressively richer filters.

**Producer:** hash-join build side.
**Consumer:** parquet GPU scan (`parquet_gpu_ingestible`).
**Routing:** DuckDB-paired (`DynamicTableFilterSet*` route key).
**Coordination:** implicit (meta-pipeline ordering).

### 1.1 Foundational wiring

Wires the scaffolding into operators end-to-end against a degenerate single-zone (N=1) zone-map filter — equivalent to a global min/max bound. Validates the channel + router + consumer-merge plumbing in a real query.

What this PR adds:

- `sirius_physical_plan_generator::dynamic_filter_channels` map (the router)
- `sirius_physical_table_scan::sirius_dynamic_filters` field (consumer endpoint)
- `sirius_physical_parquet_scan::sirius_dynamic_filters` field (consumer endpoint, propagated from table_scan)
- `sirius_physical_hash_join::probe_target` struct + `probe_targets` vector (producer endpoint)
- Plan-gen wiring in `sirius_plan_get.cpp` and `sirius_plan_comparison_join.cpp`
- Producer-side push in `sirius_physical_hash_join::finalize_operator()`: `cudf::reduce` to compute global (min, max) per join key, emit a single-zone `sirius_dynamic_zone_map_filter`, push into each probe target's channel
- Consumer-side merge in `parquet_gpu_ingestible`: replace the current "Dynamic table filters are not supported" `throw` with a call to `merge_ast_dynamic_filters_into_tree`, AND-conjoin with the static filter, install via `reader_options::set_filter`
- E2E TPC-H test (Q14, Q19) verifying row-group pruning fires

### 1.2 Multi-zone zone maps

Producer keeps per-build-partition bounds rather than reducing to a global min/max. Same filter class (`sirius_dynamic_zone_map_filter`), now with N>1 zones. AST grows to `O(partitions)` per filter; the consumer adds a fallback-to-global threshold when AST size exceeds a budget.

This sub-stage adds the runtime apply path (`sirius_dynamic_filter::apply`) so the next sub-stage (bloom) can plug in. No new filter kind here — the same zone-map filter gains a trivial `apply` implementation.

### 1.3 Bloom filter

`sirius_dynamic_bloom_filter`. Producer builds a GPU bloom over its build keys (sized by build cardinality and a target false-positive rate). Consumer applies post-decode via the 1.2 path.

Bloom filters are runtime-only — there is no AST node that evaluates "is this hash in this bitset" without a custom kernel.

**Build heuristic: only build the bloom if it fits in L2 cache.** A bloom probe is bandwidth-bound — every probed row issues a small number of random loads against the bitset. If the bitset fits in L2, those loads land in L2 and the probe runs at L2 bandwidth; if it spills, every probe pays full HBM bandwidth and the bloom is no longer worth its construction cost. The producer queries the GPU's L2 size at build time (`cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)`) and skips bloom when the sized bitset would exceed it; if the FPR target cannot be met within that budget, it falls back to zone-map / IN-list filters.

### 1.4 IN-list / hash set

`sirius_dynamic_in_list_filter` with both lowering paths:

- **Small set** (< N values, configurable; proposed N = 32): `to_ast` emits `OR(col = v_i)`. AST-pushable, prunes at parquet row-group level.
- **Large set**: `apply` uses `cudf::contains(col, set_column)`. Post-decode only.

The same filter object exposes both paths; the consumer picks based on the AST-size budget.

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

This adds explicit lifecycle methods on the channel (`signal_ready` / `wait_ready` / `ready`) and may version filters so consumers can opt into "latest" semantics. Phases 1-3 do not call any of this. The phase is speculative and included so the channel's lifecycle is not silently locked to "push once at finalization" by earlier phases. No work is planned until a concrete use case justifies it.

---

## Open questions

1. **Meta-pipeline ordering verification.** Phase 1.1's correctness depends on the build pipeline finalizing before the scan task runs. We believe this holds for join → scan pushdown via meta-pipeline structure, but it has not been verified end-to-end. Phase 1.1's E2E test is the validation.
2. **AST-size threshold for multi-zone maps.** Phase 1.2's fallback-to-global threshold should be tuned with a microbenchmark on TPC-H Q14 / Q19 once 1.2 lands.
3. **Bloom build-cost gate.** Phase 1.3 sizes bloom against L2, but a separate lower bound on build cardinality (below which bloom is dominated by zone-map / IN-list) is unspecified. Candidate: skip bloom when build cardinality < 10k.

## References

- `src/include/op/sirius_dynamic_filter.hpp`, `src/op/sirius_dynamic_filter.cpp`, `test/cpp/operator/test_sirius_dynamic_filter.cpp` — framework API, implementation, and tests added by this PR
- `src/include/expression_executor/gpu_expression_translator_internal.hpp` — existing AST construction patterns (`cudf::ast::tree::emplace`, scalar lifetime)
- `duckdb/src/include/duckdb/execution/operator/join/join_filter_pushdown.hpp` — `JoinFilterPushdownInfo`, `JoinFilterPushdownFilter` (consumed by Phase 1.1)
