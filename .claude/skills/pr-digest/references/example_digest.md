# PR Digest — #1277: Dynamic filters: SIP (sideways information passing) across non-scan probes

> **This digest explains what the PR does and where it lives — it is not a critique or an approval
> recommendation.** Nothing here should be read as "no concerns found"; no concern-hunting was
> performed.
>
> **A note on line numbers:** every `#LNNN` link below points at the file as it stands on the PR
> branch (`kk/issue-1010-dynamic-filter-sip`) at the time this digest was written. Line numbers
> drift as a branch gets rebased/amended — treat them as "accurate as of this digest," not as a
> permanent address.

| | |
|---|---|
| **PR** | [sirius-db/sirius#1277](https://github.com/sirius-db/sirius/pull/1277) |
| **Title** | dynamic filters: SIP |
| **Author** | Kevin Kristensen ([@kevkrist](https://github.com/kevkrist)) |
| **Branch** | `kk/issue-1010-dynamic-filter-sip` → `dev` |
| **Scope** | 59 commits, 100 files, +/- ~roughly evenly split between new planner/runtime code and tests |
| **Closes** | #1010 ("extend pushdown to non-scan probes"), #1125 (SIP consumer roadmap C1a–C4) |
| **Supersedes** | #1179 (design/delivery split), #1193 (abandoned planning-sidecar scaffold), #1244 (R1a — producer-key seam, now folded in), #1267 (R1b — live domain-coverage gate, now folded in) |

## PR description (author's own words)

> Decouples Sirius dynamic-filter routing from DuckDB planner metadata. Sirius now admits
> equality keys and traces each key independently through safe projections, filters, grouping
> keys, join sides, and scan or join-edge endpoints. This enables SIP across nested joins,
> grouped inputs, correlated subqueries, and materialized CTEs while preserving type, null, and
> outer-join boundaries.
>
> Discovery requires either visible build-side filtering or an opaque `CTE_REF`/`DELIM_GET` root
> (optionally projection-wrapped). Unfiltered visible joins and aggregates therefore cannot
> publish large no-op filters, while opaque Q11/Q17 routes remain eligible.
>
> The PR also adds the domain-coverage publication gate, marginal keep-ratio suppression,
> publication counters, consolidated `enable_dynamic_filter` configuration, and end-to-end/
> plan-shape coverage. DuckDB target discovery remains only as a test parity oracle.
>
> A significant part of the diff is file relocation into the dynamic-filter module.

**Plain-language summary:** Phase‑1 dynamic filters only pruned a leaf table scan on the probe
side of a join (build a hash table from the small side, derive a membership/zone-map filter,
apply it at the scan feeding the probe side of the *same* join). This PR lets that filter also
prune a probe input that is itself the output of another join, aggregate, or CTE — i.e. it can
travel *sideways* through the plan tree to reach a fact-table scan that sits several operators
below the join that actually produced the filter.

## Key changes at a glance

1. **[SIP-capable planning pipeline in `plan_comparison_join`](#key-change-1--sip-capable-planning-pipeline-in-plan_comparison_join-ranked-1-this-is-the-feature)** — Three new planner
   components (key admission, target discovery, domain-coverage evidence) decide which join keys
   can publish a filter and walk the probe subtree to find where that filter should attach. When
   the walk lands somewhere other than a table scan, it splices a new filter operator into the
   plan tree — that splice is the SIP feature.
2. **[Publisher decoupled from DuckDB's `JoinFilterPushdownInfo`](#key-change-2--publisher-decoupled-from-duckdbs-joinfilterpushdowninfo-hardened-one-shot-publication-lifecycle)** — Runtime filter publication
   now reads only the Sirius-owned plan built at planning time, not DuckDB's join metadata. The
   publish hook also gained a GPU-replica residency check, out-of-memory handling that fails the
   filter instead of the query, and a counters sink.
3. **[Consolidated `enable_dynamic_filter` setting + shared threshold validator](#key-change-3--consolidated-enable_dynamic_filter-setting--shared-domain-coverage-threshold-validator)** — Renames
   `enable_dynamic_filter_pushdown` to `enable_dynamic_filter` and routes both the YAML and SQL
   `SET` config paths through one validation predicate. This closes a real gap where YAML accepted
   a threshold value that would later fail planning on every GPU hash join.

Roughly 60 further files moved into the new `dynamic_filter/` module — see
[Relocations](#relocations) for which of those are pure moves you can safely skip and which hide
real logic changes.

---

## Key Change 1 — SIP-capable planning pipeline in `plan_comparison_join` (ranked #1: this *is* the feature)

### What it does

Three new planner components, all wired together inside the existing join-planning entry point
[`sirius_physical_plan_generator::plan_comparison_join`](../src/planner/sirius_plan_comparison_join.cpp#L399):

1. **Key admission**
   ([dynamic_filter_key_admission.hpp](../src/include/planner/dynamic_filter/dynamic_filter_key_admission.hpp),
   [dynamic_filter_key_admission.cpp](../src/planner/dynamic_filter/dynamic_filter_key_admission.cpp))
   — decides, per join condition, whether an equality key is statically eligible to become a
   dynamic-filter producer at all (must be plain equality, non-cast/non-computed on both sides, a
   supported build storage type).
2. **Target discovery**
   ([dynamic_filter_target_discovery.hpp](../src/include/planner/dynamic_filter/dynamic_filter_target_discovery.hpp),
   [dynamic_filter_target_discovery.cpp](../src/planner/dynamic_filter/dynamic_filter_target_discovery.cpp))
   — walks the probe subtree *downward* from each admitted key's ordinal through projections,
   filters, `GROUP BY` keys, unions, and join edges to find every place a consumer could attach.
   If the walk reaches a `TABLE_SCAN`, the filter binds there as before (the "scan route"). If it
   terminates somewhere else that safely tracks the key (e.g. above an intermediate join whose
   probe side is another join's output), the code **splices a new `sirius_physical_dynamic_filter`
   consumer operator directly into the probe subtree at that point** (the "direct route") — this
   splice is the SIP mechanism.
3. **Domain-coverage evidence**
   ([build_key_domain.hpp](../src/include/planner/dynamic_filter/build_key_domain.hpp),
   [build_key_domain.cpp](../src/planner/dynamic_filter/build_key_domain.cpp)) — before
   `create_plan` consumes the logical children, walks the *build* side back to its base table and
   records `NodeStatistics::max_cardinality` as a row-count upper bound for the key's column, used
   later by the publisher to gate/suppress filters that wouldn't be selective (see Key Change 2).

### Why (from PR description / issue #1010)

> The highest-value follow-up is sideways information passing: let a build-side filter prune a
> downstream join's probe input, so filters from dimension joins reach the fact table through
> intermediate joins.

Discovery is deliberately conservative:
[`join_block_descent`](../src/planner/dynamic_filter/dynamic_filter_target_discovery.cpp#L94) /
[`descent_steps`](../src/planner/dynamic_filter/dynamic_filter_target_discovery.cpp#L117) only
continue descent through join types that provably don't null-pad or duplicate the traced side —
see
[`probe_block_is_value_preserving`](../src/planner/dynamic_filter/dynamic_filter_target_discovery.cpp#L34)
and
[`build_block_is_value_preserving`](../src/planner/dynamic_filter/dynamic_filter_target_discovery.cpp#L53)
— so semantics (NULLs from outer joins, row multiplication) can't be silently broken by an
inserted filter.

### Where it's invoked

*Provenance tag on every line: **[new]** = function/class didn't exist before this PR,
**[modified]** = existing function whose body/signature this PR changed,
**[existing]** = present before this PR and untouched by it (shown only for call-chain context).*

- DuckDB logical plan (`LOGICAL_COMPARISON_JOIN`) — *(external DuckDB construct, not Sirius code — no tag)*
  - → `sirius_physical_plan_generator::CreatePlan()` [dispatch switch] — **[existing]**, this `case` arm is untouched by the PR
    - → [`plan_comparison_join(op)`](../src/planner/sirius_plan_comparison_join.cpp#L399) — **[modified]**
      - → [`build_key_domain_cardinalities(op, ...)`](../src/planner/sirius_plan_comparison_join.cpp#L423) — **[new]** — domain evidence, gated on `build_evidence`
      - → [`classify_join_key_shapes(op.conditions)`](../src/planner/sirius_plan_comparison_join.cpp#L439) — **[new]** — captures shape before materialization
      - → [`admit_dynamic_filter_keys(...)`](../src/planner/sirius_plan_comparison_join.cpp#L488) — **[new]**
      - → for each admitted key:
        - → [`trace_probe_key(*left, key.probe_key_ordinal, policy)`](../src/planner/sirius_plan_comparison_join.cpp#L506) — **[new]** — scan-route attempt
        - → [`place_endpoint(std::move(left), ...)`](../src/planner/sirius_plan_comparison_join.cpp#L555) — **[new]** — direct-route splice, only if the scan-route attempt found no scan terminal

`plan_comparison_join` is also reached indirectly via
[`sirius_plan_delim_join.cpp:76`](../src/planner/sirius_plan_delim_join.cpp#L76)
(`plan_comparison_join(op)`) — **[existing]** call site, untouched by this PR, but it calls into
the now-**[modified]** `plan_comparison_join` above — so `DELIM_JOIN`-shaped correlated-subquery
plans get the same admission/discovery pass for free. This is what lets SIP reach through
"materialized CTEs" per the PR description.

### Diff — target discovery core (new file, trimmed)

[dynamic_filter_target_discovery.cpp](../src/planner/dynamic_filter/dynamic_filter_target_discovery.cpp)
(new, 325 lines). **Trimmed**: license header and the exhaustive `SiriusPhysicalOperatorType`
switch's ~90 pass-through `case` labels inside
[`descent_steps`](../src/planner/dynamic_filter/dynamic_filter_target_discovery.cpp#L117) are
elided below.

```diff
+std::optional<descent_step> join_block_descent(
+  duckdb::JoinType join_type,
+  std::vector<cudf::size_type> const& probe_block_output_columns,
+  std::vector<cudf::size_type> const& build_block_output_columns,
+  std::size_t output_ordinal,
+  descent_policy policy)
+{
+  auto const probe_block_size = probe_block_output_columns.size();
+  if (output_ordinal < probe_block_size) {
+    if (!probe_block_is_value_preserving(join_type)) { return std::nullopt; }
+    return descent_step{
+      .child_index   = 0,
+      .child_ordinal = static_cast<std::size_t>(probe_block_output_columns[output_ordinal])};
+  }
+  if (!policy.descend_build_blocks) { return std::nullopt; }
+  if (!build_block_is_value_preserving(join_type)) { return std::nullopt; }
+  auto const build_ordinal = output_ordinal - probe_block_size;
+  if (build_ordinal >= build_block_output_columns.size()) { return std::nullopt; }
+  return descent_step{
+    .child_index   = 1,
+    .child_ordinal = static_cast<std::size_t>(build_block_output_columns[build_ordinal])};
+}
+
+// ... descent_steps(): per-operator-type switch selecting the descent_step(s) for
+// PROJECTION / HASH_GROUP_BY / HASH_JOIN / FILTER / UNION / DYNAMIC_FILTER; every other
+// operator type returns {} (i.e. "stop here, this is a terminal") ...
+
+void trace_probe_key_into(sirius::op::sirius_physical_operator& node,
+                          std::size_t ordinal,
+                          descent_policy policy,
+                          std::vector<route_terminal>& terminals)
+{
+  auto const steps = descent_steps(node, ordinal, policy);
+  if (!steps_are_followable(node, steps)) {
+    terminals.push_back(route_terminal{.node = &node, .ordinal = ordinal});
+    return;
+  }
+  for (auto const& step : steps) {
+    trace_probe_key_into(*node.children[step.child_index], step.child_ordinal, policy, terminals);
+  }
+}
+
+endpoint_placement place_endpoint(duckdb::unique_ptr<sirius::op::sirius_physical_operator> subtree,
+                                  std::size_t a0,
+                                  descent_policy policy,
+                                  endpoint_factory const& make_endpoint)
+{
+  assert(subtree != nullptr);
+  auto const steps = descent_steps(*subtree, a0, policy);
+  if (steps_are_followable(*subtree, steps)) {
+    // Ascending child order keeps site ordinals aligned with trace_probe_key() terminals.
+    endpoint_placement result;
+    for (auto const& step : steps) {
+      auto& child_slot = subtree->children[step.child_index];
+      auto placed =
+        place_endpoint(std::move(child_slot), step.child_ordinal, policy, make_endpoint);
+      child_slot = std::move(placed.subtree);
+      result.site_ordinals.insert(
+        result.site_ordinals.end(), placed.site_ordinals.begin(), placed.site_ordinals.end());
+    }
+    result.subtree = std::move(subtree);
+    return result;
+  }
+  auto endpoint = make_endpoint(*subtree);
+  endpoint->children.push_back(std::move(subtree));
+  return endpoint_placement{.subtree = std::move(endpoint), .site_ordinals = {a0}};
+}
```

*(Full new-file diff: `gh pr diff 1277 -- src/planner/dynamic_filter/dynamic_filter_target_discovery.cpp src/include/planner/dynamic_filter/dynamic_filter_target_discovery.hpp`.)*

### Diff — key admission (new file, trimmed to the admission decision)

[dynamic_filter_key_admission.cpp](../src/planner/dynamic_filter/dynamic_filter_key_admission.cpp),
[`admit_scan_route_key`](../src/planner/dynamic_filter/dynamic_filter_key_admission.cpp#L57)
(lines 57–101; the full function is shown, trimmed only inside its body where marked):

```diff
+std::optional<op::dynamic_filter_publish_plan::admitted_key> admit_scan_route_key(
+  sirius::join_condition const& condition,
+  op::dynamic_filter_condition_shape shape,
+  std::size_t condition_index,
+  std::size_t domain_cardinality,
+  std::optional<std::size_t> build_side_unique_column)
+{
+  // Null-equal keys could turn a pruned LEFT join match into an accepted NULL-padded row.
+  if (condition.comparison != sirius::comparison_type::equal) { return std::nullopt; }
+  if (side_blocks_scan_route(shape.probe) || side_blocks_scan_route(shape.build)) {
+    return std::nullopt;
+  }
+  // ... elided: extract build_ref / probe_side references and their cudf storage types ...
+  return op::dynamic_filter_publish_plan::admitted_key{
+    .planner_condition_index      = condition_index,
+    .build_key_ordinal            = build_key_ordinal,
+    .probe_key_ordinal            = probe_key_ordinal,
+    .storage_type                 = *storage_type,
+    .probe_storage_type           = probe_storage_type,
+    .key_shape                    = shape,
+    .build_key_domain_cardinality = domain_cardinality,
+    .build_key_proven_unique =
+      build_side_unique_column == std::optional{static_cast<std::size_t>(build_key_ordinal)}};
+}
```

### Diff — call site wiring in `plan_comparison_join` (trimmed)

[sirius_plan_comparison_join.cpp](../src/planner/sirius_plan_comparison_join.cpp), lines
[483](../src/planner/sirius_plan_comparison_join.cpp#L483)–[555](../src/planner/sirius_plan_comparison_join.cpp#L555).
**Trimmed**: the scan-route binding loop body and its `SIRIUS_LOG_WARN` calls are elided.

```diff
     auto const build_side_unique_column =
       build_side_unique_cols.size() == 1
         ? std::optional<std::size_t>{static_cast<std::size_t>(*build_side_unique_cols.begin())}
         : std::nullopt;
+    auto admitted_keys = admit_dynamic_filter_keys(
+      conditions, condition_key_shapes, condition_domains, build_side_unique_column);
+
+    std::vector<sirius::op::dynamic_filter_publish_plan::probe_target> targets;
+    std::size_t scan_target_count = 0;
+    bool const discovery_runs     = build_evidence &&
+                                op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
+                                !gpu_spaces.empty() && !host_spaces.empty();
+    if (discovery_runs) {
+      bool const scan_bind_armed = scan_route_join_type_admissible(op.join_type);
+      descent_policy const policy{.descend_build_blocks = true};
+      for (std::size_t key_index = 0; key_index < admitted_keys.size(); ++key_index) {
+        auto const& key       = admitted_keys[key_index];
+        auto const& condition = conditions[key.planner_condition_index];
+        auto const terminals =
+          trace_probe_key(*left, static_cast<std::size_t>(key.probe_key_ordinal), policy);
+        // ... elided: bind to a TABLE_SCAN terminal if one is found (scan route) ...
+        if (scan_bound) { continue; }
+        if (!direct_route_admissible(op.join_type, condition.comparison, key.key_shape,
+                                     key.probe_storage_type, key.storage_type)) {
+          continue;
+        }
+        auto placed = place_endpoint(
+          std::move(left), static_cast<std::size_t>(key.probe_key_ordinal), policy,
+          [&](sirius::op::sirius_physical_operator const& site)
+            -> duckdb::unique_ptr<sirius::op::sirius_physical_operator> {
+            auto channel  = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
+            auto endpoint = duckdb::make_uniq<sirius::op::scan::sirius_physical_dynamic_filter>(
+              site.types, site.estimated_cardinality, channel,
+              op_params.dynamic_filter_keep_threshold,
+              sirius::op::scan::dynamic_filter_apply_mode::membership_masks_only);
+            site_channels.push_back(std::move(channel));
+            return endpoint;
+          });
+        left = std::move(placed.subtree);
+      }
+    }
```

---

## Key Change 2 — Publisher decoupled from DuckDB's `JoinFilterPushdownInfo`; hardened one-shot publication lifecycle

### What it does

[`sirius_physical_hash_join`](../src/include/op/sirius_physical_hash_join.hpp) used to require a
`duckdb::JoinFilterPushdownInfo` (DuckDB's own join-filter metadata) at
[construction time](../src/op/sirius_physical_hash_join.cpp#L278) and threw if dynamic filtering
was enabled without it. That parameter, the member `filter_pushdown`, and the class-based
`dynamic_filter_publisher{filter_pushdown, ...}.publish(...)` are all removed. Publication is now
driven entirely by the plan-time `dynamic_filter_publish_plan` built in Key Change 1, consumed by
a new free function
[`sirius::op::publish_dynamic_filters(plan, build_view, stream)`](../src/op/dynamic_filter/dynamic_filter_publisher.cpp#L71).

Alongside the decoupling, the one-shot "publish when the whole build side arrives" hook
([`push_data_batch_partitioned`](../src/op/sirius_physical_hash_join.cpp#L2092)) picked up three
behaviors that didn't exist before:
- a **replica-residency check**
  ([`_dynamic_filter_plan.has_replica_on_device(...)`](../src/op/sirius_physical_hash_join.cpp#L2158))
  before using the build batch as a filter source, instead of assuming any GPU-resident batch is
  usable;
- an **OOM-safe catch** around publication — `rmm::out_of_memory` now fails the *publication*
  (marks state `FAILED`) without failing the *query*, since dynamic filters are advisory;
- a `dynamic_filter_stats*` counters sink threaded through the constructor, incremented at every
  branch of the publication state machine (attempts, skips, domain-gate skips, failures, etc.).

### Why (from PR description / #1244)

> Runtime dynamic-filter publication no longer reads DuckDB's `JoinFilterPushdownInfo`.
> Admitted-key metadata is decided at plan time and carried on `dynamic_filter_publish_plan`; the
> publisher consumes only that... Behavior-preserving for every accepted configuration.

The OOM handling is new behavior, not just a refactor — the code comment at the catch site is
explicit about why it fails the publication rather than retrying:

> FAILED, not reopen: retrying a sibling delivery under the same memory pressure is the storm this
> catch exists to avoid.

### Where it's invoked

- [`push_data_batch_partitioned(port_id="build", batch, partition_idx)`](../src/op/sirius_physical_hash_join.cpp#L2092) — **[modified]**
  (claims the publication slot under `op_state_mutex`, checks `_build_arrives_whole`)
  - → [`publish_dynamic_filters(build_view, publish_stream)`](../src/op/sirius_physical_hash_join.cpp#L2025) [member function] — **[modified]**
    - → [`sirius::op::publish_dynamic_filters(_dynamic_filter_plan, build_view, stream)`](../src/op/sirius_physical_hash_join.cpp#L2035) — **[new]** free function, implemented in [dynamic_filter_publisher.cpp](../src/op/dynamic_filter/dynamic_filter_publisher.cpp); builds membership/zone-map filters from the build table and pushes them into each `probe_target`'s `filter_set`

Both `push_data_batch_partitioned` and the member `publish_dynamic_filters` already existed before
this PR (same names, same call relationship) — what changed is their bodies: the residency check,
OOM handling, and stats sink described above, plus routing to the new free function instead of the
old `dynamic_filter_publisher` class.

`push_data_batch_partitioned` is a pipeline callback invoked once per hash join whenever a batch
lands on its `"build"` port; the *whole-build-in-one-batch* precondition (`_build_arrives_whole`)
is set upstream by the `PARTITION` operator's sizing decision.

### Diff — API surface (new free function replaces the class)

[dynamic_filter_publisher.hpp](../src/include/op/dynamic_filter/dynamic_filter_publisher.hpp) (new file, shown in full):

```diff
+struct dynamic_filter_publication_outcome {
+  std::size_t keys_considered            = 0;
+  std::size_t keys_with_known_domain     = 0;
+  std::size_t keys_build_exceeded_domain = 0;
+  std::size_t skipped_targets_drained    = 0;
+  std::size_t keys_skipped_domain_gate   = 0;
+  std::size_t keys_skipped_type_mismatch = 0;
+  std::size_t membership_filters_built   = 0;
+  std::size_t zone_map_filters_built     = 0;
+  std::size_t active_targets             = 0;
+  std::size_t filters_pushed             = 0;
+};
+
+/**
+ * @brief Builds and publishes filters from a complete hash-join build table
+ * @pre @p plan is enabled
+ */
+[[nodiscard]] dynamic_filter_publication_outcome publish_dynamic_filters(
+  dynamic_filter_publish_plan const& plan,
+  cudf::table_view const& build_view,
+  rmm::cuda_stream_view stream);
```

### Diff — constructor no longer takes DuckDB pushdown metadata (trimmed)

[sirius_physical_hash_join.cpp](../src/op/sirius_physical_hash_join.cpp), constructor starting at
[line 278](../src/op/sirius_physical_hash_join.cpp#L278). **Trimmed** to the signature and the
changed body; unrelated parameters and the join-condition-reordering body in between are elided.

```diff
 sirius_physical_hash_join::sirius_physical_hash_join(
   ...
   duckdb::vector<sirius::logical_type> delim_types,
   std::size_t estimated_cardinality,
-  duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> pushdown_info_p,
   uint64_t max_build_hash_table_bytes,
   dynamic_filter_publish_plan dynamic_filter_plan,
   uint64_t hash_partition_bytes,
-  uint64_t max_broadcast_join_size)
+  uint64_t max_broadcast_join_size,
+  dynamic_filter_stats* dynamic_filter_stats_sink)
   ...
-  filter_pushdown = std::move(pushdown_info_p);
-  if (_dynamic_filter_plan.enabled() && !filter_pushdown) {
-    throw std::invalid_argument(
-      "[sirius_physical_hash_join] An enabled dynamic-filter publication plan requires join "
-      "filter-pushdown metadata");
+  _dynamic_filter_stats = dynamic_filter_stats_sink;
+  if (_dynamic_filter_stats != nullptr && _dynamic_filter_plan.enabled()) {
+    _dynamic_filter_stats->producers_enabled.fetch_add(1, std::memory_order_relaxed);
   }
```

### Diff — OOM-safe, residency-checked publication (trimmed to the new control flow)

[sirius_physical_hash_join.cpp](../src/op/sirius_physical_hash_join.cpp), around
[`publish_dynamic_filters`](../src/op/sirius_physical_hash_join.cpp#L2025) and the residency check
inside [`push_data_batch_partitioned`](../src/op/sirius_physical_hash_join.cpp#L2092) starting near
[line 2155](../src/op/sirius_physical_hash_join.cpp#L2155). **Trimmed**: stats-accumulation and
logging call bodies are elided.

```diff
   try {
-    if (filter_pushdown && _dynamic_filter_plan.enabled()) {
-      dynamic_filter_publisher{
-        *filter_pushdown, _dynamic_filter_plan, key_casts, right_key_col_indices}
-        .publish(build_view, stream);
+    if (_dynamic_filter_plan.enabled()) {
+      auto const outcome =
+        sirius::op::publish_dynamic_filters(_dynamic_filter_plan, build_view, stream);
+      if (_dynamic_filter_stats != nullptr) { /* accumulate outcome.* into stats, relaxed */ }
     }
     _dynamic_filter_publication_state.store(dynamic_filter_publication_state::FINISHED, ...);
+  } catch (rmm::out_of_memory const& oom) {
+    // Dynamic filters are optional; device OOM fails publication without failing the query.
+    // FAILED, not reopen: retrying a sibling delivery under the same memory pressure is the
+    // storm this catch exists to avoid.
+    _dynamic_filter_publication_state.store(dynamic_filter_publication_state::FAILED, ...);
+    SIRIUS_LOG_WARN(... "device memory exhaustion; continuing without filters" ...);
   } catch (...) {
     _dynamic_filter_publication_state.store(dynamic_filter_publication_state::FAILED, ...);
     throw;
   }
```

```diff
-  if (!ms || build_ro->get_current_tier() != ::cucascade::memory::Tier::GPU) { return; }
+  bool const gpu_resident =
+    ms != nullptr && build_ro.get_current_tier() == ::cucascade::memory::Tier::GPU;
+  bool const source_usable =
+    gpu_resident && _dynamic_filter_plan.has_replica_on_device(ms->get_device_id());
+  if (!source_usable) {
+    // ... elided: log which case it was, then reopen the publication slot for another delivery ...
+    std::scoped_lock lg(op_state_mutex);
+    _dynamic_filter_publication_state.store(dynamic_filter_publication_state::OPEN, ...);
+    return;
+  }
```

---

## Key Change 3 — Consolidated `enable_dynamic_filter` setting + shared domain-coverage-threshold validator

### What it does

The YAML/SQL setting `enable_dynamic_filter_pushdown` is renamed to `enable_dynamic_filter`
(single on/off switch for the whole feature, matching the fact that scan-route and direct-route
publication now share one plan-time gate). The `dynamic_filter_domain_coverage_threshold` setting
is validated by a new shared predicate type,
[`config::valid_domain_coverage_threshold`](../src/include/sirius_config.hpp#L78), used by both
the YAML reader and the SQL `SET` handler.

### Why (from PR description / #1244)

> One deliberate exception: the YAML surface now rejects a
> `dynamic_filter_domain_coverage_threshold` that the SQL `SET` surface already rejected. On
> `dev` that value was accepted from YAML and reached a constructor built for every GPU hash
> join, so a non-positive or NaN setting failed planning for every such query.

I.e. this is a genuine bug fix bundled into the rename, not pure renaming — previously the two
entry points (YAML config file vs. `SET dynamic_filter_domain_coverage_threshold = ...`) enforced
different validity rules for the same value, and the looser one (YAML) could pass through a value
that later crashed planning universally.

### Where it's invoked

- YAML config load: [sirius_config.cpp:273](../src/sirius_config.cpp#L273) (`operator_params::from_yaml`) — **[modified]** (existing function; the two `r.optional(...)` lines shown in the diff below changed)
- SQL `SET` handler: [sirius_extension.cpp:2103](../src/sirius_extension.cpp#L2103) (`SetDynamicFilterDomainCoverageThreshold`) — **[modified]** (existing function; now calls the shared `valid_domain_coverage_threshold` predicate instead of an inline check)
- Extension option default: [sirius_extension.cpp:2299](../src/sirius_extension.cpp#L2299) (`AddExtensionOption("enable_dynamic_filter", ...)`) — **[modified]**, renamed from `AddExtensionOption("enable_dynamic_filter_pushdown", ...)`; its callback was also renamed from `SetEnableDynamicFilterPushdown` to `SetEnableDynamicFilter` (same body)
- The validator itself, `config::valid_domain_coverage_threshold` ([sirius_config.hpp:78](../src/include/sirius_config.hpp#L78)) — **[new]** struct, is what both of the above now call into

### Diff

[sirius_config.cpp](../src/sirius_config.cpp), lines
[270](../src/sirius_config.cpp#L270)–[278](../src/sirius_config.cpp#L278):

```diff
   r.optional("enable_runtime_distinct_build_probe", opt.enable_runtime_distinct_build_probe);
-  r.optional("enable_dynamic_filter_pushdown", opt.enable_dynamic_filter_pushdown);
+  r.optional("enable_dynamic_filter", opt.enable_dynamic_filter);
   r.optional("enable_dynamic_zone_map_filter", opt.enable_dynamic_zone_map_filter);
   r.optional("dynamic_filter_domain_coverage_threshold",
              opt.dynamic_filter_domain_coverage_threshold,
-             yaml::greater_than<double>{0.0});
+             config::valid_domain_coverage_threshold{});
```

[sirius_extension.cpp](../src/sirius_extension.cpp), lines
[2103](../src/sirius_extension.cpp#L2103)–[2106](../src/sirius_extension.cpp#L2106) (SQL `SET`
handler reusing the same predicate):

```diff
+  if (!sirius::config::valid_domain_coverage_threshold{}(threshold)) {
+    throw InvalidInputException("dynamic_filter_domain_coverage_threshold %s, got %f",
+                                sirius::config::valid_domain_coverage_threshold::description(),
+                                threshold);
+  }
```

---

## Relocations

Most of this PR's raw diff volume is the dynamic-filter substrate moving into
`op/dynamic_filter/` and `planner/dynamic_filter/`. **Read this section before opening any of
those files** — several are pure moves you can skip entirely, but three of them carry real logic
changes buried in the relocation noise.

**How these were classified:** for each move, old and new content were diffed with comment lines
and blank lines stripped, so only substantive differences survive. Git detected five of these as
renames (with a `similarity index`); the rest appear in the diff as a delete/add pair because
their content changed too much for rename detection, and were paired up by name and reconstructed
from the base branch.

### Pure moves — safe to skip

Content is identical apart from the file's own path, `#include` paths following the move, and
doc-comment reformatting (the PR converts `///` Doxygen blocks to `/** */` and condenses the
prose). No behavioral difference.

| Old path | New path | Note |
|---|---|---|
| `src/include/op/dynamic_filter_device.hpp` | [dynamic_filter_device.hpp](../src/include/op/dynamic_filter/dynamic_filter_device.hpp) | One comment reworded; code byte-identical. |
| `src/include/op/dynamic_filter_replica_reservation.hpp` | [dynamic_filter_replica_reservation.hpp](../src/include/op/dynamic_filter/dynamic_filter_replica_reservation.hpp) | Include path + doc reformat. See the verification note below. |
| `src/include/op/dynamic_filter_replica_space.hpp` | [dynamic_filter_replica_space.hpp](../src/include/op/dynamic_filter/dynamic_filter_replica_space.hpp) | Comment condensing only. |
| `src/include/op/dynamic_filter_replica_transfer.hpp` | [dynamic_filter_replica_transfer.hpp](../src/include/op/dynamic_filter/dynamic_filter_replica_transfer.hpp) | `replica_transfer_route` / `replica_transfer_policy` enums collapsed to one line each; enumerators and their order unchanged. |

**Verification note on `dynamic_filter_replica_reservation.hpp`:** the moved version's docs gained
`@throw std::invalid_argument if @p bytes is zero`, which reads like new behavior. It isn't —
checking `dev` at [line 68](../src/include/op/dynamic_filter/dynamic_filter_replica_reservation.hpp#L68),
the `if (bytes == 0) { throw std::invalid_argument(...) }` guard was already there and the PR only
documented it. Worth stating explicitly, since a newly-appeared `@throw` in a moved file is exactly
the kind of thing that looks like a silent behavior change.

### Renamed, logic intact

- `dynamic_filter_publisher` (class) → [`sirius::op::publish_dynamic_filters`](../src/op/dynamic_filter/dynamic_filter_publisher.cpp#L71)
  (free function). This is a genuine API reshape rather than a rename alone — it's covered in
  detail as **Key Change 2**; listed here only so a reader who searches for the old class name
  knows where it went.

### Moved *with* logic changes — do not skip these

- **`src/op/sirius_dynamic_filter.cpp` → [sirius_dynamic_filter.cpp](../src/op/dynamic_filter/sirius_dynamic_filter.cpp)**
  (git similarity 90%). Three real changes hide in this move:
  1. A new `sirius_dynamic_zone_map_filter::supports(cudf::data_type)` type allowlist that
     excludes floating-point types — this is the PR's "suppress dynamic zone maps for
     floating-point join keys" behavior.
  2. New validation in zone construction: throws `std::invalid_argument` if a zone's min and max
     types differ, if zones don't all share one bound type, or if the bound type isn't in the
     allowlist above. Previously unvalidated.
  3. `_consumer_col_remap` and `set_consumer_column_remap()` were **deleted** — the
     producer-index-to-consumer-column translation they performed is gone, consistent with
     routing now being decided at plan time (Key Change 1). Anyone who depended on that remap
     needs to know it no longer exists.
- **`src/include/op/sirius_dynamic_filter.hpp` → [sirius_dynamic_filter.hpp](../src/include/op/dynamic_filter/sirius_dynamic_filter.hpp)**
  (668 lines deleted, 421 added). The header counterpart of the above; the size drop is mostly the
  removed remap API plus doc condensing, but the declaration changes track the `.cpp` changes
  listed above.
- **`src/include/op/dynamic_filter_publish_plan.hpp` → [dynamic_filter_publish_plan.hpp](../src/include/op/dynamic_filter/dynamic_filter_publish_plan.hpp)**.
  Moved *and* gained three new plan-time types that Key Change 1 depends on:
  `dynamic_filter_key_shape` (`direct`/`cast`/`computed`), `dynamic_filter_condition_shape`, and
  `dynamic_filter_route_class` (`scan`/`direct`).
- **`src/planner/duckdb_join_filter_candidate_adapter.cpp` → [duckdb_join_filter_candidate_adapter.cpp](../src/planner/dynamic_filter/duckdb_join_filter_candidate_adapter.cpp)**
  (git similarity 81%) and its [header](../src/include/planner/dynamic_filter/duckdb_join_filter_candidate_adapter.hpp).
  A 63-line block was deleted in the move: `detail::clone_sirius_filter_pushdown_info` plus the
  `structurally_aligned` / `preserve_aligned` plan-copy helpers. That deletion is what turns this
  adapter from a runtime participant into the test-only parity oracle the PR description mentions.

## Not covered in this pass (flagged, not guessed)

- **`duckdb_join_filter_candidate_adapter`**
  ([duckdb_join_filter_candidate_adapter.hpp](../src/include/planner/dynamic_filter/duckdb_join_filter_candidate_adapter.hpp),
  [duckdb_join_filter_candidate_adapter.cpp](../src/planner/dynamic_filter/duckdb_join_filter_candidate_adapter.cpp),
  new/relocated, version-pinned DuckDB metadata adapter) — per the PR description this now exists
  "only as a test parity oracle." It's exercised by
  [test_duckdb_join_filter_candidate_adapter.cpp](../test/cpp/planner/test_duckdb_join_filter_candidate_adapter.cpp)
  and
  [test_dynamic_filter_discovery_parity.cpp](../test/cpp/planner/test_dynamic_filter_discovery_parity.cpp)
  but doesn't sit on the runtime path documented above. The Relocations section records *what* was
  deleted from it during the move; what parity those tests actually assert, and why the adapter is
  worth keeping alive as dead-weight-looking code, was not traced.
- **Marginal keep-ratio suppression** and the **domain-coverage gate's actual pass/fail
  arithmetic** (mentioned in the PR description, and the source of `keys_skipped_domain_gate` /
  `keys_build_exceeded_domain` in `dynamic_filter_publication_outcome`) live inside
  [dynamic_filter_publisher.cpp](../src/op/dynamic_filter/dynamic_filter_publisher.cpp) but weren't
  traced through in this pass — Key Change 2 shows only the call site and the counters, not the
  gate logic itself.
- The **test files** (roughly half this PR's line count — `test_dynamic_filter_target_discovery.cpp`
  at 713 lines, `test_dynamic_filter_discovery_parity.cpp` at 895, and a dozen more) were not read.
  They're where the admission/discovery edge cases are actually pinned down, so a reader who wants
  to know exactly which join shapes are covered should start there rather than from this digest.
- Within the **Relocations** section, the two largest moved files
  ([sirius_dynamic_filter.cpp](../src/op/dynamic_filter/sirius_dynamic_filter.cpp) and its header)
  were classified by diffing comment-stripped content, which reliably surfaces the logic changes
  listed but would not catch a subtle reordering of otherwise-identical statements.
