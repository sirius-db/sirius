# Track C foundation: PR C1a (version-pinned adapter + immutable publish plan) and PR C1b (strong target types, telemetry, shadow selectivity)

Companion to [issue-1010-dynamic-filter-sip-design.md](../issue-1010-dynamic-filter-sip-design.md); baseline dev 506a1d9f.

PR IDs covered: **C1a**, **C1b**.

---

## 0. Grep-verified enumeration: every non-legacy read of `JoinFilterPushdownInfo` / join `filter_pushdown` / `DynamicTableFilterSet` in `src/`

**A. `JoinFilterPushdownInfo` (producer metadata)**

| # | Site | What it does today |
|---|---|---|
| 1 | `src/include/transparent/sirius_optimizer_extension.hpp:43-68` (doc refs continue to :79) | declares `detail::preserved_counts` (:46-50), `clone_filter_pushdown_info` (:56-57), `preserve_dynamic_filter_metadata` (:64-66) |
| 2 | `src/transparent/sirius_optimizer_extension.cpp:45-61` | clone impl: copies `join_condition`, `build_side_has_filter`, `probe_info` sharing `dynamic_filters` shared_ptr (:54), drops `min_max_aggregates` (:58-59) |
| 3 | `src/transparent/sirius_optimizer_extension.cpp:63-90` | `preserve_dynamic_filter_metadata` parallel walk: GET `dynamic_filters` share (:74-75), join clone (:81-82) |
| 4 | `copy_logical_plan` callers: `sirius_optimizer_extension.cpp:158` (optimizer hook), `src/transparent/physical_sirius_execution.cpp:124` (re-execute), `src/sirius_context.cpp:851` (prepare-time validation copy) | invoke the preservation walk |
| 5 | `src/planner/sirius_plan_comparison_join.cpp:332` | dead `build_key_domain_cardinalities` reads `filter_pushdown->join_condition` |
| 6 | `sirius_plan_comparison_join.cpp:368` | `op.filter_pushdown ? build_key_domain_cardinalities(...)` gate |
| 7 | `sirius_plan_comparison_join.cpp:422,426` | wiring gate: non-null + `build_side_has_filter` |
| 8 | `sirius_plan_comparison_join.cpp:442-443,444,450-458` | iterates `probe_info`, uses `pi.dynamic_filters.get()` as channel key, copies `pi.columns[*].probe_column_index.column_index` / `.storage_type` |
| 9 | `sirius_plan_comparison_join.cpp:514` | `std::move(op.filter_pushdown)` into the physical join (commented-out relics :587,:595) |
| 10 | `src/include/op/sirius_physical_hash_join.hpp:85` (ctor param), `:100` (member) | ownership |
| 11 | `src/op/sirius_physical_hash_join.cpp:33` (include), `:214,228` (store), `:229-233` (ctor invariant enabled-plan⇒pushdown), `:1333` (claim `filter_pushdown && _dynamic_filter_plan.enabled()`), `:1335` (publisher ctor arg), `:1364` (claim inside `push_data_batch_partitioned`) | the runtime deref C1a removes |
| 12 | `src/include/op/dynamic_filter_publisher.hpp:19` (include), `:43,47,58` (`_filter_pushdown` ref member) | publisher holds a live DuckDB ref |
| 13 | `src/op/dynamic_filter_publisher.cpp:121,123,124` (per-key vectors sized by `join_condition.size()`), `:159-172` (key loop: `cond_idx` indexes `_key_casts`; cast skip :163-171; index skip `cond_idx >= _right_key_col_indices.size()` :172 — this is what drops range-comparison ordinals today), `:337` (summary log key count) | runtime iteration over DuckDB metadata |
| 14 | `src/include/op/sirius_physical_nested_loop_join.hpp:59,101`; `src/op/sirius_physical_nested_loop_join.cpp:126,146` | **dead**: the only NLJ constructions (`sirius_plan_comparison_join.cpp:600-607`, `test/cpp/operator/test_physical_mark_join.cpp:151`) use the projection-map overload; pushdown is never passed |

**B. `DynamicTableFilterSet` (channel identity)**

| # | Site | What it does today |
|---|---|---|
| 15 | `src/include/planner/sirius_physical_plan_generator.hpp:32,80-91` + `src/planner/sirius_physical_plan_generator.cpp:54-65` | router: pointer-keyed channel map, central enable gate (:61) via `dynamic_filter_pushdown_enabled` (:39-44) |
| 16 | `src/planner/sirius_plan_get.cpp:264-268` | scan side: copies `op.dynamic_filters`, creates channel by pointer key, logs raw address |
| 17 | `src/include/op/sirius_physical_table_scan.hpp:100-103` (member, assigned only at #16) | identity/lifetime anchor |
| 18 | `src/include/op/sirius_physical_parquet_scan.hpp:84-87` + `src/op/sirius_physical_parquet_scan.cpp:65` | copies the anchor from the table scan |
| 19 | `src/include/op/sirius_physical_duckdb_scan.hpp:79` | **dead**: never assigned nor read anywhere in non-legacy `src/` |

**Excluded name collisions** (not part of this contract): `src/sirius_extension.cpp:1321` (`TableFunction::filter_pushdown` static-filter capability flag); `src/op/scan/parquet_gpu_ingestible.cpp` `disable_filter_pushdown` (:116-124,193-207,479-497,559,637,704-727 + `.hpp:125,202`) — reader-side static-filter flag; everything under `src/legacy/` and `src/include/legacy/`.

**Verified index-space fact underpinning the adapter.** Pinned DuckDB calls `PhysicalComparisonJoin::ReorderConditions(join.conditions)` *before* recording `join_condition` (`duckdb/src/optimizer/join_filter_pushdown_optimizer.cpp:224,249-250`); that reorder is equality-first stable with equality = `COMPARE_EQUAL|COMPARE_NOT_DISTINCT_FROM` (`duckdb/src/execution/operator/join/physical_comparison_join.cpp:34-73`), semantically identical to Sirius's `reorder_join_conditions` (`sirius_physical_hash_join.cpp:171-202`, `is_equality` :79-83). **DuckDB pushes both equality and range comparisons** into `join_condition` — `COMPARE_EQUAL` plus `COMPARE_LESSTHAN/LESSTHANOREQUALTO/GREATERTHAN/GREATERTHANOREQUALTO` are admitted (`join_filter_pushdown_optimizer.cpp:230-236`); only `COMPARE_NOT_DISTINCT_FROM` is skipped (:237-239). Consequences: (a) every recorded `cond_idx` is a valid index into Sirius's post-reorder `conditions`; (b) equality conditions form a prefix, so `key_casts`/`right_key_col_indices` (populated per equality condition, ctor :284-340) are validly indexed by `cond_idx` **iff** `cond_idx < right_key_col_indices.size()` — range ordinals land past the equality prefix and are exactly the ones today's runtime skip at `dynamic_filter_publisher.cpp:172` drops, while their targets are still wired at plan time (site #8). Producer join types: DuckDB admits INNER/RIGHT/SEMI (`join_filter_pushdown_optimizer.cpp:207-219`); RIGHT/RIGHT_SEMI/RIGHT_ANTI and MIXED_JOIN never enter `BUILD_PROBE` (`sirius_physical_hash_join.cpp:445-448`, `update_join_exec_mode`), and eq+range joins run as MIXED_JOIN (:344-346) — so today's *claiming* producers are exactly {INNER, SEMI} pure-equality BUILD_PROBE joins. The design's INNER-or-SEMI claim assertion is behavior-preserving; the plan-time wiring set (which includes RIGHT and eq+range producers) must not narrow.

**Verified dead-signal fact.** `build_key_domain_cardinalities` (`sirius_plan_comparison_join.cpp:329-352`) requires `BOUND_COLUMN_REF` keys (:338), but it runs after `ColumnBindingResolver` (`sirius_physical_plan_generator.cpp:126-132` always precedes dispatch; every entry path funnels through it — `sirius_context.cpp:856,861`, `physical_sirius_execution.cpp:148-149`, `sirius_extension.cpp:536`, `sirius_ffi.cpp:171`), where keys are `BoundReferenceExpression` (the join ctor asserts BOUND_REF/BOUND_CAST, :304-337). So domains are always 0 and the publisher gates at `dynamic_filter_publisher.cpp:177` and `:214` never fire. Replacing them with `std::optional` + shadow logging is a no-behavior-change.

---

# PR C1a — version-pinned adapter + immutable publish plan

## Goal

Concentrate every pinned-DuckDB metadata read behind one version-pinned `duckdb_join_filter_candidate_adapter` (preservation + extraction entry points), snapshot the producer's routing/key decisions into immutable Sirius values at plan/construction time, and rewrite the publication claim and publisher so runtime never dereferences `JoinFilterPushdownInfo` (design "Version-pinned adapter" :296-338, "Immutable Sirius plan" :782-817).

**Gate promise** (design phasing row :962): same admitted producers/keys, scan targets, materialization choices, channel lifecycle, and results under fixed config.

**Non-goals**: per-key fan-out that changes target behavior (C1c), enforcing the repaired selectivity gate (C1d — enforcement-only, lives in the C1cde cluster and depends on C1b), removing the `build_side_has_filter` wiring gate (C1e), SIP targets/lineage/registry (C3), consumer/mask work (C2), scheduler changes (A-track), the DuckDB pin bump (B1). No new config flags — observationally idempotent.

## Deliverables

**New `src/include/op/dynamic_filter_identity.hpp`** (shared with A1 — see Dependencies):

```cpp
namespace sirius::op {
// Query-relative monotonic IDs (design "Publication, target, channel, and filter identity").
// Assigned from per-plan-generator counters; exactly one constructed plan executes per query,
// so log consumers may treat them as query-relative.
struct dynamic_filter_publication_plan_id { std::uint32_t value = 0; };
struct dynamic_filter_target_id           { std::uint32_t value = 0; };
// channel_id / filter_id slots reserved for A1; add here, not elsewhere.
struct admitted_dynamic_filter_key {          // design :330-335, three index spaces
  std::size_t duckdb_filter_ordinal;  // j in join_condition and probe_info[t].columns
  std::size_t condition_index;        // DuckDB-reordered == Sirius-reordered condition index
  std::size_t sirius_key_ordinal;     // compact ordinal after Sirius narrowing
};
}
```

**New `src/include/planner/duckdb_join_filter_candidate_adapter.hpp` / `src/planner/duckdb_join_filter_candidate_adapter.cpp`:**

```cpp
namespace sirius::planner {

enum class duckdb_candidate_kind : std::uint8_t { absent, statistics_only, admitted, malformed };

struct duckdb_probe_target_candidate {
  duckdb::DynamicTableFilterSet const* channel_key;   // opaque identity; never dereferenced
  struct probe_column { std::size_t column_index; duckdb::LogicalType storage_type; };
  std::vector<probe_column> columns;                   // validated: size == condition_indexes.size()
};

struct duckdb_join_filter_candidate {
  duckdb_candidate_kind kind = duckdb_candidate_kind::absent;
  bool build_subtree_has_filter_hint = false;          // snapshot of build_side_has_filter (design :292)
  // FULL DuckDB arity: every join_condition ordinal is kept, equality or not. DuckDB pushes
  // range comparisons too (join_filter_pushdown_optimizer.cpp:230-236); non-equality ordinals
  // are narrowed per key at finalization (the mirror of dynamic_filter_publisher.cpp:172),
  // never rejected structurally. Validated in-range and unique.
  std::vector<std::size_t> condition_indexes;
  // Per-ordinal comparison snapshot aligned with condition_indexes (telemetry/debug-assert use
  // only; lets C1b label "not admitted: non_equality" vs "not admitted: cast").
  std::vector<duckdb::ExpressionType> condition_comparisons;
  std::vector<duckdb_probe_target_candidate> targets;  // null-channel targets dropped individually
};

struct preserved_counts { std::size_t joins = 0; std::size_t gets = 0; };  // moved from transparent::detail

class duckdb_join_filter_candidate_adapter {
public:
  // Entry point 1: preservation while the optimized logical plan is copied (design :301,305-313).
  static void preserve_dynamic_filter_metadata(duckdb::LogicalOperator& original,
                                               duckdb::LogicalOperator& copy,
                                               preserved_counts& counts);
  [[nodiscard]] static duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo>
    clone_filter_pushdown_info(duckdb::JoinFilterPushdownInfo const& src);

  // Entry point 2: extraction post-resolver, before create_plan(children)/condition wrap (design :302-303,315-325).
  [[nodiscard]] static duckdb_join_filter_candidate extract(duckdb::LogicalComparisonJoin const& op);

  // Scan-side identity read (completes "one adapter owns every read").
  [[nodiscard]] static duckdb::shared_ptr<duckdb::DynamicTableFilterSet>
    scan_channel_identity(duckdb::LogicalGet const& get);
};
}
```

**Changed `src/include/op/dynamic_filter_publish_plan.hpp` (+ .cpp):**

```cpp
struct dynamic_filter_key_plan {                       // design :787-792, embedded key superset
  admitted_dynamic_filter_key key;
  cudf::size_type build_column_index;                  // resolved right_key_col_indices[condition_index]
  cudf::data_type build_type;                          // EMPTY == resolve from runtime column (fallback)
  std::optional<std::size_t> build_key_domain_cardinality;  // always nullopt in C1a; shadow value in C1b
};

class dynamic_filter_publish_plan final {
public:
  struct probe_target {                                // unchanged shape in C1a + id
    dynamic_filter_target_id target_id;
    std::shared_ptr<sirius_dynamic_filter_set> filter_set;
    std::vector<std::size_t> probe_col_idx;            // full DuckDB arity, indexed by duckdb_filter_ordinal
    std::vector<cudf::data_type> probe_col_type;
  };
  dynamic_filter_publish_plan() = default;
  dynamic_filter_publish_plan(dynamic_filter_publication_plan_id id,
                              std::vector<probe_target> probe_targets,
                              bool emit_zone_map_filters,
                              std::size_t duckdb_key_count,                    // join_condition.size() snapshot (log parity)
                              std::vector<dynamic_filter_replica_space> replica_spaces,
                              double domain_coverage_threshold = k_default_domain_coverage_threshold);
  // Exactly-once finalization by the owning join's ctor after condition reorder/key extraction:
  void finalize_keys(std::vector<dynamic_filter_key_plan> keys);              // throws if called twice
  [[nodiscard]] bool enabled() const noexcept;                                // unchanged: !_probe_targets.empty()
  [[nodiscard]] std::span<dynamic_filter_key_plan const> keys() const noexcept;
  [[nodiscard]] std::size_t duckdb_key_count() const noexcept;
  [[nodiscard]] dynamic_filter_publication_plan_id id() const noexcept;
  // Producer eligibility used by the claim's release-mode fail-closed check (design :342-354,816).
  // Join-type predicate ONLY — zero admitted keys is a legitimate state (all keys cast-skipped or
  // non-equality) in which the publisher still runs and emits its terminal summary line.
  [[nodiscard]] bool validate_producer_eligibility(duckdb::JoinType join_type) const noexcept;
  // existing accessors kept: probe_targets(), emit_zone_map_filters(), replica_spaces(), domain_coverage_threshold()
  // REMOVED: build_key_domain_cardinalities()
};

// Fresh pure predicate, NOT derived from prove_unique_columns' switch (design :356-359):
[[nodiscard]] constexpr bool is_dynamic_filter_producing_join_type(duckdb::JoinType t) noexcept
{ return t == duckdb::JoinType::INNER || t == duckdb::JoinType::SEMI; }
```

**Changed `src/include/op/dynamic_filter_publisher.hpp`:**

```cpp
class dynamic_filter_publisher final {
public:
  explicit dynamic_filter_publisher(dynamic_filter_publish_plan const& plan) : _plan(plan) {}
  void publish(cudf::table_view const& build_view, rmm::cuda_stream_view stream) const;
private:
  dynamic_filter_publish_plan const& _plan;   // ONLY member; JoinFilterPushdownInfo/key_casts/right_key_col_indices gone
};
```

**Changed `sirius_physical_hash_join`** (`src/include/op/sirius_physical_hash_join.hpp`): drop the `duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> pushdown_info` ctor parameter (:85) and the `filter_pushdown` member (:100); `_dynamic_filter_plan` loses `const` qualification (:254) but is finalized exactly once in the ctor and never mutated afterward (documented invariant). Add a public const accessor (test/introspection surface — `_dynamic_filter_plan` is `protected`, and today only `publishes_dynamic_filters()` at :167-170 is public):

```cpp
[[nodiscard]] dynamic_filter_publish_plan const& dynamic_filter_plan() const noexcept
{ return _dynamic_filter_plan; }
```

## Step-by-step changes

**Step 1 — adapter module (new files + move preservation).**
- Create `src/include/planner/duckdb_join_filter_candidate_adapter.hpp` and `src/planner/duckdb_join_filter_candidate_adapter.cpp` (add the .cpp to the library sources next to `src/op/dynamic_filter_publish_plan.cpp` at `CMakeLists.txt:282`).
- Move the bodies of `clone_filter_pushdown_info` (`src/transparent/sirius_optimizer_extension.cpp:45-61`) and `preserve_dynamic_filter_metadata` (`:63-90`) into the adapter verbatim (pointer-identity share at :54 and :75 is the load-bearing invariant, design :305-313). `src/transparent/sirius_optimizer_extension.cpp` keeps `copy_logical_plan` (:94-103) delegating to the adapter; delete the entire `detail` namespace block from `src/include/transparent/sirius_optimizer_extension.hpp:43-68` (`preserved_counts` :46-50 moves to the adapter header, clone decl :56-57, preserve decl :64-66) and update the doc comments through :79. Callers `sirius_optimizer_extension.cpp:158`, `physical_sirius_execution.cpp:124`, `sirius_context.cpp:851` are signature-stable — no edits beyond includes.
- Implement `extract(duckdb::LogicalComparisonJoin const&)`:
  - null `filter_pushdown` → `absent`; non-null with empty `probe_info` → `statistics_only` (design :270-272);
  - **whole-candidate `malformed` is reserved for structural corruption only**: an out-of-range `condition_indexes` entry (`>= op.conditions.size()`), a duplicate entry, or a probe target whose `pi.columns.size() != join_condition.size()` (arity corruption breaks the ordinal alignment for every consumer; unreachable under the pinned DuckDB, which pushes `pushdown_columns` and `join_condition` in lockstep, `join_filter_pushdown_optimizer.cpp:243-250` — see Risks for the honest consequence if it ever fired);
  - **non-equality pushed conditions are NOT malformed**: all `join_condition` ordinals are kept at full arity with their `condition_comparisons` snapshot; range ordinals are narrowed per key at finalization (Step 3), reproducing today's runtime skip at `dynamic_filter_publisher.cpp:172`. This preserves plan shape for eq+range joins (targets wired, channels registered, scan operators inserted) exactly as today;
  - per-target validation dropping only that target: null `pi.dynamic_filters` (design :317);
  - copies out values only: `column_index` = `col.probe_column_index.column_index`, `storage_type` as `duckdb::LogicalType` value (the cudf conversion with the EMPTY fallback stays in the planner, mirroring `sirius_plan_comparison_join.cpp:452-457`).
- Implement `scan_channel_identity(LogicalGet const&)` returning `get.dynamic_filters` (trivial; centralizes read #16).

**Step 2 — planner rewiring, `src/planner/sirius_plan_comparison_join.cpp`.**
- Delete the dead `build_key_domain_cardinalities` (:325-352) and its call/gate (:367-368). Keep `trace_binding_to_get` (:280-323) — move it to the adapter .cpp now as a file-static kept for C1b's preservation-time domain walk.
- Replace the direct `op.filter_pushdown` block (:420-496): call `auto candidate = duckdb_join_filter_candidate_adapter::extract(op);` **before** `create_plan(*op.children[...])` (:372-373, same constraint as today's :365-368 comment). Wiring predicate stays byte-equivalent: `candidate.kind == admitted && candidate.build_subtree_has_filter_hint` plus the GPU/HOST-space check (:432-440). **Log-emission equivalence**: the "build side is unfiltered" INFO (:426-430) today fires for *any* non-null `filter_pushdown` with `build_side_has_filter == false`, including probe_info-empty candidates — key it on `candidate.kind != absent && !candidate.build_subtree_has_filter_hint` (NOT on `kind == admitted`) so `statistics_only` candidates keep their line. Keep the GPU/HOST INFO (:438-440) and the wired-targets INFO (:488-492) as-is. Channel creation loop (:442-461) now iterates `candidate.targets`, calling `get_or_create_dynamic_filter_channel(tgt.channel_key)` (:444 unchanged router) and `channel->register_producer()` (:446) — **unchanged for all DuckDB-admitted producers incl. RIGHT joins and eq+range MIXED_JOIN joins**, because `register_producer()` feeds `has_producers()` which controls scan-operator insertion (`src/pipeline/sirius_pipeline_converter.cpp:277-279,294-304`).
- Assign `dynamic_filter_target_id`s from a per-generator monotonic counter (new `std::uint32_t next_dynamic_filter_target_id_ = 0;` + `next_dynamic_filter_publication_plan_id_` members beside the channel map, `src/include/planner/sirius_physical_plan_generator.hpp:80-91`; a generator instance is per-plan-construction — `sirius_context.cpp:843`, `physical_sirius_execution.cpp:148` — and exactly one constructed plan executes per query, making the IDs query-relative for log consumers).
- Construct the plan with the new ctor: `{plan_id, targets, op_params.enable_dynamic_zone_map_filter, candidate.condition_indexes.size(), replica_spaces, op_params.dynamic_filter_domain_coverage_threshold}` (replaces :497-502; `build_key_domains` argument gone; `duckdb_key_count` = full `join_condition.size()` always, including non-equality ordinals).
- Hash-join construction (:504-516): remove the `std::move(op.filter_pushdown)` argument (:514). Pass `candidate.condition_indexes` (+ `condition_comparisons`) into the plan draft as a private "pending key candidates" field `std::vector<std::size_t> _pending_condition_indexes` cleared by finalization.

**Step 3 — hash-join ctor finalization, `src/op/sirius_physical_hash_join.cpp`.**
- Ctor (:204-233): drop `pushdown_info_p` param, delete `:228` and the invariant `:229-233`. After key extraction completes (:284-340), call a new private static
  `resolve_dynamic_filter_keys(std::span<std::size_t const> condition_indexes, std::span<key_cast_info const> key_casts, std::span<cudf::size_type const> right_key_col_indices, duckdb::vector<sirius::logical_type> const& rhs_types) -> std::vector<dynamic_filter_key_plan>`
  which, per DuckDB ordinal `j` with `cond_idx = condition_indexes[j]`, reproduces today's runtime skips exactly: skip (not-admitted) if `cond_idx >= right_key_col_indices.size()` (mirror of `dynamic_filter_publisher.cpp:172`; this is the branch that drops range-comparison ordinals — they sit past the equality prefix); skip if `key_casts[cond_idx].cast_left || .cast_right` (mirror of :163-171, keep the DEBUG log text incl. `k`/`cond_idx`); otherwise emit `{ {j, cond_idx, compact_ordinal++}, right_key_col_indices[cond_idx], build_type, std::nullopt }` where `build_type = get_cudf_type(rhs_types[right_key_col_indices[cond_idx]])` wrapped in try/catch → `EMPTY` (runtime resolves from `col.type()`). Debug-assert (never fires under the pinned DuckDB): any `cond_idx < right_key_col_indices.size()` refers to an equality condition. Then `_dynamic_filter_plan.finalize_keys(std::move(keys))`.
- Claim rewrite (:1358-1367): condition becomes
  ```cpp
  claim = _dynamic_filter_publication_state.load(...) == OPEN &&
          _join_mode == HASH_JOIN_MODE::BUILD_PROBE &&
          _dynamic_filter_plan.enabled();
  ```
  (the removed `filter_pushdown &&` is redundant today given the ctor invariant at :229-233 — provably equivalent). Inside `publish_dynamic_filters` (:1332-1337): release-mode fail-closed eligibility check **scoped strictly to the join-type predicate** —
  ```cpp
  if (_dynamic_filter_plan.enabled()) {
    if (!_dynamic_filter_plan.validate_producer_eligibility(join_type)) {
      SIRIUS_LOG_ERROR("[sirius_physical_hash_join] dynamic-filter eligibility violated at claim "
                       "(join_type={}); skipping publication.", ...);
    } else {
      dynamic_filter_publisher{_dynamic_filter_plan}.publish(build_view, stream);
    }
  }
  ```
  `validate_producer_eligibility(join_type)` = `is_dynamic_filter_producing_join_type(join_type)` — genuinely unreachable today (only INNER/SEMI reach a claimed BUILD_PROBE publish, §0). **The publisher runs even when `keys()` is empty** (all keys cast-skipped or non-equality): the key loop no-ops and the terminal INFO line is still emitted with its `duckdb_key_count`, preserving the log-diff gate's multiset identity — today the publisher runs and prints "Pushed 0 dynamic filter(s) …" in that state (`dynamic_filter_publisher.cpp:330-337`). The FINISHED/FAILED state transitions (:1338-1344) are unchanged. Also update :1333 (`if (filter_pushdown && ...)` → `if (_dynamic_filter_plan.enabled())`). Remove include :33 if now unused.
- Header (`src/include/op/sirius_physical_hash_join.hpp`): delete member :100, ctor param :85; `_dynamic_filter_plan` (:254) drops `const`; add the public `dynamic_filter_plan()` accessor next to `publishes_dynamic_filters()` (:167-170); doc-comment (:230-243) unchanged.

**Step 4 — publisher rewrite, `src/op/dynamic_filter_publisher.{hpp,cpp}`.**
- hpp: remove includes :19 and :21 (`join_filter_pushdown.hpp`, `sirius_physical_hash_join.hpp`), members `_filter_pushdown`, `_key_casts`, `_right_key_col_indices` (:43-61) → single-arg ctor.
- cpp: keep :74-114 unchanged (empty-build skip :77-81, drained-targets skip :83-91, source-space resolution :95-111, L2 probe :114). Key loop :159-272 iterates `_plan.keys()` (admitted only; may be empty — loop no-ops, function still reaches the terminal INFO): the cast skip (:163-171) and index skip (:172) are gone (resolved at finalization); the domain gates (:176-190 membership, :210-231 zone-map) now read `key.build_key_domain_cardinality` — `nullopt` in C1a ⇒ never fire ⇒ identical to today's always-zero behavior. Column access: `build_view.column(key.build_column_index)`. **Effective build type is always the runtime `col.type()`** (today's behavior, :194); the plan-time `key.build_type` is used only as a drift *detector*: if `key.build_type != EMPTY && key.build_type != col.type()`, `SIRIUS_LOG_WARN` (existing `[sirius_physical_hash_join]` prefix of this file) **and continue with `col.type()`** — never skip the key (skipping would change behavior in a direction the plan cannot prove impossible; see Risks item 5). Per-key vectors sized `_plan.keys().size()`, indexed by `sirius_key_ordinal`.
- Fan-out :303-329: iterate targets; the arity-mismatch guard (:310-317) **stays WARN + skip-target in release** (a throw would propagate through `publish_dynamic_filters`' catch → FAILED + rethrow, `sirius_physical_hash_join.cpp:1340-1344`, aborting the query — "fail closed" means skip, not crash), comparing `tgt.probe_col_idx.size() == _plan.duckdb_key_count()`; add a debug-only assert (adapter guarantees it; same never-fires observable in release). Push loop iterates `_plan.keys()` and indexes target vectors by `key.key.duckdb_filter_ordinal`; zone-map type check `tgt.probe_col_type[key.key.duckdb_filter_ordinal] == effective_build_type` (:319-321 semantics preserved).
- Summary INFO (:330-337): keep format and **keep the keys count = `_plan.duckdb_key_count()`** (not admitted count) so fixed-config log-diff is byte-stable. This is the machine-diffed terminal line of the gate protocol; it keeps its existing `[sirius_physical_hash_join]` prefix and INFO level.

**Step 5 — scan side + cleanups.**
- `src/planner/sirius_plan_get.cpp:264-268`: replace direct reads with `auto identity = duckdb_join_filter_candidate_adapter::scan_channel_identity(op); node->dynamic_filters = identity; if (identity) { node->sirius_dynamic_filters = get_or_create_dynamic_filter_channel(identity.get()); ... }` (log line unchanged). Router `get_or_create_dynamic_filter_channel` (`sirius_physical_plan_generator.cpp:54-65`) and its map (`hpp:80-91`) unchanged — scan channels stay keyed by preserved `DynamicTableFilterSet*` (design :558-562).
- Dead-code deletions (no behavior change): NLJ pushdown ctor overload + member (`sirius_physical_nested_loop_join.hpp:59,:101`; `.cpp:126,:146` — never populated, §0 #14); `sirius_physical_duckdb_scan.hpp:79` member (never assigned). Optional; if deferred, note in PR.
- Test call-site mechanical fixes for the ctor signature: `test/cpp/operator/test_no_history_peak_memory_estimate.cpp:67-77`, `test/cpp/operator/test_physical_mark_join.cpp:90-101`, `test/cpp/operator/test_physical_concat.cpp:95-106` (each currently passes a trailing `nullptr` pushdown arg — delete it).
- Update `test/cpp/transparent/test_preserve_dynamic_filter_metadata.cpp` includes/namespaces to the adapter (**7** existing TEST_CASEs at :72,:83,:97,:113,:127,:146,:174 must pass unmodified in assertion content — they are the pointer-identity contract tests the design names).

## Tests

All new files must be added to `TEST_SOURCES` (`CMakeLists.txt:562` list, `add_executable` :693) or the `check-orphan-tests` pre-commit hook fails (`.pre-commit-config.yaml:81-85`).

1. `test/cpp/planner/test_duckdb_join_filter_candidate_adapter.cpp` — tag `[dynamic_filter][adapter]`, no-DB logical-op construction (pattern of `test/cpp/transparent/test_preserve_dynamic_filter_metadata.cpp:116-183`), CPU-only:
   - `extract classifies null / statistics_only / admitted` (design :986);
   - `extract keeps range-comparison ordinals at full arity` (eq+range join → `admitted`, `condition_indexes.size()` == full join_condition arity, `condition_comparisons[j]` records the range type — NOT malformed);
   - `extract rejects out-of-range join_condition index` and `...duplicate index` → `malformed`; `target arity corruption -> malformed` (hand-corrupted `pi.columns`);
   - `extract drops null-channel-identity target, keeps sibling target`;
   - `ordinal alignment: duckdb ordinal -> condition index -> compact sirius ordinal` incl. an eq+range mixed-condition join (range ordinal present in candidate, absent from finalized keys) and an eq+INDF join (INDF never recorded by DuckDB, design :987);
   - `preservation keeps DynamicTableFilterSet pointer identity join<->get` and `two producing joins share one scan set` (design :983-984) — adapter-namespace versions extending the 7 existing cases in `test_preserve_dynamic_filter_metadata.cpp` (those move/update to the adapter API, assertions unchanged).
2. `test/cpp/planner/test_dynamic_filter_plan_snapshot.cpp` — tag `[dynamic_filter][planner]`, SQL-through-planner harness copied from `test/cpp/planner/test_distinct_hash_join_detection.cpp:41-158` (Parser→CreatePlan→Optimize→create_plan under `SIRIUS_CONFIG_FILE=test/cpp/config/data/minimal.yaml`, `SIRIUS_DISABLE=1` after DB creation; needs GPU-capable binary but no kernel launches). Assertions read the join's plan via the new `dynamic_filter_plan()` accessor:
   - `filtered build inner join wires 1 target with admitted key` (asserts `publishes_dynamic_filters()`, `keys()`/target arity, `duckdb_key_count`);
   - `unfiltered build side wires nothing` (mirror of `sirius_plan_comparison_join.cpp:426-430`);
   - `cast build key: target wired, key not admitted, plan enabled` (arity preservation, `keys().empty()`);
   - `eq+range join: target wired at full arity, range ordinal not admitted, plan enabled` (guards F1 plan-shape parity; the join runs as MIXED_JOIN and never claims, but channels/scan operators must exist exactly as today);
   - `statistics_only probe_info wires nothing` (and still emits the "unfiltered" INFO when the hint is false);
   - `RIGHT join still registers channel producer` (guards converter elision behavior at `sirius_pipeline_converter.cpp:278`);
   - `pushdown disabled via config wires no channels` (fixture flip per `test_dynamic_filter_router.cpp:53-57`).
3. `test/cpp/operator/test_dynamic_filter_publisher_plan.cpp` — tag `[dynamic_filter][publisher]`, **needs a CUDA device** (style of `test/cpp/operator/test_sirius_dynamic_filter.cpp:42-58`): construct `sirius_dynamic_filter_set` channels + a hand-built finalized plan + small cudf build tables; assert:
   - `admitted int32 key publishes in_list under L2, bloom above` (mirrors :244-261 policy);
   - `no admitted keys -> publisher runs, pushes nothing, terminal summary reflects duckdb_key_count` (empty `keys()`, publication state ends FINISHED);
   - `drained target skipped, sibling target receives filters` (:83-91,:307);
   - `zone-map pushed only on matching probe/build type` (:319-321);
   - `plan/runtime build-type drift: WARN + publish with runtime type` (hand-built plan with a wrong `build_type`; push counts unchanged vs EMPTY-typed control);
   - `claim eligibility: MARK join_type fails closed` (via `validate_producer_eligibility`; join-type predicate only).
4. Existing suites as regression oracle (must pass unmodified except the three ctor call-site arg deletions): `test_dynamic_filter_router.cpp` (7 cases), `test_sirius_dynamic_filter.cpp`, `test_sirius_dynamic_filter_mgpu.cpp` (2-GPU, self-skipping), `test/cpp/integration/test_gpu_execution_tpch.cpp` (GPU, `[integration]`).

## Gate & rollback

**Fixed-config compatibility protocol** (gate evidence; log level stated per pass, per program conventions):
- (a) `pixi run make test` green on the `gpu-2xl4` runner (whole Catch2 binary, `test.yml:133-136`).
- (b) TPC-H SF1 `test/tpch_performance/benchmark_and_validate.sh` result validation + timing, run at **INFO** log level (timing passes are always INFO; DEBUG lines are excluded from timing runs).
- (c) paired-run log diff, base vs PR, fixed config, two legs:
  - **Leg 1 (INFO, may reuse the timing pass)**: per query, identical multisets of the publisher terminal line (`dynamic_filter_publisher.cpp:330-337`, INFO — counts preserved by `duckdb_key_count`).
  - **Leg 2 (DEBUG, separate non-timed pass)**: identical multisets of the scan apply line and selectivity-gate line (`src/op/scan/dynamic_filter_merge.cpp:158,:216` — both `SIRIUS_LOG_DEBUG`; invisible at INFO, so this leg requires a dedicated DEBUG run whose timings are discarded).
  - Both legs segmented by the QueryBegin/QueryEnd anchors the log analyzer already parses (`tools/log_analyzer/patterns.py:46-51`). Script lives in the PR description / scratch, not the repo.

**Merge gate C1a** (design row :962): (a)-(c) show identical wired-target counts, pushed-filter counts, apply-row trajectories, and query results across TPC-H SF1 + clickbench sample under fixed config (dynamic filters ON and OFF); all listed suites green; adapter tests green. No new flags — `enable_dynamic_filter_pushdown` / `enable_dynamic_zone_map_filter` / thresholds keep their exact consumption points (`sirius_physical_plan_generator.cpp:38-45`, `sirius_plan_comparison_join.cpp:415,499,502`, `sirius_pipeline_converter.cpp:299`).

**Rollback**: flag-free refactor — revert story is `git revert` of the single PR commit; no config, YAML schema, or on-disk state involved.

## Dependencies

1. **C1a → C1b** strictly ordered (C1b reshapes C1a's plan/target types and reuses the adapter entry points).
2. **A1 (Track A)**: independent files except `dynamic_filter_identity.hpp` — whichever PR lands first creates it; the other consumes. Log-analyzer pattern registration + `SHAPE_VERSION` bump is A1's scope.
3. **A2** (`dynamic_filter_build_priority` flag) touches `task_scheduler.*` only — its `collect_filter_build_pipelines` reads `publishes_dynamic_filters()` (`src/pipeline/task_scheduler.cpp:190`), whose semantics (`_dynamic_filter_plan.enabled()`, `sirius_physical_hash_join.hpp:167-170`) C1a preserves; no ordering constraint.
4. **B1**: no code dependency; C1a changes no admitted candidates, so the vulnerable pin does not block it (design :947-949).

## Size

~600–750 prod LOC touched (adapter ~330 new; plan value ~150; publisher ~120 diff; hash join ~120 diff; planner ~140 diff; transparent −60/+20; identity header ~40; NLJ/duckdb-scan dead-code −55) + ~800–950 test LOC (3 new files, 1 updated, 3 mechanical fixes). **Recommend splitting** into C1a-1 (adapter module + preservation move + adapter/preserve tests; zero runtime change) and C1a-2 (value snapshot + publisher/claim rewrite + planner rewiring + parity tests) — each reviewable in one pass.

## Risks & mitigations

1. **Index-space misalignment** (DuckDB ordinal vs reordered condition vs equality ordinal). The scheme rests on DuckDB recording `join_condition` post-`ReorderConditions` (`join_filter_pushdown_optimizer.cpp:224,249-250`) and Sirius's reorder being a stable no-op on already-ordered input (`sirius_physical_hash_join.cpp:173-185`). *Mitigation*: extraction fails closed (malformed) on out-of-range/duplicate indexes; finalization's `cond_idx >= right_key_col_indices.size()` skip fences every non-equality/non-prefix ordinal out of the `key_casts`/`right_key_col_indices` index space (identical to today's runtime fence at `dynamic_filter_publisher.cpp:172`), plus a debug assert that admitted `cond_idx` refer to equality conditions; adapter tests cover eq+range and eq+INDF layouts.
2. **Wiring side-effect drift**: channel `register_producer()` (`sirius_plan_comparison_join.cpp:446`) drives scan-operator insertion (`sirius_pipeline_converter.cpp:277-279,294-304`); narrowing wiring (e.g., excluding RIGHT joins or eq+range joins at plan time) would silently change plans. *Mitigation*: keep the wiring predicate byte-equivalent and the candidate at full DuckDB arity; INNER/SEMI narrowing exists only at the claim (proven unreachable difference); RIGHT-join and eq+range plan-shape tests.
3. **Log-diff instability breaking the gate itself**: the summary line's key count currently equals `join_condition.size()` (`dynamic_filter_publisher.cpp:337`); a rewrite keyed on admitted count — or skipping the publisher on empty `keys()` — would change/remove the line. *Mitigation*: `duckdb_key_count` snapshot preserved for that line; publisher runs even with zero admitted keys (Step 3/4).
4. **`DynamicTableFilterSet` lifetime after removing the join's ownership**: sets remain owned by scans (`sirius_plan_get.cpp:264`, `sirius_physical_parquet_scan.cpp:65`) and by the logical plan during planning; runtime never dereferences the pointer (map keys only, planning-scoped). *Mitigation*: comment the invariant on the router map (`sirius_physical_plan_generator.hpp:80-84`); if any GET path is found that plans without a table-scan node retaining the set, fall back to keeping an opaque ownership-only `unique_ptr` member on the join (no accessor).
5. **Plan-time `build_type` vs runtime `col.type()` divergence** (decimal/timestamp conversions). Today's behavior uses the runtime type unconditionally; the plan snapshot cannot be proven drift-free for every type mapping. *Mitigation*: runtime `col.type()` remains authoritative; `key.build_type` is EMPTY-fallback + WARN-only drift detector (never skips a key); publisher parity test over INT32/INT64/DECIMAL/VARCHAR keys plus an injected-drift test.
6. **`_dynamic_filter_plan` losing `const`**: mutation-after-finalize would break the immutability contract (design :782-817). *Mitigation*: `finalize_keys` throws on second call; the new public accessor is const-ref only; comment on the member (`sirius_physical_hash_join.hpp:253-254`).
7. **Defensive `malformed` on target-arity corruption is a (theoretical) plan-shape change**: today an arity-corrupt target would still wire and be WARN-skipped at fan-out (`dynamic_filter_publisher.cpp:310-317`); under C1a the whole candidate is rejected and no channel is registered. This is unreachable under the pinned DuckDB (columns and `join_condition` pushed in lockstep, `join_filter_pushdown_optimizer.cpp:243-250`) and is the design's intended structural fail-closed (design :315-320); stated here honestly rather than claimed "no behavior change". The runtime WARN+skip guard is retained as a second fence regardless.

---

# PR C1b — strong target-key value types, materialization telemetry, shadow selectivity signal

## Goal

Strong target-key value types (`scan_target_key`, target variant, strong IDs), materialization telemetry (`duckdb_candidate` / `membership_materialization` / `zone_map_materialization` / `duckdb_build_subtree_has_filter_hint`, design :283-294), and a live-but-shadow selectivity signal replacing the dead domain path with a pre-resolver snapshot and `std::nullopt` unknown semantics (design :840-848).

**Ownership (cross-cluster resolution): C1b OWNS the shadow selectivity signal end to end** — the preservation-time snapshot pass, the registry, the flow into `dynamic_filter_key_plan.build_key_domain_cardinality`, and the would-suppress *recording* (log-only). The C1cde cluster's **C1d is enforcement-only** (flipping shadow suppression to actual skips) and **depends on C1b landing**; it adds no signal plumbing of its own.

**Gate** (design row :963): C1a criteria repeated; C1b additionally makes policy observable with behavior unchanged.

## Deliverables

```cpp
// dynamic_filter_publish_plan.hpp
struct scan_target_key {                                  // strong per-target key (design :794-798)
  admitted_dynamic_filter_key key;
  cudf::size_type probe_column_index;
  cudf::data_type probe_column_type;
};
struct scan_publish_target {
  dynamic_filter_target_id target_id;
  std::shared_ptr<sirius_dynamic_filter_set> channel;
  std::vector<scan_target_key> keys;                      // aligned to admitted keys (sirius_key_ordinal)
};
struct join_probe_target_key { admitted_dynamic_filter_key key;
                               cudf::size_type probe_column_index; cudf::data_type probe_column_type; };
struct join_probe_publish_target { dynamic_filter_target_id target_id;
                                   std::shared_ptr<sirius_dynamic_filter_set> channel;
                                   std::vector<join_probe_target_key> keys; };   // declared; unconstructed until C3
using dynamic_filter_publish_target = std::variant<scan_publish_target, join_probe_publish_target>;  // design :806-807
```

```cpp
// new src/include/planner/dynamic_filter_candidate_snapshot_registry.hpp (+ .cpp)
// Query-local logical pairing (design G9 row, :935); owned by SiriusContext, cleared in QueryBegin/QueryEnd.
class dynamic_filter_candidate_snapshot_registry {
public:
  void record(duckdb::JoinFilterPushdownInfo const* cloned_info,
              std::vector<std::optional<std::size_t>> per_key_domains);   // 0/untraceable stored as nullopt
  [[nodiscard]] std::optional<std::vector<std::optional<std::size_t>>>
    take(duckdb::JoinFilterPushdownInfo const* info);
  void clear();
};
```

Materialization-telemetry value enums (in `dynamic_filter_publish_plan.hpp` or a small `dynamic_filter_telemetry.hpp`): `membership_materialization {exact_set, bloom, none_unsupported_type}`, `zone_map_materialization {emitted, none_disabled, none_invalid_minmax, none_shadow_would_suppress}` — used only to format log lines in C1b.

## Step-by-step changes

**Step 1 — target restructure (`dynamic_filter_publish_plan.{hpp,cpp}`, `sirius_plan_comparison_join.cpp`, `dynamic_filter_publisher.cpp`).**
- Replace `probe_target` with the `dynamic_filter_publish_target` variant. Planner builds full-arity *draft* scan targets (probe column index/type per DuckDB ordinal, from `candidate.targets[t].columns`); `finalize_keys` (hash-join ctor) compacts each scan target's per-ordinal columns into `std::vector<scan_target_key>` aligned to admitted keys. Publisher fan-out becomes a `std::visit` (exhaustive; `join_probe_publish_target` branch `throw std::logic_error` until C3) iterating `tgt.keys` by `sirius_key_ordinal`. Push set is provably identical (non-admitted ordinals never have built filters today — non-equality and cast ordinals are skipped before any filter is materialized, C1a Step 3). The C1a arity WARN+skip guard translates to the compacted form (keys-vector length vs plan `keys().size()`).
- The converter/scan/channel code is untouched; `filter_set`→`channel` rename confined to the plan/publisher/planner.

**Step 2 — shadow selectivity signal (owned here, enforced in C1d).**
- New `dynamic_filter_candidate_snapshot_registry` (files above; register the .cpp in CMake). Owner: `SiriusContext` member; `clear()` called in `QueryBegin` (`src/sirius_context.cpp:165-209`) and `QueryEnd` (:211-265). SiriusContext serializes query lifecycles (`sirius_context.hpp:267-277`), so the query-local registry is race-safe.
- Preservation-time walk: extend `duckdb_join_filter_candidate_adapter::preserve_dynamic_filter_metadata` with an optional `(duckdb::ClientContext&, dynamic_filter_candidate_snapshot_registry*)`; in the join branch (old :78-84), after cloning, run the resurrected domain walk **on the original join** (pre-resolver: conditions are `BOUND_COLUMN_REF`, exactly what the dead code at `sirius_plan_comparison_join.cpp:329-352` required) reusing `trace_binding_to_get` (moved to the adapter in C1a) and the pre-filter cardinality read (:343-349); store per-DuckDB-ordinal `optional<size_t>` (0/untraceable → `nullopt`, design :845-846) keyed by the **clone's** `JoinFilterPushdownInfo*` (stable from clone through `extract`, since the copy's `unique_ptr` moves don't change the pointee). All three `copy_logical_plan` callers get the registry from the `SiriusContext` already in hand (`sirius_optimizer_extension.cpp:111,146`, `physical_sirius_execution.cpp:132`, `sirius_context.cpp` member scope).
- Extraction attaches: `extract` gains an optional registry param; on hit, per-ordinal domains flow into `dynamic_filter_key_plan.build_key_domain_cardinality` at finalization. **Registry misses (all stay `nullopt`, harmless)** occur on every path where no preservation copy runs: the explicit non-transparent paths `sirius_extension.cpp:295,387,686` and `sirius_ffi.cpp:164`, **and the transparent replan fallback** `physical_sirius_execution.cpp:132-147` (re-parse/re-optimize when `logical_plan_` proved non-copyable — `extract` then sees native, non-clone `JoinFilterPushdownInfo` pointers).
- Publisher gates become shadow: at :176-190 and :214-231 sites compute `would_suppress` from the optional and threshold, **log only** (`shadow_suppress_membership=`/`shadow_suppress_zone_map=` k=v fields), never skip. Enforcement is C1d (C1cde cluster), which consumes this signal as-is.

**Step 3 — materialization telemetry.**
- Plan-time, one machine-parsed line per producing join at INFO: `[dynf_summary] candidate: plan_id={} kind={absent|statistics_only|admitted|malformed} build_hint={} keys_total={} keys_admitted={} targets={}` (emitted from `sirius_plan_comparison_join.cpp`; uses the `[dynf_summary]` prefix because it is a machine-parsed summary line, per program logging conventions — human-oriented lines in that file keep `[sirius_plan_comparison_join]`). `keys_total` = `duckdb_key_count`; `keys_admitted` = finalized `keys().size()` is only known post-ctor, so the planner logs `keys_total`/`targets` and the hash-join ctor logs a companion `[dynf_summary] finalized: plan_id={} keys_admitted={}` line — both INFO, both per-join, both stable-count (safe for timing passes).
- Publish-time: extend the per-key DEBUG (`dynamic_filter_publisher.cpp:262-271`) to k=v: `plan_id= key= cond_idx= build_rows= membership={exact_set|bloom|none(unsupported_type)} zone_map={emitted|none(disabled)|none(invalid_minmax)} domain={n|unknown} shadow_suppress_membership={0|1} shadow_suppress_zone_map={0|1} set_bytes= bloom_bytes= l2_bytes=`. **DEBUG level — excluded from timing runs**; coverage claims about these fields come from a separate non-timed DEBUG pass. Keep the terminal INFO (:330) format-stable. New/changed lines follow the log_analyzer contract style (`tools/log_analyzer/patterns.py` anchors, `k=v` numeric fields); pattern/`SHAPE_VERSION` updates are A1's deliverable — coordinate field names (`plan_id`, `target_id`) with A1's event vocabulary (design :1059-1069) so A1 doesn't re-churn.

## Tests

5. Extend `test_duckdb_join_filter_candidate_adapter.cpp`: `registry: preservation records per-ordinal domains keyed by clone pointer; extract attaches; miss -> all nullopt; zero cardinality -> nullopt` (design :845-846). Miss matrix covers all three no-preservation paths: explicit non-transparent planning, FFI, **and the transparent replan fallback** (simulate by extracting from a native, never-preserved `JoinFilterPushdownInfo` — the `physical_sirius_execution.cpp:132-147` shape).
6. Extend `test_dynamic_filter_plan_snapshot.cpp`: `shadow domains flow from preservation through finalized key plans` (uses `copy_logical_plan` + registry, then plans the copy; asserted via `dynamic_filter_plan()`).
7. Extend `test_dynamic_filter_publisher_plan.cpp` (GPU): `high domain coverage logs would-suppress but still publishes` — push counts identical with domain nullopt vs domain small vs domain huge (shadow proof, design :843-844).
8. Variant dispatch: `join_probe_publish_target present -> publish throws logic_error` (until C2/C3).

## Gate & rollback

- **Merge gate C1b** (design row :963): C1a's fixed-config protocol repeated (same log levels per leg: INFO timing/terminal-line legs; separate non-timed DEBUG leg for per-key shadow fields and scan-apply lines), plus: shadow `k=v` fields present in the DEBUG pass for every admitted key on a TPC-H run; at least one query demonstrates a non-`unknown` domain with `would_suppress=1` while push counts remain unchanged vs C1a base (policy observable, behavior unchanged); `[dynf_summary]` lines present per producing join in the INFO pass.
- **Proceed gate**: C1c/C1e may only build on the merged C1b types; **C1d additionally consumes C1b's shadow signal (enforcement-only)**; B1 is deferred (decision 2026-07-08, sirius-db/sirius#1123) and blocks nothing; while on the unpatched pin, LIMIT/TOP-N-shaped tests use explicit expected rows or a filters-disabled reference, never an unpatched CPU run.
- **Rollback**: flag-free; `git revert` of the single PR commit. The registry is query-local and self-clearing; reverting C1b leaves C1a intact (C1b is strictly additive over C1a's types except the target restructure, which is why C1b must not be squashed into C1a).

## Dependencies

1. **C1a → C1b** strictly ordered.
2. **A1**: k=v field names (`plan_id`, `target_id`) and the `[dynf_summary]` vocabulary must match A1's event vocabulary; log_analyzer pattern registration + `SHAPE_VERSION` bump is A1's scope. Coordinate before C1b merges.
3. **C1d (C1cde cluster)**: enforcement-only consumer of C1b's shadow signal; blocked on C1b. C2/C3 consume C1b's `join_probe_publish_target`/variant and the adapter; blocked on C1b.

## Size

~450–550 prod (variant/target restructure ~180; registry ~150; preservation walk ~80; telemetry ~60) + ~380–480 test. Single PR.

## Risks & mitigations

1. **Registry key aliasing across copies**: three `copy_logical_plan` sites can register multiple snapshots per query (prepare-validation + execute copies). *Mitigation*: key by the clone pointer actually consumed by `extract` (`take()` removes the entry), `clear()` at QueryBegin/QueryEnd (`sirius_context.cpp:165-265`); registry test covers double-copy and the replan-fallback miss.
2. **Preservation-time domain walk cost/exceptions**: `get->function.cardinality` (old :343-347) can allocate/throw inside the optimizer hook. *Mitigation*: wrap per-key in try/catch → `nullopt`; walk only joins that already have `filter_pushdown`; it replaces an identical-shape walk that previously ran per plan anyway (:367-368).
3. **Telemetry perturbing the timing gate**: per-key lines are DEBUG and excluded from timing runs; the two new per-join `[dynf_summary]` INFO lines are constant-count per plan (no per-batch fan-out) and are diffed, not timed, in leg 1.

---

# Review resolution appendix

One line per adversarial-review finding → how it was applied.

1. **BLOCKER — non-equality pushed conditions made the whole candidate `malformed`** → Fixed as directed: `extract()` keeps ALL `join_condition` ordinals at full arity (DuckDB pushes range comparisons, `join_filter_pushdown_optimizer.cpp:230-236`); non-equality ordinals are narrowed per key at finalization via the `cond_idx >= right_key_col_indices.size()` mirror of today's runtime skip (`dynamic_filter_publisher.cpp:172`); `malformed` reserved for out-of-range/duplicate indexes and target-arity corruption; `duckdb_key_count = join_condition.size()` always; §0 index-space paragraph corrected; eq+range plan-shape test added. Note: per-target arity corruption moved from "drop target" to `malformed` per the directive and design :315-320 — a theoretical plan-shape change vs today's runtime WARN+skip, unreachable under the pinned DuckDB and stated honestly as C1a Risk 7 (the runtime WARN+skip fence is retained regardless).
2. **MAJOR — `!keys().empty()` eligibility clause broke log parity and emitted spurious ERROR** → Fixed as directed: the publisher runs even with zero admitted keys (loop no-ops; terminal INFO keeps its multiset identity via `duckdb_key_count`); `validate_producer_eligibility` and the release-mode ERROR are scoped strictly to the join-type predicate (genuinely unreachable); publisher test 3 updated to assert the terminal line + FINISHED state on empty keys.
3. **MINOR — fan-out arity assertion would abort the query** → Fixed: WARN + skip-target in release (unchanged failure semantics vs today's :310-317), debug-only assert.
4. **MINOR — build-type drift "WARN + skip-key" inverted today's fallback** → Fixed: runtime `col.type()` stays authoritative; plan-time `build_type` is a WARN-only drift detector, never skips; injected-drift publisher test added.
5. **MINOR — "8 existing cases at :72-190"** → Fixed: 7 TEST_CASEs (:72,:83,:97,:113,:127,:146,:174; file is 191 lines) — verified against the file.
6. **MINOR — snapshot tests needed an accessor** → Fixed: public `[[nodiscard]] dynamic_filter_publish_plan const& dynamic_filter_plan() const noexcept` added to C1a deliverables on `sirius_physical_hash_join` (member is `protected` at `hpp:254`; only `publishes_dynamic_filters()` was public).
7. **MINOR — registry-miss enumeration omitted the transparent replan fallback** → Fixed: `physical_sirius_execution.cpp:132-147` added to the C1b miss enumeration and to the registry test matrix (test 5).
8. **MINOR — `detail` decl citation ":52-57"** → Fixed: the block spans `sirius_optimizer_extension.hpp:43-68` (`preserved_counts` :46-50, clone :56-57, preserve :64-66) — verified; Step 1 now deletes the whole block including `preserved_counts`.
9. **MINOR — "build side is unfiltered" INFO would vanish for statistics_only candidates** → Fixed: the log is keyed on `candidate.kind != absent && !build_subtree_has_filter_hint` (not `kind == admitted`), preserving today's emission set; snapshot test 2's statistics_only case asserts the line.
10. **MINOR — gate protocol (c) silent about log level** → Fixed: protocol now states levels per pass — leg 1 (publisher terminal line) at INFO and reusable for timing; leg 2 (scan-apply `dynamic_filter_merge.cpp:158` and selectivity-gate `:216`, both `SIRIUS_LOG_DEBUG`) in a separate non-timed DEBUG pass; timing numbers only ever from INFO passes.

Directive-level additions beyond the numbered findings: (a) cross-cluster ownership stated — C1b owns the shadow selectivity signal (snapshot pass + would-suppress recording); C1cde's C1d is enforcement-only and depends on C1b landing; (b) program logging conventions applied — machine-parsed per-join summary lines use `[dynf_summary]` at INFO, other new lines keep their file's component prefix, per-key lines are DEBUG and excluded from timing runs; (c) doc restructured into per-PR sections (Goal / Deliverables / Step-by-step / Tests / Gate & rollback / Dependencies / Size / Risks) with the standard header, content preserved.
