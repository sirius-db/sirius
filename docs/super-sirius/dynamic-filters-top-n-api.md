# Dynamic Filters — Top-N API and Test Specification

> **Status: Stages 1–4 implemented; Stages 5–7 proposed.** Companion to
> [dynamic-filters-top-n.md](dynamic-filters-top-n.md), which owns the design narrative and the
> normative contract. This document turns that contract into a header-level API surface and an
> example-driven high-level test plan. Stage numbers refer to the main doc's seven-stage rollout.
> The Stage 1–4 surfaces below are implemented and behind `enable_top_n_dynamic_filter` (default
> false) — producers publish, with scan binds and first-key endpoints live, LEX endpoints
> deferred to Stage 7, and set operations a trace terminal; the Stage 5–7 surfaces remain
> proposed. Declarations here stay the authority — where an
> implemented signature differs, the doc is corrected, not the code.

## Scope and untouched surface

The append-only hash-join path does not change. The following existing declarations stay
byte-identical:

| File | Untouched declarations |
|---|---|
| `src/include/op/dynamic_filter/sirius_dynamic_filter.hpp` | `sirius_dynamic_filter`, `sirius_device_replicable`, `sirius_ast_lowerable`, `sirius_mask_applicable`, zone-map/IN-list/Bloom classes, `push_filter`, `filters_for_column`, `filtered_columns`, `empty`, `ignore_columns`, `register_producer`, `has_producers`, `close_for_new_filters`, `accepting_filters`, `has_filters`, `filter_count`, `column_ref_resolver_fn` (declaration text unchanged; it moved earlier in the header so the Stage-3 capability classes can reference it — no signature or semantic change). `merge_ast_dynamic_filters_into_tree` was on this list until the implementation deleted it: its by-value per-column re-fetch reproduced the parquet snapshot lifetime defect and it had no production callers — `scan::merge_dynamic_filters_into_ast` over a snapshot is the sole AST merge path. |
| `src/include/op/dynamic_filter/dynamic_filter_publisher.hpp` | Everything — the join publisher is not a refinement producer |
| `src/include/op/dynamic_filter/dynamic_filter_replica_space.hpp` | Everything — reused as-is by Stage 3 replication |
| `src/planner/dynamic_filter/*` join admission/evidence/domain | Everything — the Top-N trace is separate |

Every change below is additive: a new file, a new member, or a new overload. The enum extension
(`sirius_dynamic_filter_kind::RANGE`, `::LEX_RANGE`) is append-at-end and does not renumber
existing values. The channel's single-ordinal push/store/lookup contract is preserved:
multi-column filters register under a primary ordinal and resolve the rest through the consumers'
existing column-reference resolver.

---

## Part 1 — API surface by stage

### Stage 1 — Coordinator and sink self-consumption *(implemented)*

#### New file: `src/include/op/dynamic_filter/exact_host_scalar.hpp`

One exact typed host value. Introduced here for the coordinator's boundary; reused verbatim by
the Stage 3 RANGE filter. `std::variant` keeps exactness explicit — no value ever rounds through
`double` (the same rule the zone-map replication follows).

**Allowlist rule.** The variant covers the admitted types' **physical representations**, not their
logical types; `storage_type` is what distinguishes `DECIMAL64` from `INT64` when both ride an
`int64_t`. Admitting a logical type therefore widens the variant only when its physical
representation is not already present — `DECIMAL(5–18)` needed no new alternative. The next
widening, `DECIMAL128`, is not a variant edit alone: `component::value`, the kernel's width
switch, and the variant must change **together**, because a value the variant can hold but the
kernel cannot load is the silently-wrong case the allowlist exists to prevent.

```cpp
namespace sirius::op {

/**
 * @brief Exact host-side scalar for the Top-N type allowlist
 *
 * Carries the value and its cuDF storage type together so comparison and device-scalar
 * construction cannot disagree about representation. The variant holds **physical
 * representations**, not logical types, so admitting a new logical type widens it only when that
 * type's physical representation is not already present — `DECIMAL(5-18)` needed no new
 * alternative, because its scaled integer *is* an `int32_t`/`int64_t` and the scale rides in
 * `storage_type` (`fixed_point_scalar<T>::value()` returns `rep_type`). `DECIMAL128` would need
 * one, since no `int128` alternative exists. See the main doc, "Range and lexicographic filters".
 * Immutable after construction; freely copyable; no device state.
 */
class exact_host_scalar final {
 public:
  using value_type = std::variant<std::int8_t, std::int16_t, std::int32_t, std::int64_t>;

  exact_host_scalar(value_type value, cudf::data_type storage_type) noexcept;

  [[nodiscard]] value_type const& value() const noexcept;
  [[nodiscard]] cudf::data_type storage_type() const noexcept;

  /**
   * @brief Three-way comparison in the given SQL order
   *
   * @pre Both operands share one storage type; the coordinator guarantees this by construction.
   * @return Negative/zero/positive when `*this` orders before/equal-to/after @p other under
   * @p order. Null ordering is not this class's concern — a null boundary never constructs one.
   */
  [[nodiscard]] int compare(exact_host_scalar const& other, cudf::order order) const noexcept;

 private:
  value_type _value;
  cudf::data_type _storage_type;
};

}  // namespace sirius::op
```

`DATE` rides `std::int32_t` through its exact cuDF physical representation; the variant is
physical, the `storage_type` is logical.

```cpp
namespace sirius::op {

/**
 * @brief One ORDER BY key's frozen semantics
 */
struct top_n_key_semantics {
  cudf::data_type storage_type;
  cudf::order order;
  cudf::null_order null_order;
};

/**
 * @brief The exact host boundary tuple: one optional component per ORDER BY key
 *
 * A disengaged component records that the boundary row's key is null there — legal in tail
 * positions, and the reason the type is not `std::vector<exact_host_scalar>`. Immutable and
 * copyable, like its component type.
 */
class exact_host_key_tuple final {
 public:
  explicit exact_host_key_tuple(std::vector<std::optional<exact_host_scalar>> components);

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::optional<exact_host_scalar> const& component(std::size_t i) const;

  /**
   * @brief Lexicographic three-way comparison under per-key semantics
   *
   * Honors each key's direction and null placement; null components order per `null_order`.
   *
   * @pre `semantics.size() == size()` and both tuples share component storage types.
   */
  [[nodiscard]] int lex_compare(exact_host_key_tuple const& other,
                                std::span<top_n_key_semantics const> semantics) const noexcept;
};

}  // namespace sirius::op
```

#### New file: `src/include/op/dynamic_filter/top_n_threshold_coordinator.hpp`

```cpp
namespace sirius::op {

/**
 * @brief Result of offering one local Top-N boundary to the coordinator
 */
enum class threshold_offer_result {
  ACCEPTED_FOR_PUBLICATION,  ///< Tightest so far; this call owns the publisher loop (Stage 4)
  COALESCED,                 ///< Tightest so far; an active publisher will flush it (Stage 4)
  NOT_TIGHTER,               ///< Boundary does not lexicographically strengthen the tightest; ignored
  NO_ACCEPTING_TARGET,       ///< Tightened `tightest_seen` only; no channel target exists/accepts
  UNSUPPORTED_BOUNDARY,      ///< Null first component or otherwise unpublishable Kth-row tuple
  REJECTED_STATE             ///< Coordinator is FINISHING, FINISHED, or CANCELLED
};

/**
 * @brief One local result's proof of a legal boundary
 *
 * Created only by the `sirius_physical_top_n::execute` seam after the stream-ordered
 * device-to-host copies of every key column of row `K - 1` have completed (main doc, "Witness
 * handoff"). The witness co-owns the result batch so the K retained rows cannot be released
 * while an asynchronous publication still refers to them.
 */
struct top_n_threshold_witness {
  exact_host_key_tuple boundary;                    ///< Completed exact host Kth-row key tuple
  std::shared_ptr<cucascade::data_batch> witnesses; ///< The K-row local result, kept alive
};

/**
 * @brief Execution-owned Top-N threshold policy: monotonic boundary, revisions, publication
 *
 * One coordinator per Top-N producer per execution, shared by the local and merge operators via
 * `std::shared_ptr`. Holds checked K, per-key semantics, the tightest host boundary tuple, the
 * tightest pending candidate, a monotonic revision counter, and metrics. Tightness is
 * `exact_host_key_tuple::lex_compare` over the full tuple. It does not discover targets, inspect
 * DuckDB metadata, schedule scans, or decide final output.
 *
 * Threading: `offer`, `tightest_boundary`, `finish`, and `cancel` are thread-safe. Host
 * comparison happens under one short internal mutex; filter construction and replication (Stage
 * 4) run outside it with at most one publisher loop active. All state is execution-scoped and
 * starts empty (main doc, "Execution-scoped state").
 */
class top_n_threshold_coordinator final {
 public:
  /**
   * @brief Construct with frozen semantics from the planner
   *
   * @param[in] k Checked `limit + offset`
   * @param[in] keys Complete ORDER BY semantics, in key order; `keys[0]` is the first-key layer's
   * key
   * @param[in] lex_admitted True when every key's type is admitted — enables the LEX layer and
   * the lexicographic prefilter; false degrades both to the first-key comparison
   * @param[in] stats Non-owning counter sink owned by `SiriusContext`; may be null
   */
  top_n_threshold_coordinator(std::size_t k,
                              std::vector<top_n_key_semantics> keys,
                              bool lex_admitted,
                              dynamic_filter_stats* stats);

  top_n_threshold_coordinator(top_n_threshold_coordinator const&)            = delete;
  top_n_threshold_coordinator& operator=(top_n_threshold_coordinator const&) = delete;

  /**
   * @brief Offer a K-witness boundary; monotonically tightens `tightest_seen`
   *
   * Stage 1 semantics: tighten the shared boundary for the sink prefilter and return
   * `NO_ACCEPTING_TARGET`. Stage 4 adds the publisher loop and the remaining results.
   */
  threshold_offer_result offer(top_n_threshold_witness witness);

  /**
   * @brief Current tightest boundary tuple for sink self-consumption, or empty before K witnesses
   *
   * A mutex-guarded host copy; deliberately not a channel read. The sink prefilter builds the
   * strict LEX predicate from it (the degraded inclusive first-key comparison when
   * `lex_admitted` is false) with task-local scalars. Staleness is safe — a stale boundary
   * prunes less, never more.
   */
  [[nodiscard]] std::optional<exact_host_key_tuple> tightest_boundary() const;

  /**
   * @brief Synchronous producer-side drain called by merge/finalization
   *
   * Transitions OPEN -> FINISHING, rejects later offers, joins/starts the publisher until pending
   * work is empty (Stage 4), then transitions to FINISHED. Never called or awaited by consumers.
   */
  void finish();

  /**
   * @brief Reject further offers and publication; safe under teardown ordering
   */
  void cancel() noexcept;
};

}  // namespace sirius::op
```

#### Modified file: `src/include/op/sirius_physical_top_n.hpp`

Additive members only; the existing public fields and `execute` signature are unchanged. Only the
coordinator is mirrored on `sirius_physical_top_n_merge` (its delegating constructor already
copies shared state from the local operator, exactly as it shares `dynamic_filter` today); the
gate is non-copyable and per-operator, and the merge never prefilters.

```cpp
class sirius_physical_top_n : public sirius_physical_operator {
 public:
  // ... existing declarations unchanged ...

  /**
   * @brief Execution coordinator for threshold refinement; null when the producer is ineligible
   *
   * Set by the planner after eligibility validation. Shared with `sirius_physical_top_n_merge`,
   * which calls `finish()` after its child barrier.
   */
  std::shared_ptr<top_n_threshold_coordinator> threshold_coordinator;

  /**
   * @brief Per-execution keep-ratio state for the sink prefilter
   *
   * Reuses `scan::dynamic_filter_gate` unchanged: the prefilter records rows before/after each
   * measured batch and stops prefiltering when unselective. A tightened boundary re-arms one
   * measurement through the gate's growth rule (Stage 2 makes that rule generation-based).
   */
  scan::dynamic_filter_gate prefilter_gate;
};
```

The prefilter itself and the witness extraction are `execute()`-internal (a static helper in
`sirius_physical_top_n.cpp`), not public API: read `tightest_boundary()`, marshal it into the
boundary kernel's launch parameters, run one fused predicate+compaction pass (below), gather
unless every row passed, then `compute_top_n_table`, then extract every key column of row `K - 1`
and offer the witness. No AST, no device scalars, no BOOL8 mask on this path.

#### New files: `src/include/op/dynamic_filter/top_n_boundary_filter.hpp`, `src/cuda/top_n_boundary_filter.cu`

The one device row-filter for Top-N thresholds, shared by the Stage-1 sink prefilter, the
Stage-4 native/endpoint capability implementations, and the Stage-5 inclusive forms. CUB
device-algorithm style like the existing `src/cuda` filters; no raw kernel launches.

```cpp
namespace sirius::op::detail {

/**
 * @brief Launch-parameter form of one boundary: POD, passed by value, no device state
 *
 * Components hold the exact value widened to int64 (sign-extension is exact for the signed
 * allowlist), an engaged flag (null tail components), and the key's direction, output-order null
 * placement, and physical width. `strict` selects the row producer's predicate; inclusive covers
 * the degraded first-key form and the group-key producer. More components than
 * `k_max_components` degrade to the inclusive prefix form, which is sound standalone.
 */
struct boundary_filter_params {
  static constexpr std::size_t k_max_components = 8;
  // per-component: int64 value, engaged, descending, nulls_first, width; plus count and
  // strictness — exact layout is implementation detail.
};

/**
 * @brief One fused pass: per-row lexicographic compare against the boundary, compacting passing
 * row indices
 *
 * @return `filtered == nullptr` when every row passed (caller forwards the batch unchanged);
 * otherwise the gathered surviving table. `rows_kept` is always valid.
 */
struct boundary_filter_result {
  std::unique_ptr<cudf::table> filtered;
  cudf::size_type rows_kept;
};

[[nodiscard]] boundary_filter_result apply_boundary_filter(
  cudf::table_view const& batch,
  std::span<cudf::size_type const> key_columns,
  boundary_filter_params const& params,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::op::detail
```

#### Modified file: `src/include/op/dynamic_filter/dynamic_filter_stats.hpp`

The observability hook the tests depend on. New fields in `dynamic_filter_stats` (atomics) and
`dynamic_filter_stats_snapshot` (plain), following the existing timing-class prose. Read through
the existing `SiriusContext::get_dynamic_filter_stats_snapshot()`; no new accessor is needed.

```cpp
  // --- Top-N refinement (Stage 1) ---
  std::atomic<std::uint64_t> top_n_producers_eligible{0};   ///< Plan-time fact, like producers_enabled
  std::atomic<std::uint64_t> top_n_producers_rejected{0};   ///< Failed eligibility (keys/shape/K)
  std::atomic<std::uint64_t> top_n_producers_first_key_only{0};  ///< Tail key type degraded LEX away
  std::atomic<std::uint64_t> top_n_offers{0};               ///< Witness offers reaching the coordinator
  std::atomic<std::uint64_t> top_n_offers_not_tighter{0};   ///< Lost the lexicographic compare
  std::atomic<std::uint64_t> top_n_offers_unsupported{0};   ///< Null first boundary component
  std::atomic<std::uint64_t> top_n_prefilter_rows_in{0};    ///< Rows entering measured prefilters
  std::atomic<std::uint64_t> top_n_prefilter_rows_out{0};   ///< Rows surviving measured prefilters
  std::atomic<std::uint64_t> top_n_prefilter_disabled{0};   ///< Keep-ratio disables

  // --- Top-N publication and endpoints (Stage 4); per layer where meaningful ---
  std::atomic<std::uint64_t> top_n_first_key_scan_targets{0};     ///< Plan-time
  std::atomic<std::uint64_t> top_n_lex_scan_targets{0};           ///< Plan-time; all keys one scan
  std::atomic<std::uint64_t> top_n_endpoint_sites_placed{0};      ///< Plan-time; either layer
  /// Plan-time; endpoints and post-decode-only scan binds
  std::atomic<std::uint64_t> top_n_sites_skipped_no_work_saved{0};
  std::atomic<std::uint64_t> top_n_lex_endpoint_sites_deferred{0};///< Plan-time; staged gap, not a cost decision
  std::atomic<std::uint64_t> top_n_first_key_subsumed_by_lex{0};  ///< Plan-time; dedup fired
  std::atomic<std::uint64_t> top_n_revisions_published{0};        ///< Boundary updates fanned out
  std::atomic<std::uint64_t> top_n_lex_filters_pushed{0};
  std::atomic<std::uint64_t> top_n_first_key_filters_pushed{0};
  std::atomic<std::uint64_t> top_n_revisions_failed{0};           ///< Replica failure; old retained
  std::atomic<std::uint64_t> top_n_revisions_stale{0};
```

Delta/direction assertion rules from the sibling doc carry over: plan-time facts are exact per
plan construction (and the transparent path constructs twice per query); delivery-adjacent
counters are asserted as deltas or directions only.

#### Modified: extension configuration (`src/sirius_extension.cpp`, `operator_params`)

```text
enable_top_n_dynamic_filter (BOOLEAN, default false)
  Enable Top-N threshold refinement: sink self-consumption now, channel publication in later
  stages. Independent of enable_dynamic_filter (the join path); both may be toggled in tests.
```

One flag covers all six stages; stages change what an enabled producer does, not the switch.

---

### Stage 2 — Channel foundation *(implemented)*

#### Modified file: `src/include/op/dynamic_filter/sirius_dynamic_filter.hpp`

Additions to the channel; every existing member is unchanged.

```cpp
/**
 * @brief One column's visible filters inside a coherent snapshot
 */
struct column_filter_snapshot {
  std::size_t column;  ///< Consumer output ordinal — the channel's one coordinate
  std::vector<std::shared_ptr<sirius_dynamic_filter const>> filters;  ///< Insertion order
};

/**
 * @brief Coherent view of a channel: generation bound atomically to filter pointers
 *
 * The only legal input for predicate construction once refinement is enabled. Owning copies keep
 * superseded filters alive for in-flight consumers (main doc, "Coherent snapshots").
 * Generation-to-pointer coherence is the guarantee; `logical_filter_count` may lag `columns` by
 * an in-flight append — the pre-existing outside-mutex count bump is deliberately untouched for
 * join byte-equivalence.
 *
 * **The snapshot must outlive every artifact derived from it**, not merely the derivation. A
 * lowered AST tree references device scalars owned by the filters the snapshot holds, so the
 * snapshot — not just the tree — has to stay alive across the consuming call. Holding it in a
 * narrower scope than the `read_parquet`/evaluate call it feeds is a use-after-free.
 *
 * **Only a replacing producer can expose that error, which is why it can lie dormant.** For an
 * append-only producer the channel keeps its sole reference forever, so a prematurely dropped
 * snapshot reference costs nothing and the bug is unobservable. A refinement slot replaces its
 * filter on every accepted revision: once the channel's reference moves to the new filter, a
 * snapshot that has already died leaves refcount 0 and frees device scalars underneath a live
 * reader. Exactly this sat latent in the parquet consumer through two phases — the snapshot was
 * scoped to an inner block while its tree was correctly kept for the read — and surfaced as a
 * SIGSEGV only when Top-N became the first producer to supersede a filter.
 *
 * Consequence for whoever adds the *next* replacing producer: re-audit every consumer's snapshot
 * scope. That the existing consumers are correct is not evidence of anything — it is an artifact
 * of nothing having replaced before. A workload that republishes hard is what makes the window
 * reachable: the regression stress uses a descending variant republishing ~4,340 times per run
 * (~26,000 boundary replacements) against ~18 for ascending, and the fix was accepted at 2400
 * clean iterations where ~1000 had reproduced the original crash.
 */
struct dynamic_filter_snapshot {
  std::uint64_t generation = 0;
  std::size_t logical_filter_count = 0;
  std::vector<column_filter_snapshot> columns;
};

/**
 * @brief Publisher result for a refinement-slot replacement
 */
enum class refinement_publish_result { ACCEPTED, STALE, CLOSED, IGNORED };

/**
 * @brief Capability handle for replacing one refinement slot's filter
 *
 * Move-only and bound to one (channel, slot); it cannot retarget. Exactly one policy-owning
 * coordinator holds it — the slot supplies sequencing, stale-write rejection, and atomic
 * visibility, never semantic-strengthening checks (main doc, "Versioned refinement slots").
 * Thread-safe; outlives nothing: the channel is co-owned via `shared_ptr`.
 */
class dynamic_filter_refinement_publisher final {
 public:
  dynamic_filter_refinement_publisher(dynamic_filter_refinement_publisher&&) noexcept;
  dynamic_filter_refinement_publisher(dynamic_filter_refinement_publisher const&) = delete;

  /**
   * @brief Install @p ready_filter at @p producer_revision; rejects stale/closed/ignored
   *
   * An accepted call installs the immutable filter, bumps the channel generation, and counts
   * `filter_count` only for the slot's first value. A null @p ready_filter returns `IGNORED`.
   */
  refinement_publish_result publish(
    std::uint64_t producer_revision,
    std::shared_ptr<sirius_dynamic_filter const> ready_filter) const;
};

class sirius_dynamic_filter_set : public std::enable_shared_from_this<sirius_dynamic_filter_set> {
 public:
  // ... existing members unchanged ...

  /**
   * @brief Plan-time only: create a stable refinement slot at @p primary_ordinal
   *
   * @pre The set is owned by a `shared_ptr` (every production channel is) — the returned
   * publisher takes shared co-ownership via `shared_from_this`, so a late publish after
   * consumer teardown reports `CLOSED` instead of touching freed state; a non-shared set throws
   * `std::bad_weak_ptr`.
   *
   * Also registers a producer (see @ref register_producer). Each call mints a distinct slot;
   * separate producers targeting one channel receive separate slots. @p referenced_ordinals
   * lists every additional consumer ordinal a multi-column filter in this slot may reference
   * (empty for single-column slots); a slot whose primary or referenced ordinal is ignored via
   * @ref ignore_columns rejects publications. Storage and lookup remain keyed by the primary
   * ordinal — the join path's single-ordinal contract is untouched.
   */
  [[nodiscard]] dynamic_filter_refinement_publisher register_refinement_slot(
    std::size_t primary_ordinal, std::vector<std::size_t> referenced_ordinals = {});

  /**
   * @brief Coherent snapshot of columns, filter pointers, count, and generation
   */
  [[nodiscard]] dynamic_filter_snapshot snapshot() const;

  /**
   * @brief Lock-free advisory change hint; never pair with separate filter reads
   */
  [[nodiscard]] std::uint64_t generation() const noexcept;
};
```

#### Modified file: `src/include/op/scan/dynamic_filter_gate.hpp`

Replacement does not grow `filter_count()`, so the gate's growth rule becomes marker-based. On
LP64 a count-typed and a generation-typed `record_keep_ratio` are the same function type, so
there is exactly **one** marker-typed recorder; the marker's domain is fixed per gate instance —
the Top-N prefilter gate only ever receives its coordinator's boundary-update count, scan and
endpoint gates only ever receive channel generations. Mixing domains on one instance is a
programming error, enforced by call-site discipline, not runtime tagging.

```cpp
class dynamic_filter_gate {
 public:
  // ... existing members unchanged during migration ...

  /// Snapshot-based applicability: work is due when filters exist and the gate is active or the
  /// snapshot's generation has advanced past the disabling decision's marker.
  [[nodiscard]] bool applicable(sirius::op::dynamic_filter_snapshot const& snap) const;

  /// Record one split's keep ratio against the monotonic marker it measured under (channel
  /// generation, or the prefilter's coordinator update count — one domain per instance). An
  /// older completing measurement cannot overwrite a newer-marker decision; ACTIVE stays
  /// terminal.
  void record_keep_ratio(std::size_t rows_before,
                         std::size_t rows_after,
                         std::uint64_t observed_marker);
```

The retained set-based `applicable`/gated-view pair is internally generation-domain since Stage
2 (bit-identical on append-only channels); its full retirement rides the Stage-4 consumer
cleanup.

#### Modified file: `src/include/op/scan/dynamic_filter_merge.hpp`

Snapshot-consuming overloads of `scan::merge_dynamic_filters_into_ast`,
`apply_dynamic_filters_to_view`, and `apply_dynamic_filters_gated_view` (same semantics,
`dynamic_filter_snapshot const&` in place of the live set); the resolver-based
`merge_ast_dynamic_filters_into_tree` in the channel header was untouched at this stage (it was
deleted later — see the untouched-declarations table's note). Parquet and native
consumers switch to one snapshot per checkpoint; the set-based overloads retire in the Stage-4
consumer cleanup.

---

### Stage 3 — Range and lexicographic filters *(implemented)*

#### Modified file: `src/include/op/dynamic_filter/sirius_dynamic_filter.hpp`

Both layer filters live beside the zone map — same header, same AST-and-replication pattern,
PIMPL-free because they own only per-device `cudf::scalar`s like the zone map does.

```cpp
enum class sirius_dynamic_filter_kind { ZONE_MAP, IN_LIST, BLOOM, RANGE, LEX_RANGE };

enum class range_bound_side { LOWER, UPPER };

/**
 * @brief What a RANGE predicate does with null probe values
 */
enum class dynamic_filter_null_policy { ADMIT, REJECT };

/**
 * @brief Immutable one-sided range filter: keeps rows on one side of an exact boundary
 *
 * Lowers to `col < B` / `col <= B` / `col > B` / `col >= B`, wrapped per @ref
 * dynamic_filter_null_policy (`IS NULL OR pred` to admit, bare comparison to reject). Not a
 * synthetic zone map: one meaningful side, no sentinel bound (main doc, "Range and lexicographic
 * filters").
 *
 * @pre The boundary's storage type equals the consumer column's type.
 */
class sirius_dynamic_range_filter final : public sirius_dynamic_filter,
                                          public sirius_ast_lowerable,
                                          public sirius_compaction_applicable,
                                          public sirius_device_replicable {
 public:
  /**
   * @throw std::invalid_argument if @p bound's type is outside the admitted allowlist
   * @throw std::runtime_error if the current CUDA device cannot be identified
   */
  sirius_dynamic_range_filter(exact_host_scalar bound,
                              range_bound_side side,
                              bool inclusive,
                              dynamic_filter_null_policy null_policy);
  ~sirius_dynamic_range_filter() noexcept override;

  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  {
    return sirius_dynamic_filter_kind::RANGE;
  }

  [[nodiscard]] cudf::ast::expression const& to_ast(cudf::ast::tree& tree,
                                                    cudf::ast::expression const& column_ref,
                                                    int device_id = -1) const override;

  /**
   * @brief Materialize the boundary scalar on every planned consumer device
   *
   * Unlike the join filters' best-effort per-target policy, RANGE replication is all-or-nothing:
   * any target failure throws, the caller installs nothing, and the previous revision stays
   * visible on every device (main doc, "Multi-GPU publication"). Replica construction must go
   * through reservation admission, so an exhausted target budget is a deterministic failure
   * injection point.
   */
  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  [[nodiscard]] exact_host_scalar const& bound() const noexcept;
  [[nodiscard]] range_bound_side side() const noexcept;
  [[nodiscard]] bool inclusive() const noexcept;
  [[nodiscard]] dynamic_filter_null_policy null_policy() const noexcept;
};

/**
 * @brief Capability mixin: filter applies itself on device by fused predicate + compaction
 *
 * The device row-wise sibling of the AST path: one kernel pass, no BOOL8 mask. Implemented by
 * RANGE and LEX_RANGE over `detail::apply_boundary_filter`; consumers dispatch on the capability
 * and never see the kernel. AST lowering remains solely for the parquet reader checkpoint.
 *
 * This path requires **no device replicas**: the boundary rides the kernel's launch parameters,
 * so compaction works on any device from host state alone. `is_available_on_device` therefore
 * gates only the AST path's literal scalars.
 *
 * **Known gap — the consumer leg of the type contract is open.** The caller must ensure the
 * batch's key columns carry the same storage types the filter's boundary was built from; the
 * gated-apply path currently checks only the ordinal range, so a mismatch reads the consumer's
 * columns at the producer's widths. This is reachable in principle because cuDF derives a
 * decimal's width from the parquet *physical* type rather than the logical precision: a file
 * storing `DECIMAL(9,2)` as physical INT64 — legal parquet, though not a DuckDB or Spark
 * default — decodes as `DECIMAL64` while the catalog maps `DECIMAL32`. The fix is to pass the
 * batch through unfiltered on any type mismatch, matching the producer-side guard.
 */
class sirius_compaction_applicable {
 public:
  virtual ~sirius_compaction_applicable() = default;

  /**
   * @brief Filter @p batch in one fused pass using the device-local replica for @p device_id
   *
   * @param[in] key_columns The filter's component columns in @p batch, primary first; a RANGE
   * caller passes its single channel column.
   * @return Null `filtered` when nothing was dropped or no replica applies; `rows_kept` valid.
   */
  [[nodiscard]] virtual detail::boundary_filter_result apply_compact(
    cudf::table_view const& batch,
    std::span<cudf::size_type const> key_columns,
    int device_id,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;
};

/**
 * @brief Capability mixin: filter lowers itself against several consumer columns
 *
 * The multi-column sibling of @ref sirius_ast_lowerable. Instead of one pre-emplaced column
 * reference, the consumer supplies its existing @ref column_ref_resolver_fn; the filter resolves
 * each ordinal it references.
 */
class sirius_multi_column_ast_lowerable {
 public:
  virtual ~sirius_multi_column_ast_lowerable() = default;

  /**
   * @brief Lower to a BOOL fragment referencing every component column
   *
   * Nodes are owned by @p tree; device scalars referenced by literals are owned by the filter.
   *
   * @param[in] resolver Maps a consumer output ordinal to an AST column expression already
   * emplaced in @p tree. Invoked once per referenced ordinal; must stay valid for the call.
   */
  [[nodiscard]] virtual cudf::ast::expression const& to_ast(
    cudf::ast::tree& tree, column_ref_resolver_fn const& resolver, int device_id = -1) const = 0;

  /**
   * @brief Consumer ordinals this filter references, primary first
   */
  [[nodiscard]] virtual std::span<std::size_t const> referenced_ordinals() const noexcept = 0;
};

/**
 * @brief One LEX component's semantics and its consumer-side column binding
 */
struct lex_component_semantics {
  std::size_t consumer_ordinal;  ///< In the target's output space; component 0 is the primary
  top_n_key_semantics key;
};

/**
 * @brief Immutable lexicographic boundary filter over the full ORDER BY tuple
 *
 * Lowers to the prefix-disjunction `T0 OR (E0 AND T1) OR ...` with the per-component null
 * derivations from the main doc's table; the inclusive form appends the all-equal disjunct
 * `E0 AND ... AND En` (group-key producer — boundary-tied rows are never dropped). A null tail
 * component contributes `IS NULL` / `IS NOT NULL` terms and owns no device scalar; the first
 * component must be non-null. Never decomposed into per-column filters — the no-tail lemma at
 * the representation level.
 *
 * @pre `boundary.size() == components.size() >= 2` (a single-key producer publishes RANGE).
 * @pre `boundary.component(0)` is engaged.
 */
class sirius_dynamic_lex_range_filter final : public sirius_dynamic_filter,
                                              public sirius_multi_column_ast_lowerable,
                                              public sirius_compaction_applicable,
                                              public sirius_device_replicable {
 public:
  sirius_dynamic_lex_range_filter(exact_host_key_tuple boundary,
                                  std::vector<lex_component_semantics> components,
                                  bool inclusive);
  ~sirius_dynamic_lex_range_filter() noexcept override;

  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  {
    return sirius_dynamic_filter_kind::LEX_RANGE;
  }

  [[nodiscard]] cudf::ast::expression const& to_ast(
    cudf::ast::tree& tree,
    column_ref_resolver_fn const& resolver,
    int device_id = -1) const override;
  [[nodiscard]] std::span<std::size_t const> referenced_ordinals() const noexcept override;

  /**
   * @brief All-or-nothing, like RANGE: one scalar per non-null component per planned device, or
   * throw and install nothing
   */
  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  [[nodiscard]] exact_host_key_tuple const& boundary() const noexcept;
  [[nodiscard]] std::vector<lex_component_semantics> const& components() const noexcept;
  [[nodiscard]] bool inclusive() const noexcept;
};
```

---

### Stage 4 — External consumers *(implemented)*

#### New file: `src/include/op/dynamic_filter/top_n_dynamic_filter_publish_plan.hpp`

```cpp
namespace sirius::op {

/**
 * @brief Which layer a target receives
 */
enum class top_n_filter_layer { FIRST_KEY, LEX };

/**
 * @brief Immutable routing and placement for one Top-N producer
 *
 * Frozen at plan time; owns no boundary and no mutable revision. Reuses
 * `dynamic_filter_replica_space` but not the join-specific admitted-key or source-policy plan.
 * `replica_spaces` covers every planned consumer device, including endpoint devices. A
 * FIRST_KEY target subsumed by a LEX target at the same site is never planned.
 */
class top_n_dynamic_filter_publish_plan final {
 public:
  struct key {
    std::size_t child_ordinal;        ///< At the Top-N child; traces remap it per site
    top_n_key_semantics semantics;
    bool type_admitted;               ///< Per-key allowlist verdict
  };

  struct target {
    dynamic_filter_refinement_publisher publisher;  ///< Bound to its (primary) consumer ordinal
    top_n_filter_layer layer;
    /// The key components' ordinals in **this** target's output space, primary first, mirroring
    /// the slot's declared ordinals. Each target's trace remaps the keys independently, so a LEX
    /// filter — which owns the ordinals it references — is site-specific and is built per target
    /// (see "Per-target LEX construction" below). A FIRST_KEY target carries one entry, which
    /// the publisher's push coordinate already supplies; it is recorded for symmetry and
    /// plan-shape assertions.
    std::vector<std::size_t> component_ordinals;
  };

  std::size_t k;                      ///< Checked limit + offset
  std::vector<key> keys;              ///< Complete ORDER BY, in order; keys[0] feeds FIRST_KEY
  std::vector<target> targets;        ///< Scan channels and endpoint channels, uniformly
  std::vector<dynamic_filter_replica_space> replica_spaces;
};

}  // namespace sirius::op
```

The plan is held `const` by the local operator. The coordinator receives it through a plan-time
setter, not the constructor:

```cpp
  /**
   * @brief Install the frozen publish plan; plan-time only
   *
   * @throw std::logic_error if the coordinator has left the open state or has already accepted
   * an offer — the precondition is load-bearing, not defensive.
   */
  void set_publish_plan(top_n_dynamic_filter_publish_plan plan);
```

**Why the precondition is load-bearing.** The publisher loop reads the plan's targets without
holding `_mu`, which is sound only because every plan write happens-before every such read.
`_boundary_updates` is incremented under `_mu` strictly before `_pending` is assigned, so a
successful plan write completes in a critical section that precedes the first offer's critical
section, which precedes that offer's `publisher_loop()` in program order; the mutex edge orders
the write before every unlocked read, and the same edge carries `_target_closed` across
successive publisher owners. Accepting a plan after an offer would break that chain and make the
lock-free reads a data race. A future author restoring constructor injection must preserve the
same ordering guarantee — deleting the throwing precondition without replacing it silently
removes what makes the reads safe.

A producer with no discovered target simply never receives a plan; the self-consumption-only path
is unchanged.

**Per-target LEX construction.** The two layers are deliberately asymmetric. A single-column
`RANGE` is ordinal-free: its consumer ordinal is the channel's push coordinate, so **one
immutable object is shared across every accepting first-key target** — the same property the join
path relies on with its per-target `key_binding::channel_push_ordinal`.
`LEX_RANGE` cannot: only its primary ordinal can be the push coordinate, so the remaining
components' ordinals live inside the filter (`lex_component_semantics::consumer_ordinal`), and
each target's trace remaps them differently. One LEX object therefore cannot serve two sites with
different mappings. The publisher builds **one LEX filter per accepting LEX target**, from the
same boundary tuple and semantics, differing only in `component_ordinals`.

Any plan pairing a scan bind with an endpoint forces this, as would set-operation fan-out once it
is supported — each branch binds its own scan at its own ordinals. The design always implied
site-specific LEX filters; what was under-specified is that this makes construction per-target,
and that only ordinal-free layers literally "fan the same object".

**LEX endpoints are deferred.** `place_endpoint` traces one ordinal, so it cannot address an
all-keys stop point that differs from key zero's. Stage 4 therefore delivers LEX **only at scan
binds**; a non-scan LEX terminal is classified `LEX_ENDPOINT_DEFERRED`, counted separately, and
nothing is spliced. This is always sound — the sink prefilter applies the identical predicate, so
only pruning is lost — and it is a staged gap, not a cost decision, which is why it does not fold
into the skipped-site counter.

Siting one needs a multi-ordinal `place_endpoint` that guarantees three things the single-ordinal
form never had to: every component ordinal survives the *same* hop into the *same* child (a set
that splits across a join's two blocks stops at the join output — exactly the arrive-together
semantics), each ordinal is remapped independently at the splice, and the sited operator's input
schema addresses all of them. A span-taking form with the current signature as a delegating
wrapper would keep the join path byte-equivalent, but it generalizes a routine on the join's
critical path for a narrow near-term win: with the minimal hop set almost everything between an
all-keys stop point and the sink is pass-through, so the cost gate would skip most such sites
anyway. The exception is real but uncommon — a residual `FILTER` above the arrive-together point,
which DuckDB usually pushes below the join. The work therefore belongs with the trace widening
that makes endpoints broadly material (Stage 7), not with Stage 4.

Cost: N accepting LEX targets means N filters and N replica sets per revision. Each carries one
host boundary tuple and at most one device scalar per non-null component per device — realistic N
is one or two (branch count), so this is immaterial against the join filters' hash sets and Bloom
bitsets. A cheaper formulation exists if N ever grows — make the filter ordinal-free over
component indices and carry the per-site mapping in the slot, which already declares its
referenced ordinals — but it would change the implemented Stage-2 snapshot surface for no
measured benefit and is deliberately not taken now.

#### Modified file: `src/include/planner/dynamic_filter/dynamic_filter_target_discovery.hpp`

The Top-N self-trace reuses `descent_steps` / `trace_probe_key` / `place_endpoint` unchanged in
shape; a policy bit restricts hops to the minimal set.

```cpp
struct descent_policy {
  // ... existing fields unchanged ...

  /**
   * @brief Restrict hops to the Top-N self-trace set
   *
   * Accepts plain-reference projections, FILTER pass-through/gather, and endpoints; refuses
   * joins, aggregates, set operations, and every other operator, so each trace yields exactly
   * one terminal. Positional UNION stays a fan-out hop for the join policy but is a terminal
   * here: Sirius rejects set operations at planning, so a fan-out branch would be untestable
   * machinery. Stage 7 widens this set per proven hop, not by flipping the bit off.
   */
  bool top_n_self_trace = false;
};

/**
 * @brief Trace the full ordinal set toward the deepest schema where every key coexists
 *
 * The all-keys counterpart of `trace_probe_key` for the LEX layer: a hop is accepted only when
 * every ordinal in @p key_ordinals survives it, each remapped independently. The terminal
 * carries all remapped ordinals, primary (key zero) first. Terminates at worst at @p root
 * itself, which always exists.
 */
[[nodiscard]] std::vector<multi_key_route_terminal> trace_top_n_all_keys(
  sirius::op::sirius_physical_operator& root,
  std::span<std::size_t const> key_ordinals,
  descent_policy policy);

/**
 * @brief Classify one Top-N trace terminal into its target kind
 *
 * A terminal is worth siting only if it saves work nothing else already saves (main doc, "Siting
 * rule"). At least one of these must hold:
 *
 *  - `consumer_skips_reads` — the consumer turns the predicate into data never read (Parquet's
 *    reader `set_filter` prunes row groups by statistics). Sited unconditionally; the saving is
 *    upstream of any pass. This is a plan-time answer a pinned-cache serve can falsify at
 *    runtime; siting stands, and consumption flips post-decode at prepare (Phase 1 section).
 *  - `material_hops_above > 0` — real per-row work sits between the site and the sink, so the
 *    site's compaction pass buys more than it costs.
 *
 * A terminal meeting neither is `SKIPPED_NO_WORK_SAVED`. This subsumes the old
 * `ENDPOINT_SKIPPED_IMMATERIAL` rule and extends it to scan binds, which were previously exempt
 * on the false premise that binding to a scan always avoids reads — true for Parquet, false for
 * the DuckDB-native scan, whose only filter path is post-decode. A native scan-sited target
 * duplicates the sink's own self-consumption pass and measurably regresses.
 *
 * `consumer_skips_reads` is a property of the consumer's filter path, not of its format tag:
 * ask the target whether a predicate reduces what it reads, so a future backend that gains
 * read-time filtering is admitted without editing this rule.
 *
 * A FIRST_KEY terminal coinciding with a planned LEX target's site is marked subsumed.
 */
enum class top_n_target_kind {
  SCAN_BIND,
  ENDPOINT_SITE,
  /// Neither read-skipping nor material downstream work: the site would duplicate the sink's own
  /// pass. Replaces `ENDPOINT_SKIPPED_IMMATERIAL` and now also covers post-decode-only scans.
  SKIPPED_NO_WORK_SAVED,
  SUBSUMED_BY_LEX,
  /// A LEX terminal that is not a scan. Siting one requires addressing every component ordinal
  /// at the splice, which the single-ordinal `place_endpoint` cannot express; deferred to the
  /// trace-widening stage (see "LEX endpoints are deferred"). Distinct from
  /// `SKIPPED_NO_WORK_SAVED` so a staged gap never hides inside a cost-gate decision.
  LEX_ENDPOINT_DEFERRED
};

[[nodiscard]] top_n_target_kind classify_top_n_terminal(route_terminal const& terminal,
                                                        top_n_filter_layer layer,
                                                        std::size_t material_hops_above,
                                                        bool consumer_skips_reads);
```

Endpoint splicing itself is the existing `place_endpoint` with an `endpoint_factory` that
constructs `sirius_physical_dynamic_filter`; Top-N layers apply through
`sirius_compaction_applicable` (join-path zone maps keep the AST row-mask mode).

```cpp
/**
 * @brief Whether a boundary key may be compared against a target site's column
 *
 * Wired at all three target sites. The rule is plain type identity —
 * `key_storage_type == site_column_type` — for **every** type, with no fixed-point carve-out: an
 * `is_fixed_point` exception was written and then deleted during review because no reachable case
 * needed it, and three assertions that had pinned the weaker behavior became `REQUIRE_FALSE`.
 *
 * This is what makes `exact_host_scalar::compare`'s "operands share one storage type"
 * precondition true in practice. That precondition is otherwise unenforced: the comparison widens
 * to `int64` and never consults the storage type, so two decimals of different scale would
 * compare as raw integers and prune wrongly with no error anywhere.
 */
[[nodiscard]] bool boundary_key_matches_site_type(cudf::data_type key_storage_type,
                                                  cudf::data_type site_column_type) noexcept;
```

#### New file: `src/include/planner/top_n_key_types.hpp`

The per-key type allowlist as a **pure function** with its own unit test, rather than a predicate
buried in the planner:

```cpp
/**
 * @brief Exact cuDF storage type for an admitted ORDER BY key type, or empty when refused
 *
 * Admits `TINYINT`/`SMALLINT`/`INTEGER`/`BIGINT`/`DATE` and `DECIMAL` of precision 5–18
 * (p ≤ 9 → `DECIMAL32`, p ≤ 18 → `DECIMAL64`). Refuses p ≤ 4 (INT16-backed, no cuDF counterpart)
 * and p ≥ 19 (`DECIMAL128`: the kernel's width switch handles 1/2/4/8 only).
 */
[[nodiscard]] std::optional<cudf::data_type> admitted_key_storage_type(
  duckdb::LogicalType const& type);
```

It lives here so each rule is falsifiable in isolation. The p ≥ 19 refusal in particular is
**reachable in production**: the native scan's precision gate does not cover parquet, so a Top-N
over a parquet-backed `DECIMAL(38,4)` builds a plan, reaches admission, and is refused with no
slot registered — pinned by test. Without that refusal a decimal128 key would reach a width
switch that returns 0 in release rather than failing.

#### Modified file: `src/include/op/scan/sirius_physical_dynamic_filter.hpp`

The installation-site hook for tests, and the third planner role.

```cpp
/// @brief Why the planner installed this operator — exposed for plan-shape tests and telemetry.
enum class dynamic_filter_endpoint_provenance { scan_route, join_edge, top_n_endpoint };

class sirius_physical_dynamic_filter : public sirius_physical_operator {
 public:
  // Existing constructor unchanged; provenance defaults preserve current call sites.
  sirius_physical_dynamic_filter(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::shared_ptr<sirius::op::sirius_dynamic_filter_set> filters,
    double gate_keep_threshold     = dynamic_filter_gate::k_default_keep_threshold,
    dynamic_filter_apply_mode mode = dynamic_filter_apply_mode::membership_masks_only,
    dynamic_filter_endpoint_provenance provenance = dynamic_filter_endpoint_provenance::scan_route);

  [[nodiscard]] dynamic_filter_endpoint_provenance provenance() const noexcept;
};
```

A Top-N endpoint is this operator with `top_n_endpoint` provenance, dispatching Top-N filters
through the fused-kernel capability — no new operator type, matching the Phase 2 precedent.

Stage-4 consumer cleanup: retire the set-based gate `applicable`/gated-view overloads — already
internally generation-domain since Stage 2 — once the last consumer takes snapshots.

---

### Stage 5 — Aggregate group-key producer

#### Modified file: `src/include/op/dynamic_filter/top_n_threshold_coordinator.hpp`

Distinct-key mode (main doc, "The group-key producer"): the mode is fixed at construction and
everything downstream of the boundary — tightening, publisher loop, `finish`, `cancel` — is
shared with row mode.

**Admission cap.** A group-key producer is admitted only for `k <= k_max_group_key`, a structural
constant of **1024** on the coordinator (a named constant, not a configuration option — the same
treatment as `boundary_filter_params::k_max_components`). The justification is **collapsing
pruning value, not rising cost**: measured per-batch cost is nearly flat in K (≈16–21% from K=10
to K=1000 — `distinct` and `sort` dominate and are K-independent), while rows kept climb from
0.2% at K=10 to 20% at K=1000 for 5000 distinct groups, and to 83% when groups are scarce. Beyond
the cap the producer buys almost nothing, so refusing beats optimizing. The row producer takes no
cap: it extracts one row's key tuple per batch regardless of K. A refusal increments
`top_n_group_producers_rejected` alongside the other eligibility refusals rather than earning its
own counter — these are mutually exclusive plan-time verdicts of one kind, and K is visible in
the query, so a separate name would add no diagnostic power. (Contrast `LEX_ENDPOINT_DEFERRED`,
which does have its own counter because a *staged capability gap* must not hide inside a
*tuning decision*; two eligibility refusals are the same kind.)

```cpp
/**
 * @brief Which witness discipline the coordinator runs
 */
enum class top_n_producer_kind { ROW, GROUP_KEY };

/**
 * @brief One batch's best distinct ORDER-BY key values, host-extracted
 *
 * Sorted best-first, deduplicated, at most K entries.
 *
 * The witness owns **completed host values and nothing else**: extraction synchronizes its
 * stream before construction, and the copies read from a table local to the extraction rather
 * than from the caller's batch. It therefore holds no device reference, and keeping the input
 * batch alive is not required — the durability question is moot for this producer, not merely
 * satisfied by some other mechanism.
 *
 * The requirement that still binds any *future* producer: a witness that retains a device
 * reference must keep it valid until the offer completes, by co-ownership or by a lock the seam
 * holds across both. Such a producer adds the handle it needs; this one carries none.
 */
struct top_n_distinct_key_witness {
  std::vector<exact_host_key_tuple> best_keys;
};

class top_n_threshold_coordinator final {
 public:
  // Row-mode constructor unchanged; GROUP_KEY mode selects the distinct-key discipline. `stats`
  // is the Stage-1 non-owning counter sink and keeps its position.
  top_n_threshold_coordinator(std::size_t k,
                              std::vector<top_n_key_semantics> keys,
                              bool lex_admitted,
                              dynamic_filter_stats* stats,
                              top_n_producer_kind kind);

  /**
   * @brief GROUP_KEY mode only: merge a batch's distinct keys into the bounded witness set
   *
   * Union by key value under the coordinator mutex, truncated to the K best; the boundary is the
   * set's Kth element once the set is full and only tightens afterwards. Sub-K batches
   * contribute, unlike row-mode offers.
   */
  threshold_offer_result offer(top_n_distinct_key_witness witness);
};
```

#### Modified files: `src/include/op/sirius_physical_grouped_aggregate.hpp`, `src/op/sirius_physical_grouped_aggregate.cpp`

The producer seam in the partial aggregate sink, mirroring the Top-N seam: a
`threshold_coordinator` shared_ptr (set by the planner; null when ineligible) and a per-operator
prefilter gate. Per input batch, `execute()`-internal helpers (not public API): gated inclusive
prefilter against `tightest_boundary()` before hash insert, then bounded distinct-key extraction
— sort/unique on the ORDER-BY key columns, truncate to K, host copies on the task stream with
observed completion — and the distinct-key offer. The merge aggregate is untouched; only the
Top-N's merge calls `finish()`.

#### Modified file: `src/include/op/dynamic_filter/top_n_dynamic_filter_publish_plan.hpp`

`top_n_producer_kind kind;` joins the plan. A `GROUP_KEY` plan roots its traces at the
aggregate's input, forces both layers inclusive, and is otherwise identical — targets, replica
spaces, and slots are reused unchanged.

#### Modified file: `src/include/op/dynamic_filter/dynamic_filter_stats.hpp`

```cpp
  // --- Top-N group-key producer (Stage 5) ---
  std::atomic<std::uint64_t> top_n_group_producers_eligible{0};  ///< Plan-time fact
  std::atomic<std::uint64_t> top_n_group_producers_rejected{0};  ///< Aggregate-output key, filter,
                                                                 ///< K unrepresentable, or K above
                                                                 ///< the group-key admission cap
  std::atomic<std::uint64_t> top_n_group_offers{0};              ///< Distinct-key offers merged
  std::atomic<std::uint64_t> top_n_group_witness_set_full{0};    ///< Boundary became defined
  std::atomic<std::uint64_t> top_n_group_prefilter_rows_in{0};
  std::atomic<std::uint64_t> top_n_group_prefilter_rows_out{0};
```

Three ROW-named counters deliberately have **no** group twin. `top_n_prefilter_disabled` and
`top_n_offers_unsupported` are shared: both record a condition of the coordinator/gate machinery
that is identical in either mode, and a producer's kind is already known from the plan, so
splitting them would add names without adding information. `top_n_producers_first_key_only` has
no group equivalent and is never bumped by a group-key producer: tail-type degradation is a
row-producer concept, and a group-key producer whose tail type is unadmitted simply publishes the
first-key layer under the same eligibility counters. Read a ROW-named counter moving during a
group-key query as shared accounting, not as a mislabelled bug.

#### Threading and lifetime summary

| Element | Mutability | Guard | Lifetime |
|---|---|---|---|
| `exact_host_scalar` / `exact_host_key_tuple` | Immutable | None | Value types |
| `top_n_key_semantics` / `lex_component_semantics` | Immutable | None | Value types, frozen at plan time |
| `top_n_threshold_witness` / `top_n_distinct_key_witness` | Immutable after creation | None | Until offer/publication completes |
| `top_n_threshold_coordinator` | Mutable | One internal mutex; single publisher loop | Execution-scoped, shared_ptr from both operators |
| `top_n_dynamic_filter_publish_plan` | Immutable | None | Owned `const` by the plan |
| Refinement slot / generation | Mutable | Channel mutex | Channel lifetime (co-owned) |
| `dynamic_filter_snapshot` | Immutable | None | Consumer-held; keeps filters alive |
| `sirius_dynamic_range_filter` / `sirius_dynamic_lex_range_filter` | Immutable after replication | None post-publication | Until last snapshot releases them |
| `prefilter_gate` / endpoint gate | Mutable | Gate's existing internal locks | Execution-scoped |
| `dynamic_filter_stats` additions | Atomics | Per-field relaxed | Connection lifetime |

### Phase 1 — Pinned-serve consumption flip *(implemented with this change)*

A pinned-cache-served parquet scan runs no reader, so the Stage-4 assumption "AST filters
already ran as scan-time row-group pruning" fails at runtime and `membership_masks_only`
left Top-N boundaries and join zone maps unapplied. The fix is a prepare-time latch plus a
monotone mode promotion in the wrapper (main doc, "Pinned-cache-served scans").

#### New file: `src/include/op/scan/read_time_filter_bypass.hpp`

```cpp
/// One-way, per-execution latch: this scan's batches will not pass through a read-time
/// dynamic-filter phase in this execution. Created by the plan generator when it wraps a
/// scan; co-owned by the scan operator and its wrapper. Single writer:
/// sirius_scan_manager::prepare_for_query marks it strictly before pipeline execution, so
/// consumers never observe a mid-query change and the keep-ratio gate trains under one
/// mode. Plans are per-execution, so the latch needs no reset.
class read_time_filter_bypass {
 public:
  void mark_bypassed() noexcept;
  [[nodiscard]] bool bypassed() const noexcept;
};
```

#### Modified: `src/include/op/scan/sirius_physical_dynamic_filter.hpp`

Two appended, defaulted constructor parameters — `sirius::op::dynamic_filter_stats* stats`
(non-owning, `SiriusContext` lifetime, the hash-join contract) and
`std::shared_ptr<read_time_filter_bypass> read_bypass` (null for `join_edge` /
`top_n_endpoint` provenance) — plus:

```cpp
[[nodiscard]] dynamic_filter_apply_mode effective_mode() const noexcept;
[[nodiscard]] std::shared_ptr<read_time_filter_bypass const> read_bypass() const noexcept;
```

`effective_mode()` returns the plan-time mode, promoted to `include_ast_row_masks` when the
latch is bypassed; promotion is monotone and settled before the first batch. (The Stage-4
snippet above that says "Existing constructor unchanged" is superseded by this appendix.)

#### Modified: `src/include/op/scan/sirius_gpu_scan_operator.hpp`

Appended defaulted `read_bypass` constructor parameter,
`void mark_served_from_pinned_cache() noexcept` (called by `prepare_for_query` at the
cache-hit commit point, after `validate_pinned_entry_for_serving` — a validation fallback to
disk is never marked), and a `read_bypass()` accessor for plan-shape tests.

#### Modified: `src/include/op/dynamic_filter/dynamic_filter_stats.hpp`

```cpp
std::atomic<std::uint64_t> post_decode_apply_rows_in{0};
std::atomic<std::uint64_t> post_decode_apply_rows_out{0};
```

Delivery-time; incremented by the wrapper only when a gated apply produced a result table.
They cover every provenance and capability, so a test isolating one capability uses a
channel carrying only that capability (a Top-N-only channel has no membership filters, so on
a pinned scan these counters move only if the flip engaged — the discriminator the
integration mutation check relies on). Appended at the end of the struct and snapshot.

**Serve-path invariant the flip rests on:** one scan operator in one execution is served
either entirely from resident pinned chunks or entirely through the reader
(`worker_loop` runs `process_cached_entries` XOR `process_provider_inputs`); parquet pins
carry no MVCC/delta side channel. Only duckdb-native pins append insert-delta splits, and
the native wrapper already runs `include_ast_row_masks`.

**Known gap — per-split reader bypass is still unconsumed.** On the fresh-read path a
`disable_filter_pushdown` split (BYTE_ARRAY-decimal probe) skips reader-side dynamic AST
while the wrapper stays membership-only: the pinned-serve miss in miniature, per split. No
writer available to the tests can emit that encoding (the probe itself is the declared
untested conservatism from Phase 0), and the per-scan latch deliberately does not express
per-split facts; if the encoding ever becomes constructible, this needs batch-level
signaling. **Known gap — WI-0b** (reader-path runtime gate) remains specified and
unimplemented; the flip neither implements nor replaces it.

---

## Part 2 — High-level test shape

### Layering

- **SQLLogic (`test/sql/top-n-dynamic-filter.test`)** — result equivalence only, across setting
  toggles (`enable_top_n_dynamic_filter` on/off, `gpu_execution` on/off). SQLLogic cannot observe
  plans or counters, so it carries the broad equivalence matrix and the negative cases cheaply.
- **Catch2 planner (`test/cpp/planner/test_top_n_dynamic_filter_plan_shape.cpp`,
  `[plan_tree_shape][isolated_context]`)** — installation-site assertions via the existing
  `generate_sirius_plan` + `collect`/`find_first` helpers plus the new `provenance()` getter.
  Plan shape is deterministic and device-free, so every siting/skip decision is pinned here.
- **Catch2 integration (`test/cpp/integration/test_gpu_execution_top_n_dynamic_filter.cpp`,
  `[integration][gpu_execution][dynamic_filter]`)** — GPU-vs-CPU equivalence plus pruning-effect
  assertions through before/after `sirius::test::get_dynamic_filter_stats_snapshot(con)` deltas,
  following the existing dynamic-filter integration tests.
**Standing criterion — a guard needs a test that triggers it.** A guard added because a failure
was hard to reach is exactly the guard most likely to be unfalsifiable, and this project has now
caught five: the negative-value guard, the K cap, the boundary-key type check, the
component-width guard, and the parquet pushdown-safety probe — each unpinned, each now either
tested or removed. Treat "I could not easily construct the failing case" as the reason to write
the test, not the excuse for skipping it; if the case is genuinely unconstructible through a
built plan, move the rule to a pure function and test it there (the allowlist's home is exactly
this).

**Deleting a guard's test is as consequential as deleting the guard.** The pushdown probe had two
regression tests when it was written against a real cuDF failure. A later file move removed them
wholesale, and the guard then sat unfalsifiable for two months while the upstream behavior it
defended against was fixed — so by the time anyone looked, a guard that had been *justified* was
indistinguishable from one that never was, and the only way to tell was to reconstruct the
original failure. That is the second failure mode: an untested guard does not merely risk being
wrong, it destroys the evidence of why it exists. When a move or refactor drops a guard's test,
it is deleting the guard's rationale; carry the test or delete the guard deliberately.

- **Catch2 kernel parity (`test/cpp/operator/test_top_n_boundary_filter.cpp`,
  `[dynamic_filter][top_n]`)** — `apply_boundary_filter` against a host reference over the full
  case matrix: per-type × direction × null-order × strict/inclusive × component count, null tail
  components, and empty / all-pass / all-fail batches. The per-type prefilter equivalence tests
  are the safety net and must pass unchanged against the kernel.

### Observability contract used by the assertions

Installation site: plan-shape layer (`DYNAMIC_FILTER` node presence, position, `provenance()`).
Pruning effect and lifecycle: stats deltas (`top_n_offers`, `top_n_prefilter_rows_*`,
`top_n_revisions_published`, the per-layer `top_n_{first_key,lex}_*` counters,
`top_n_endpoint_sites_*`, and the post-decode consumer pair
`post_decode_apply_rows_in/out`). Batch arrival order is not deterministic,
so row-level pruning counters are asserted as directions (`rows_out <= rows_in`, deltas `> 0`
only where a single-batch shape forces them), matching the sibling doc's counter-contract rules.

### Scenario matrix

Data: `topn_facts(id INT, v INT, w INT, grp INT, pay VARCHAR)` seeded deterministically (about
10k rows; `v` a permutation so thresholds tighten; `w` a second key with duplicates in `v` so
lexicographic tails matter; `w` nullable in the null-tail variants), written once as
DuckDB-native and once as Parquet. TPC-H `lineitem`/`orders` (SF0.01 from the existing
integration data) are reused where marked. "Site" names the expected installation; a single-key
shape publishes its strict predicate as RANGE, and multi-key shapes publish per the layer rules.

| # | Scenario | Query shape | Expected site and layer (stage it becomes real) | Test layers |
|---|---|---|---|---|
| 1 | Parquet scan, single key | `SELECT * FROM topn_parquet ORDER BY v LIMIT 10` | Strict RANGE in the scan's reader AST (S4) | plan, integration, sqllogic |
| 2 | Native scan, single key | Same over the native table | **No target sited**: the native scan cannot skip reads and nothing material separates it from the sink, so the siting rule refuses it and `top_n_sites_skipped_no_work_saved` increments. Self-consumption alone covers the shape | plan, integration |
| 3 | Aggregate disruptor | `SELECT grp, sum(v) s FROM t GROUP BY grp ORDER BY s DESC LIMIT 5` | No endpoint — site immaterial, self-consumption covers (S1/S4 skip assert); endpoint above the aggregate read-out only when material work intervenes, e.g. a dimension join between aggregate and Top-N (S7) | plan, integration |
| 4 | Expression-key projection | `SELECT a + b AS k FROM t2 ORDER BY k LIMIT 10` | Trace stops at the materializing projection; endpoint skipped as immaterial (S4 skip assert); self-consumption covers | plan, integration |
| 5 | Join disruptor, single key | `SELECT o.*, l.v FROM t l JOIN dim o ON ... ORDER BY l.v LIMIT 10` | Endpoint above the join is immaterial → skipped (S4); widened probe-block hop reaches the scan → scan bind (S7) | plan, integration |
| 6 | Set-operation terminal | `SELECT v, w FROM t1 UNION ALL SELECT v, w FROM t2 ORDER BY v, w LIMIT 10` | Unreachable while Sirius rejects set operations at planning: the query falls back to CPU, so there is no Sirius plan to assert on. Deferred with set-operation support, when the trace regains the fan-out hop and this becomes "LEX into each branch's scan" | deferred |
| 7 | Pass-through hops | `SELECT v FROM t WHERE grp <> 3 ORDER BY v LIMIT 10` (filter + plain projection) | Still the scan (S4) | plan, sqllogic |
| 8 | Self-consumption only | Any eligible shape with the channel stages disabled or no target | No `DYNAMIC_FILTER` node; `top_n_offers` delta > 0, prefilter direction asserts, results unchanged (S1) | integration, sqllogic |
| 9 | All keys, one scan | `SELECT * FROM topn_parquet ORDER BY v, w LIMIT 10` | LEX in the scan's reader AST; `top_n_first_key_subsumed_by_lex` delta > 0, no separate first-key filter on that channel (S4) | plan, integration, sqllogic |
| 10 | **Split keys across a join** | `SELECT l.v, o.w FROM t l JOIN dim o ON l.id = o.id ORDER BY l.v, o.w LIMIT 10` | Both traces stop at the join with the minimal hop set, so **neither layer reaches `l`'s scan**; the strict predicate is applied by the sink prefilter and the all-keys terminal is `LEX_ENDPOINT_DEFERRED` (assert the deferral counter, not an endpoint). Reaching `l`'s scan needs a hop through the join's probe block, which is provable but publishes to a channel already closed — see the main doc's reach ceiling | plan, integration, sqllogic |
| 11 | Mixed directions and null orders | `ORDER BY v DESC, w ASC NULLS FIRST LIMIT 10` and the transposed combos | Same sites as 9; per-component `T_i`/`E_i` derivations pinned by equivalence sweep (S4) | sqllogic, integration |
| 12 | Null tail boundary | Data forcing row K−1's `w` to null under both null orders | Publication proceeds through the derivation table; equivalence exact; `top_n_offers_unsupported` unchanged (S4) | integration, sqllogic |
| 13 | Unsupported tail type | `ORDER BY v, pay LIMIT 10` (VARCHAR tail) | First-key layer only: `top_n_producers_first_key_only` delta > 0, no LEX target, RANGE still reaches `v`'s scan (S4) | plan, integration |
| 14 | Negatives | See below | No producer / no publication (S1) | sqllogic + one integration counter case |
| 15 | **TopN over aggregate, integer keys (marquee)** | `SELECT grp, min(v) FROM topn_parquet GROUP BY grp ORDER BY grp LIMIT 5` | Group-key producer: inclusive RANGE into the scan's reader AST; row producer self-consumes above the barrier; `top_n_group_offers` and `…_witness_set_full` deltas > 0 (S5) | plan, integration, sqllogic |
| 16 | Aggregate-output key (Q3 shape) | `SELECT grp, sum(v) s FROM t GROUP BY grp ORDER BY s LIMIT 5` | No group-key producer: `top_n_group_producers_rejected` delta; scenario 3's skip asserts unchanged (S5) | plan, integration |
| 17 | Filter between aggregate and Top-N | Scenario 15 plus `HAVING min(v) > 0` | No group-key producer (S5) | plan, integration |
| 18 | Boundary-tie preservation | Scenario 15 data with duplicated boundary `grp` values and a `sum` aggregate | Exact equivalence — tied rows provably kept; inclusive predicates asserted via results, not counters (S5) | integration, sqllogic |
| 19 | Pinned-cache-served scan | Scenario 1's query after `CALL pin_table(...)`, GPU and HOST tiers, plus an unpinned rerun | Scan target sited exactly as in 1, but no reader runs: the prepare-time `read_time_filter_bypass` promotes the wrapper to `include_ast_row_masks`; boundary applies post-decode and `post_decode_apply_rows_in/out` move; the unpinned rerun keeps them at zero on a Top-N-only channel (both directions pinned — under-flip and over-flip each fail one leg) (Phase 1) | plan, operator, integration |

Negative sub-cases (14): a tie-preserving rank shape — pinned DuckDB v1.5.4 has no `WITH TIES`
grammar, so the negative is expressed as `RANK() OVER (ORDER BY …) <= n`, asserting no producer
counter moves — **first**-key-null boundary — `v`
nullable under `NULLS LAST` with fewer than K non-null keys (`top_n_offers_unsupported` delta, no
publication, exact results), table smaller than K (publish nothing), `LIMIT 7 OFFSET 5` (boundary
is the 12th row's tuple — a single-batch shape pins K arithmetic via offer count and result
equivalence), `ORDER BY pay, v` (VARCHAR **first** key: `top_n_producers_rejected`, no producer —
contrast with scenario 13 where only the tail degrades), `LIMIT 0` (no producer, empty result).

### Per-scenario anatomy (scenario 1 as the template)

```text
Schema   : topn_parquet(id INT, v INT, grp INT) — 10k rows, v = permutation of 1..10k,
           written as 8 row groups so reader pruning is observable.
Query    : SELECT id, v FROM topn_parquet ORDER BY v LIMIT 10;
Plan     : [plan] GPU_SCAN present; no DYNAMIC_FILTER node with provenance top_n_endpoint;
           the scan's channel has a registered Top-N producer (plan-shape helper).
Runtime  : [integration] gpu result == cpu result (enable_top_n_dynamic_filter=false rerun,
           then gpu_execution=false rerun); delta(top_n_offers) > 0;
           delta(top_n_revisions_published) >= 1; delta(top_n_revisions_failed) == 0.
Pruning  : direction-only — parquet rows-decoded telemetry where cuDF exposes it, else the
           prefilter row counters.
Stage    : equivalence rows runnable from S1 (filter inert), site/publication asserts from S4.
```

Scenarios 3–5 and 10 assert the *skip* explicitly at Stage 4 — the skipped-site counter
delta and absence of a `top_n_endpoint` node — so the cost-gate behavior is pinned, not
accidental. Their Stage 7 variants (join between aggregate and Top-N; probe-block descent) flip
the same assertions to `top_n_endpoint_sites_placed`/scan-bind and are written up front but
tagged `[!mayfail]` until Stage 7 lands.

### Multi-GPU shape

Two layers, mirroring `test_sirius_dynamic_filter_mgpu.cpp`:

- **Focused operator test** (`test/cpp/operator/test_sirius_dynamic_range_filter_mgpu.cpp`,
  skips below two devices): construct `sirius_dynamic_range_filter` and a
  `sirius_dynamic_lex_range_filter` with a null tail component against a two-GPU memory manager;
  assert replicas ready on both devices after `replicate_to_devices` (the null component owns no
  scalar on either); induce a target reservation denial and assert the all-or-nothing contract —
  the call throws, and a slot that held revision N still serves revision N on both devices for
  both layers (`top_n_revisions_failed` delta at the coordinator level).
- **End-to-end 2-GPU run** (integration, `integration-2gpu.yaml`): scenario 1, 9, and 10 queries
  on two GPUs; bit-identical results against single-GPU and CPU runs;
  `top_n_revisions_failed == 0`.

Every multi-GPU case above skips on a single-GPU host, so none of them is evidence until a
two-device run reports it green; a stage whose multi-GPU assertions have only been compiled is
not multi-GPU-verified. See the implementation notes for which runs are outstanding.

### SQLLogic sketch

`test/sql/top-n-dynamic-filter.test`: `require sirius` and the transparent path
(`gpu_buffer_init` is `SIRIUS_ENABLE_LEGACY`-only and must not appear); a file-backed database
via `load __TEST_DIR__/...` with `CHECKPOINT` after each `CREATE` — the native-scan GPU path
rejects in-memory tables ("requires a single-file block manager"). Then: create the native and
Parquet tables; foreach `enable_top_n_dynamic_filter` in (false, true) run the full query list
(scenarios 1, 3, 4, 5, 7, 9, 10, the scenario 11 direction/null-order sweep, 12, 15, 18, and
all negatives) with results compared against stored expected output; the `gpu_execution=false`
pass pins the CPU baseline. This file is runnable — and must pass with the filter inert — from
Stage 1 onward, which is what makes it the regression floor for every later stage.
