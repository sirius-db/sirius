# Dynamic Filters — Top-N API and Test Specification

> **Status: proposed (design only).** Companion to
> [dynamic-filters-top-n.md](dynamic-filters-top-n.md), which owns the design narrative and the
> normative contract. This document turns that contract into a header-level API surface and an
> example-driven high-level test plan. Stage numbers refer to the main doc's six-stage rollout.
> Nothing here is implemented.

## Scope and untouched surface

The append-only hash-join path does not change. The following existing declarations stay
byte-identical:

| File | Untouched declarations |
|---|---|
| `src/include/op/dynamic_filter/sirius_dynamic_filter.hpp` | `sirius_dynamic_filter`, `sirius_device_replicable`, `sirius_ast_lowerable`, `sirius_mask_applicable`, zone-map/IN-list/Bloom classes, `push_filter`, `filters_for_column`, `filtered_columns`, `empty`, `ignore_columns`, `register_producer`, `has_producers`, `close_for_new_filters`, `accepting_filters`, `has_filters`, `filter_count`, `merge_ast_dynamic_filters_into_tree`, `column_ref_resolver_fn` |
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

### Stage 1 — Coordinator and sink self-consumption

#### New file: `src/include/op/dynamic_filter/exact_host_scalar.hpp`

One exact typed host value. Introduced here for the coordinator's boundary; reused verbatim by
the Stage 3 RANGE filter. `std::variant` keeps exactness explicit — no value ever rounds through
`double` (the same rule the zone-map replication follows).

```cpp
namespace sirius::op {

/**
 * @brief Exact host-side scalar for the Top-N type allowlist
 *
 * Carries the value and its cuDF storage type together so comparison and device-scalar
 * construction cannot disagree about representation. The variant covers exactly the admitted
 * types (main doc, "Range and lexicographic filters"); widening the allowlist widens the variant.
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
   */
  top_n_threshold_coordinator(std::size_t k,
                              std::vector<top_n_key_semantics> keys,
                              bool lex_admitted);

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

Additive members only; the existing public fields and `execute` signature are unchanged. The same
two members are mirrored on `sirius_physical_top_n_merge` (its delegating constructor already
copies shared state from the local operator, exactly as it shares `dynamic_filter` today).

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
`sirius_physical_top_n.cpp`), not public API: read `tightest_boundary()`, build the strict LEX
predicate — or the degraded inclusive first-key comparison — as a task-local AST, evaluate with
`cudf::compute_column` + `apply_boolean_mask`, then `compute_top_n_table`, then extract every key
column of row `K - 1` and offer the witness.

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
  std::atomic<std::uint64_t> top_n_endpoint_sites_skipped{0};     ///< Plan-time; immaterial gap
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

### Stage 2 — Channel foundation

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
   * `filter_count` only for the slot's first value.
   */
  refinement_publish_result publish(
    std::uint64_t producer_revision,
    std::shared_ptr<sirius_dynamic_filter const> ready_filter) const;
};

class sirius_dynamic_filter_set {
 public:
  // ... existing members unchanged ...

  /**
   * @brief Plan-time only: create a stable refinement slot at @p primary_ordinal
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

Replacement does not grow `filter_count()`, so the gate's growth rule becomes generation-based.
The count-based overloads remain during migration and are removed when the last consumer moves.

```cpp
class dynamic_filter_gate {
 public:
  // ... existing members unchanged during migration ...

  /// Generation-aware applicability: work is due when filters exist and the gate is active or the
  /// channel generation has advanced past the disabling decision's generation.
  [[nodiscard]] bool applicable(sirius::op::dynamic_filter_snapshot const& snap) const;

  /// Record one split's keep ratio against the snapshot generation it measured. An older
  /// completing measurement cannot overwrite a newer-generation decision; ACTIVE stays terminal.
  void record_keep_ratio(std::size_t rows_before,
                         std::size_t rows_after,
                         std::uint64_t observed_generation);
```

#### Modified file: `src/include/op/scan/dynamic_filter_merge.hpp`

Snapshot-consuming overloads of `merge_ast_dynamic_filters_into_tree` and
`apply_dynamic_filters_gated_view` (same semantics, `dynamic_filter_snapshot const&` in place of
the live set). Parquet and native consumers switch to one snapshot per checkpoint; the set-based
overloads then retire.

---

### Stage 3 — Range and lexicographic filters

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
   * visible on every device (main doc, "Multi-GPU publication").
   */
  void replicate_to_devices(std::span<dynamic_filter_replica_space const> spaces) override;
  [[nodiscard]] bool is_available_on_device(int device_id) const noexcept override;

  [[nodiscard]] exact_host_scalar const& bound() const noexcept;
  [[nodiscard]] range_bound_side side() const noexcept;
  [[nodiscard]] bool inclusive() const noexcept;
  [[nodiscard]] dynamic_filter_null_policy null_policy() const noexcept;
};

/**
 * @brief Capability mixin: filter lowers itself against several consumer columns
 *
 * The multi-column sibling of @ref sirius_ast_lowerable. Instead of one pre-emplaced column
 * reference, the consumer supplies its existing @ref column_ref_resolver_fn; the filter resolves
 * each ordinal it references. `merge_ast_dynamic_filters_into_tree` dispatches on this capability
 * exactly as it does on the single-column one and already owns the resolver it needs.
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
 * @brief Immutable strict lexicographic boundary filter over the full ORDER BY tuple
 *
 * Lowers to the prefix-disjunction `T0 OR (E0 AND T1) OR ...` with the per-component null
 * derivations from the main doc's table. A null tail component contributes `IS NULL` /
 * `IS NOT NULL` terms and owns no device scalar; the first component must be non-null.
 * Never decomposed into per-column filters — the no-tail lemma at the representation level.
 *
 * @pre `boundary.size() == components.size() >= 2` (a single-key producer publishes RANGE).
 * @pre `boundary.component(0)` is engaged.
 */
class sirius_dynamic_lex_range_filter final : public sirius_dynamic_filter,
                                              public sirius_multi_column_ast_lowerable,
                                              public sirius_device_replicable {
 public:
  sirius_dynamic_lex_range_filter(exact_host_key_tuple boundary,
                                  std::vector<lex_component_semantics> components);
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
};
```

---

### Stage 4 — External consumers

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
  };

  std::size_t k;                      ///< Checked limit + offset
  std::vector<key> keys;              ///< Complete ORDER BY, in order; keys[0] feeds FIRST_KEY
  std::vector<target> targets;        ///< Scan channels and endpoint channels, uniformly
  std::vector<dynamic_filter_replica_space> replica_spaces;
};

}  // namespace sirius::op
```

The plan is held `const` by the local operator; the coordinator receives it at construction
(extending the Stage 1 constructor with an optional plan — the Stage 1 form remains for
self-consumption-only producers).

#### Modified file: `src/include/planner/dynamic_filter/dynamic_filter_target_discovery.hpp`

The Top-N self-trace reuses `descent_steps` / `trace_probe_key` / `place_endpoint` unchanged in
shape; a policy bit restricts hops to the minimal set.

```cpp
struct descent_policy {
  // ... existing fields unchanged ...

  /**
   * @brief Restrict hops to the Top-N self-trace set
   *
   * Accepts plain-reference projections, FILTER pass-through/gather, positional UNION fan-out,
   * and endpoints; refuses joins, aggregates, and every other operator. Stage 6 widens this set
   * per proven hop, not by flipping the bit off.
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
 * A supported scan terminal is a scan bind (for the all-keys trace: every component must land on
 * a supported decoded scan column). Any other terminal is an endpoint site, marked immaterial
 * when only pass-through hops separate it from the Top-N input — the planner then skips it
 * because sink self-consumption applies the same predicate (main doc, "Sited endpoint"). A
 * FIRST_KEY terminal coinciding with a planned LEX target's site is marked subsumed.
 */
enum class top_n_target_kind {
  SCAN_BIND,
  ENDPOINT_SITE,
  ENDPOINT_SKIPPED_IMMATERIAL,
  SUBSUMED_BY_LEX
};

[[nodiscard]] top_n_target_kind classify_top_n_terminal(route_terminal const& terminal,
                                                        top_n_filter_layer layer,
                                                        std::size_t material_hops_above);
```

Endpoint splicing itself is the existing `place_endpoint` with an `endpoint_factory` that
constructs `sirius_physical_dynamic_filter` in `include_ast_row_masks` mode.

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

A Top-N endpoint is this operator with `include_ast_row_masks` and `top_n_endpoint` — no new
operator type, matching the Phase 2 precedent.

#### Threading and lifetime summary

| Element | Mutability | Guard | Lifetime |
|---|---|---|---|
| `exact_host_scalar` / `exact_host_key_tuple` | Immutable | None | Value types |
| `top_n_key_semantics` / `lex_component_semantics` | Immutable | None | Value types, frozen at plan time |
| `top_n_threshold_witness` | Immutable after creation | None | Until offer/publication completes |
| `top_n_threshold_coordinator` | Mutable | One internal mutex; single publisher loop | Execution-scoped, shared_ptr from both operators |
| `top_n_dynamic_filter_publish_plan` | Immutable | None | Owned `const` by the plan |
| Refinement slot / generation | Mutable | Channel mutex | Channel lifetime (co-owned) |
| `dynamic_filter_snapshot` | Immutable | None | Consumer-held; keeps filters alive |
| `sirius_dynamic_range_filter` / `sirius_dynamic_lex_range_filter` | Immutable after replication | None post-publication | Until last snapshot releases them |
| `prefilter_gate` / endpoint gate | Mutable | Gate's existing internal locks | Execution-scoped |
| `dynamic_filter_stats` additions | Atomics | Per-field relaxed | Connection lifetime |

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

### Observability contract used by the assertions

Installation site: plan-shape layer (`DYNAMIC_FILTER` node presence, position, `provenance()`).
Pruning effect and lifecycle: stats deltas (`top_n_offers`, `top_n_prefilter_rows_*`,
`top_n_revisions_published`, the per-layer `top_n_{first_key,lex}_*` counters, and
`top_n_endpoint_sites_*`). Batch arrival order is not deterministic,
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
| 2 | Native scan, single key | Same over the native table | RANGE at the scan's post-decode operator, `include_ast_row_masks` (S4) | plan, integration |
| 3 | Aggregate disruptor | `SELECT grp, sum(v) s FROM t GROUP BY grp ORDER BY s DESC LIMIT 5` | No endpoint — site immaterial, self-consumption covers (S1/S4 skip assert); endpoint above the aggregate read-out only when material work intervenes, e.g. a dimension join between aggregate and Top-N (S6) | plan, integration |
| 4 | Expression-key projection | `SELECT a + b AS k FROM t2 ORDER BY k LIMIT 10` | Trace stops at the materializing projection; endpoint skipped as immaterial (S4 skip assert); self-consumption covers | plan, integration |
| 5 | Join disruptor, single key | `SELECT o.*, l.v FROM t l JOIN dim o ON ... ORDER BY l.v LIMIT 10` | Endpoint above the join is immaterial → skipped (S4); widened probe-block hop reaches the scan → scan bind (S6) | plan, integration |
| 6 | UNION fan-out | `SELECT v, w FROM t1 UNION ALL SELECT v, w FROM t2 ORDER BY v, w LIMIT 10` | Both branches: LEX into each scan; first-key targets subsumed per branch (S4) | plan, integration |
| 7 | Pass-through hops | `SELECT v FROM t WHERE grp <> 3 ORDER BY v LIMIT 10` (filter + plain projection) | Still the scan (S4) | plan, sqllogic |
| 8 | Self-consumption only | Any eligible shape with the channel stages disabled or no target | No `DYNAMIC_FILTER` node; `top_n_offers` delta > 0, prefilter direction asserts, results unchanged (S1) | integration, sqllogic |
| 9 | All keys, one scan | `SELECT * FROM topn_parquet ORDER BY v, w LIMIT 10` | LEX in the scan's reader AST; `top_n_first_key_subsumed_by_lex` delta > 0, no separate first-key filter on that channel (S4) | plan, integration, sqllogic |
| 10 | **Split keys across a join (marquee)** | `SELECT l.v, o.w FROM t l JOIN dim o ON l.id = o.id ORDER BY l.v, o.w LIMIT 10` | First-key inclusive RANGE into `l`'s scan (`top_n_first_key_filters_pushed`); all-keys trace stops at the join output — skipped as immaterial, strict predicate at the sink prefilter (S4); probe-block widening sites the LEX endpoint below a second join above (S6) | plan, integration, sqllogic |
| 11 | Mixed directions and null orders | `ORDER BY v DESC, w ASC NULLS FIRST LIMIT 10` and the transposed combos | Same sites as 9; per-component `T_i`/`E_i` derivations pinned by equivalence sweep (S4) | sqllogic, integration |
| 12 | Null tail boundary | Data forcing row K−1's `w` to null under both null orders | Publication proceeds through the derivation table; equivalence exact; `top_n_offers_unsupported` unchanged (S4) | integration, sqllogic |
| 13 | Unsupported tail type | `ORDER BY v, pay LIMIT 10` (VARCHAR tail) | First-key layer only: `top_n_producers_first_key_only` delta > 0, no LEX target, RANGE still reaches `v`'s scan (S4) | plan, integration |
| 14 | Negatives | See below | No producer / no publication (S1) | sqllogic + one integration counter case |

Negative sub-cases (14): `LIMIT 10 WITH TIES` (no producer), **first**-key-null boundary — `v`
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

Scenarios 3–5 and 10 assert the *skip* explicitly at Stage 4 — `top_n_endpoint_sites_skipped`
delta and absence of a `top_n_endpoint` node — so the cost-gate behavior is pinned, not
accidental. Their Stage 6 variants (join between aggregate and Top-N; probe-block descent) flip
the same assertions to `top_n_endpoint_sites_placed`/scan-bind and are written up front but
tagged `[!mayfail]` until Stage 6 lands.

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

### SQLLogic sketch

`test/sql/top-n-dynamic-filter.test`: load extension; create the native and Parquet tables;
foreach `enable_top_n_dynamic_filter` in (false, true) run the full query list (scenarios 1, 3,
4, 5, 6, 7, 9, 10, the scenario 11 direction/null-order sweep, 12, and all negatives) with
results compared against stored expected output; the `gpu_execution=false` pass pins the CPU
baseline. This file is runnable — and must pass with the filter inert — from Stage 1 onward,
which is what makes it the regression floor for every later stage.
