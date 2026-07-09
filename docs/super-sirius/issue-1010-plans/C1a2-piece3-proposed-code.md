# C1a-2 piece 3 — proposed code: builder, planning view, and the one-shot freeze seam

Companion to [C1ab-adapter-foundation.md](C1ab-adapter-foundation.md). This is a **proposal for
you to insert** (per the working protocol: production code is shown, tests/validation are handled
separately). Code below follows the verified project conventions: leading-underscore members,
`sirius::internal_exception` for internal-error guards, `.value` formatting on strong IDs.

**What this piece does in one sentence:** the hash join stops being handed a finished
`dynamic_filter_publish_plan` plus raw DuckDB metadata at construction, and instead retains a
mutable Sirius-only *builder* that the engine freezes exactly once — after pipeline conversion,
before task creation — into an immutable plan published through a `single_assignment` slot that
runtime can only read after the freeze.

Section A is fully insertable now (new types, no call-site changes; everything compiles unused).
Section B is the integration round (constructor/planner/engine rewiring — piece 4's edits begin
here) shown so you can see where every type lands; don't insert B until A is in and its tests are
green.

Deviations from the plan doc's letter, called out inline and worth recording as resolutions 22-23
if you accept them:

1. `prepare_dynamic_filter_plans` takes the producer enumeration as an explicit
   `std::span<sirius_physical_hash_join* const>` first parameter. The plan doc's signature has
   only `grouped_additions`, but also says "the engine … enumerates every retained C1 builder" —
   the enumeration has to reach the seam somehow, and passing it keeps the seam a pure function
   instead of giving it tree-walking knowledge.
2. The builder does not carry a separate `duckdb_key_count` field; it is
   `key_candidates.size()` by construction (the adapter's arity fence already guarantees
   `condition_indexes.size() == probe columns arity`), exposed as `duckdb_key_count()`. Carrying
   both would just create one more equality to validate.

---

## A1 — new file `src/include/sirius/single_assignment.hpp`

One-shot publication slot with an explicit two-phase API: `prepare_assignment` does every
fallible thing and mints a move-only token; `commit_assignment(token)` is statically `noexcept`.
A token destroyed uncommitted rolls its slot back, so a multi-slot preparation that fails midway
leaves **zero** slots changed (the plan's "preparation failure before any slot changes" gate).

```cpp
/*
 * Copyright 2026, Sirius Contributors.
 * (Apache-2.0 header as in sibling files)
 */

#pragma once

#include <sirius/exception.hpp>

#include <atomic>
#include <type_traits>
#include <utility>

namespace sirius {

/// @brief A write-once publication slot with a two-phase commit.
///
/// Lifecycle: exactly one successful `prepare_assignment` (all validation/allocation happens
/// there), then exactly one `commit_assignment` of the returned token — allocation-free and
/// statically `noexcept`. Destroying an uncommitted token rolls the slot back to empty, so a
/// caller preparing N slots that throws on slot k leaves slots 1..k-1 unchanged.
///
/// Concurrency: preparation and commit happen on the single planning/freeze thread; readers may
/// be concurrent afterwards. `is_assigned()`/`get()` acquire; commit releases — a reader that
/// observes the slot assigned is guaranteed to see the committed value.
///
/// A second `prepare_assignment` (or checked `assign`) on a pending/assigned slot is an internal
/// error and throws. A second commit is structurally impossible: tokens are move-only, minted
/// only by `prepare_assignment`, and consumed by commit.
template <class T>
class single_assignment final {
  static_assert(std::is_nothrow_default_constructible_v<T> &&
                  std::is_nothrow_move_assignable_v<T>,
                "single_assignment requires a nothrow default-constructible / move-assignable "
                "payload so the commit path cannot throw");

  enum class slot_state : std::uint8_t { empty, pending, assigned };

 public:
  class assignment_token final {
   public:
    assignment_token(assignment_token&& other) noexcept
      : _slot(std::exchange(other._slot, nullptr)), _value(std::move(other._value))
    {
    }
    assignment_token& operator=(assignment_token&&) = delete;
    assignment_token(assignment_token const&)       = delete;
    assignment_token& operator=(assignment_token const&) = delete;

    /// Uncommitted token: roll the slot back so no observable state changed.
    ~assignment_token()
    {
      if (_slot != nullptr) { _slot->_state.store(slot_state::empty, std::memory_order_release); }
    }

   private:
    friend class single_assignment;
    assignment_token(single_assignment* slot, T value) noexcept
      : _slot(slot), _value(std::move(value))
    {
    }

    single_assignment* _slot;  ///< null once committed or moved-from
    T _value;
  };

  single_assignment() = default;
  // Non-copyable/non-movable: tokens hold the slot's address.
  single_assignment(single_assignment const&)            = delete;
  single_assignment& operator=(single_assignment const&) = delete;

  /// @brief Phase one: every fallible step. Claims the slot and carries @p value in the token.
  [[nodiscard]] assignment_token prepare_assignment(T value)
  {
    auto expected = slot_state::empty;
    if (!_state.compare_exchange_strong(
          expected, slot_state::pending, std::memory_order_acq_rel)) {
      throw sirius::internal_exception(
        "[single_assignment] prepare_assignment on a slot that is already pending or assigned");
    }
    return assignment_token{this, std::move(value)};
  }

  /// @brief Phase two: publish the already-built value. Allocation-free and non-throwing.
  void commit_assignment(assignment_token&& token) noexcept
  {
    auto* slot = std::exchange(token._slot, nullptr);
    // A foreign or already-consumed token is unreachable through the type system (move-only,
    // minted by this slot's prepare); the null check makes a moved-from token a no-op rather
    // than UB.
    if (slot != this) { return; }
    _value = std::move(token._value);
    _state.store(slot_state::assigned, std::memory_order_release);
  }

  /// @brief The ordinary checked one-shot assignment (prepare + commit in one step).
  void assign(T value) { commit_assignment(prepare_assignment(std::move(value))); }

  [[nodiscard]] bool is_assigned() const noexcept
  {
    return _state.load(std::memory_order_acquire) == slot_state::assigned;
  }

  /// @brief The committed value. Reading before the freeze is an internal error, never a silent
  /// default (the plan's "runtime access requires a frozen slot").
  [[nodiscard]] T const& get() const
  {
    if (!is_assigned()) {
      throw sirius::internal_exception(
        "[single_assignment] read before the one-shot assignment was committed");
    }
    return _value;
  }

 private:
  std::atomic<slot_state> _state{slot_state::empty};
  T _value{};
};

}  // namespace sirius
```

Notes for your review:

- `commit_assignment` is `noexcept` by declaration *and* by construction (`static_assert` on the
  payload, move-assign only). The compile-time `noexcept` assertion the test gate requires can be
  `static_assert(noexcept(slot.commit_assignment(std::declval<...>())))` in the test file.
- Double-*prepare* throws (fallible phase — allowed); double-*commit* cannot be expressed without
  a second token, and a second `prepare` to get one throws. This is C.41/I.5 "establish
  invariants in construction, make misuse unrepresentable" applied to the freeze.

---

## A2 — additions to `src/include/op/dynamic_filter_publish_plan.hpp`

New includes at the top: `"op/dynamic_filter_identity.hpp"`, `<sirius/single_assignment.hpp>`,
`<optional>`, `<span>`, `<cstdint>`. Forward declaration `class sirius_physical_hash_join;`.
Everything below goes inside `namespace sirius::op`, after the existing
`dynamic_filter_publish_plan` class (which is unchanged in this piece — the builder produces it).

```cpp
//===----------------------------------------------------------------------===//
// Key decisions and resolved keys (C1a-2)
//===----------------------------------------------------------------------===//

/// @brief The canonical per-candidate narrowing decision, recorded once by the hash-join
/// constructor after equality-key extraction. C1b/C3 include this definition; nobody redeclares
/// it (and the alternate spelling dynamic_filter_filter_id does not exist anywhere).
enum class dynamic_filter_key_decision : std::uint8_t {
  admitted,      ///< equality key with a resolved build column; carries one dynamic_filter_key_plan
  non_equality,  ///< recorded range/non-equality comparison: valid candidate, no runtime key
  cast,          ///< equality under a build-side cast: rejected (Phase 1 behavior preserved)
  unresolved,    ///< recorded ordinal with no matching resolved build column
};

/// @brief One DuckDB-recorded candidate ordinal, reduced to Sirius values at the planner
/// boundary (comparisons arrive as is_equality — no DuckDB enum crosses this header).
struct dynamic_filter_key_candidate {
  duckdb_filter_ordinal duckdb_ordinal;  ///< position j in the recorded vectors
  join_condition_index condition_index;  ///< the value stored at join_condition[j] — indexes the
                                         ///< identically-reordered (equality-first) conditions
  bool is_equality = false;
};

/// @brief One admitted key, fully resolved to the build input.
struct dynamic_filter_key_plan {
  sirius_key_ordinal ordinal;            ///< compact: unique, contiguous from zero
  duckdb_filter_ordinal duckdb_ordinal;  ///< provenance; addresses full-arity target vectors
  join_condition_index condition_index;
  std::size_t build_column_index = 0;    ///< into the captured build input width
  cudf::data_type build_type{};          ///< plan-time type; runtime stays authoritative (WARN-only drift)
};

//===----------------------------------------------------------------------===//
// Sanctioned pre-freeze planning view (the ONLY C3 bind-time read surface)
//===----------------------------------------------------------------------===//

struct dynamic_filter_planning_ordinal_view {
  duckdb_filter_ordinal duckdb_ordinal;
  dynamic_filter_key_decision decision;
  std::optional<dynamic_filter_key_plan> admitted_key;  ///< engaged iff decision == admitted
  std::optional<cudf::data_type> build_type;            ///< engaged iff decision == admitted
};

struct dynamic_filter_planning_view {
  dynamic_filter_publication_plan_id publication_plan_id;
  bool wired   = false;  ///< the preserved Phase 1 wiring predicate's verdict
  bool enabled = false;  ///< whether the resolved builder can produce an enabled plan
  std::span<dynamic_filter_planning_ordinal_view const> by_duckdb_ordinal;  ///< exactly
                                                                            ///< duckdb_key_count()
};

//===----------------------------------------------------------------------===//
// The mutable planner-side builder
//===----------------------------------------------------------------------===//

/// @brief Everything the planner knows about one producing join's publication, mutable only
/// between plan_comparison_join (construction) and the hash-join constructor (key resolution);
/// frozen exactly once by prepare/commit_dynamic_filter_plans. Runtime never sees this type.
class dynamic_filter_publish_plan_builder final {
 public:
  /// Full-DuckDB-arity scan target draft: the C1a-1 adapter's copied values plus minted identity.
  /// (C1b compacts these to per-admitted-key entries; C1a-2 keeps full arity end to end.)
  struct scan_target_draft {
    dynamic_filter_target_id target_id;
    dynamic_filter_channel_id channel_id;
    std::shared_ptr<sirius_dynamic_filter_set> channel;
    std::vector<std::size_t> probe_col_idx;
    std::vector<cudf::data_type> probe_col_type;
  };

  dynamic_filter_publish_plan_builder(dynamic_filter_publication_plan_id publication_plan_id,
                                      bool wired,
                                      std::vector<scan_target_draft> scan_targets,
                                      bool emit_zone_map_filters,
                                      double domain_coverage_threshold,
                                      std::vector<dynamic_filter_replica_space> replica_spaces,
                                      std::vector<dynamic_filter_key_candidate> key_candidates);

  /// @brief Single-shot key resolution, called by the hash-join constructor after its normal
  /// equality-key extraction. @p decisions has one entry per candidate in candidate order;
  /// @p resolved_keys has exactly one entry per admitted decision, in the same relative order.
  /// A second call, or a call after freeze, throws.
  void resolve_keys(std::vector<dynamic_filter_key_decision> decisions,
                    std::vector<dynamic_filter_key_plan> resolved_keys,
                    std::size_t build_input_column_count);

  [[nodiscard]] dynamic_filter_publication_plan_id publication_plan_id() const noexcept
  {
    return _publication_plan_id;
  }
  [[nodiscard]] bool wired() const noexcept { return _wired; }
  [[nodiscard]] bool keys_resolved() const noexcept { return _keys_resolved; }
  [[nodiscard]] std::size_t duckdb_key_count() const noexcept { return _key_candidates.size(); }

  /// @brief The immutable value read surface backing sirius_physical_hash_join::planning_view().
  /// Valid only after resolve_keys; spans reference builder-owned storage that lives until the
  /// join (which owns this builder) is destroyed.
  [[nodiscard]] dynamic_filter_planning_view planning_view() const;

  /// @brief The full final-validation ladder; returns the immutable runtime plan. Called only by
  /// prepare_dynamic_filter_plans (and the planner-test seam). Throws sirius::internal_exception
  /// naming the violated invariant. A disabled result is a valid, installable plan with zero
  /// live targets.
  [[nodiscard]] std::shared_ptr<dynamic_filter_publish_plan const> finalize() const;

 private:
  dynamic_filter_publication_plan_id _publication_plan_id;
  bool _wired = false;
  std::vector<scan_target_draft> _scan_targets;
  bool _emit_zone_map_filters = false;
  double _domain_coverage_threshold = dynamic_filter_publish_plan::k_default_domain_coverage_threshold;
  std::vector<dynamic_filter_replica_space> _replica_spaces;
  std::vector<dynamic_filter_key_candidate> _key_candidates;

  // Filled by resolve_keys:
  bool _keys_resolved = false;
  std::vector<dynamic_filter_key_decision> _decisions;
  std::vector<dynamic_filter_key_plan> _resolved_keys;
  std::size_t _build_input_column_count = 0;
  // Domain evidence is deliberately absent in C1a-2 (the dead post-resolver walk is deleted, and
  // its runtime effect was already "all gates off"); C1b captures real evidence pre-resolver and
  // threads it through here. finalize() materializes all-zero cardinalities meanwhile so the
  // publisher's coverage gates keep their exact Phase 1 (off) behavior.

  mutable std::vector<dynamic_filter_planning_ordinal_view> _planning_view_storage;
};

//===----------------------------------------------------------------------===//
// The one-shot freeze seam (generic producer boundary; C3b supplies additions later)
//===----------------------------------------------------------------------===//

/// @brief SIP probe target. Placeholder in C1a-2: C1b defines the compact key-carrying fields
/// before C3 constructs any.
struct join_probe_publish_target {};

/// @brief One producer's validated, grouped target additions (C3b's only input to the seam).
struct dynamic_filter_target_addition {
  sirius_physical_hash_join* producer = nullptr;
  std::vector<join_probe_publish_target> targets;
};

/// @brief Canonical frozen-topology descriptor: strong IDs and decisions only — never object
/// addresses (operator IDs reset per query; pointers move). Owned by the cached prepared
/// execution record; the digest is a fast reject and equality of the full descriptor is the
/// real verification.
struct dynamic_filter_frozen_descriptor {
  struct producer_record {
    dynamic_filter_publication_plan_id publication_plan_id;
    bool enabled = false;
    std::vector<std::uint8_t> decisions;  ///< dynamic_filter_key_decision per DuckDB ordinal
    std::vector<dynamic_filter_target_id> target_ids;
    std::vector<dynamic_filter_channel_id> channel_ids;
  };
  std::vector<producer_record> producers;  ///< in enumeration order

  [[nodiscard]] std::uint64_t digest() const noexcept;  // FNV-1a over the canonical encoding
  friend bool operator==(dynamic_filter_frozen_descriptor const&,
                         dynamic_filter_frozen_descriptor const&) = default;
};

/// @brief Owns one prebuilt immutable plan and prepared slot assignment per enumerated producer,
/// plus the topology descriptor. Move-only; destroying it uncommitted rolls every slot back.
class prepared_dynamic_filter_plans final {
 public:
  prepared_dynamic_filter_plans(prepared_dynamic_filter_plans&&) noexcept = default;
  prepared_dynamic_filter_plans(prepared_dynamic_filter_plans const&)     = delete;

  [[nodiscard]] dynamic_filter_frozen_descriptor const& descriptor() const noexcept
  {
    return _descriptor;
  }

 private:
  friend prepared_dynamic_filter_plans prepare_dynamic_filter_plans(
    std::span<sirius_physical_hash_join* const>, std::span<dynamic_filter_target_addition const>);
  friend void commit_dynamic_filter_plans(prepared_dynamic_filter_plans&&) noexcept;

  prepared_dynamic_filter_plans() = default;

  using runtime_slot =
    sirius::single_assignment<std::shared_ptr<dynamic_filter_publish_plan const>>;
  struct prepared_producer {
    sirius_physical_hash_join* join = nullptr;
    runtime_slot::assignment_token token;
  };
  std::vector<prepared_producer> _producers;
  dynamic_filter_frozen_descriptor _descriptor;
};

/// @brief Fallible phase: finalize EVERY enumerated producer's builder (disabled, scan-only,
/// zero-admitted, and all-rejected joins included — registry presence is never the condition for
/// assigning the slot), fold in @p grouped_additions (empty in C1a-2), build the descriptor, and
/// prepare every slot assignment. Throws with zero slots changed on any validation failure.
[[nodiscard]] prepared_dynamic_filter_plans prepare_dynamic_filter_plans(
  std::span<sirius_physical_hash_join* const> producers,
  std::span<dynamic_filter_target_addition const> grouped_additions);

/// @brief No-throw phase: publish every prepared plan through its slot.
void commit_dynamic_filter_plans(prepared_dynamic_filter_plans&& prepared) noexcept;

/// @brief Cached re-execution: digest fast-reject, then full descriptor comparison against the
/// already-frozen topology. Never assigns. Mismatch is an internal error (throws).
void verify_frozen_dynamic_filter_topology(dynamic_filter_frozen_descriptor const& cached,
                                           dynamic_filter_frozen_descriptor const& current);
```

---

## A3 — `src/op/dynamic_filter_publish_plan.cpp` additions (the validation ladder)

Append to the existing file (which keeps the current constructor untouched). The ladder is the
plan doc's final-validation list, numbered; every throw is `sirius::internal_exception` with the
invariant named.

```cpp
dynamic_filter_publish_plan_builder::dynamic_filter_publish_plan_builder(
  dynamic_filter_publication_plan_id publication_plan_id,
  bool wired,
  std::vector<scan_target_draft> scan_targets,
  bool emit_zone_map_filters,
  double domain_coverage_threshold,
  std::vector<dynamic_filter_replica_space> replica_spaces,
  std::vector<dynamic_filter_key_candidate> key_candidates)
  : _publication_plan_id(publication_plan_id),
    _wired(wired),
    _scan_targets(std::move(scan_targets)),
    _emit_zone_map_filters(emit_zone_map_filters),
    _domain_coverage_threshold(domain_coverage_threshold),
    _replica_spaces(std::move(replica_spaces)),
    _key_candidates(std::move(key_candidates))
{
}

void dynamic_filter_publish_plan_builder::resolve_keys(
  std::vector<dynamic_filter_key_decision> decisions,
  std::vector<dynamic_filter_key_plan> resolved_keys,
  std::size_t build_input_column_count)
{
  if (_keys_resolved) {
    throw sirius::internal_exception(
      "[dynamic_filter_publish_plan_builder] resolve_keys called twice on one builder");
  }
  _decisions                = std::move(decisions);
  _resolved_keys            = std::move(resolved_keys);
  _build_input_column_count = build_input_column_count;
  _keys_resolved            = true;
}

dynamic_filter_planning_view dynamic_filter_publish_plan_builder::planning_view() const
{
  if (!_keys_resolved) {
    throw sirius::internal_exception(
      "[dynamic_filter_publish_plan_builder] planning_view before resolve_keys");
  }
  if (_planning_view_storage.empty() && !_key_candidates.empty()) {
    _planning_view_storage.reserve(_key_candidates.size());
    std::size_t next_key = 0;
    for (std::size_t j = 0; j < _key_candidates.size(); ++j) {
      dynamic_filter_planning_ordinal_view v{};
      v.duckdb_ordinal = _key_candidates[j].duckdb_ordinal;
      v.decision       = j < _decisions.size() ? _decisions[j]
                                               : dynamic_filter_key_decision::unresolved;
      if (v.decision == dynamic_filter_key_decision::admitted &&
          next_key < _resolved_keys.size()) {
        v.admitted_key = _resolved_keys[next_key];
        v.build_type   = _resolved_keys[next_key].build_type;
        ++next_key;
      }
      _planning_view_storage.push_back(v);
    }
  }
  dynamic_filter_planning_view view{};
  view.publication_plan_id = _publication_plan_id;
  view.wired               = _wired;
  view.enabled             = !_scan_targets.empty() && !_resolved_keys.empty();
  view.by_duckdb_ordinal   = _planning_view_storage;
  return view;
}

std::shared_ptr<dynamic_filter_publish_plan const>
dynamic_filter_publish_plan_builder::finalize() const
{
  auto const fail = [this](char const* what) {
    throw sirius::internal_exception(
      "[dynamic_filter_publish_plan_builder] publication plan {} failed final validation: {}",
      _publication_plan_id.value,
      what);
  };

  if (!_keys_resolved) { fail("keys were never resolved"); }

  // 1. decision count equals candidate count (identity match is positional by construction).
  if (_decisions.size() != _key_candidates.size()) {
    fail("decision count != candidate count");
  }
  // 2./4. every admitted decision has exactly one key; key count == admitted-decision count.
  std::size_t const admitted_count = static_cast<std::size_t>(
    std::count(_decisions.begin(), _decisions.end(), dynamic_filter_key_decision::admitted));
  if (_resolved_keys.size() != admitted_count) {
    fail("resolved key count != admitted decision count");
  }
  // 3. rejected decisions have no Sirius ordinal — enforced structurally: keys exist only in
  //    _resolved_keys, and (5.) their compact ordinals must be unique and contiguous from zero,
  //    in candidate order.
  {
    std::size_t next_key = 0;
    for (std::size_t j = 0; j < _decisions.size(); ++j) {
      if (_decisions[j] != dynamic_filter_key_decision::admitted) { continue; }
      auto const& key = _resolved_keys[next_key];
      if (key.ordinal.value != static_cast<std::size_t>(next_key)) {
        fail("Sirius key ordinals are not contiguous in candidate order");
      }
      // 6./7. provenance bijection: the key names exactly its candidate's ordinal spaces.
      if (key.duckdb_ordinal != _key_candidates[j].duckdb_ordinal ||
          key.condition_index != _key_candidates[j].condition_index) {
        fail("admitted key does not match its candidate's ordinals");
      }
      // 8. each build column is inside the captured build input width.
      if (key.build_column_index >= _build_input_column_count) {
        fail("admitted key's build column exceeds the build input width");
      }
      ++next_key;
    }
  }
  // 6. DuckDB ordinals unique and below key count (candidate side).
  {
    std::vector<bool> seen(_key_candidates.size(), false);
    for (auto const& cand : _key_candidates) {
      auto const j = static_cast<std::size_t>(cand.duckdb_ordinal.value);
      if (j >= _key_candidates.size() || seen[j]) {
        fail("duplicate or out-of-range DuckDB ordinal");
      }
      seen[j] = true;
    }
  }
  // 9. full-arity scan target column/type vectors equal DuckDB key count.
  for (auto const& target : _scan_targets) {
    if (target.probe_col_idx.size() != _key_candidates.size() ||
        target.probe_col_type.size() != _key_candidates.size()) {
      fail("scan target arity != DuckDB key count");
    }
  }
  // 10. enabled plans: nonzero IDs, unique target IDs, non-null channels.
  bool const enabled = !_scan_targets.empty() && !_resolved_keys.empty();
  if (!_publication_plan_id.is_valid()) { fail("publication plan ID is zero"); }
  {
    std::vector<dynamic_filter_target_id> ids;
    for (auto const& target : _scan_targets) {
      if (!target.target_id.is_valid() || !target.channel_id.is_valid()) {
        fail("target or channel ID is zero");
      }
      if (target.channel == nullptr) { fail("scan target has a null channel"); }
      ids.push_back(target.target_id);
    }
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
      fail("duplicate target IDs");
    }
  }
  // 11. replica spaces: reuse the existing ctor validation (GPU/HOST tier + unique devices) by
  //     construction below — dynamic_filter_publish_plan's constructor still enforces it.
  // 12. disabled plans contain no live target but are still built and installed.

  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  if (enabled) {
    targets.reserve(_scan_targets.size());
    for (auto const& draft : _scan_targets) {
      targets.push_back(dynamic_filter_publish_plan::probe_target{
        draft.channel, draft.probe_col_idx, draft.probe_col_type});
    }
  }
  // Domain evidence is null in C1a-2: all-zero cardinalities keep the coverage gates exactly off,
  // byte-for-byte matching the dead pre-C1a-2 walk's runtime effect.
  return std::make_shared<dynamic_filter_publish_plan const>(
    std::move(targets),
    _emit_zone_map_filters,
    std::vector<std::size_t>(_key_candidates.size(), 0),
    _replica_spaces,
    _domain_coverage_threshold);
}
```

`prepare_dynamic_filter_plans` / `commit_dynamic_filter_plans` / `verify_frozen…` then compose
what already exists (finalize + token per join + descriptor build; commit loops
`join->commit_runtime_plan(std::move(token))`). Their bodies are mechanical given A1/A2 and the
hash-join slot below; I'll draft them in the same round as B so they compile against the real
member names you pick.

---

## B — integration sketch (next round; anchors from the surface maps)

**B1 `sirius_physical_hash_join`** (hpp:75-87 ctor, hpp:100 `filter_pushdown`, hpp:248
`_dynamic_filter_plan`):

- ctor drops `duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> pushdown_info` and
  `dynamic_filter_publish_plan dynamic_filter_plan = {}`; gains
  `std::unique_ptr<dynamic_filter_publish_plan_builder> plan_builder` (nullptr for the
  delegating 7-arg ctor and the three operator-test fixtures → a disabled builder is synthesized
  internally so *every* join freezes a plan).
- after the existing equality-key extraction (cpp:284-340): walk
  `plan_builder->…key_candidates` and record per candidate —
  `!is_equality → non_equality`; `key_casts[cond_idx].cast_right/left → cast`;
  `cond_idx >= right_key_col_indices.size() → unresolved` (today's SILENT skip at
  publisher.cpp:172 becomes a plan-time decision — add no new log); else `admitted` with
  `build_column_index = right_key_col_indices[cond_idx]` and the build-side cudf type. Then one
  `plan_builder->resolve_keys(...)`.
- members: `filter_pushdown` and `_dynamic_filter_plan` deleted; add
  `std::unique_ptr<dynamic_filter_publish_plan_builder> _plan_builder;` and
  `sirius::single_assignment<std::shared_ptr<dynamic_filter_publish_plan const>> _runtime_plan;`
  plus `planning_view()`, `commit_runtime_plan(token)`, and a runtime
  `dynamic_filter_plan()` that returns `*_runtime_plan.get()` (throws before freeze — the map's
  hazard "runtime access before freeze must be an internal error, not a silent disabled read").
- claim path (cpp:1341-1367, 1370-1427): `filter_pushdown && _dynamic_filter_plan.enabled()`
  becomes `_runtime_plan.get()->enabled()`; the enabled-requires-pushdown `invalid_argument`
  guard (cpp:229-233) is subsumed by finalize().
- delete the dead NLJ pushdown member/overload.

**B2 `plan_comparison_join`** (sirius_plan_comparison_join.cpp:471-629): replace every
`op.filter_pushdown` read with `candidate = generator.candidate_cache.find(op)`; keep
`rhs_cardinality` locally computed; keep all three "Not wiring dynamic filter(s)" INFO lines
byte-identical (the unfiltered-build line keyed off `candidate->build_subtree_has_filter_hint`,
firing for statistics_only too); keep `register_producer()` for every currently wired shape;
mint the publication ID through a generator-owned allocator + producer memo and channel IDs
through a channel memo (both new generator members beside `dynamic_filter_channels`); delete the
dead `build_key_domain_cardinalities` walk; drop the `join_filter_pushdown.hpp` include.

**B3 engine seam** (sirius_engine.cpp: tail of `initialize_internal`, after
`convert()`/wiring at :355-361 and before `execute()`'s `create_query` at :162): collect
`sirius_physical_hash_join*` from the operator tree, `prepare_dynamic_filter_plans(joins, {})`,
`commit_dynamic_filter_plans(std::move(prepared))`. Freeze-before-any-build-batch is guaranteed
because task creation happens strictly later in `execute()`. The cached re-execution
verify path and the descriptor's home (`sirius_prepared_statement_data`) land with piece 5's
execution coordinator — the transparent path rebuilds its plan every execution and never
exercises it (map hazard), so its tests must drive repeated `sirius_execute_query` on one
prepared statement.

**Byte-compatibility checklist for B** (from the maps, verify in review): publisher INFO line's
`{} keys` = `duckdb_key_count()` including the "Pushed 0" case; `active_targets` counts a
WARN+skipped mismatched target *before* the skip; no new log on the unresolved-ordinal path; the
DEBUG cast-skip line keeps logging the original `cond_idx`; `total_pushed` keeps counting filters
(zone-map + membership separately), not keys.
