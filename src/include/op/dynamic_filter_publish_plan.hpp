/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

//===----------------------------------------------------------------------===//
// The producer side of dynamic filters, from planning to runtime
//
// This header tells the whole story of how one hash join becomes a dynamic-filter
// producer. Four modules cooperate, in this order:
//
//   1. PLANNING (src/planner/sirius_plan_comparison_join.cpp)
//      The planner reads the join's snapshot from the candidate cache (it never
//      touches DuckDB's raw metadata — that is the adapter's job) and creates a
//      dynamic_filter_publish_plan_builder: "here are the scan targets, the key
//      candidates, and the policy knobs I know about".
//
//   2. KEY RESOLUTION (src/op/sirius_physical_hash_join.cpp, constructor)
//      The hash join works out which candidate keys it can actually publish
//      filters for, and records one dynamic_filter_key_decision per candidate
//      into the builder. After this, the builder is complete but still mutable
//      in principle — nothing at runtime may read it.
//
//   3. FREEZE (src/sirius_engine.cpp calls freeze_or_verify_dynamic_filter_plans
//      after pipelines are built and before any task can run)
//      Every builder is validated and turned into an immutable
//      dynamic_filter_publish_plan, which is published into the join's
//      write-once slot (sirius::single_assignment). This happens for EVERY hash
//      join — joins with nothing to publish get a valid "disabled" plan — so
//      runtime code can always ask "what is my plan?" and get a loud error only
//      if the engine forgot to freeze.
//
//   4. RUNTIME (src/op/dynamic_filter_publisher.cpp)
//      When the build side of the join is complete, the publisher reads ONLY the
//      frozen plan — no DuckDB types, no planner state — builds the filters, and
//      pushes them into each target's channel, where the consuming table scans
//      pick them up.
//
// Track C3 (sideways information passing) plugs into exactly two seams defined
// here: it reads planning facts through dynamic_filter_planning_view (step 2.5),
// and it hands extra targets to the freeze through dynamic_filter_target_addition
// (step 3). It never touches anything else.
//===----------------------------------------------------------------------===//

// sirius
#include <op/dynamic_filter_identity.hpp>
#include <op/dynamic_filter_replica_space.hpp>
#include <sirius/single_assignment.hpp>

// cudf
#include <cudf/types.hpp>

// standard library
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace sirius::op {

class sirius_physical_hash_join;
class sirius_dynamic_filter_set;

//===----------------------------------------------------------------------===//
// Key decisions and resolved keys
//===----------------------------------------------------------------------===//

/// @brief One join key that DuckDB recorded as a dynamic-filter candidate, reduced
/// to Sirius values.
///
/// DuckDB records, per producing join, a list of "here is a join condition whose
/// build values could filter a probe-side scan". Each entry in that list is one of
/// these. The comparison kind is already reduced to a plain bool at the planner
/// boundary, so no DuckDB enum ever reaches runtime code.
struct dynamic_filter_key_candidate {
  duckdb_filter_ordinal duckdb_ordinal;  ///< Position in DuckDB's recorded list (0, 1, 2, ...).
  join_condition_index condition_index;  ///< Which join condition this candidate points at. The
                                         ///< join conditions are sorted equality-first by both
                                         ///< DuckDB and Sirius, so this index means the same
                                         ///< condition on both sides.
  bool is_equality = false;              ///< True for `a = b` conditions; false for ranges.
};

/// @brief The verdict on one candidate, decided once in the hash-join constructor.
///
/// Only `admitted` keys produce filters at runtime. The other three all mean "no
/// filter for this key", but they are kept distinct because later telemetry (C1b)
/// wants to report WHY a key was rejected.
enum class dynamic_filter_key_decision : std::uint8_t {
  admitted,      ///< Equality key with a resolved build column: filters will be built for it.
  non_equality,  ///< A range comparison: a valid candidate, but Sirius publishes none for it yet.
  cast,          ///< The key is used through a type cast: rejected so build and probe values
                 ///< stay type-identical (the runtime skip DEBUG line documents each one).
  unresolved,    ///< The recorded condition index has no matching resolved build column.
};

/// @brief One admitted key, fully resolved: everything the runtime publisher needs
/// to read the right build column.
struct dynamic_filter_key_plan {
  sirius_key_ordinal ordinal;            ///< Compact position among admitted keys only: 0, 1, ...
                                         ///< with no gaps, in candidate order.
  duckdb_filter_ordinal duckdb_ordinal;  ///< Which candidate this key came from — also the
                                         ///< position in every full-arity target column vector.
  join_condition_index condition_index;
  std::size_t build_column_index = 0;  ///< Column in the join's build (right-side) input that
                                       ///< holds this key's values.
  cudf::data_type build_type{};        ///< The build column's type as known AT PLAN TIME. The
                                       ///< runtime column type stays authoritative; if the two
                                       ///< ever disagree the publisher warns and trusts runtime.
};

//===----------------------------------------------------------------------===//
// Sanctioned pre-freeze planning view (the ONLY C3 bind-time read surface)
//===----------------------------------------------------------------------===//

/// @brief What planning decided about one DuckDB-recorded candidate, as a plain value.
///
/// Rejected entries carry their identity too (`duckdb_ordinal`, `condition_index`):
/// a consumer that copies or filters these entries must not have to remember array
/// positions to know which key an entry describes, and the runtime DEBUG line for
/// cast-rejected keys needs the condition index of a key that has no
/// `admitted_key` to read it from.
struct dynamic_filter_planning_ordinal_view {
  duckdb_filter_ordinal duckdb_ordinal;
  join_condition_index condition_index;
  dynamic_filter_key_decision decision;
  std::optional<dynamic_filter_key_plan> admitted_key;  ///< engaged iff decision == admitted
  std::optional<cudf::data_type> build_type;            ///< engaged iff decision == admitted
};

/// @brief The read-only summary of one producer that C3 route discovery is allowed
/// to look at between key resolution and the freeze. It exposes values only — never
/// the mutable builder.
struct dynamic_filter_planning_view {
  dynamic_filter_publication_plan_id publication_plan_id;
  bool wired   = false;  ///< Did the Phase 1 wiring predicate admit this producer?
  bool enabled = false;  ///< Has at least one live scan target. NOTE: an enabled plan whose every
                         ///< key was rejected still runs the publisher and reports "Pushed 0" —
                         ///< enabled means "will attempt publication", not "has publishable keys".
  std::span<dynamic_filter_planning_ordinal_view const> by_duckdb_ordinal;  ///< exactly
                                                                            ///< duckdb_key_count()
};

//===----------------------------------------------------------------------===//
// dynamic_filter_publish_plan — the immutable runtime plan
//===----------------------------------------------------------------------===//
/// @brief Immutable description of one hash join's dynamic-filter publication,
/// created exactly once by the freeze and read-only forever after.
///
/// The planner owns routing and placement decisions. The runtime publisher consumes
/// this value but cannot mutate its targets, policy, or device set. Replica
/// placements cover every active GPU space, each paired with its planned HOST
/// staging space. The build GPU's space is included because it sources filter
/// construction and the remote transfers, not because a second copy is made there;
/// only other GPUs receive replicas. Their owner follows the lifetime contract on
/// @ref dynamic_filter_replica_space.
///
/// A plan is "enabled" when it has at least one live scan target. Disabled plans
/// are still real, installed values (a join with nothing to publish holds one), so
/// runtime code never has to treat "no plan" as a separate case from "plan that
/// publishes nothing".
class dynamic_filter_publish_plan final {
 public:
  /// One consuming scan: the shared channel the filters travel through, plus the
  /// probe-side column each key lands on. The column vectors are FULL ARITY: one
  /// entry per DuckDB-recorded candidate, addressed by `duckdb_ordinal`, whether or
  /// not that candidate was admitted. (C1b later compacts these to admitted keys.)
  struct probe_target {
    std::shared_ptr<sirius_dynamic_filter_set> filter_set;
    std::vector<std::size_t> probe_col_idx;
    std::vector<cudf::data_type> probe_col_type;
    dynamic_filter_target_id target_id{};    ///< Strong identity for logs and topology checks.
    dynamic_filter_channel_id channel_id{};  ///< Producers sharing one scan share this value.
  };

  /// Default fraction of a key's domain a build may cover and still publish that key's filters.
  static constexpr double k_default_domain_coverage_threshold = 0.9;

  /// A default-constructed plan is the canonical "disabled" plan: no targets, no
  /// keys, publishes nothing. Used for joins that were never dynamic-filter
  /// producers (tests, non-producer plan shapes).
  dynamic_filter_publish_plan() = default;

  /// The full constructor, normally reached only through
  /// dynamic_filter_publish_plan_builder::finalize(). Validates replica placements
  /// (an enabled plan needs at least one GPU space, every entry must pair a GPU
  /// space with a HOST staging space) and de-duplicates them by device.
  dynamic_filter_publish_plan(
    dynamic_filter_publication_plan_id publication_plan_id,
    std::vector<dynamic_filter_planning_ordinal_view> ordinals,
    std::vector<probe_target> probe_targets,
    bool emit_zone_map_filters,
    std::vector<std::size_t> build_key_domain_cardinalities,
    std::vector<dynamic_filter_replica_space> replica_spaces,
    double domain_coverage_threshold = k_default_domain_coverage_threshold);

  [[nodiscard]] bool enabled() const noexcept { return !_probe_targets.empty(); }
  [[nodiscard]] dynamic_filter_publication_plan_id publication_plan_id() const noexcept
  {
    return _publication_plan_id;
  }
  /// How many candidates DuckDB recorded for this join. This is the number the
  /// terminal INFO line reports as "keys", and the length of every full-arity
  /// vector in this plan.
  [[nodiscard]] std::size_t duckdb_key_count() const noexcept { return _ordinals.size(); }
  /// One record per DuckDB-recorded candidate, in recorded order: the decision,
  /// the condition index, and (for admitted keys) the resolved key. This is the
  /// same data the pre-freeze planning view exposed — frozen.
  [[nodiscard]] std::vector<dynamic_filter_planning_ordinal_view> const& ordinals() const noexcept
  {
    return _ordinals;
  }
  [[nodiscard]] std::vector<probe_target> const& probe_targets() const noexcept
  {
    return _probe_targets;
  }
  [[nodiscard]] bool emit_zone_map_filters() const noexcept { return _emit_zone_map_filters; }
  /// Per recorded candidate, aligned with @ref ordinals: the unfiltered cardinality
  /// of the base table the build key traces to, or 0 when unknown (the publish
  /// coverage gates are then off for that key). All zeros in C1a-2 — C1b restores
  /// real evidence captured before column-binding resolution.
  [[nodiscard]] std::vector<std::size_t> const& build_key_domain_cardinalities() const noexcept
  {
    return _build_key_domain_cardinalities;
  }
  [[nodiscard]] std::vector<dynamic_filter_replica_space> const& replica_spaces() const noexcept
  {
    return _replica_spaces;
  }
  [[nodiscard]] double domain_coverage_threshold() const noexcept
  {
    return _domain_coverage_threshold;
  }

 private:
  dynamic_filter_publication_plan_id _publication_plan_id{};  ///< invalid (0) for disabled
                                                              ///< plans of non-producer joins
  std::vector<dynamic_filter_planning_ordinal_view> _ordinals;
  std::vector<probe_target> _probe_targets;
  bool _emit_zone_map_filters = false;
  std::vector<std::size_t> _build_key_domain_cardinalities;
  double _domain_coverage_threshold = k_default_domain_coverage_threshold;
  /// Non-owning GPU/HOST placements. See @ref dynamic_filter_replica_space for the lifetime
  /// contract.
  std::vector<dynamic_filter_replica_space> _replica_spaces;
};

//===----------------------------------------------------------------------===//
// The mutable planner-side builder
//===----------------------------------------------------------------------===//

/// @brief Everything the planner knows about one producing join's publication,
/// gathered in two steps and then frozen.
///
/// Step 1 (plan_comparison_join): construction, with the scan targets, policy, and
/// key candidates. Step 2 (the hash-join constructor): resolve_keys(), recording
/// the per-candidate decisions. After that the builder just waits for the freeze;
/// runtime never sees this type.
class dynamic_filter_publish_plan_builder final {
 public:
  /// A scan target as the planner drafted it: the shared channel plus full-arity
  /// column routing copied from the adapter's snapshot, and the minted identity.
  /// (C1b compacts these to per-admitted-key entries; C1a-2 keeps full arity end
  /// to end.)
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
  /// A second call throws.
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
  [[nodiscard]] std::vector<dynamic_filter_key_candidate> const& key_candidates() const noexcept
  {
    return _key_candidates;
  }
  [[nodiscard]] std::vector<scan_target_draft> const& scan_targets() const noexcept
  {
    return _scan_targets;
  }

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
  /// The per-candidate records that both planning_view() and finalize() expose — built once,
  /// eagerly, at the end of resolve_keys(), so the pre-freeze view and the frozen plan are the
  /// same values by construction.
  [[nodiscard]] std::vector<dynamic_filter_planning_ordinal_view> const& ordinal_records() const
  {
    return _ordinal_records;
  }

  dynamic_filter_publication_plan_id _publication_plan_id;
  bool _wired = false;
  std::vector<scan_target_draft> _scan_targets;
  bool _emit_zone_map_filters = false;
  double _domain_coverage_threshold =
    dynamic_filter_publish_plan::k_default_domain_coverage_threshold;
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

  std::vector<dynamic_filter_planning_ordinal_view> _ordinal_records;  ///< built by resolve_keys
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

/// @brief A value snapshot of the frozen dynamic-filter topology: which producers
/// exist, what they decided, and which targets/channels they feed — as strong IDs
/// only, never object addresses (operator IDs reset per query; pointers move).
///
/// Two uses: cached-plan re-execution compares the topology it froze against the
/// topology the current builders describe (they must match exactly — a cached plan
/// cannot change shape without replanning), and C3b later adds its routing
/// descriptor beside this. The digest is only a fast reject; real verification
/// compares full descriptors.
struct dynamic_filter_frozen_descriptor {
  struct producer_record {
    dynamic_filter_publication_plan_id publication_plan_id;
    bool enabled = false;
    std::vector<std::uint8_t> decisions;  ///< dynamic_filter_key_decision per DuckDB ordinal
    std::vector<dynamic_filter_target_id> target_ids;
    std::vector<dynamic_filter_channel_id> channel_ids;

    friend bool operator==(producer_record const&, producer_record const&) = default;
  };
  std::vector<producer_record> producers;  ///< in enumeration order

  [[nodiscard]] std::uint64_t digest() const noexcept;  // FNV-1a over the canonical encoding
  friend bool operator==(dynamic_filter_frozen_descriptor const&,
                         dynamic_filter_frozen_descriptor const&) = default;
};

/// @brief Everything a freeze prepared but has not yet published: one immutable plan and one
/// write-once-slot token per enumerated producer, plus the topology descriptor. Move-only.
/// If this object is destroyed without being committed, every token rolls its slot back —
/// a failed freeze changes nothing.
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

/// @brief Fallible phase of the freeze: finalize EVERY enumerated producer's builder (disabled,
/// scan-only, zero-admitted, and all-C3-rejected joins included — a join with no builder gets the
/// canonical disabled plan), fold in @p grouped_additions (empty until C3b), build the topology
/// descriptor, and prepare every slot assignment. Throws with zero slots changed on any failure.
[[nodiscard]] prepared_dynamic_filter_plans prepare_dynamic_filter_plans(
  std::span<sirius_physical_hash_join* const> producers,
  std::span<dynamic_filter_target_addition const> grouped_additions);

/// @brief No-throw phase of the freeze: publish every prepared plan through its slot.
void commit_dynamic_filter_plans(prepared_dynamic_filter_plans&& prepared) noexcept;

/// @brief The topology the current builders/joins describe (what a freeze WOULD produce).
[[nodiscard]] dynamic_filter_frozen_descriptor describe_planned_dynamic_filter_topology(
  std::span<sirius_physical_hash_join* const> producers);

/// @brief The topology the already-frozen plans actually carry.
[[nodiscard]] dynamic_filter_frozen_descriptor describe_frozen_dynamic_filter_topology(
  std::span<sirius_physical_hash_join* const> producers);

/// @brief Value verification for cached re-execution: digest fast-reject, then full descriptor
/// comparison. Never assigns anything. A mismatch is an internal error (throws).
void verify_frozen_dynamic_filter_topology(dynamic_filter_frozen_descriptor const& cached,
                                           dynamic_filter_frozen_descriptor const& current);

/// @brief The engine's one entry point, called after pipeline conversion and before any task can
/// run. First execution: prepare + commit a frozen plan for every producer. Re-execution of a
/// cached plan (slots already assigned): verify the frozen topology still matches the builders
/// and reuse it — never assign twice. A mix of frozen and unfrozen producers is an internal
/// error.
void freeze_or_verify_dynamic_filter_plans(std::span<sirius_physical_hash_join* const> producers);

}  // namespace sirius::op
