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
// dynamic_filter_publish_plan
//===----------------------------------------------------------------------===//
/// @brief Immutable plan-time description of one hash join's dynamic-filter publication.
///
/// The planner owns routing and placement decisions. The runtime publisher consumes this value but
/// cannot mutate its targets, policy, or device set after operator construction. Replica placements
/// cover every active GPU space, each paired with its planned HOST staging space. The build GPU's
/// space is included because it sources filter construction and the remote transfers, not because a
/// second copy is made there; only other GPUs receive replicas. Their owner follows the lifetime
/// contract on @ref dynamic_filter_replica_space.
class dynamic_filter_publish_plan final {
 public:
  struct probe_target {
    std::shared_ptr<sirius_dynamic_filter_set> filter_set;
    std::vector<std::size_t> probe_col_idx;
    std::vector<cudf::data_type> probe_col_type;
  };

  /// Default fraction of a key's domain a build may cover and still publish that key's filters.
  static constexpr double k_default_domain_coverage_threshold = 0.9;

  dynamic_filter_publish_plan() = default;
  dynamic_filter_publish_plan(
    std::vector<probe_target> probe_targets,
    bool emit_zone_map_filters,
    std::vector<std::size_t> build_key_domain_cardinalities,
    std::vector<dynamic_filter_replica_space> replica_spaces,
    double domain_coverage_threshold = k_default_domain_coverage_threshold);

  [[nodiscard]] bool enabled() const noexcept { return !_probe_targets.empty(); }
  [[nodiscard]] std::vector<probe_target> const& probe_targets() const noexcept
  {
    return _probe_targets;
  }
  [[nodiscard]] bool emit_zone_map_filters() const noexcept { return _emit_zone_map_filters; }
  /// Per pushed key, aligned with the pushdown info's join_condition: the unfiltered cardinality
  /// of the base table the build key traces to, or 0 when untraceable (coverage gates off).
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
  std::vector<probe_target> _probe_targets;
  bool _emit_zone_map_filters = false;
  std::vector<std::size_t> _build_key_domain_cardinalities;
  double _domain_coverage_threshold = k_default_domain_coverage_threshold;
  /// Non-owning GPU/HOST placements. See @ref dynamic_filter_replica_space for the lifetime
  /// contract.
  std::vector<dynamic_filter_replica_space> _replica_spaces;
};

//===----------------------------------------------------------------------===//
// Key decisions and resolved keys
//===----------------------------------------------------------------------===//

/// @brief A candidate key for a dynamic filter, represented ordinally.
struct dynamic_filter_key_candidate {
  duckdb_filter_ordinal duckdb_ordinal;
  join_condition_index condition_index;
  bool is_equality = false;
};

/// @brief A decision regarding a dynamic_filter_key_candidate.
enum class dynamic_filter_key_decision : std::uint8_t {
  admitted,
  non_equality,
  cast,
  unresolved,
};

/// @brief An admitted key, fully resolved to the build input.
struct dynamic_filter_key_plan {
  sirius_key_ordinal ordinal;
  duckdb_filter_ordinal duckdb_ordinal;
  join_condition_index condition_index;
  std::size_t build_column_index = 0;
  cudf::data_type build_type{};
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

}  // namespace sirius::op
