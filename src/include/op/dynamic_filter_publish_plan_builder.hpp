/*
 * Copyright 2026, Sirius Contributors.
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

/**
 * @file
 * @brief The C1a-2a canonical planner sidecar with invariants established by its callers.
 *
 * The generator (`sirius_physical_plan_generator`) captures one immutable DuckDB candidate
 * (`duckdb_join_filter_candidate`, produced by the `duckdb_join_filter_candidate_adapter`) per
 * logical join. No runtime code consumes this sidecar in C1a-2a.
 *
 * Flow:
 *
 * 1. `plan_comparison_join` constructs a descriptor containing canonical scan targets, static key
 *    evidence, policy, and replica-space descriptions.
 * 2. The `sirius_physical_hash_join` constructor resolves one decision per candidate and one
 *    compact key plan per admitted decision.
 * 3. `planning_view()` exposes channel-free scan-target and ordinal values to later planning.
 *
 * The per-key value types, all aligned by DuckDB filter ordinal `j`:
 *
 *   dynamic_filter_key_candidate[j]     IN: static evidence, snapshotted before computed keys
 *        |                              are rewritten (is_equality, direct/uncast reference,
 *        |                              supported membership type; false fails closed)
 *        |
 *        | sirius_physical_hash_join ctor
 *        v
 *   dynamic_filter_key_decision[j]      admitted | non_equality | not_direct_uncast |
 *        |                              unsupported_membership_type | unresolved
 *        |  admitted decisions only
 *        v
 *   dynamic_filter_key_plan             compact sirius_key_ordinal plus the resolved build
 *        |                              batch column index and type
 *        |
 *        |  dynamic_filter_publish_plan_builder::resolve_keys()
 *        v
 *   dynamic_filter_planning_ordinal_view[j]   decision + admitted_key, engaged iff admitted
 *
 * Example (assuming the optimized plan keeps `orders` as probe/left and `customers` as
 * build/right):
 *
 *   orders.customer_id = customers.customer_id
 *
 * DuckDB filter ordinal `j = 0` ties together two different physical coordinate systems:
 *
 *   plan_comparison_join captures
 *     scan_target_draft[t].probe_col_idx[0]             = 2
 *     dynamic_filter_key_candidate[0].condition_index   = 0
 *
 *   sirius_physical_hash_join later resolves
 *     dynamic_filter_key_plan.duckdb_ordinal            = 0
 *     dynamic_filter_key_plan.build_batch_column_index  = 0
 *     dynamic_filter_key_plan.build_type                = INT32
 *
 *   planning_view() exposes
 *     scan_targets[t].probe_col_idx[0]                            = 2
 *     by_duckdb_ordinal[0].admitted_key->build_batch_column_index = 0
 *
 * Probe position `2` indexes the target scan's DuckDB `column_ids` vector; build position `0`
 * indexes the right-child input-batch schema. They are not interchangeable: shared DuckDB ordinal
 * `0` associates the scan target with the resolved build key. A same-provenance multi-key join
 * repeats this mapping independently for each `j`; it does not construct a tuple-valued membership
 * filter.
 *
 * Each scan target `t` is likewise one struct on the way in but two on the way out. The
 * constructor splits the draft so that view readers get channel-free values and can never reach
 * the channel object.
 *
 *   scan_target_draft[t] = { target ID, channel ID, probe columns, scan-created channel }
 *                              |                                      |
 *                              v                                      v
 *   _scan_targets[t]                                    _scan_target_channels[t]
 *     channel-free planning values,                       existing channel object;
 *     exposed by planning_view()                          private, C1a-2b freeze only
 */

// sirius
#include <op/dynamic_filter_identity.hpp>
#include <op/dynamic_filter_publish_plan.hpp>

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

class sirius_dynamic_filter_set;

/**
 * @brief One DuckDB-recorded key with the static evidence required for fail-closed narrowing.
 *
 * Static admission evidence is captured before Sirius rewrites computed equality-key expressions
 * into references to temporary columns appended below the join. This preserves whether each key
 * was originally a direct, uncast column reference.
 *
 * The three admission flags — `is_equality`, `has_direct_uncast_keys`, and
 * `has_supported_membership_type` — default to false, and admission requires each to be set true
 * explicitly; missing evidence therefore fails closed.
 */
struct dynamic_filter_key_candidate {
  duckdb_filter_ordinal duckdb_ordinal;
  join_condition_index condition_index;
  bool is_equality                   = false;
  bool has_direct_uncast_keys        = false;
  bool has_supported_membership_type = false;
};

/** @brief Recorded resolution outcome for one DuckDB filter ordinal. */
enum class dynamic_filter_key_decision : std::uint8_t {
  admitted,
  non_equality,
  not_direct_uncast,
  unsupported_membership_type,
  unresolved,  ///< Final fail-closed outcome: physical resolution could not prove admission.
};

/** @brief One admitted key resolved to the build input. */
struct dynamic_filter_key_plan {
  sirius_key_ordinal ordinal;
  duckdb_filter_ordinal duckdb_ordinal;
  join_condition_index condition_index;
  std::size_t build_batch_column_index = 0;  ///< Column in the right/build child's input batch.
  cudf::data_type build_type{};
};

/** @brief Resolved sidecar values for one DuckDB-recorded key. */
struct dynamic_filter_planning_ordinal_view {
  duckdb_filter_ordinal duckdb_ordinal;
  join_condition_index condition_index;
  dynamic_filter_key_decision decision = dynamic_filter_key_decision::unresolved;
  /// Engaged if and only if decision is admitted.
  std::optional<dynamic_filter_key_plan> admitted_key;
};

/** @brief Channel-free values describing one canonical scan-consumer edge. */
struct dynamic_filter_planning_scan_target {
  dynamic_filter_target_id target_id;
  dynamic_filter_channel_id channel_id;
  std::vector<std::size_t> probe_col_idx;
  std::vector<cudf::data_type> probe_col_type;
};

/** @brief Execution-mode claim required by future canonical v1 publication. */
enum class dynamic_filter_publication_claim : std::uint8_t { build_probe };

/**
 * @brief The only sanctioned read surface for a resolved C1a-2a sidecar.
 *
 * The spans borrow builder-owned storage and must not outlive the builder or its owning hash join.
 */
struct dynamic_filter_planning_view {
  static constexpr auto required_publication_claim = dynamic_filter_publication_claim::build_probe;

  dynamic_filter_publication_plan_id publication_plan_id;
  bool build_subtree_has_filter_hint = false;  ///< Observation only; not sidecar route admission.
  /// At least one canonical scan target and one admitted key; sidecar eligibility only.
  bool enabled = false;
  std::span<dynamic_filter_planning_scan_target const> scan_targets;
  std::span<dynamic_filter_planning_ordinal_view const> by_duckdb_ordinal;
};

/**
 * @brief Single-shot owner of one canonical planner-side publication model.
 *
 * Key resolution builds every exposed record locally and commits only after all fallible work
 * succeeds. The builder is held at a stable address behind `unique_ptr` and cannot be copied or
 * moved.
 */
class dynamic_filter_publish_plan_builder final {
 public:
  /** @brief Input-only target draft containing the scan-created channel for the future freeze. */
  struct scan_target_draft {
    dynamic_filter_target_id target_id;
    dynamic_filter_channel_id channel_id;
    std::shared_ptr<sirius_dynamic_filter_set> channel;
    std::vector<std::size_t> probe_col_idx;
    std::vector<cudf::data_type> probe_col_type;
  };

  /** @brief Complete input to one canonical planner-side publication model. */
  struct descriptor {
    dynamic_filter_publication_plan_id publication_plan_id;
    bool build_subtree_has_filter_hint = false;
    std::size_t join_condition_count   = 0;
    std::vector<scan_target_draft> scan_targets;
    bool emit_zone_map_filters = false;
    double domain_coverage_threshold =
      dynamic_filter_publish_plan::k_default_domain_coverage_threshold;
    std::vector<dynamic_filter_replica_space> replica_spaces;
    std::vector<dynamic_filter_key_candidate> key_candidates;
  };

  /**
   * @brief Take ownership of one producer's descriptor.
   *
   * Preconditions on @p draft:
   *
   * - `publication_plan_id` is valid and `join_condition_count` is nonzero;
   * - `key_candidates` is non-empty, entry `j` carries `duckdb_ordinal` `j`, and every
   *   `condition_index` is unique and less than `join_condition_count`;
   * - every scan target has a valid unique target ID, a valid channel ID, a non-null channel, and
   *   exactly one probe column and type per key candidate;
   * - channel objects are distinct, and their channel IDs are unique;
   * - `domain_coverage_threshold` is greater than zero;
   * - every replica pairs a GPU space with a HOST staging space, at most one per GPU device.
   */
  explicit dynamic_filter_publish_plan_builder(descriptor draft);
  ~dynamic_filter_publish_plan_builder() = default;

  dynamic_filter_publish_plan_builder(dynamic_filter_publish_plan_builder const&) = delete;
  dynamic_filter_publish_plan_builder& operator=(dynamic_filter_publish_plan_builder const&) =
    delete;
  dynamic_filter_publish_plan_builder(dynamic_filter_publish_plan_builder&&)            = delete;
  dynamic_filter_publish_plan_builder& operator=(dynamic_filter_publish_plan_builder&&) = delete;

  /**
   * @brief Return the resolved, channel-free planning values.
   *
   * @throws sirius::internal_exception if key resolution has not completed.
   */
  [[nodiscard]] dynamic_filter_planning_view planning_view() const;

 private:
  friend class sirius_physical_hash_join;
  friend struct dynamic_filter_publish_plan_builder_test_access;

  /**
   * @brief Resolve one decision per candidate and one compact plan per admitted key.
   *
   * Private: the hash-join constructor (friend) is the sole production caller. A second call
   * throws.
   *
   * Preconditions:
   *
   * - @p decisions has exactly one entry per key candidate, in candidate order;
   * - @p resolved_keys has exactly one entry per admitted decision, in the same relative order;
   * - resolved key `k` carries compact ordinal `k` and its admitted candidate's `duckdb_ordinal`
   *   and `condition_index`;
   * - every `build_batch_column_index` is less than @p build_input_column_count;
   * - every resolved key's `build_type` equals the type of its selected build column and the
   *   corresponding `probe_col_type` in every scan target;
   * - a decision is admitted only when its candidate has `is_equality`,
   *   `has_direct_uncast_keys`, and `has_supported_membership_type` set, and at least one scan
   *   target exists.
   */
  void resolve_keys(std::vector<dynamic_filter_key_decision> decisions,
                    std::vector<dynamic_filter_key_plan> resolved_keys,
                    std::size_t build_input_column_count);

  [[nodiscard]] std::vector<dynamic_filter_key_candidate> const& key_candidates() const noexcept
  {
    return _key_candidates;
  }
  [[nodiscard]] std::vector<dynamic_filter_planning_scan_target> const& scan_targets()
    const noexcept
  {
    return _scan_targets;
  }
  [[nodiscard]] std::vector<dynamic_filter_planning_ordinal_view> const& ordinal_records() const
  {
    return _ordinal_records;
  }

  dynamic_filter_publication_plan_id _publication_plan_id;
  bool _build_subtree_has_filter_hint = false;
  std::vector<dynamic_filter_planning_scan_target> _scan_targets;
  std::vector<std::shared_ptr<sirius_dynamic_filter_set>> _scan_target_channels;
  bool _emit_zone_map_filters = false;
  double _domain_coverage_threshold =
    dynamic_filter_publish_plan::k_default_domain_coverage_threshold;
  std::vector<dynamic_filter_replica_space> _replica_spaces;
  std::vector<dynamic_filter_key_candidate> _key_candidates;

  bool _keys_resolved = false;
  std::vector<dynamic_filter_key_decision> _decisions;
  std::vector<dynamic_filter_key_plan> _resolved_keys;
  std::size_t _build_input_column_count = 0;
  std::vector<dynamic_filter_planning_ordinal_view> _ordinal_records;
};

}  // namespace sirius::op
