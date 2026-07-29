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

#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"

#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::op {

class sirius_dynamic_filter_set;

//===----------------------------------------------------------------------===//
// dynamic_filter_publish_plan
//===----------------------------------------------------------------------===//

/**
 * @brief Shape of one join-condition side, classified before computed-key materialization
 *
 * `sirius_plan_comparison_join` materializes computed equality keys into plain bound references
 * backed by an injected projection before the conditions reach `sirius_physical_hash_join`, so a
 * post-materialization condition cannot distinguish a computed key from a direct column reference.
 * The planner therefore classifies each condition side while the original expression is still
 * visible and carries the immutable result here and on the physical join. Admission consumes the
 * carried classification and never re-derives shape from post-materialization conditions.
 */
enum class dynamic_filter_key_shape : std::uint8_t {
  direct,   ///< A plain bound column reference
  cast,     ///< A cast wrapping a bound column reference
  computed  ///< Any other expression; materialized into a projection column by the planner
};

/**
 * @brief Both sides' shapes for one join condition
 */
struct dynamic_filter_condition_shape {
  dynamic_filter_key_shape probe = dynamic_filter_key_shape::direct;  ///< Probe (left) side
  dynamic_filter_key_shape build = dynamic_filter_key_shape::direct;  ///< Build (right) side

  [[nodiscard]] bool operator==(dynamic_filter_condition_shape const&) const = default;
};

/**
 * @brief Which kind of consumer a target channel feeds
 *
 * Both classes push the consumer operator's own output ordinal -- the coordinate the channel
 * stores and every lookup uses. A scan route pushes the bound scan's output ordinal, the discovery
 * walk's exit ordinal at that scan; a direct route pushes `planner::place_endpoint()`'s site
 * ordinal, in the output schema of the operator the endpoint wraps. Either exit ordinal is derived
 * by mapping the producing join's probe-key entry ordinal through each accepted descent step, so
 * entry and exit ordinals differ whenever a hop was accepted.
 */
enum class dynamic_filter_route_class : std::uint8_t {
  scan,   ///< A GPU scan consumer; zone-map capable
  direct  ///< A join-edge endpoint consumer; membership only
};

/**
 * @brief Publication policy transported from configuration
 *
 * `config::valid_domain_coverage_threshold` validates the threshold before planning;
 * `dynamic_filter_publish_plan` stores it without revalidation.
 *
 * @note This is only effective for DuckDB native tables which carry the requisite statistics.
 */
struct dynamic_filter_publication_policy {
  /// Whether publication constructs zone-map filters alongside membership filters
  bool emit_zone_map_filters = false;
  /// Fraction of a key's domain a build may cover and still publish that key's filters
  double domain_coverage_threshold = 0.9;
};

/**
 * @brief Immutable publication plan for one `sirius_physical_hash_join`
 *
 * `sirius_plan_comparison_join` builds a dense array of admitted keys, sparse bindings from each
 * target to that array, publication policy, and device-replica placements. The plan is consumed by
 * `publish_dynamic_filters()`.
 */
class dynamic_filter_publish_plan final {
 public:
  /**
   * @brief One statically admitted build key, in admitted order
   *
   * Static legality -- join semantics, comparison, bound-key shape, compatible type -- is decided
   * once at plan time by the admission helper; the runtime publisher re-checks only runtime facts
   * (join mode, complete build, live targets) and never re-derives legality.
   */
  struct admitted_key {
    /**
     * @brief Index into the planner's original condition vector, recorded before
     * `wrap_join_conditions` and before the physical join's equality-first condition reordering
     *
     * Provenance and uniqueness only. This index is in the planner's original condition order;
     * anything the physical join derives from its own `conditions` vector -- `key_casts`,
     * `right_key_col_indices` -- is in reordered equality-ordinal space and must never be
     * subscripted with a value from this space.
     */
    std::size_t planner_condition_index = 0;
    /**
     * @brief Build-child output column holding the key values; indexes the build table view the
     * publisher receives
     */
    cudf::size_type build_key_ordinal = 0;
    /**
     * @brief Probe-child output column holding the probe-side key values
     *
     * This is the entry ordinal the discovery walk (`planner::trace_probe_key()` /
     * `planner::place_endpoint()`) starts from; targets publish at
     * `key_binding::channel_push_ordinal`, the walk's exit ordinal. Every accepted trace step can
     * remap the value, so the two ordinals are equal only when the walk accepts no step.
     *
     * Always a real bound reference: admission admits no key whose probe side is not one, so this
     * ordinal never needs a sentinel reading.
     */
    cudf::size_type probe_key_ordinal = 0;
    /**
     * @brief Build key storage type recorded at plan time
     *
     * The publisher validates that the runtime build column carries this exact type before
     * constructing filters from it. Never `cudf::type_id::EMPTY` in a constructed plan -- admission
     * does not admit a key whose type it cannot represent.
     */
    cudf::data_type storage_type{cudf::type_id::EMPTY};
    /**
     * @brief Probe-side key storage type recorded at plan time
     *
     * `cudf::type_id::EMPTY` when the probe side carries no representable bound reference.
     * `planner::direct_route_admissible` consumes it; a `direct` endpoint filters the probe-side
     * column, so its admissibility is decided against this type, not the build type.
     */
    cudf::data_type probe_storage_type{cudf::type_id::EMPTY};
    /**
     * @brief Carried pre-materialization classification of this condition's sides
     */
    dynamic_filter_condition_shape key_shape{};
    /**
     * @brief Unfiltered cardinality of the base table this build key traces to
     *
     * Stored with the admitted key. 0 when untraceable, which disables the membership coverage gate
     * for this key. The value is a true upper bound, never an estimate; see
     * `planner/dynamic_filter/build_key_domain.hpp` for the evidence contract.
     */
    std::size_t build_key_domain_cardinality = 0;
    /**
     * @brief Whether this key is proven unique in the build's base relation
     *
     * Set by admission only when the planner's proven-unique column set is exactly the singleton
     * of this condition's build column. The membership coverage gate fires solely for
     * proven-unique keys: only then is `build_rows / build_key_domain_cardinality` the coverage
     * fraction it claims to be -- for duplicate keys the same ratio measures row retention, and a
     * near-1.0 retention can coexist with a highly selective filter.
     */
    bool build_key_proven_unique = false;

    [[nodiscard]] bool operator==(admitted_key const&) const = default;
  };

  /**
   * @brief One admitted key's binding onto one target channel
   *
   * Contains only the admitted keys routed to this target. `publish_dynamic_filters()` iterates the
   * target's bindings rather than the full admitted-key array.
   */
  struct key_binding {
    /**
     * @brief Index into `admitted_keys()`
     */
    std::size_t admitted_key_index = 0;
    /**
     * @brief Push coordinate in the target channel's push space; see
     * @ref dynamic_filter_route_class
     */
    std::size_t channel_push_ordinal = 0;
    /**
     * @brief Probe-side storage type at the bound coordinate
     *
     * Zone-map filters are pushed to this binding only when this equals the build key's runtime
     * type; `cudf::type_id::EMPTY` marks a probe type with no cudf representation and suppresses
     * zone maps for this binding only.
     */
    cudf::data_type probe_storage_type{cudf::type_id::EMPTY};

    [[nodiscard]] bool operator==(key_binding const&) const = default;
  };

  /**
   * @brief One channel that receives filters from this publisher
   */
  struct probe_target {
    /**
     * @brief The endpoint channel; never null in a constructed plan
     */
    std::shared_ptr<sirius_dynamic_filter_set> filter_set;
    /**
     * @brief Which push-coordinate space `key_binding::channel_push_ordinal` values live in
     */
    dynamic_filter_route_class route_class = dynamic_filter_route_class::scan;
    /**
     * @brief Whether zone-map filters may be pushed to this target
     *
     * Membership filters are accepted by every target; zone maps only where this is true. Always
     * false for `dynamic_filter_route_class::direct` targets, which are membership-only.
     */
    bool accepts_zone_map_filters = true;
    /**
     * @brief Sparse key bindings
     *
     * An admitted key need not appear in every target. This vector may be empty when planning
     * created the channel but admitted no key for it.
     */
    std::vector<key_binding> key_bindings;
  };

  static constexpr double k_default_domain_coverage_threshold =
    0.9;  ///< Default fraction of a key's domain a build may cover and still publish its filters

  /**
   * @brief Construct the canonical disabled plan
   *
   * No admitted keys, no targets, `enabled() == false` -- every constructor invariant holds
   * vacuously. The validating constructor below is the only path that can produce an enabled plan.
   */
  dynamic_filter_publish_plan() = default;

  /**
   * @brief Validate, canonicalize, and freeze one hash join's publication metadata
   *
   * The plan persists five distinct coordinates: original planner condition index, dense
   * admitted-key index, runtime build-table ordinal, probe-child entry ordinal, and target-specific
   * channel push ordinal. The constructor validates every represented relationship, but it cannot
   * verify that a channel push ordinal is meaningful in the target channel's schema; the discovery
   * walk (`trace_probe_key()` / `place_endpoint()`) owns that precondition.
   *
   * @p replica_spaces is sorted by device ID and deduplicated. Duplicate admitted keys for the same
   * planner condition are rejected.
   *
   * @throw std::invalid_argument if the plan has probe targets but no replica space
   * @throw std::invalid_argument if a replica space is not a GPU space with HOST-tier staging
   * @throw std::invalid_argument if two admitted keys name the same planner condition
   * @throw std::invalid_argument if an admitted key has a negative build ordinal or an EMPTY
   * storage type (either is the signature of an upstream conversion defect)
   * @throw std::invalid_argument if a probe target has a null channel, is a membership-only
   * (direct) target accepting zone maps, binds an admitted key that does not exist, or binds the
   * same admitted key more than once
   *
   * @param[in] admitted_keys Statically admitted build keys, in admitted order
   * @param[in] probe_targets Endpoint channels with their sparse key bindings
   * @param[in] replica_spaces GPU/HOST placements for device-local filter replicas
   * @param[in] policy Publication policy transported from configuration and validated at its
   * ingress; see @ref dynamic_filter_publication_policy
   */
  dynamic_filter_publish_plan(std::vector<admitted_key> admitted_keys,
                              std::vector<probe_target> probe_targets,
                              std::vector<dynamic_filter_replica_space> replica_spaces,
                              dynamic_filter_publication_policy policy = {});

  /**
   * @brief Whether this producer publishes at all
   *
   * A plan with admitted keys but no target is disabled. A plan with a target but no admitted key
   * is enabled and completes a publication attempt without emitting a filter.
   *
   * @return True when at least one probe target exists
   */
  [[nodiscard]] bool enabled() const noexcept { return !_probe_targets.empty(); }
  /**
   * @brief Admitted build keys in admitted order
   *
   * @return The dense admitted-key array; a key's position in it is the admitted-key index that
   * `key_binding::admitted_key_index` refers to
   */
  [[nodiscard]] std::vector<admitted_key> const& admitted_keys() const noexcept
  {
    return _admitted_keys;
  }
  /**
   * @brief Target channels and their sparse bindings
   *
   * @return Targets in planner-defined order
   */
  [[nodiscard]] std::vector<probe_target> const& probe_targets() const noexcept
  {
    return _probe_targets;
  }
  /**
   * @brief Whether the publisher also constructs zone-map filters
   *
   * @return The configured zone-map publication setting
   */
  [[nodiscard]] bool emit_zone_map_filters() const noexcept
  {
    return _policy.emit_zone_map_filters;
  }
  /**
   * @brief Canonical device-replica placements
   *
   * @return Non-owning placements sorted by GPU device ID
   */
  [[nodiscard]] std::vector<dynamic_filter_replica_space> const& replica_spaces() const noexcept
  {
    return _replica_spaces;
  }
  /**
   * @brief Domain coverage at or above which publication skips an eligible key
   *
   * @return The validated configuration value
   */
  [[nodiscard]] double domain_coverage_threshold() const noexcept
  {
    return _policy.domain_coverage_threshold;
  }

 private:
  std::vector<admitted_key> _admitted_keys;
  std::vector<probe_target> _probe_targets;
  dynamic_filter_publication_policy _policy{};
  std::vector<dynamic_filter_replica_space>
    _replica_spaces;  ///< Non-owning GPU/HOST placements; see @ref dynamic_filter_replica_space for
                      ///< the lifetime contract
};

}  // namespace sirius::op
