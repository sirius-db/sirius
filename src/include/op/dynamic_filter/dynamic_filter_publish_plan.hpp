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

/**
 * @brief Shape of one join-condition side, classified before computed-key materialization
 *
 * The planner materializes computed equality keys into plain bound references before the
 * conditions reach the physical join, so shape is classified while the original expression is
 * still visible and carried here; admission never re-derives it from post-materialization
 * conditions.
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
 * ordinal, in the output schema of the operator the endpoint wraps.
 */
enum class dynamic_filter_route_class : std::uint8_t {
  scan,   ///< A GPU scan consumer; zone-map capable
  direct  ///< A join-edge endpoint consumer; membership only
};

/**
 * @brief Publication policy transported from configuration
 *
 * Validated at the configuration surfaces before planning and stored here without revalidation.
 * The coverage gate arms only for unique keys with DuckDB-native cardinality evidence.
 */
struct dynamic_filter_publication_policy {
  /// Default coverage fraction at or above which publication skips a key
  static constexpr double k_default_domain_coverage_threshold = 0.9;
  /// Default bound on the exact hash IN-list's estimated set size as a fraction of the smallest
  /// probe-GPU L2 cache; see operator_params::dynamic_filter_inlist_max_l2_fraction for the full
  /// semantics
  static constexpr double k_default_inlist_max_l2_fraction = 0.125;

  /// Whether publication constructs zone-map filters alongside membership filters
  bool emit_zone_map_filters = false;
  /// Coverage fraction at or above which publication skips a key
  double domain_coverage_threshold = k_default_domain_coverage_threshold;
  /// Estimated-set-bytes bound for the exact hash IN-list, as a fraction of the smallest
  /// probe-GPU L2 cache; larger sets publish the Bloom filter instead. No domain validation
  /// here: both configuration surfaces enforce [0, 1], and tests may legitimately construct
  /// out-of-domain plans.
  double inlist_max_l2_fraction = k_default_inlist_max_l2_fraction;
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
     * `cudf::type_id::EMPTY` when the bound probe reference has no cuDF representation.
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
     * 0 when untraceable, which disables the membership coverage gate for this key. The value is a
     * true upper bound, never an estimate; see `planner/dynamic_filter/build_key_domain.hpp` for
     * the evidence contract.
     */
    std::size_t build_key_domain_cardinality = 0;
    /**
     * @brief Whether this key is proven unique in the build's base relation
     *
     * The coverage gate uses this only when the planner proves this build column unique. For a
     * non-unique key, the row-count ratio is not a distinct-key coverage fraction.
     */
    bool build_key_proven_unique = false;

    [[nodiscard]] bool operator==(admitted_key const&) const = default;
  };

  /**
   * @brief One admitted key's binding onto one target channel
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
     * For scan bindings, zone-map filters are pushed only when this equals the build key's runtime
     * type; `cudf::type_id::EMPTY` suppresses zone maps for that binding. Direct bindings require
     * INT32 or INT64 storage matching the admitted build key.
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
     * An admitted key need not appear in every target. The constructor permits an empty vector.
     */
    std::vector<key_binding> key_bindings;
  };

  /**
   * @brief Construct the disabled plan
   *
   * No admitted keys, no targets, `enabled() == false`; only the validating constructor can
   * produce an enabled plan.
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
   * @p replica_spaces is sorted by device ID and deduplicated.
   *
   * @throw std::invalid_argument if the plan has probe targets but no replica space
   * @throw std::invalid_argument if a replica space is not a GPU space with HOST-tier staging
   * @throw std::invalid_argument if two admitted keys name the same planner condition
   * @throw std::invalid_argument if an admitted key has a negative build ordinal or an EMPTY
   * storage type
   * @throw std::invalid_argument if a probe target has a null channel, is a membership-only
   * (direct) target accepting zone maps, binds an admitted key that does not exist, or binds the
   * same admitted key more than once
   * @throw std::invalid_argument if a direct binding's probe storage is not INT32 or INT64, or
   * does not match the admitted build key's storage type
   *
   * @param[in] admitted_keys Statically admitted build keys, in admitted order
   * @param[in] probe_targets Endpoint channels with their sparse key bindings
   * @param[in] replica_spaces GPU/HOST placements for device-local filter replicas
   * @param[in] policy Publication policy; see @ref dynamic_filter_publication_policy
   */
  dynamic_filter_publish_plan(std::vector<admitted_key> admitted_keys,
                              std::vector<probe_target> probe_targets,
                              std::vector<dynamic_filter_replica_space> replica_spaces,
                              dynamic_filter_publication_policy policy = {});

  /**
   * @brief Whether this producer publishes at all
   *
   * A plan with a target but no admitted key is enabled and completes a publication attempt
   * without emitting a filter.
   */
  [[nodiscard]] bool enabled() const noexcept { return !_probe_targets.empty(); }
  /**
   * @brief Admitted build keys, in admitted order
   */
  [[nodiscard]] std::vector<admitted_key> const& admitted_keys() const noexcept
  {
    return _admitted_keys;
  }
  /**
   * @brief Target channels and their sparse bindings, in planner-defined order
   */
  [[nodiscard]] std::vector<probe_target> const& probe_targets() const noexcept
  {
    return _probe_targets;
  }
  /**
   * @brief Whether the publisher also constructs zone-map filters
   */
  [[nodiscard]] bool emit_zone_map_filters() const noexcept
  {
    return _policy.emit_zone_map_filters;
  }
  /**
   * @brief Device-replica placements, sorted by GPU device ID
   */
  [[nodiscard]] std::vector<dynamic_filter_replica_space> const& replica_spaces() const noexcept
  {
    return _replica_spaces;
  }
  /**
   * @brief Whether the plan holds a replica space on GPU @p gpu_device_id
   *
   * The delivery hook consults this before publishing: a build batch resident on a GPU outside the
   * replica set has no source space to allocate from and is skipped, not published.
   */
  [[nodiscard]] bool has_replica_on_device(int gpu_device_id) const noexcept;
  /**
   * @brief Domain coverage at or above which publication skips an eligible key
   */
  [[nodiscard]] double domain_coverage_threshold() const noexcept
  {
    return _policy.domain_coverage_threshold;
  }
  /**
   * @brief Estimated-set-bytes bound for the exact hash IN-list, as a fraction of probe-GPU L2
   */
  [[nodiscard]] double inlist_max_l2_fraction() const noexcept
  {
    return _policy.inlist_max_l2_fraction;
  }

  /**
   * @brief Drop replica targets on GPUs outside @p admitted_gpu_ids
   *
   * An empty list means "no subset" and leaves the plan untouched. A restriction that would erase
   * every replica space instead disables the plan (`enabled() == false`, probe targets cleared):
   * the constructor invariant "probe targets => replica spaces" holds for the plan's whole
   * lifetime. See sirius_pipeline_converter::restrict_dynamic_filter_replicas for why restriction
   * is needed.
   */
  void restrict_replicas_to(std::vector<int> const& admitted_gpu_ids);

 private:
  std::vector<admitted_key> _admitted_keys;
  std::vector<probe_target> _probe_targets;
  dynamic_filter_publication_policy _policy{};
  std::vector<dynamic_filter_replica_space>
    _replica_spaces;  ///< Non-owning GPU/HOST placements; see @ref dynamic_filter_replica_space for
                      ///< the lifetime contract
};

}  // namespace sirius::op
