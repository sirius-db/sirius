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
 * @brief Coordinate space a target channel expects at push time
 *
 * A scan endpoint's channel is installed with a column_ids-to-output-position remap
 * (`sirius_dynamic_filter_set::set_consumer_column_remap`) that `push_filter` applies once at push
 * time, so scan push ordinals stay in DuckDB `column_ids` space. A join-edge endpoint's channel
 * installs no remap, so its push ordinals are the producing join's probe-child output ordinals,
 * stored unchanged. Pushing an output ordinal through a scan channel would remap twice and can
 * silently filter a same-typed wrong column.
 */
enum class dynamic_filter_route_class : std::uint8_t {
  scan,   ///< Endpoint on a DYNAMIC_FILTER-wrapped table scan; `column_ids`-space push ordinals
  direct  ///< Endpoint at the producing join's probe edge; probe-child output push ordinals
};

/**
 * @brief Immutable plan-time description of one hash join's dynamic-filter publication
 *
 * The planner owns admission, routing, and placement: `sirius_plan_comparison_join` classifies key
 * shapes, resolves endpoint channels, and builds this value through the admission helper in
 * `planner/dynamic_filter_key_admission.hpp`. The runtime publisher (`publish_dynamic_filters` in
 * `op/dynamic_filter_publisher.hpp`) consumes it but cannot mutate its keys, targets, policy, or
 * device set after operator construction, and it is the publisher's only key/target input --
 * runtime publication does not read DuckDB metadata.
 *
 * The representation is a dense admitted-key array with sparse per-target bindings: admitted keys
 * carry everything filter construction needs (build ordinal, storage type, carried shape), while
 * each target lists only the keys that apply to it, expressed in its own channel's push-coordinate
 * space. Field naming follows the admitted-key schema in issue #1010's delivery plan -- delivered
 * by that issue's documentation unit as
 * `docs/super-sirius/issue-1010-github-delivery-plan-v2.md` (`filter_set` is that schema's
 * `endpoint_channel`; `accepts_zone_map_filters` encodes its `accepted_filter_kinds`, membership
 * filters being accepted by every target).
 *
 * Replica placements cover every active GPU space, each paired with its planned HOST staging space.
 * The build GPU's space is included because it sources filter construction and the remote
 * transfers, not because a second copy is made there; only other GPUs receive replicas. Their owner
 * follows the lifetime contract on @ref dynamic_filter_replica_space.
 *
 * Placement and routing are orthogonal: `replica_spaces()` says where a filter needs device
 * storage, while `probe_target::key_bindings` says which logical channel coordinate receives it.
 * For keys 0 and 1, bindings `A:{0->12}` and `B:{1->4, 0->7}` push key 0 to A/12 and B/7, and key 1
 * only to B/4. Each filter is constructed once, all usable replicas complete before fan-out, and a
 * missing binding means no publication to that target. Channel producer registration happens
 * earlier during plan wiring; it is lifecycle bookkeeping, not a replica-readiness signal.
 *
 * The nested structs are true aggregates constructed with designated initializers at every
 * construction site; member declaration order is therefore part of their API and must stay stable.
 */
/**
 * @brief Publication policy transported from configuration
 *
 * Validated once where it enters the engine (`config::valid_domain_coverage_threshold`); the plan
 * constructor does not re-validate it. Everything outside this struct is planner-computed and
 * constructor-validated; everything inside it is config-transported and ingress-validated.
 */
struct dynamic_filter_publication_policy {
  /// Whether publication constructs zone-map filters alongside membership filters
  bool emit_zone_map_filters = false;
  /// Fraction of a key's domain a build may cover and still publish that key's filters
  double domain_coverage_threshold = 0.9;
};

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
     * `right_key_col_indices`, `condition_key_shapes` -- is in reordered equality-ordinal space
     * and must never be subscripted with a value from this space.
     */
    std::size_t planner_condition_index = 0;
    /**
     * @brief Build-child output column holding the key values; indexes the build table view the
     * publisher receives
     */
    cudf::size_type build_key_ordinal = 0;
    /**
     * @brief Build key storage type recorded at plan time
     *
     * The publisher validates that the runtime build column carries this exact type before
     * constructing filters from it. Never `cudf::type_id::EMPTY` in a constructed plan -- admission
     * does not admit a key whose type it cannot represent.
     */
    cudf::data_type storage_type{cudf::type_id::EMPTY};
    /**
     * @brief Carried pre-materialization classification of this condition's sides
     */
    dynamic_filter_condition_shape key_shape{};
    /**
     * @brief Unfiltered cardinality of the base table this build key traces to
     *
     * 0 when untraceable, which disables the membership coverage gate for this key. Carried per
     * key rather than in a parallel array so a cardinality cannot become misaligned with its key.
     * The value is a true upper bound, never an estimate; see
     * `planner/build_key_domain.hpp` for the evidence contract.
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
   * Bindings are sparse because application routes are chosen per key: a producer with one
   * scan-routed and one edge-routed key lists each key only under its own route's target, and the
   * publisher iterates a target's bindings, never the full admitted-key array.
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
   * @brief One application endpoint the publisher fans filters out to
   *
   * Deliberately no `operator==`: a defaulted form would compare `filter_set` by channel identity,
   * which tests should assert explicitly rather than through whole-value equality.
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
     * Sparse means that an admitted key need not be bound in every target: this vector lists only
     * the admitted keys that reach this one.
     *
     * May be empty: a producer whose keys are all inadmissible still mints its targets so
     * consumer-side operator wrapping and the publication state trajectory are unchanged.
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
   * The plan persists four distinct coordinates: planner condition index for provenance,
   * admitted-key index for dense filter construction, build-key ordinal for runtime build-table
   * lookup, and channel push ordinal for target-specific publication. The constructor validates
   * every relationship it can represent. One precondition is not verifiable here and is trusted
   * to the admission helper, the sole producer of bindings: each `channel_push_ordinal` must be
   * valid in its target channel's push space.
   *
   * DuckDB filter ordinal and target index are plan-construction alignment indexes, not persisted
   * key coordinates: admission consumes the former and replaces it with sparse bindings, while
   * each `probe_targets[i]` owns the bindings for target `i`.
   *
   * @p replica_spaces is canonicalized rather than rejected: any order and any duplication of a
   * device denote the same replica set, so the constructor sorts by device id and drops repeats.
   * Two admitted keys naming the same condition are different -- they denote two plans, not one --
   * and are rejected. `replica_spaces()` therefore returns the canonical form, not the input order.
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
   * Deliberately target-based: a plan with admitted keys but no live target publishes nothing, and
   * a plan with targets but no admitted keys still claims publication (and publishes nothing),
   * keeping the publication state trajectory of producers whose keys are all inadmissible
   * unchanged.
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
  [[nodiscard]] std::vector<probe_target> const& probe_targets() const noexcept
  {
    return _probe_targets;
  }
  [[nodiscard]] bool emit_zone_map_filters() const noexcept
  {
    return _policy.emit_zone_map_filters;
  }
  [[nodiscard]] std::vector<dynamic_filter_replica_space> const& replica_spaces() const noexcept
  {
    return _replica_spaces;
  }
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
