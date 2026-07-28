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

/**
 * @file dynamic_filter_scan_routes.hpp
 * @brief Plan-time routing of one producing hash join onto its scan endpoints
 *
 * `sirius_plan_comparison_join` calls @ref sirius::planner::resolve_scan_routes once per hash join,
 * between `duckdb_join_filter_candidate_adapter::extract` -- which snapshots DuckDB's join-filter
 * metadata into a `duckdb_join_filter_candidate` -- and `admit_dynamic_filter_keys`, which decides
 * which build keys those endpoints receive. Routing answers a single question: does this producer
 * wire dynamic-filter targets at all, and if so, which channels does it publish into.
 *
 * The only side effect is channel minting and the producer registration that follows it, and both
 * happen through the injected channel factory: `sirius_plan_comparison_join` passes
 * `sirius_physical_plan_generator::get_or_create_dynamic_filter_channel`, and tests pass a stub, so
 * routing is reachable without a GPU or a `duckdb::ClientContext`. Everything else here is a pure
 * function of the candidate snapshot.
 *
 * Nothing in this file logs. The caller reports @ref
 * sirius::planner::dynamic_filter_routing_decision, because the wording and the build-side
 * cardinality those messages carry belong to the planner.
 */

#pragma once

#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "planner/dynamic_filter/duckdb_join_filter_candidate_adapter.hpp"
#include "planner/dynamic_filter/dynamic_filter_key_admission.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace duckdb {
class DynamicTableFilterSet;
}  // namespace duckdb

namespace sirius::planner {

/**
 * @brief Why -- or why not -- one producing join wires dynamic-filter targets
 *
 * Exactly one value per producer. The join-edge placement helper of issue #1010's next unit extends
 * the enumerators rather than the shape.
 */
enum class dynamic_filter_routing_decision : std::uint8_t {
  wired,               ///< At least one endpoint channel was minted and registered
  no_duckdb_metadata,  ///< The candidate is `duckdb_candidate_kind::absent`
  metadata_malformed,  ///< The candidate is `duckdb_candidate_kind::malformed`
  /// The candidate is `duckdb_candidate_kind::statistics_only`, or it is admitted and carries no
  /// `build_subtree_has_filter_hint`
  build_side_unfiltered,
  no_device_placement,  ///< The caller found no GPU/HOST space pair for device-local replicas
  no_live_channel       ///< Every target DuckDB named failed to mint a channel
};

/**
 * @brief The routing outcome for one producing join
 *
 * `targets` and `target_inputs` are index-aligned and describe the same endpoints in the same
 * order: entry `t` of each is the target that DuckDB's `probe_info[t]` named, minus any entry whose
 * channel could not be minted. Both are empty unless `decision ==
 * dynamic_filter_routing_decision::wired`.
 */
struct resolved_scan_routes {
  /**
   * @brief The routing decision this producer reached
   */
  dynamic_filter_routing_decision decision = dynamic_filter_routing_decision::no_duckdb_metadata;
  /**
   * @brief Endpoints for `dynamic_filter_publish_plan`, each holding a minted channel this producer
   * is registered on
   *
   * `key_bindings` is left empty here: bindings are `admit_dynamic_filter_keys`'s output, and the
   * caller moves them in afterwards.
   */
  std::vector<op::dynamic_filter_publish_plan::probe_target> targets;
  /**
   * @brief The same endpoints described in admission's input vocabulary -- channel push ordinals
   * and probe storage types, per DuckDB filter ordinal
   */
  std::vector<dynamic_filter_scan_target_input> target_inputs;
  /**
   * @brief The filter-ordinal to condition-index mapping DuckDB recorded, or empty when it recorded
   * none
   *
   * Read the vector as that mapping: `hinted_condition_indexes[f]` is the position, in the join's
   * original condition vector, of the condition DuckDB derived its filter ordinal `f` from.
   *
   * Non-empty for every admitted candidate, including those that stop short of wiring: DuckDB's
   * naming of a condition set restricts admission whether or not this producer ends up with a
   * target. `duckdb_join_filter_candidate_adapter::extract` never admits a candidate with an empty
   * index list, so empty is unambiguously "no hint".
   */
  std::vector<std::size_t> hinted_condition_indexes;
};

namespace dynamic_filter_scan_routes::detail {

/**
 * @brief The decision reached before any channel is minted, or nothing when routing proceeds to
 * minting
 *
 * Exposed only so @ref sirius::planner::resolve_scan_routes can keep its non-generic reasoning out
 * of the header; not part of the routing API.
 *
 * @param[in] candidate The snapshot of this join's DuckDB join-filter metadata
 * @param[in] device_placement_available Whether device-local replicas can be placed
 * @return The terminal decision, or `std::nullopt` to continue with channel minting
 */
[[nodiscard]] std::optional<dynamic_filter_routing_decision> stop_before_minting(
  duckdb_join_filter_candidate const& candidate, bool device_placement_available) noexcept;

/**
 * @brief Describe one DuckDB-named probe target in admission's input vocabulary
 *
 * Exposed only so @ref sirius::planner::resolve_scan_routes can keep its non-generic reasoning out
 * of the header; not part of the routing API.
 *
 * @param[in] target_candidate One probe target of an admitted candidate
 * @return The target's columns, in DuckDB filter-ordinal order
 */
[[nodiscard]] dynamic_filter_scan_target_input to_scan_target_input(
  duckdb_probe_target_candidate const& target_candidate);

}  // namespace dynamic_filter_scan_routes::detail

/**
 * @brief Resolve DuckDB-named scan endpoints into Sirius-owned target descriptors, registering this
 * producer on every channel it mints
 *
 * A candidate that DuckDB did not admit, whose build subtree carries no filter hint, or whose
 * replicas have nowhere to live stops before @p mint_channel is ever called: registering a producer
 * on a channel this join then abandons would leave the channel expecting a publication that never
 * comes. That is why device placement is a parameter decided by the caller rather than a memory
 * manager this helper queries.
 *
 * A target whose channel mints as null is dropped -- the master switch being off mints null for
 * every target, which is how a disabled producer wires nothing -- and a candidate all of whose
 * targets drop reaches `dynamic_filter_routing_decision::no_live_channel` with no targets, which is
 * a quiet outcome rather than a fault.
 *
 * @tparam ChannelFactory Callable mapping a `duckdb::DynamicTableFilterSet` identity to the Sirius
 * channel that pairs this producer with its consumer scan, or to null when no channel is to exist
 *
 * @param[in] candidate The snapshot of this join's DuckDB join-filter metadata
 * @param[in] device_placement_available Whether the caller located both a GPU and a HOST memory
 * space to place device-local filter replicas in
 * @param[in,out] mint_channel The channel factory; invoked once per named target, and its own state
 * (the plan generator's channel map, a test stub's call log) is updated by each call
 * @return This producer's routing outcome
 */
template <class ChannelFactory>
  requires std::invocable<ChannelFactory&, duckdb::DynamicTableFilterSet const*> &&
           std::convertible_to<
             std::invoke_result_t<ChannelFactory&, duckdb::DynamicTableFilterSet const*>,
             std::shared_ptr<op::sirius_dynamic_filter_set>>
[[nodiscard]] resolved_scan_routes resolve_scan_routes(
  duckdb_join_filter_candidate const& candidate,
  bool device_placement_available,
  ChannelFactory&& mint_channel)
{
  resolved_scan_routes routes;
  if (candidate.kind() == duckdb_candidate_kind::admitted) {
    routes.hinted_condition_indexes = candidate.condition_indexes();
  }
  if (auto const stopped = dynamic_filter_scan_routes::detail::stop_before_minting(
        candidate, device_placement_available)) {
    routes.decision = *stopped;
    return routes;
  }

  routes.targets.reserve(candidate.targets().size());
  routes.target_inputs.reserve(candidate.targets().size());
  for (auto const& target_candidate : candidate.targets()) {
    std::shared_ptr<op::sirius_dynamic_filter_set> channel =
      mint_channel(target_candidate.channel_identity().get());
    if (!channel) { continue; }
    channel->register_producer();
    routes.target_inputs.push_back(
      dynamic_filter_scan_routes::detail::to_scan_target_input(target_candidate));
    routes.targets.push_back(op::dynamic_filter_publish_plan::probe_target{
      .filter_set               = std::move(channel),
      .route_class              = op::dynamic_filter_route_class::scan,
      .accepts_zone_map_filters = true,
      .key_bindings             = {}});
  }
  routes.decision = routes.targets.empty() ? dynamic_filter_routing_decision::no_live_channel
                                           : dynamic_filter_routing_decision::wired;
  return routes;
}

}  // namespace sirius::planner
