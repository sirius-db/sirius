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

#include "planner/dynamic_filter/dynamic_filter_scan_routes.hpp"

#include "cudf/cudf_utils.hpp"
#include "helper/type_conversions.hpp"

#include <cudf/types.hpp>

namespace sirius::planner::dynamic_filter_scan_routes::detail {

std::optional<dynamic_filter_routing_decision> stop_before_minting(
  duckdb_join_filter_candidate const& candidate, bool device_placement_available) noexcept
{
  switch (candidate.kind()) {
    case duckdb_candidate_kind::absent: return dynamic_filter_routing_decision::no_duckdb_metadata;
    case duckdb_candidate_kind::malformed:
      return dynamic_filter_routing_decision::metadata_malformed;
    case duckdb_candidate_kind::statistics_only:
      return dynamic_filter_routing_decision::build_side_unfiltered;
    case duckdb_candidate_kind::admitted: break;
  }

  // An unfiltered build is (for FK-shaped joins) the whole key domain -- its filter keeps every
  // probe row by construction, so wiring a producer target for it only buys overhead. DuckDB's hint
  // covers the delim-join case where the effective build data is children[0].
  if (!candidate.build_subtree_has_filter_hint()) {
    return dynamic_filter_routing_decision::build_side_unfiltered;
  }
  if (!device_placement_available) { return dynamic_filter_routing_decision::no_device_placement; }
  return std::nullopt;
}

dynamic_filter_scan_target_input to_scan_target_input(
  duckdb_probe_target_candidate const& target_candidate)
{
  dynamic_filter_scan_target_input target_input;
  target_input.columns.reserve(target_candidate.columns().size());
  for (auto const& column : target_candidate.columns()) {
    // An unmappable probe type is a routine classification, not an error: it leaves the binding's
    // type EMPTY, which suppresses zone maps for that binding only.
    auto const probe_type = sirius::try_get_cudf_type(sirius::from_duckdb(column.storage_type));
    target_input.columns.push_back(
      {.channel_push_ordinal = column.column_index,
       .probe_storage_type   = probe_type.value_or(cudf::data_type{cudf::type_id::EMPTY})});
  }
  return target_input;
}

}  // namespace sirius::planner::dynamic_filter_scan_routes::detail
