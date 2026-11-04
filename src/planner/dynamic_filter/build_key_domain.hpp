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

#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"
#include "planner/scan_column_origin.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace duckdb {
class ClientContext;
class LogicalComparisonJoin;
class LogicalGet;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::scan_manager {
class sirius_scan_manager;
}

namespace sirius::planner {

/**
 * @brief Copies domain evidence for each original join condition through row-subset lineage
 *
 * Unknown or mismatched sources return empty evidence. Call before `create_plan` moves the logical
 * children, while the planning window keeps the pin registry stable.
 */
[[nodiscard]] std::vector<op::dynamic_filter_domain_evidence> build_key_domain_evidence_for(
  duckdb::LogicalComparisonJoin const& join,
  duckdb::ClientContext& context,
  scan_manager::sirius_scan_manager const& registry);

/**
 * @brief Callable returning domain evidence for one base-scan output column
 *
 * Production converts callback failures to unknown evidence; custom-source exceptions propagate.
 * Implementations must not answer from `LogicalGet::estimated_cardinality`, which may reflect
 * filters and understate the domain.
 */
template <class Source>
concept base_table_domain_evidence_source =
  std::invocable<Source const&, duckdb::LogicalGet const&, std::size_t> &&
  std::same_as<std::invoke_result_t<Source const&, duckdb::LogicalGet const&, std::size_t>,
               op::dynamic_filter_domain_evidence>;

namespace detail {

/**
 * @brief Returns one base-scan column per original condition, or nullopt for an untraceable key
 *
 * Call after type and binding resolution and before `create_plan` moves the children.
 */
[[nodiscard]] std::vector<std::optional<scan_column_origin>> resolve_build_key_scan_columns(
  duckdb::LogicalComparisonJoin const& join);

}  // namespace detail

/**
 * @brief Returns domain evidence per original condition; unresolved keys have unknown evidence
 *
 * Each distinct scan-column pair is queried once.
 *
 * @pre @p join still owns both logical children; call before `create_plan`
 */
template <base_table_domain_evidence_source Source>
[[nodiscard]] std::vector<op::dynamic_filter_domain_evidence> build_key_domain_evidence_for(
  duckdb::LogicalComparisonJoin const& join, Source const& evidence_for)
{
  auto const columns = detail::resolve_build_key_scan_columns(join);
  std::vector<op::dynamic_filter_domain_evidence> domains(columns.size());
  std::vector<std::pair<scan_column_origin, op::dynamic_filter_domain_evidence>> memo;
  for (std::size_t condition_index = 0; condition_index < columns.size(); ++condition_index) {
    if (!columns[condition_index]) { continue; }
    auto const column = *columns[condition_index];
    auto const hit    = std::ranges::find_if(memo, [&](auto const& entry) {
      return entry.first.get == column.get && entry.first.ordinal == column.ordinal;
    });
    if (hit != memo.end()) {
      domains[condition_index] = hit->second;
      continue;
    }
    domains[condition_index] = evidence_for(*column.get, column.ordinal);
    memo.emplace_back(column, domains[condition_index]);
  }
  return domains;
}

}  // namespace sirius::planner
