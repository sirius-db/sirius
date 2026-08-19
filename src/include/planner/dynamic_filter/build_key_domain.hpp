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

namespace sirius::planner {

/**
 * @brief Callable returning an exact unfiltered row count, a true upper bound, or `std::nullopt`
 *
 * Production converts callback failures to `std::nullopt`; custom-source exceptions propagate.
 * Implementations must not answer from `LogicalGet::estimated_cardinality`, which may reflect
 * filters and understate the domain.
 */
template <class Source>
concept base_table_cardinality_source =
  std::invocable<Source const&, duckdb::LogicalGet const&> &&
  std::same_as<std::invoke_result_t<Source const&, duckdb::LogicalGet const&>,
               std::optional<std::size_t>>;

namespace detail {

/**
 * @brief Traces a column through value-preserving row subsets, or returns null when unresolved
 */
[[nodiscard]] duckdb::LogicalGet const* resolve_pass_through_scan(
  duckdb::LogicalOperator const& subtree, std::size_t output_ordinal) noexcept;

/**
 * @brief Returns one base scan per original condition, or null for an untraceable build key
 *
 * Call after type and binding resolution and before `create_plan` moves the children.
 */
[[nodiscard]] std::vector<duckdb::LogicalGet const*> resolve_build_key_scans(
  duckdb::LogicalComparisonJoin const& join);

}  // namespace detail

/**
 * @brief Returns one domain bound per original condition; 0 means unknown
 *
 * Each distinct scan is queried once.
 *
 * @pre @p join still owns both logical children; call before `create_plan`
 */
template <base_table_cardinality_source Source>
[[nodiscard]] std::vector<std::size_t> build_key_domain_cardinalities(
  duckdb::LogicalComparisonJoin const& join, Source const& evidence_for)
{
  auto const scans = detail::resolve_build_key_scans(join);
  std::vector<std::size_t> domains(scans.size(), 0);
  std::vector<std::pair<duckdb::LogicalGet const*, std::size_t>> memo;
  for (std::size_t condition_index = 0; condition_index < scans.size(); ++condition_index) {
    if (scans[condition_index] == nullptr) { continue; }
    auto const hit =
      std::ranges::find(memo, scans[condition_index], &decltype(memo)::value_type::first);
    if (hit != memo.end()) {
      domains[condition_index] = hit->second;
      continue;
    }
    domains[condition_index] = evidence_for(*scans[condition_index]).value_or(std::size_t{0});
    memo.emplace_back(scans[condition_index], domains[condition_index]);
  }
  return domains;
}

/**
 * @brief Domain evidence for DuckDB-native scans
 *
 * Uses `NodeStatistics::max_cardinality`; unsupported scans and callback failures return
 * `std::nullopt`.
 */
class duckdb_base_table_cardinality {
 public:
  explicit duckdb_base_table_cardinality(duckdb::ClientContext& context) noexcept;

  [[nodiscard]] std::optional<std::size_t> operator()(duckdb::LogicalGet const& get) const noexcept;

 private:
  duckdb::ClientContext* _context;
};

}  // namespace sirius::planner
