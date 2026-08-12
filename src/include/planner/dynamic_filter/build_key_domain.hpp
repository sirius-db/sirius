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
 * @file build_key_domain.hpp
 * @brief Plan-time domain evidence for hash-join build keys
 *
 * A positional lineage walk resolves each plain build key through value-preserving, row-subset
 * operators to a base scan. A cardinality source then supplies an exact row count or upper bound.
 * Unresolved keys receive 0, which disables their coverage gate.
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
 * @brief A source of base-table domain evidence
 *
 * Answers the unfiltered row count of the scanned base table as an exact figure or a true upper
 * bound, never an estimate that may under-state the domain. Answers `std::nullopt` when the scan
 * cannot vouch for such a figure. The production source converts callback failures to
 * `std::nullopt`; exceptions from a caller-supplied source propagate.
 */
template <class Source>
concept base_table_cardinality_source =
  std::invocable<Source const&, duckdb::LogicalGet const&> &&
  std::same_as<std::invoke_result_t<Source const&, duckdb::LogicalGet const&>,
               std::optional<std::size_t>>;

// Implementation details of build_key_domain_cardinalities.
namespace detail {

/**
 * @brief The base scan producing @p output_ordinal of @p subtree, or null when unresolved
 *
 * Follows the ordinal down through operators that (a) produce it by value-preserving pass-through
 * and (b) emit rows that are an injective image of the traced child's rows. Refuses -- returns
 * null -- at every other operator, at a computed expression, and at a scan that is not a plain
 * base-table scan. Refusal is the only failure mode.
 *
 * @param[in] subtree Root of the logical subtree the ordinal is an output of
 * @param[in] output_ordinal Position in `subtree.types`
 */
[[nodiscard]] duckdb::LogicalGet const* resolve_pass_through_scan(
  duckdb::LogicalOperator const& subtree, std::size_t output_ordinal) noexcept;

/**
 * @brief The base scan behind each of @p join's build keys, indexed by planner condition index
 *
 * Entry `i` is null unless `join.conditions[i].right` is a plain bound reference that
 * @ref resolve_pass_through_scan resolves through `join.children[1]`.
 *
 * The tree must have been through `ResolveOperatorTypes` and `ColumnBindingResolver`, and the
 * call must precede `create_plan`, which moves data out of `join.children`.
 *
 * @param[in] join The producing join, with both logical children still intact
 * @return One scan pointer per condition, in original planner condition order
 */
[[nodiscard]] std::vector<duckdb::LogicalGet const*> resolve_build_key_scans(
  duckdb::LogicalComparisonJoin const& join);

}  // namespace detail

/**
 * @brief Unfiltered base-table cardinality behind each build key; 0 = not established
 *
 * The source is consulted once per distinct resolved scan, not once per key. The result owns its
 * values and remains valid after `create_plan()` moves state out of the logical children.
 *
 * @param[in] join The producing join, with both logical children still intact
 * @return One cardinality per condition, in original planner condition order
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
 * @brief The production evidence source: DuckDB-native table scans only
 *
 * Yields `NodeStatistics::max_cardinality` for a DuckDB-native table scan and converts unsupported
 * scans or callback failures to `std::nullopt`. It does not use
 * `LogicalGet::estimated_cardinality`, which may reflect filters.
 */
class duckdb_base_table_cardinality {
 public:
  explicit duckdb_base_table_cardinality(duckdb::ClientContext& context) noexcept;

  /**
   * @brief The scanned base table's unfiltered row upper bound, or `std::nullopt` when refused
   */
  [[nodiscard]] std::optional<std::size_t> operator()(duckdb::LogicalGet const& get) const noexcept;

 private:
  duckdb::ClientContext* _context;
};

}  // namespace sirius::planner
