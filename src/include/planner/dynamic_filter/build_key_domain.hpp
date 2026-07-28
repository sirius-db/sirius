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
 * `sirius_plan_comparison_join` asks one question here before it hands conditions to
 * `admit_dynamic_filter_keys`: for each join condition, what is the unfiltered row count of the
 * base table its build key comes from? The answer becomes
 * `admitted_key::build_key_domain_cardinality`, the denominator of the membership
 * domain-coverage gate in `dynamic_filter_source_policy.hpp`; 0 means the count could not be
 * established, which disables the gate for that key and the key publishes as it would without
 * evidence.
 *
 * The answer is produced in two decoupled halves:
 *
 *  - **Mechanism** -- a positional lineage walk (`detail::resolve_pass_through_scan`) that follows
 *    one output ordinal down through operators whose rows are an injective image of the traced
 *    child's rows and whose columns pass values through unchanged. The walk refuses -- silently,
 *    its only failure mode -- at every operator it does not model, so an unmodelled shape can
 *    never yield a wrong table.
 *  - **Policy** -- a @ref base_table_cardinality_source that turns a resolved scan into a row
 *    count. The production source, @ref duckdb_base_table_cardinality, answers only for
 *    DuckDB-native table scans, where the figure is a true upper bound; everything else is
 *    refused. Tests substitute a stub source.
 *
 * The gate's safety rests on the source's contract: the returned count must never under-state the
 * domain, because an under-stated denominator inflates the coverage ratio and suppresses a filter
 * that would have paid for itself. Refusing (`std::nullopt` -> 0 -> gate off) is always safe; the
 * key then publishes exactly as today.
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
 * cannot vouch for such a figure; the walk then reports 0 for that key and the coverage gate stays
 * off. Sources must not throw; the production source converts its own callback failures to
 * `std::nullopt`.
 */
template <class Source>
concept base_table_cardinality_source =
  std::invocable<Source const&, duckdb::LogicalGet const&> &&
  std::same_as<std::invoke_result_t<Source const&, duckdb::LogicalGet const&>,
               std::optional<std::size_t>>;

/// Internal: declared here so the walk is unit-testable without a ClientContext. Not part of the
/// module's public surface.
namespace detail {

/**
 * @brief The base scan producing @p output_ordinal of @p subtree, or null when unresolved
 *
 * Follows the ordinal down through operators that (a) produce it by value-preserving pass-through
 * and (b) emit rows that are an injective image of the traced child's rows. Refuses -- returns
 * null -- at every other operator, at a computed expression, and at a scan that is not a plain
 * base-table scan. Refusal is the only failure mode; the walk neither throws nor logs.
 *
 * @param[in] subtree Root of the logical subtree the ordinal is an output of
 * @param[in] output_ordinal Position in `subtree.types`
 * @return The producing base scan, or null
 */
[[nodiscard]] duckdb::LogicalGet const* resolve_pass_through_scan(
  duckdb::LogicalOperator const& subtree, std::size_t output_ordinal) noexcept;

/**
 * @brief The base scan behind each of @p join's build keys, indexed by planner condition index
 *
 * Entry `i` is null unless `join.conditions[i].right` is a plain bound reference that
 * @ref resolve_pass_through_scan resolves through `join.children[1]`.
 *
 * The tree must have been through `ResolveOperatorTypes` and `ColumnBindingResolver` (true for
 * every tree `sirius_physical_plan_generator` sees), and the call must precede `create_plan`,
 * which moves data out of `join.children`.
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
 * The value `admit_dynamic_filter_keys` records on each admitted key as
 * `build_key_domain_cardinality`. The source is consulted once per distinct resolved scan, not
 * once per key. The result is an eager vector because the caller consumes it after `create_plan`
 * has moved data out of the logical children -- a lazy view over `join.conditions` would dangle.
 *
 * @param[in] join The producing join, with both logical children still intact
 * @param[in] evidence_for The domain-evidence source; see @ref base_table_cardinality_source
 * @return One cardinality per condition, in original planner condition order
 */
template <base_table_cardinality_source Source>
[[nodiscard]] std::vector<std::size_t> build_key_domain_cardinalities(
  duckdb::LogicalComparisonJoin const& join, Source const& evidence_for)
{
  auto const scans = detail::resolve_build_key_scans(join);
  std::vector<std::size_t> domains(scans.size(), 0);
  std::vector<std::pair<duckdb::LogicalGet const*, std::size_t>> memo;  // tiny n; beats a map
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
 * Yields `NodeStatistics::max_cardinality` for a native table scan -- committed rows plus
 * transaction-local inserts, a true upper bound -- and converts everything else, including a
 * throwing callback, to `std::nullopt`. Refusal is the only failure mode, extended from the walk
 * to the source. Deliberately never reads `LogicalGet::estimated_cardinality`, which the
 * join-order optimizer overwrites with a post-filter figure. Parquet scans are refused as declared
 * scope: without a catalog entry a Parquet build key cannot be proven unique by declared
 * constraint, so the gate could not arm on the evidence anyway.
 */
class duckdb_base_table_cardinality {
 public:
  explicit duckdb_base_table_cardinality(duckdb::ClientContext& context) noexcept;

  /**
   * @brief The scanned base table's unfiltered row upper bound, or `std::nullopt` when refused
   *
   * @param[in] get The resolved base scan
   * @return The row count, or `std::nullopt` for any scan that is not a DuckDB-native table scan
   */
  [[nodiscard]] std::optional<std::size_t> operator()(duckdb::LogicalGet const& get) const noexcept;

 private:
  duckdb::ClientContext* _context;
};

}  // namespace sirius::planner
