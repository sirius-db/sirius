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

#include <span>
#include <vector>

namespace duckdb {
class LogicalComparisonJoin;
class LogicalGet;
class LogicalOperator;
}  // namespace duckdb

namespace sirius {
struct operator_params;
}  // namespace sirius

namespace sirius::planner {

/**
 * @brief Mirrors DuckDB's join-filter test for GET filters, FILTER, TOP_N, or a containing subtree
 */
[[nodiscard]] bool build_subtree_is_filtering(duckdb::LogicalOperator const& op);

/**
 * @brief Tests for a CTE_REF behind projections or ordinary deduplication.
 *
 * Deduplication has nonempty groups, no aggregate outputs or grouping functions, and one
 * complete grouping set (explicit or implicit). Each wrapper has exactly one non-null child.
 * This admits build evidence only; it does not resolve CTE producers or move filter targets.
 */
[[nodiscard]] bool build_relation_is_opaque(duckdb::LogicalOperator const& op);

/**
 * @brief Tests for a SEMI or ANTI comparison join (either side variant) in the subtree.
 *
 * Such a join only removes rows, so it reduces the build like a filter does. DuckDB's join-filter
 * test does not count it; without this, a table reduced only by a join (the shape
 * `delim_join_as_cte` gives a correlated subquery's domain) publishes no filter.
 */
[[nodiscard]] bool build_subtree_has_filtering_join(duckdb::LogicalOperator const& op);

/**
 * @brief Whether a RIGHT_SEMI or RIGHT_ANTI join can publish dynamic filters under @p params.
 *
 * Such a join sizes its partitions from the whole probe input, so its filters must be out before
 * the probe is read. That takes dynamic filters, Bloom accumulation and a non-zero Bloom budget.
 */
[[nodiscard]] bool right_family_publication_enabled(sirius::operator_params const& params);

/**
 * @brief The base scans a dynamic filter from @p join is expected to be bound to.
 *
 * Traces the probe side of every plain equality whose probe key is a bare reference, on a join
 * type that binds scans, with `origin_policy::filter_route`. The answer is a prediction made
 * before the probe side is planned: key admission or the physical plan can still leave a scan
 * unbound. Null-safe equalities are left out because their route depends on the admitted key.
 *
 * @pre Types are resolved on @p join.
 */
[[nodiscard]] std::vector<duckdb::LogicalGet const*> expected_probe_targets(
  duckdb::LogicalComparisonJoin const& join, sirius::operator_params const& params);

/**
 * @brief The scans of @p expected found anywhere in the subtree @p op.
 *
 * A join above expects to filter such a scan at run time, so a build side that contains one is
 * reduced even when it carries no filter of its own. Under demand-driven scheduling that build
 * side is normally requested after the join above has finished its own build. Nothing holds the
 * scan back: one that starts early is read unfiltered, and the join it feeds then publishes a
 * filter that removes less, which the run-time gates can decline or switch off.
 */
[[nodiscard]] std::vector<duckdb::LogicalGet const*> expected_targets_in_build(
  duckdb::LogicalOperator const& op, std::span<duckdb::LogicalGet const* const> expected);

}  // namespace sirius::planner
