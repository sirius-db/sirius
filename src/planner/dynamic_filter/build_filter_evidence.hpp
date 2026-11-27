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
 * Traces the probe key of every plain equality that key admission and the membership filters would
 * accept (a bare probe column, no cast on either side, a supported key type that the probe column
 * can be compared with), on a join type that binds scans, with `origin_policy::filter_route`.
 *
 * The answer is a prediction made before the probe side is planned. The join may still turn out
 * not to be a hash join, the physical plan may lack the route, and at run time the filter may not
 * be published or may remove nothing. A join below that relies on a predicted filter is therefore
 * checked twice: the generator counts a scan to which no join bound a filter
 * (`expected_targets_unbound`), and the relying join publishes nothing from a build that arrives
 * unreduced (`admitted_key::expected_reduced_scan_rows`, `unreduced_build_gate_fires`).
 *
 * Null-safe equalities are left out because their route depends on the admitted key, and
 * distributed plans because a rank holds only its share of a build.
 *
 * @pre Types and column bindings are resolved on @p join.
 */
[[nodiscard]] std::vector<duckdb::LogicalGet const*> expected_probe_targets(
  duckdb::LogicalComparisonJoin const& join, sirius::operator_params const& params);

/**
 * @brief The scan of @p expected whose rows the build side @p build consists of, or null.
 *
 * The build must be that scan behind projections and filters only, so that it holds exactly the
 * rows the scan emits. A join above expects to filter the scan at run time, which
 * reduces the build even when it carries no filter of its own. Anything else between the two
 * (an aggregate, DISTINCT, a limit, a join) would make the build's size say nothing about whether
 * the expected filter arrived, so it is not counted.
 *
 * Under demand-driven scheduling the build side is normally requested after the join above has
 * finished its own build, but nothing holds the scan back, and the expected filter may remove
 * nothing. The caller therefore records the scan's row count in the publish plan
 * (`admitted_key::expected_reduced_scan_rows`), and the join publishes no filter from a build that
 * arrives about that large (`unreduced_build_gate_fires`).
 */
[[nodiscard]] duckdb::LogicalGet const* expected_target_behind_build(
  duckdb::LogicalOperator const& build, std::span<duckdb::LogicalGet const* const> expected);

}  // namespace sirius::planner
