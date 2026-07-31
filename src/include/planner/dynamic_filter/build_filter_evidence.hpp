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
 * @file build_filter_evidence.hpp
 * @brief Sirius-owned structural evidence about a producing join's build side
 *
 * The evidence is computed by `sirius_plan_comparison_join` from the logical build child before
 * `create_plan` moves data out of the logical nodes, alongside the domain and uniqueness walks. Two
 * independent predicates: `build_subtree_is_filtering`, a byte-faithful mirror of DuckDB's
 * `JoinFilterPushdownOptimizer::IsFiltering`, gates the scan route and contributes to the join-edge
 * route; `build_relation_is_derived` widens the join-edge route only.
 */

#pragma once

namespace duckdb {
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

/**
 * @brief Whether @p op's subtree contains an operator that removes or reorders-and-truncates rows
 *
 * Mirrors DuckDB's `JoinFilterPushdownOptimizer::IsFiltering` exactly: true for a `LOGICAL_GET`
 * with a non-empty `table_filters`, for a `LOGICAL_FILTER`, for a `LOGICAL_TOP_N`, or for any
 * subtree containing one of those; false otherwise. Sirius plans over the same optimized logical
 * tree DuckDB's optimizer annotated, so this walk computes the same value DuckDB's
 * `build_side_has_filter` hint carried.
 *
 * @param[in] op Root of the subtree to inspect
 * @return True when the subtree carries filter evidence
 */
[[nodiscard]] bool build_subtree_is_filtering(duckdb::LogicalOperator const& op);

/**
 * @brief Whether @p op presents, through projections, a derived leaf relation
 *
 * True exactly when @p op is a `LOGICAL_DELIM_GET` or `LOGICAL_CTE_REF`, or a chain of
 * `LOGICAL_PROJECTION`s over one. Such a build is opaque to `build_subtree_is_filtering`: the
 * mirror bottoms out at the childless reference, so its false verdict there means "evidence
 * unavailable", not "whole key domain" -- the delim scan is a duplicate-eliminated correlation
 * domain and the CTE reference a materialized subplan, both computed derivations rather than
 * base-table images. The join-edge route treats that opacity as permission to wire; the scan route
 * (mirroring DuckDB) treats it as denial.
 *
 * Root-down on purpose: containing a derived leaf under visible unfiltered structure (a join, an
 * aggregate) does not make the presented relation derived, and admitting such shapes re-wires
 * measured losers (TPC-H q15). This is a plausibility heuristic, not a selectivity proof: a derived
 * relation can still span the probe key's domain (an unfiltered correlation domain, a bare-copy
 * MATERIALIZED CTE). A wrong true costs bounded apply overhead -- the consumer-side keep-ratio gate
 * and per-filter permanent skip contain it, and the producing join stays authoritative -- never
 * correctness.
 *
 * @param[in] op Root of the producing join's logical build child
 * @return True when the build presents a derived leaf relation
 */
[[nodiscard]] bool build_relation_is_derived(duckdb::LogicalOperator const& op);

}  // namespace sirius::planner
