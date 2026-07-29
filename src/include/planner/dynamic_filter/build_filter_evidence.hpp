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
 * @brief Sirius-owned evidence that a producing join's build subtree filters its rows
 *
 * `sirius_plan_comparison_join` gates the SCAN route on this evidence: an unfiltered build is (for
 * FK-shaped joins) the whole key domain, so its filter keeps every probe row by construction and a
 * scan target for it would only buy overhead. The evidence is computed from the logical build
 * child before `create_plan` moves data out of the logical nodes, alongside the domain and
 * uniqueness walks.
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

}  // namespace sirius::planner
