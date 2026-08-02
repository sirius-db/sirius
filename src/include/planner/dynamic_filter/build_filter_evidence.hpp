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
 * @brief Classifies logical join builds for dynamic-filter routing
 *
 * Either `build_subtree_is_filtering` or `build_relation_is_derived` can arm scan and join-edge
 * target discovery.
 */

#pragma once

namespace duckdb {
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

/**
 * @brief Reports whether @p op's subtree contains filter evidence
 *
 * Mirrors DuckDB's `JoinFilterPushdownOptimizer::IsFiltering` exactly: true for a `LOGICAL_GET`
 * with a non-empty `table_filters`, for a `LOGICAL_FILTER`, for a `LOGICAL_TOP_N`, or for any
 * subtree containing one of those; false otherwise.
 *
 * @param[in] op Root of the subtree to inspect
 * @return True when the subtree carries filter evidence
 */
[[nodiscard]] bool build_subtree_is_filtering(duckdb::LogicalOperator const& op);

/**
 * @brief Reports whether a logical relation is a projected derived leaf
 *
 * Returns true for a `LOGICAL_DELIM_GET` or `LOGICAL_CTE_REF` root, or for one reached through only
 * `LOGICAL_PROJECTION` wrappers. Any other operator stops the classification, even if one of its
 * descendants is a matching leaf.
 *
 * This is structural evidence only; it does not imply that the relation filters any rows.
 *
 * @param[in] op Root of the logical relation to classify
 * @return True if the relation is a derived leaf with only projection wrappers
 */
[[nodiscard]] bool build_relation_is_derived(duckdb::LogicalOperator const& op);

}  // namespace sirius::planner
