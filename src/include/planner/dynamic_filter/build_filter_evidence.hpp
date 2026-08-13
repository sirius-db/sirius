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
 * Either `build_subtree_is_filtering` or `build_relation_is_opaque` can arm scan and join-edge
 * target discovery. Sirius owns both checks and does not consume DuckDB pushdown metadata:
 * `build_subtree_is_filtering` mirrors DuckDB's `JoinFilterPushdownOptimizer::IsFiltering`, while
 * `build_relation_is_opaque` covers build roots whose defining subtree is unavailable here.
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
 */
[[nodiscard]] bool build_subtree_is_filtering(duckdb::LogicalOperator const& op);

/**
 * @brief Reports whether a logical build root hides its defining subtree
 *
 * Returns true only for a `LOGICAL_DELIM_GET` or `LOGICAL_CTE_REF` root, optionally wrapped in one
 * or more valid single-child `LOGICAL_PROJECTION` operators. Other roots return false even when
 * they contain an opaque leaf below a non-projection operator.
 */
[[nodiscard]] bool build_relation_is_opaque(duckdb::LogicalOperator const& op);

}  // namespace sirius::planner
