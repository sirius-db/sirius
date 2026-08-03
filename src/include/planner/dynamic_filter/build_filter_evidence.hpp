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
 * @brief Reports whether a logical relation is derived rather than a base-table image
 *
 * Returns true when the subtree contains a derivation marker anywhere below the root, recursing
 * through every other operator. A derivation marker is either a childless derived leaf
 * (`LOGICAL_DELIM_GET`, `LOGICAL_CTE_REF`) or a reducing operator (`LOGICAL_COMPARISON_JOIN`,
 * `LOGICAL_ANY_JOIN`, `LOGICAL_DELIM_JOIN`, `LOGICAL_AGGREGATE_AND_GROUP_BY`, `LOGICAL_DISTINCT`,
 * `LOGICAL_INTERSECT`, `LOGICAL_EXCEPT`). `LOGICAL_UNION` is deliberately not a marker, because a
 * union does not reduce its inputs' key sets.
 *
 * This is structural evidence only; it does not imply that the relation filters any rows. The known
 * false-positive class is the cardinality-preserving enrichment join: a build that joins a table to
 * a dimension on that dimension's key is structurally derived, yet emits every key its larger input
 * held. Two gates downstream contain it. `publish_dynamic_filters` applies the build-key
 * domain-coverage gate, which skips a key whose build covers most of the traced base table's domain
 * before any membership structure is built; the trace descends through joins to the underlying
 * `LOGICAL_GET`, so an enrichment join's key still resolves to its base domain.
 * `dynamic_filter_gate` then measures what an applied filter actually keeps and retires one that
 * prunes too little.
 *
 * @param[in] op Root of the logical relation to classify
 * @return True when the subtree contains a derivation marker
 */
[[nodiscard]] bool build_relation_is_derived(duckdb::LogicalOperator const& op);

}  // namespace sirius::planner
