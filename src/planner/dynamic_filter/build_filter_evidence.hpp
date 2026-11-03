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

namespace duckdb {
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

/**
 * @brief Mirrors DuckDB's join-filter test for GET filters, FILTER, TOP_N, or a containing subtree
 */
[[nodiscard]] bool build_subtree_is_filtering(duckdb::LogicalOperator const& op);

/**
 * @brief Tests for a DELIM_GET or CTE_REF behind projections or ordinary deduplication.
 *
 * Deduplication has nonempty groups, no aggregate outputs or grouping functions, and one
 * complete grouping set (explicit or implicit). Each wrapper has exactly one non-null child.
 * This admits build evidence only; it does not resolve CTE producers or move filter targets.
 */
[[nodiscard]] bool build_relation_is_opaque(duckdb::LogicalOperator const& op);

}  // namespace sirius::planner
