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

#include <duckdb/common/unique_ptr.hpp>

#include <vector>

namespace duckdb {
class ClientContext;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

/// @brief Rewrites Sirius's copy of DuckDB's optimized plan to reuse a grouped SUM from an IN subquery.
/// 
/// The SQL below illustrates the equivalent plan shapes:
/// @code{.sql}
/// -- BEFORE: scan lineitem in both the outer query and the IN subquery.
/// SELECT o.o_orderkey, SUM(l.l_quantity) AS quantity
/// FROM orders o
/// JOIN lineitem l ON l.l_orderkey = o.o_orderkey
/// WHERE o.o_orderkey IN (
///   SELECT l_orderkey
///   FROM lineitem
///   GROUP BY l_orderkey
///   HAVING SUM(l_quantity) > 300
/// )
/// GROUP BY o.o_orderkey;
///
/// -- AFTER: carry the subquery's per-order subtotal into the outer SUM.
/// SELECT o.o_orderkey, SUM(q.quantity) AS quantity
/// FROM orders o
/// JOIN (
///   SELECT l_orderkey, SUM(l_quantity) AS quantity
///   FROM lineitem
///   GROUP BY l_orderkey
///   HAVING SUM(l_quantity) > 300
/// ) q ON q.l_orderkey = o.o_orderkey
/// GROUP BY o.o_orderkey;
/// @endcode
///
/// The subquery emits one row per qualifying key. The inner join keeps the same orders as the IN
/// filter and supplies the subtotal for the same lineitem rows. Reuse requires matching scan rows
/// and a SUM argument type accepted by `sum_combines_exactly`.
///
/// Call after `convert_mark_joins` and before `ColumnBindingResolver`. The rewrite checks the full
/// match before editing the plan. If it throws, discard the Sirius plan and fall back to DuckDB's
/// original plan.
///
/// @return The subtrees the rewrites removed (each a `LogicalGet`), now owned by the caller.
///         DuckDB's own plan still reads them, which read-view validation must be told about
///         (`read_view_registry::record_skipped_scans`). Empty when nothing was rewritten.
[[nodiscard]] std::vector<duckdb::unique_ptr<duckdb::LogicalOperator>> rewrite_repeated_scans(
  duckdb::ClientContext& context, duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

}  // namespace sirius::planner
