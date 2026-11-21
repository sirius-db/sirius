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

#include <cstddef>

namespace duckdb {
class Binder;
class ClientContext;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

// Rewrites of the delim CTEs DuckDB's binder produces under `delim_join_as_cte`, applied to
// DuckDB's own plan before its optimizer runs, so that join ordering sees their result. There is no
// plan copy to fall back on at that point, so each edit is decided and built in full first and
// then applied with moves that cannot fail: the plan is valid after every edit. The guarantee is
// per edit, not per call. A rewrite that throws while it looks for its next edit keeps the edits
// it has already made.

/// Answer a correlated `EXISTS` from a scalar aggregate subquery over the same rows.
///
///   CTE N { Projection <- Filter(mark) <- MARK join(X, S) ;
///           ... LEFT join-back(scan of N, Aggregate(GROUP BY key) <- S restricted to N's keys) }
///
/// The aggregate has a group for a key exactly when S has a row for it, which is what the `EXISTS`
/// tests. So the MARK join and its filter are removed, the mark column becomes the constant TRUE,
/// and the join-back becomes an INNER join. The subquery's tables are then joined once, not twice.
///
/// Declines unless the two sides are provably the same rows keyed the same way with the same NULL
/// handling, and unless X is selective and, at the most rows its joins can produce (the product
/// of its scans' rows), small next to S: without the `EXISTS` the CTE holds all of X. Returns the
/// number of `EXISTS` joins removed.
std::size_t answer_exists_from_aggregate(duckdb::ClientContext& context,
                                         duckdb::LogicalOperator& plan);

/// Restrict every table a delim subquery ties to its correlation column, not just the one the SQL
/// names.
///
///   Filter(x = y, ...) <- SEMI(L, scan of N's distinct keys) ON x = key
///
/// DuckDB attaches the outer rows' keys to `x` only and never derives that `y` is one of them too,
/// so `y`'s table stays unrestricted and is joined late. With `delim_join_as_cte = false` the keys
/// are an ordinary relation and DuckDB's filter combiner derives `y = key` itself. This adds
/// `SEMI(..., scan of the keys) ON y = key` for each such `y` of another relation. Every row the
/// filter keeps has `x = y`, so the added join removes only rows the filter removes anyway.
/// Returns the number of joins added.
std::size_t carry_subquery_keys_across_equalities(duckdb::ClientContext& context,
                                                  duckdb::Binder& binder,
                                                  duckdb::LogicalOperator& plan);

}  // namespace sirius::planner
