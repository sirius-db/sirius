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
 * @file scan_reuse_rewrites.hpp
 * @brief Logical-plan rewrites for queries that read the same rows twice, applied to Sirius's copy
 * of DuckDB's optimized plan before physical planning (docs/plans/sf10k-scan-elimination.md).
 *
 * Each rule checks the conditions under which its rewrite returns the same rows, and leaves the
 * plan alone otherwise. They hold on one GPU and distributed alike.
 */

#pragma once

#include <duckdb/common/unique_ptr.hpp>

#include <cstddef>
#include <functional>

namespace duckdb {
class ClientContext;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

/// How many times each rule rewrote a plan.
struct scan_reuse_counts {
  /// An IN-subquery's grouped aggregate reused for the outer query's SUM over the same table
  /// (TPC-H Q18).
  std::size_t semi_join_aggregates = 0;
  /// A correlated MIN/MAX/AVG subquery that read the outer query's rows again (TPC-H Q17, Q2).
  std::size_t correlated_aggregates = 0;

  [[nodiscard]] std::size_t total() const noexcept
  {
    return semi_join_aggregates + correlated_aggregates;
  }
};

/// Called with each subtree a rewrite drops, before it goes: DuckDB's own plan still scans what it
/// reads, which read-view validation must account for (read_view_registry::record_skipped_scans).
using removed_subtree_callback = std::function<void(duckdb::LogicalOperator const&)>;

/**
 * @brief Rewrites @p plan in place wherever a rule applies.
 *
 * Expects bindings, not resolved references: call before ColumnBindingResolver.
 *
 * Reuse a SEMI join's grouped aggregate (TPC-H Q18). Shape:
 *
 *     AGGREGATE[G; SUM(L.c)...] ← INNER JOIN L.k = o (GET L, Y)
 *
 * where Y holds, through INNER joins and filters, `SEMI JOIN o = x (·, B)` and B is, through
 * projections and filters, `AGGREGATE[x = T.k; ..., SUM(T.c), ...]` over a plain scan T of the same
 * table as L with the same filters. Rewritten: the SEMI join becomes INNER, carrying the subquery's
 * SUM up, the join with L goes, and each SUM(L.c) becomes SUM(SUM(T.c)). Valid because the
 * subquery's keys are unique (one group per key), every Y row joined all L rows of its key, and
 * SUM(T.c) over a key is their sum. G and the aggregates may reference no other column of L.
 *
 * Read the outer rows again from a CTE in a correlated MIN/MAX/AVG subquery (TPC-H Q17, Q2).
 * Shape: a DELIM JOIN whose outer side is an inner-join region O (scans, INNER equality joins,
 * filters, projections of references) and whose subquery is projections and filters over one
 * aggregate of MIN, MAX and AVG over a region R ⋈ DELIM_GET. When O = R ⋈ E with E joined to R only
 * on the delim columns (O's scans are R's plus E's; R's equalities and filters are O's on R's
 * scans; no filter reads both), O's rows with delim key k are R's rows with that key, each repeated
 * the number of E rows with key k, and those aggregates do not change. Rewritten: O is materialized
 * once as a CTE, the DELIM JOIN reads it, and the subquery's R becomes the CTE (still joined with
 * the DELIM_GET).
 */
scan_reuse_counts rewrite_repeated_scans(duckdb::ClientContext& context,
                                         duckdb::unique_ptr<duckdb::LogicalOperator>& plan,
                                         removed_subtree_callback const& removed = {});

}  // namespace sirius::planner
