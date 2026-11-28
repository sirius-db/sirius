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

#include "duckdb/common/unique_ptr.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace duckdb {
class Binder;
class ClientContext;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

/// The thresholds `push_partial_aggregates_below_joins` judges a push by.
struct partial_aggregate_pushdown_options {
  /// The share of its unfiltered rows a scan must be estimated to keep for a join against it to
  /// count as keeping the rows of its other side. 0 accepts any other side.
  double min_kept_share = 0.7;
  /// The columns the pushed aggregate may group by beyond those it is joined on.
  std::uint64_t max_extra_keys = 1;
  /// The rows each group of the pushed aggregate must be estimated to combine, at least, for it to
  /// be pushed where the aggregate above does not group by the same columns. The estimate is the
  /// input's row estimate over the most groups its keys can form, from the scan statistics. 0 never
  /// trusts the estimate, so only the grouping decides.
  double min_rows_per_key = 8;
};

/// What `push_partial_aggregates_below_joins` did to a plan.
struct partial_aggregate_pushdown_result {
  std::size_t pushed = 0;  ///< The aggregates pushed
  /// One entry per aggregate over a join that was left alone: every reason found, separated by
  /// "; ". The same text is logged at debug level.
  std::vector<std::string> declined;
};

/// Aggregate before a join that only adds group-by payload. For `AGGREGATE <- JOIN ... <- S`,
/// where every aggregate argument comes from the subtree S and a join above S brings in
/// variable-width columns the aggregate groups by, insert `AGGREGATE(GROUP BY the S columns used
/// above S)` directly above S and make the original aggregate combine its results. The join then
/// repeats the payload once per group of S instead of once per row (TPC-H q10 joins `customer`'s
/// strings to one row per customer instead of one per lineitem).
///
/// Rows of S that agree on every column used above S pair with the same rows of every join above,
/// so combining them first changes no result, whether or not any join key is unique. Limited to
/// SUM over an integer or a DECIMAL of up to 64 bits or over a floating-point value, MIN and MAX,
/// whose combined result has the original type, so nothing above the aggregate changes.
///
/// Pushed only where that pays: where the aggregate above groups by the pushed aggregate's keys,
/// or by columns that fix them (one an inner join equates with a key, or one proven unique in the
/// joined table that holds such a column), or where the scan statistics show at least
/// `min_rows_per_key` rows per key. Never where a key is proven unique, since such an aggregate
/// would combine nothing.
///
/// An aggregate over a join that is left alone reports why. The reasons are short fixed names
/// (`aggregate:count_star`, `other-side:filtered`, ...), and every independent one is listed, so a
/// plan's report tells which limits would all have to go for the rule to apply.
///
/// Works on the optimized, unresolved plan.
partial_aggregate_pushdown_result push_partial_aggregates_below_joins(
  duckdb::ClientContext& context,
  duckdb::Binder& binder,
  duckdb::unique_ptr<duckdb::LogicalOperator>& plan,
  partial_aggregate_pushdown_options const& options = {});

}  // namespace sirius::planner
