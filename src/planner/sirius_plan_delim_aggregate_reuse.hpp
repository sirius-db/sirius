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

namespace duckdb {
class Binder;
class ClientContext;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

/// Compute each decorrelated subquery aggregate (`MIN`, `MAX` or `AVG` per correlation key) from
/// the outer rows its delim join already materializes, instead of from a second scan of the
/// subquery's tables, when the subquery's rows are provably the outer rows of each key. Works on
/// the optimized, unresolved plan in either of DuckDB's delim-join representations: a
/// `DELIM_JOIN` becomes a materialized CTE of its outer rows read twice, and an aggregate under an
/// existing delim CTE reads that CTE. Returns the number of aggregates rewritten.
std::size_t reuse_delim_rows_for_subquery_aggregates(
  duckdb::ClientContext& context,
  duckdb::Binder& binder,
  duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

}  // namespace sirius::planner
