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

/// Aggregate before a join that only adds group-by payload. For `AGGREGATE <- JOIN ... <- S`,
/// where every aggregate argument comes from the subtree S and an inner join above S brings in
/// variable-width columns the aggregate groups by, insert `AGGREGATE(GROUP BY the S columns used
/// above S)` directly above S and make the original aggregate combine its results. The join then
/// repeats the payload once per group of S instead of once per row (TPC-H q10 joins `customer`'s
/// strings to one row per customer instead of one per lineitem).
///
/// Rows of S that agree on every column used above S pair with the same rows of every join above,
/// so combining them first changes no result, whether or not any join key is unique. Limited to
/// SUM over a DECIMAL stored in at most 64 bits, MIN and MAX, whose combined result has the
/// original type, so nothing above the aggregate changes.
///
/// Works on the optimized, unresolved plan. Returns the number of aggregates pushed.
std::size_t push_partial_aggregates_below_joins(duckdb::ClientContext& context,
                                                duckdb::Binder& binder,
                                                duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

}  // namespace sirius::planner
