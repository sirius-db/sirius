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

#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/binder.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <cstddef>
#include <cstdint>

namespace sirius::transparent {

/// The most rows DuckDB may estimate for the aggregate, and for the lookup scan, of a lookup join
/// that moves. The late join and the second aggregate read that many rows, and must stay
/// negligible next to the joins below the aggregate (TPC-H q9's aggregate is estimated at 57,142
/// rows at SF100000; q10's, which groups by customer, at 24 billion).
inline constexpr duckdb::idx_t max_late_join_rows = 1'000'000;

/// Joins a small lookup table after the aggregation when its columns are only GROUP BY keys.
///
/// Below an aggregate, through projections, filters and INNER comparison joins, an INNER join on
/// one equality of two plain columns whose right side is an unfiltered scan is a lookup join when
/// the scan's columns reach nothing above it but GROUP BY keys, unchanged. The aggregate then
/// groups by the join's foreign key in place of those columns, and the scan is joined to its
/// groups instead of to the rows below the large joins, which stop carrying the looked-up columns
/// (TPC-H q9's `n_name`: the aggregate groups by `s_nationkey` and `nation` is joined to 175
/// groups).
///
/// A second aggregate over the late join merges the groups again, because nothing says the
/// looked-up columns determine the key: two lookup rows with different keys can carry the same
/// name. It needs every aggregate to merge with itself, so only `sum`, `min` and `max` qualify. A
/// row whose foreign key is NULL or has no lookup row is dropped by the late join as the lookup
/// join dropped it.
///
/// The late join and the second aggregate are extra work, worth it only when the looked-up
/// columns are large against GPU memory, which is what makes a join below the aggregate spill. A
/// lookup join therefore moves only when the rows DuckDB estimates into the aggregate, times the
/// bytes of lookup columns each stops carrying (less the foreign key it starts carrying), reach
/// @p min_carried_bytes. At most one lookup join is moved per aggregate.
///
/// @return The number of lookup joins moved above their aggregate.
std::size_t join_dimensions_after_aggregate(duckdb::ClientContext& context,
                                            duckdb::unique_ptr<duckdb::LogicalOperator>& plan,
                                            duckdb::Binder& binder,
                                            std::uint64_t min_carried_bytes);

}  // namespace sirius::transparent
