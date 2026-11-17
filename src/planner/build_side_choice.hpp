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
#include <duckdb/planner/logical_operator.hpp>

#include <cstddef>
#include <cstdint>

namespace sirius::planner {

/// Makes the filtered side of an inner join its build when the unfiltered side is the larger one.
///
/// DuckDB chooses the build side by estimated bytes, and its row estimate for a filter it cannot
/// rate is a fifth of the table. TPC-H q9 at SF100000 then builds its last join on all of orders
/// (150 billion rows, held until the lineitem side arrives) and probes with the 33 billion rows
/// that survive the part filter. Swapped, orders is scanned once, behind the join's Bloom filter.
///
/// A join is swapped only when its build has no filter at all, its probe has one, DuckDB
/// estimates more rows for the build than for the probe, and the build's estimated rows times
/// the bytes of the columns it returns reach @p min_build_bytes. The build is the side a join
/// holds whole until its probe arrives, so a build that fits in GPU memory costs nothing to hold
/// and keeps the side DuckDB chose. The sides are compared by rows and not by bytes because
/// DuckDB's estimate for the filtered side is too high (7 times for q9 at SF100000), which is
/// what made it choose the unfiltered side.
///
/// @return The number of swapped joins.
std::size_t swap_unfiltered_builds(duckdb::ClientContext& context,
                                   duckdb::LogicalOperator& plan,
                                   std::uint64_t min_build_bytes);

}  // namespace sirius::planner
