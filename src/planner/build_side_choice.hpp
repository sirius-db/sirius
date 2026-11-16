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

namespace sirius::planner {

/// Makes the filtered side of an inner join its build when the unfiltered side is the larger one.
///
/// DuckDB chooses the build side by estimated bytes, and its row estimate for a filter it cannot
/// rate is a fifth of the table. TPC-H q9 at SF100000 then builds its last join on all of orders
/// (150 billion rows, held until the lineitem side arrives) and probes with the 33 billion rows
/// that survive the part filter. Swapped, orders is scanned once, behind the join's Bloom filter.
///
/// A join is swapped only when its build has no filter at all, its probe has one, and DuckDB
/// estimates more rows for the build than for the probe.
///
/// @return The number of swapped joins.
std::size_t swap_unfiltered_builds(duckdb::ClientContext& context, duckdb::LogicalOperator& plan);

}  // namespace sirius::planner
