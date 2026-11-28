/*
 * Copyright 2025, Sirius Contributors.
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

#include <duckdb/planner/binder.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <cstddef>

namespace sirius::transparent {

/// Shrinks the columns an aggregate only counts across an outer join.
///
/// For an aggregate directly over a LEFT or RIGHT comparison join, a COUNT(col) argument that
/// comes from the null-extended side and is referenced nowhere else in the join or the aggregate
/// is replaced, below the join, by the BOOLEAN `CASE WHEN col IS NOT NULL THEN TRUE END`. The
/// flag is NULL exactly where col is, so the count is unchanged, while the join and any exchange
/// below it carry one byte per row instead of col's width (TPC-H q13's `count(o_orderkey)`).
///
/// @return The number of narrowed columns.
std::size_t narrow_outer_join_counts(duckdb::LogicalOperator& plan, duckdb::Binder& binder);

}  // namespace sirius::transparent
