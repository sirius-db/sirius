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

#include <duckdb/planner/binder.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <cstddef>

namespace sirius::transparent {

/// Computes a projection's expression below the inner join under it when that narrows the rows.
///
/// For a projection directly over an INNER comparison join, an expression whose inputs all come
/// from one join child, are used by nothing else in the projection or the join, and are wider in
/// total than its result is moved into a projection under that child, and the inputs are dropped
/// there. The join and the exchange below it then carry the result instead of the inputs (TPC-H
/// q9's `l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity`: 16 bytes for 32).
///
/// DuckDB's PROJECTION_PLACEMENT refuses any expression that can throw, which is every arithmetic
/// function: below the join it would also run on rows the join drops. This moves only DECIMAL
/// `+`, `-` and `*`, which Sirius evaluates without an overflow check wherever they sit, and only
/// in Sirius's copy of the plan, never in the plan a CPU fallback runs.
///
/// @return The number of moved expressions.
std::size_t push_narrowing_expressions(duckdb::LogicalOperator& plan, duckdb::Binder& binder);

}  // namespace sirius::transparent
