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

#include "duckdb/common/vector.hpp"
#include "duckdb/planner/expression.hpp"

#include <cudf/types.hpp>

#include <vector>

namespace sirius {
namespace op {

/// Phase 1 ranking window functions. Each maps to a cuDF rank_method:
/// ROW_NUMBER -> FIRST (1..n, no ties), RANK -> MIN (ties skip), DENSE_RANK -> DENSE (no skip).
enum class window_rank_kind { ROW_NUMBER, RANK, DENSE_RANK };

/// cuDF compute definitions extracted from one LogicalWindow's expressions.
///
/// All expressions in a single LogicalWindow handled by sirius_physical_window share the same
/// PARTITION BY / ORDER BY (guaranteed by the Phase 1 guard in create_plan), so the partition and
/// order keys are recorded once. One window_rank_kind is recorded per window expression, in the
/// same order they appear in LogicalWindow.expressions (i.e. trailing output columns).
struct window_definitions {
  std::vector<int> partition_idx;            ///< PARTITION BY column indices (into child output)
  std::vector<int> order_idx;                ///< ORDER BY column indices (into child output)
  std::vector<cudf::order> order_dirs;       ///< per ORDER BY key: ASCENDING / DESCENDING
  std::vector<cudf::null_order> order_null;  ///< per ORDER BY key: BEFORE / AFTER
  std::vector<window_rank_kind> ranks;       ///< one per window expression, in output order
};

/// Extract cuDF compute definitions from the (raw DuckDB) window expressions of one LogicalWindow.
///
/// Each element must be a BoundWindowExpression of a Phase 1 ranking type with PARTITION BY / ORDER
/// BY keys that are BoundReferenceExpressions; create_plan(LogicalWindow&) guards these conditions
/// before constructing the operator, so this routine only handles the validated shape.
window_definitions convert_duckdb_window_to_cudf(
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& window_exprs);

}  // namespace op
}  // namespace sirius
