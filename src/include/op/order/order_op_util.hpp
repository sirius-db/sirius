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

#include "duckdb/common/vector.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"

#include <cudf/types.hpp>

#include <vector>

namespace sirius {
namespace op {

/**
 * @brief Translates a DuckDB order specification into cuDF sort vectors.
 *
 * Converts a vector of BoundOrderByNode into the three parallel vectors required
 * by cuDF sort/merge APIs: column indices, sort orders, and null precedences.
 *
 * @param orders        DuckDB order nodes; each must be a BOUND_REF expression.
 * @param operator_name Name of the calling operator, used in error messages.
 * @param order_key_idx Output: zero-based column indices of the sort keys.
 * @param column_order  Output: cudf::order per sort key.
 * @param null_precedence Output: cudf::null_order per sort key.
 */
void build_order_vectors(const duckdb::vector<duckdb::BoundOrderByNode>& orders,
                         const char* operator_name,
                         std::vector<int>& order_key_idx,
                         std::vector<cudf::order>& column_order,
                         std::vector<cudf::null_order>& null_precedence);

}  // namespace op
}  // namespace sirius
