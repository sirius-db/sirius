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
#include "expression/aggregate_id.hpp"
#include "expression/ast/node.hpp"

#include <cudf/aggregation.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace sirius {
namespace op {

/**
 * @brief Map a simple Sirius aggregate_id to a single cuDF aggregation kind.
 *
 * Pure aggregate_id -> Kind mapping over the closed aggregate_id enum. DISTINCT is NOT an
 * aggregate_id: COUNT(DISTINCT ...) is the `count` id with the distinct() modifier set, and is
 * intercepted by the caller (COLLECT_SET path) before this helper runs — so there is no
 * count_distinct case to add here.
 *
 * Returns std::nullopt for ids that do not map to a single merge-able cuDF kind: `avg` (decomposes
 * into SUM + COUNT_VALID) and `first` (a whole-row distinct in the grouped aggregate, NTH_ELEMENT
 * in the ungrouped one).
 */
std::optional<cudf::aggregation::Kind> to_cudf_aggregation_kind(sirius::aggregate_id id);

/**
 * @brief Mapping from one original DuckDB aggregate expression to its position(s) in the expanded
 * cudf_aggregates vector. AVG is decomposed into SUM + COUNT_VALID (two slots), FIRST uses none,
 * all others use one. COUNT DISTINCT uses COLLECT_SET locally and MERGE_SETS during merge, then
 * counts list elements.
 */
struct AggregateSlot {
  bool is_avg            = false;
  bool is_count_distinct = false;  ///< True if this is a COUNT(DISTINCT col) aggregate
  bool is_first          = false;  ///< True if this is a FIRST, which emits no cuDF aggregation
  size_t cudf_idx;  ///< Index in cudf_aggregates. For AVG, this is the SUM slot; cudf_idx+1 is
                    ///< COUNT_VALID. Unused, and 0, for FIRST.
  cudf::data_type output_type{cudf::type_id::EMPTY};  ///< For AVG: the desired output cudf type
                                                      ///< (FLOAT64 or DECIMAL).
  int first_input_idx = -1;  ///< For FIRST: the child column whose value this slot carries
};

/**
 * @brief Result of converting DuckDB aggregate expressions to cuDF compute definitions.
 */
struct CudfAggregateDefinitions {
  std::vector<int> group_idx;                            ///< Column indices for GROUP BY keys
  std::vector<cudf::aggregation::Kind> cudf_aggregates;  ///< cuDF aggregation types (expanded: 2
                                                         ///< entries per AVG)
  std::vector<int> cudf_aggregate_idx;  ///< Column indices for aggregation inputs (expanded)

  /// For COLLECT_SET aggregates only: when non-empty, the aggregate input is a struct column
  /// synthesized from these column indices (multi-column COUNT DISTINCT). Parallel to
  /// cudf_aggregates; empty entries mean single-column (use cudf_aggregate_idx directly).
  std::vector<std::vector<int>> cudf_aggregate_struct_col_indices;

  /// One entry per original DuckDB aggregate expression, mapping to cudf_aggregates positions.
  std::vector<AggregateSlot> aggregate_slots;
  bool has_avg            = false;  ///< True if any aggregate is AVG
  bool has_count_distinct = false;  ///< True if any aggregate is COUNT(DISTINCT col)
  /// True if any aggregate is FIRST. Presence only: whole_row_distinct_select() decides whether
  /// cudf::distinct can run the list, so a second FIRST implementation can sit beside it.
  bool has_first = false;
};

/**
 * @brief Convert DuckDB aggregate expressions to cuDF compute definitions.
 *
 * This function extracts:
 * 1. GROUP BY column indices from group expressions
 * 2. Aggregation types (SUM, COUNT, MIN, MAX, etc.) from aggregate expressions
 * 3. Input column indices for each aggregate from the aggregate children
 *
 * @param groups_p DuckDB GROUP BY expressions (BoundReferenceExpression)
 * @param expressions DuckDB aggregate expressions (BoundAggregateExpression)
 * @return CudfAggregateDefinitions containing the extracted information
 * @throws std::runtime_error if an unsupported aggregate function is encountered
 */
CudfAggregateDefinitions convert_duckdb_aggregates_to_cudf(
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& groups_p,
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& expressions);

/**
 * @brief Child columns a whole-row distinct keeps, in output order
 *
 * A grouped aggregate is a whole-row distinct when every slot is a FIRST and `group_idx` together
 * with the slots' `first_input_idx` values name each of `0 .. output_width - 1` once.
 * `cudf::distinct` on the `group_idx` keys then computes the whole operator: it keeps one row per
 * group, and the returned list selects `group_idx` followed by each slot's `first_input_idx` out of
 * that row. `sirius_physical_grouped_aggregate` and `sirius_physical_grouped_aggregate_merge` both
 * route on this function, so they cannot disagree about the shape.
 *
 * An empty `aggregate_slots` returns nullopt, which keeps a zero-aggregate `SELECT DISTINCT` on the
 * ordinary groupby path.
 *
 * @param group_idx Child column read by each group key
 * @param aggregate_slots One entry per aggregate expression
 * @param output_width Number of output columns the operator declares
 * @return Child column for each output column, or nullopt when the request list is not a whole-row
 * distinct
 */
std::optional<std::vector<int>> whole_row_distinct_select(
  std::vector<int> const& group_idx,
  std::vector<AggregateSlot> const& aggregate_slots,
  std::size_t output_width);

}  // namespace op
}  // namespace sirius
