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

#include <cstddef>
#include <memory>
#include <optional>
#include <variant>
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
 * into SUM + COUNT_VALID) and `first` (a carried column in the grouped aggregate, NTH_ELEMENT in
 * the ungrouped one).
 */
std::optional<cudf::aggregation::Kind> to_cudf_aggregation_kind(sirius::aggregate_id id);

/** @brief An aggregate answered by one cuDF request whose partial is also its output. */
struct plain_slot {
  std::size_t partial_idx{};  ///< Index in cudf_aggregates
};

/** @brief AVG: SUM at `sum_idx` and COUNT_VALID at `sum_idx + 1`, divided at the merge. */
struct avg_slot {
  std::size_t sum_idx{};                              ///< Index of the SUM in cudf_aggregates
  cudf::data_type output_type{cudf::type_id::EMPTY};  ///< Quotient type: FLOAT64 or DECIMAL
};

/** @brief COUNT(DISTINCT): COLLECT_SET locally and MERGE_SETS at the merge, then counted. */
struct count_distinct_slot {
  std::size_t partial_idx{};  ///< Index in cudf_aggregates
};

/** @brief FIRST: no cuDF request; child column `input_idx` is carried at `carried_idx`. */
struct first_slot {
  int input_idx{-1};          ///< Child column whose value this slot carries
  std::size_t carried_idx{};  ///< Position in the carried block that follows the partials
};

/**
 * @brief Mapping from one original DuckDB aggregate expression to its position(s) in the expanded
 * cudf_aggregates vector. AVG is decomposed into SUM + COUNT_VALID (two slots), FIRST uses none,
 * all others use one. COUNT DISTINCT uses COLLECT_SET locally and MERGE_SETS during merge, then
 * counts list elements.
 */
using AggregateSlot = std::variant<plain_slot, avg_slot, count_distinct_slot, first_slot>;

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
  /// True if any aggregate is FIRST. Presence only; each FIRST is a column of the carried block.
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
 * @brief Child column read by each FIRST slot, indexed by its `carried_idx`
 *
 * @param aggregate_slots One entry per aggregate expression
 * @return The columns a grouped aggregate carries after its partials, in carried-block order
 */
std::vector<int> carried_inputs(std::vector<AggregateSlot> const& aggregate_slots);

}  // namespace op
}  // namespace sirius
