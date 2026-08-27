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
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

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
 * Returns std::nullopt for ids that do not map to a single merge-able cuDF kind: `avg`
 * (decomposes into SUM + COUNT_VALID) and `first` (NTH_ELEMENT, handled by the caller).
 */
std::optional<cudf::aggregation::Kind> to_cudf_aggregation_kind(sirius::aggregate_id id);

/**
 * @brief Mapping from one original DuckDB aggregate expression to its position(s) in the expanded
 * cudf_aggregates vector. AVG is decomposed into SUM + COUNT_VALID (two slots), all others use one.
 * COUNT DISTINCT uses COLLECT_SET locally and MERGE_SETS during merge, then counts list elements.
 */
struct AggregateSlot {
  bool is_avg            = false;
  bool is_count_distinct = false;  ///< True if this is a COUNT(DISTINCT col) aggregate
  size_t cudf_idx;  ///< Index in cudf_aggregates. For AVG, this is the SUM slot; cudf_idx+1 is
                    ///< COUNT_VALID.
  cudf::data_type output_type{cudf::type_id::EMPTY};  ///< For AVG: the desired output cudf type
                                                      ///< (FLOAT64 or DECIMAL).
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
 * @brief Throw if a SUM over `input` could overflow its 64-bit accumulator.
 *
 * DuckDB models integer sums as HUGEINT, but cuDF has no INT128, so the plan generator
 * downcasts them to BIGINT (`downcast_hugeint_types`) and the reduction runs in a 64-bit
 * accumulator that wraps silently on overflow. This is the loud guard on that downcast path:
 * a cheap min/max pre-check that refuses any 64-bit integer sum whose
 * `valid_rows * max(|min|, |max|)` bound exceeds the accumulator range. Conservative — it can
 * refuse a sum that would have fit — but it never lets one wrap.
 *
 * No-op for other input types: narrower integers cannot overflow an int64 accumulator within
 * one column (2^31 rows * 2^31 max < 2^63), and float/decimal sums do not take this path.
 */
void throw_if_int64_sum_could_overflow(const cudf::column_view& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @brief True when a SUM over `values_type` is order-sensitive: a FLOAT32/FLOAT64 sum.
 *
 * Floating-point addition is not associative, so the result bits of a float sum depend on the
 * order in which values are combined. Integer and decimal sums accumulate exactly and are
 * order-independent; MIN/MAX/COUNT are order-independent for every type.
 */
bool is_order_sensitive_sum(cudf::aggregation::Kind kind, cudf::data_type values_type);

/**
 * @brief Gather `input` into a canonical row order so floating-point sums are bit-stable.
 *
 * Sorts rows by `sort_col_indices` (all ascending, nulls last), which must cover the group key
 * columns plus every float SUM value column. The resulting per-group value sequence — and
 * therefore the bits an atomics-free reduction produces — is then a pure function of the row
 * multiset, independent of upstream batch order and exchange arrival order. Distributed plans
 * that evaluate the same aggregation twice and compare the sums for exact equality (TPC-H q15:
 * `sum = max(sum)`) rely on this: without it, cuDF's hash groupby accumulates float sums via
 * atomicAdd and the two evaluations diverge by ULPs, silently emptying the join.
 */
std::unique_ptr<cudf::table> canonicalize_row_order(
  const cudf::table_view& input,
  const std::vector<cudf::size_type>& sort_col_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace op
}  // namespace sirius
