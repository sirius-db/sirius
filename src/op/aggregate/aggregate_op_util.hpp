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
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_set>
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

/// The type a SUM over a column of type @p type is computed in when it must not overflow the
/// input width: the next wider decimal type for DECIMAL32 and DECIMAL64, nullopt otherwise.
std::optional<cudf::data_type> widened_decimal_sum_type(cudf::data_type type);

/// The columns among @p candidates (indices into @p table; DECIMAL32 or DECIMAL64) whose SUM over
/// this batch could overflow the column's own storage width and so must be summed over a wider
/// decimal type.
///
/// A sum of n values of magnitude at most m is at most n * m, so rows * max|value| within the
/// storage width (2^31-1 or 2^63-1) rules out wrap-around in every group. The bound is exact. It
/// lets columns with small values, such as TPC-H's DECIMAL(15,2), skip the widening copy and the
/// 128-bit accumulation. Nulls are skipped, and a column with no valid value cannot overflow.
/// All cudf::minmax passes are launched first and their results come back in one device-to-host
/// copy, so a batch costs a single stream synchronization. Throws sirius::internal_exception if a
/// candidate is not DECIMAL32 or DECIMAL64.
std::unordered_set<int> decimal_sums_needing_widening(cudf::table_view const& table,
                                                      std::vector<int> const& candidates,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr);

/// |v| as an unsigned 64-bit magnitude; the most negative value's 2^63 fits.
constexpr std::uint64_t decimal_magnitude(std::int64_t v)
{
  auto const bits = static_cast<std::uint64_t>(v);
  return v < 0 ? std::uint64_t{0} - bits : bits;
}

/// Whether a SUM of @p num_rows values of magnitude at most @p max_abs can exceed the storage
/// width of @p type (2^31-1 for DECIMAL32, 2^63-1 for DECIMAL64). Throws
/// sirius::internal_exception for any other type.
bool decimal_sum_may_overflow(cudf::data_type type,
                              cudf::size_type num_rows,
                              std::uint64_t max_abs);

/// A decimal SUM input of one batch: its column and, when the planner proved one from base-table
/// statistics (planner::resolve_decimal_sum_input_bounds), the largest |unscaled value| it holds.
struct decimal_sum_candidate {
  int column;
  std::optional<std::uint64_t> max_abs;
};

/// The columns among @p candidates (DECIMAL32 or DECIMAL64) whose SUM over this batch could
/// overflow their storage width. A column whose every candidate carries a bound is decided on the
/// host from the batch row count; the others are measured with decimal_sums_needing_widening. A
/// batch without unproven columns therefore costs no device work and no synchronization.
std::unordered_set<int> decimal_sums_to_widen(cudf::table_view const& table,
                                              std::vector<decimal_sum_candidate> const& candidates,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr);

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

}  // namespace op
}  // namespace sirius
