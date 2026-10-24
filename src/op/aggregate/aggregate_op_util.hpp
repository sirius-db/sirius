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
#include "helper/logical_type.hpp"
#include "op/aggregate/wide_integer_sum.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_set>
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
 * STDDEV_SAMP maps to Kind::M2 as an internal marker for its count/mean/M2 STRUCT.
 * The local executor expands the marker into three cuDF aggregations and packs their results.
 *
 * Returns std::nullopt for ids that do not map to a single merge-able cuDF kind: `avg` (decomposes
 * into SUM + COUNT_VALID) and `first` (a carried column in the grouped aggregate, NTH_ELEMENT in
 * the ungrouped one).
 */
std::optional<cudf::aggregation::Kind> to_cudf_aggregation_kind(sirius::aggregate_id id);

/// Whether @p id over an input of type @p input_type is a SUM that DuckDB accumulates in HUGEINT
/// (128 bits): `sum` or `avg` over INTEGER, BIGINT or HUGEINT. Its total can exceed INT64 even
/// when every input fits, so the GPU must not accumulate it in INT64 unchecked. `sum_no_overflow`
/// is excluded: DuckDB's statistics propagation already proved that the total fits in INT64.
/// Narrower inputs (TINYINT, SMALLINT, BOOLEAN) are excluded because DuckDB itself sums them in
/// INT64. On the GPU a HUGEINT input is carried as INT64, like the BIGINT result of such a SUM.
bool is_wide_integer_sum(sirius::aggregate_id id, sirius::logical_type const& input_type);

/// The type a SUM over a column of type @p type is computed in when it must not overflow the
/// input width: the next wider decimal type for DECIMAL32 and DECIMAL64, nullopt otherwise.
std::optional<cudf::data_type> widened_decimal_sum_type(cudf::data_type type);

/// The type a SUM candidate of type @p type (see sums_to_widen) is widened to: the next wider
/// decimal type for DECIMAL32 and DECIMAL64, wide_integer_sum_type() for INT64, nullopt otherwise.
std::optional<cudf::data_type> widened_sum_type(cudf::data_type type);

/// The columns among @p candidates (indices into @p table; DECIMAL32, DECIMAL64 or INT64) whose
/// SUM over this batch could overflow the column's own storage width and so must be summed over a
/// wider type (widened_sum_type).
///
/// A sum of n values of magnitude at most m is at most n * m, so rows * max|value| within the
/// storage width (2^31-1 or 2^63-1) rules out wrap-around in every group. The bound is exact. It
/// lets columns with small values, such as TPC-H's DECIMAL(15,2), skip the widening copy and the
/// 128-bit accumulation. Nulls are skipped, and a column with no valid value cannot overflow.
/// All cudf::minmax passes are launched first and their results come back in one device-to-host
/// copy, so a batch costs a single stream synchronization. Throws sirius::internal_exception if a
/// candidate is not DECIMAL32, DECIMAL64 or INT64.
std::unordered_set<int> sums_needing_widening(cudf::table_view const& table,
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
/// width of @p type (2^31-1 for DECIMAL32, 2^63-1 for DECIMAL64 and INT64). Throws
/// sirius::internal_exception for any other type.
bool sum_may_overflow(cudf::data_type type, cudf::size_type num_rows, std::uint64_t max_abs);

/// A SUM input of one batch that may need widening: its column and, when the planner proved one
/// from base-table statistics (planner::resolve_sum_input_bounds), the largest |unscaled value| it
/// holds.
struct sum_candidate {
  int column;
  std::optional<std::uint64_t> max_abs;
};

/// The columns among @p candidates (DECIMAL32, DECIMAL64 or INT64) whose SUM over this batch could
/// overflow their storage width. A column whose every candidate carries a bound is decided on the
/// host from the batch row count; the others are measured with sums_needing_widening. A batch
/// without unproven columns therefore costs no device work and no synchronization.
std::unordered_set<int> sums_to_widen(cudf::table_view const& table,
                                      std::vector<sum_candidate> const& candidates,
                                      ::cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/// Whether summing @p column (INT64 partial sums of a wide integer SUM, nulls skipped) can
/// overflow INT64 in any group: true unless sum(|value|) over the whole column provably stays
/// below 2^62. Any group's sum, and every running sum on the way, is bounded by that total, so a
/// merge over a column for which this returns false is exact in INT64. The total is accumulated in
/// double, whose relative error is far inside the factor-two margin. Synchronizes @p stream.
bool partial_sums_may_overflow(cudf::column_view const& column,
                               ::cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr);

/// The final value of a wide integer SUM: @p column as INT64. An INT64 column is returned as is.
/// A DECIMAL128 column (a sum that was accumulated over 128 bits) is cast to INT64 after checking
/// that every valid value fits; if one does not, throws sirius::not_implemented_exception so that
/// the query is replayed on DuckDB, which returns the exact HUGEINT. Synchronizes @p stream for a
/// DECIMAL128 column.
std::unique_ptr<cudf::column> narrow_wide_integer_sum(std::unique_ptr<cudf::column> column,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr);

/// The narrower type a SUM over a column of type @p type can be computed in when the batch's values
/// are small enough: DECIMAL64 at the same scale for DECIMAL128, nullopt otherwise.
std::optional<cudf::data_type> narrowed_decimal_sum_type(cudf::data_type type);

/// The columns among @p candidates (DECIMAL128) whose SUM over this batch can be computed over
/// DECIMAL64 and widened back afterwards, with the same result.
///
/// DuckDB types some sums wider than their values need. For example, DuckDB v2 types the TPC-H Q9
/// profit `DECIMAL(18,4) - DECIMAL(18,4)` as DECIMAL(19,4), which the GPU holds as DECIMAL128,
/// although every value fits 64 bits. cuDF's hash aggregation sums 128-bit values far more slowly
/// than 64-bit ones, especially over few groups. A column is narrowed only when rows * max|value|
/// is at most 2^63 - 1, the same exact proof sums_needing_widening uses, so no value and no
/// partial sum of any group can leave the DECIMAL64 range. A column whose every candidate carries a
/// plan-time bound is decided on the host. The others are measured: all cudf::minmax passes are
/// launched first and come back in one device-to-host copy, so a batch costs a single stream
/// synchronization. Nulls are skipped. Throws sirius::internal_exception if a candidate is not
/// DECIMAL128.
std::unordered_set<int> decimal_sums_to_narrow(cudf::table_view const& table,
                                               std::vector<sum_candidate> const& candidates,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr);

/** @brief An aggregate answered by one cuDF request whose partial is also its output. */
struct plain_slot {
  std::size_t partial_idx{};  ///< Index in cudf_aggregates
  /// True if this is a wide integer SUM (is_wide_integer_sum): its partial results are INT64 or,
  /// where INT64 could overflow, DECIMAL128 with scale 0.
  bool is_wide_integer_sum = false;
};

/** @brief AVG: SUM at `sum_idx` and COUNT_VALID at `sum_idx + 1`, divided at the merge. */
struct avg_slot {
  std::size_t sum_idx{};                              ///< Index of the SUM in cudf_aggregates
  cudf::data_type output_type{cudf::type_id::EMPTY};  ///< Quotient type: FLOAT64 or DECIMAL
  bool is_wide_integer_sum = false;  ///< True if the SUM is a wide integer SUM, as in plain_slot
};

/** @brief COUNT(DISTINCT): COLLECT_SET locally and MERGE_SETS at the merge, then counted. */
struct count_distinct_slot {
  std::size_t partial_idx{};  ///< Index in cudf_aggregates
};

/** @brief STDDEV_SAMP: a count/mean/M2 STRUCT, merged by MERGE_M2 and finalized after merge. */
struct stddev_samp_slot {
  std::size_t partial_idx{};  ///< Index in cudf_aggregates
};

/** @brief FIRST: no cuDF request; child column `input_idx` is carried at `carried_idx`. */
struct first_slot {
  int input_idx{-1};          ///< Child column whose value this slot carries
  std::size_t carried_idx{};  ///< Position in the carried block that follows the partials
};

/**
 * @brief Mapping from one original DuckDB aggregate expression to its position(s) in the expanded
 * cudf_aggregates vector. AVG is decomposed into SUM + COUNT_VALID (two entries), FIRST uses none,
 * all others use one. STDDEV_SAMP uses a count/mean/M2 STRUCT identified by Kind::M2 and finalizes
 * after merge. COUNT DISTINCT uses COLLECT_SET locally and MERGE_SETS during merge, then counts
 * list elements.
 */
using AggregateSlot =
  std::variant<plain_slot, avg_slot, count_distinct_slot, stddev_samp_slot, first_slot>;

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

  /// One entry per original DuckDB aggregate expression, mapping to cudf_aggregates positions or,
  /// for a FIRST, to its carried column.
  std::vector<AggregateSlot> aggregate_slots;
  /// Runtime partial schema: group keys, expanded aggregate carriers, then the FIRST carried
  /// columns.
  duckdb::vector<sirius::logical_type> local_types;
  bool has_avg            = false;  ///< True if any aggregate is AVG
  bool has_count_distinct = false;  ///< True if any aggregate is COUNT(DISTINCT col)
  /// True if any aggregate is FIRST. Presence only; each FIRST is a column of the carried block.
  bool has_first = false;
};

/// Parallel to a cudf_aggregates vector of @p num_cudf_aggregates entries: true at the SUM entry
/// of every wide integer SUM or AVG among @p slots.
std::vector<bool> wide_integer_sum_mask(std::vector<AggregateSlot> const& slots,
                                        std::size_t num_cudf_aggregates);

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

/**
 * @brief Whether every slot in @p aggregate_slots holds a `plain_slot`
 *
 * Any other `AggregateSlot` alternative, including one added later, makes the result false. An
 * empty list is all plain.
 */
[[nodiscard]] bool all_slots_plain(std::vector<AggregateSlot> const& aggregate_slots);

/**
 * @brief Whether any slot in @p aggregate_slots holds the alternative `Kind`
 *
 * @tparam Kind One of the `AggregateSlot` alternatives; any other type does not compile
 */
template <typename Kind>
[[nodiscard]] bool has_slot(std::vector<AggregateSlot> const& aggregate_slots)
{
  return std::ranges::any_of(
    aggregate_slots, [](AggregateSlot const& slot) { return std::holds_alternative<Kind>(slot); });
}

}  // namespace op
}  // namespace sirius
