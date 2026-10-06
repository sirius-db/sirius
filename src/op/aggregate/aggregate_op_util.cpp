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

#include "op/aggregate/aggregate_op_util.hpp"

#include "cudf/cudf_utils.hpp"
#include "duckdb/common/assert.hpp"
#include "expression/aggregate_id.hpp"
#include "expression/ast/node.hpp"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <algorithm>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sirius {
namespace op {

namespace {

// Single place that builds the "Unsupported aggregate function: <name>" diagnostic so the
// message (and the aggregate_id -> name lookup) is not repeated at every rejection site.
[[noreturn]] void throw_unsupported_aggregate(sirius::aggregate_id fid,
                                              std::string_view detail = {})
{
  auto const name = sirius::to_duckdb_aggregate_name(fid);
  throw std::runtime_error(detail.empty()
                             ? std::format("Unsupported aggregate function: {}", name)
                             : std::format("Unsupported aggregate function: {} {}", name, detail));
}

}  // namespace

std::optional<cudf::aggregation::Kind> to_cudf_aggregation_kind(sirius::aggregate_id id)
{
  switch (id) {
    case sirius::aggregate_id::sum:
    case sirius::aggregate_id::sum_no_overflow: return cudf::aggregation::Kind::SUM;
    case sirius::aggregate_id::count: return cudf::aggregation::Kind::COUNT_VALID;
    case sirius::aggregate_id::count_star: return cudf::aggregation::Kind::COUNT_ALL;
    case sirius::aggregate_id::min: return cudf::aggregation::Kind::MIN;
    case sirius::aggregate_id::max: return cudf::aggregation::Kind::MAX;
    case sirius::aggregate_id::avg:
    case sirius::aggregate_id::first: return std::nullopt;
  }
  return std::nullopt;
}

CudfAggregateDefinitions convert_duckdb_aggregates_to_cudf(
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& groups_p,
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& expressions)
{
  CudfAggregateDefinitions result;

  // 1. Extract group_idx from groups_p
  for (const auto& group : groups_p) {
    auto const& ref =
      sirius::ast::require_reference(group.get(), "convert_duckdb_aggregates_to_cudf group");
    result.group_idx.push_back(static_cast<int>(ref.column_index));
  }

  // 2. Extract aggregates (cudf::aggregation::Kind) from expressions
  for (const auto& aggregate : expressions) {
    auto const& aggr = sirius::ast::require_aggregate(
      aggregate.get(), "convert_duckdb_aggregates_to_cudf aggregate");
    auto const fid       = aggr.function();
    auto const& children = aggr.arguments();

    // Handle AVG specially: it expands into SUM + COUNT_VALID
    if (fid == sirius::aggregate_id::avg) {
      D_ASSERT(children.size() == 1);
      D_ASSERT(children[0]->is_reference());
      auto col_idx = static_cast<int>(children[0]->as_reference().column_index);

      size_t sum_position = result.cudf_aggregates.size();
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::SUM);
      result.cudf_aggregate_idx.push_back(col_idx);
      result.cudf_aggregate_struct_col_indices.push_back({});
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::COUNT_VALID);
      result.cudf_aggregate_idx.push_back(col_idx);
      result.cudf_aggregate_struct_col_indices.push_back({});
      result.aggregate_slots.push_back(
        AggregateSlot{true, false, sum_position, sirius::get_cudf_type(aggr.return_type())});
      result.has_avg = true;
      continue;
    }

    // Handle COUNT(DISTINCT col) and COUNT(DISTINCT (col1, col2, ...)):
    // Use COLLECT_SET locally; merge via MERGE_SETS; then count list elements.
    // For multi-column, a struct column is synthesized from the component columns.
    if (aggr.distinct() && fid == sirius::aggregate_id::count) {
      D_ASSERT(children.size() == 1);
      auto const& child = *children[0];
      size_t position   = result.cudf_aggregates.size();
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::COLLECT_SET);

      if (child.is_reference()) {
        // Single-column case: COUNT(DISTINCT col)
        result.cudf_aggregate_idx.push_back(static_cast<int>(child.as_reference().column_index));
        result.cudf_aggregate_struct_col_indices.push_back({});
      } else {
        // Multi-column case: COUNT(DISTINCT (col1, col2, ...)) — child is a struct_pack expression
        D_ASSERT(child.is_function_call());
        auto const& func_expr = child.as_function_call();
        std::vector<int> struct_indices;
        for (auto const& arg : func_expr.arguments()) {
          D_ASSERT(arg->is_reference());
          struct_indices.push_back(static_cast<int>(arg->as_reference().column_index));
        }
        D_ASSERT(!struct_indices.empty());
        result.cudf_aggregate_idx.push_back(-1);  // sentinel: struct column, see gpu_aggregate_impl
        result.cudf_aggregate_struct_col_indices.push_back(std::move(struct_indices));
      }

      result.aggregate_slots.push_back(AggregateSlot{false, true, position});
      result.has_count_distinct = true;
      continue;
    }

    auto const agg_kind = to_cudf_aggregation_kind(fid);
    if (!agg_kind) { throw_unsupported_aggregate(fid); }
    size_t current_position = result.cudf_aggregates.size();
    result.cudf_aggregates.push_back(*agg_kind);

    // 3. Extract aggregate_idx from the children of the aggregate expression
    if (children.empty()) {
      // COUNT(*) has no children - use 0 as a placeholder (will be handled by COUNT_ALL)
      if (fid == sirius::aggregate_id::count_star) {
        result.cudf_aggregate_idx.push_back(0);
      } else {
        throw_unsupported_aggregate(fid, "with no children");
      }
    } else {
      if (children.size() == 1) {
        // Extract the column index from the first child (most aggregates have one child)
        D_ASSERT(children[0]->is_reference());
        result.cudf_aggregate_idx.push_back(
          static_cast<int>(children[0]->as_reference().column_index));
      } else {
        throw_unsupported_aggregate(fid, "with " + std::to_string(children.size()) + " children");
      }
    }
    result.cudf_aggregate_struct_col_indices.push_back({});
    result.aggregate_slots.push_back(AggregateSlot{false, false, current_position});
  }

  return result;
}

namespace {

/// The unscaled value of a DECIMAL32 or DECIMAL64 scalar, widened to 128 bits.
__int128 unscaled_value(cudf::scalar const& s, cudf::type_id id, rmm::cuda_stream_view stream)
{
  if (id == cudf::type_id::DECIMAL32) {
    return static_cast<cudf::fixed_point_scalar<numeric::decimal32> const&>(s).value(stream);
  }
  return static_cast<cudf::fixed_point_scalar<numeric::decimal64> const&>(s).value(stream);
}

}  // namespace

std::optional<cudf::data_type> widened_decimal_sum_type(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::DECIMAL32: return cudf::data_type(cudf::type_id::DECIMAL64, type.scale());
    case cudf::type_id::DECIMAL64: return cudf::data_type(cudf::type_id::DECIMAL128, type.scale());
    default: return std::nullopt;
  }
}

std::unordered_set<int> decimal_sums_needing_widening(cudf::table_view const& table,
                                                      std::vector<int> const& candidates,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  struct extremes {
    int column;
    std::unique_ptr<cudf::scalar> min;
    std::unique_ptr<cudf::scalar> max;
  };
  std::unordered_set<int> widen;
  auto const num_rows = static_cast<unsigned __int128>(table.num_rows());
  if (num_rows == 0) { return widen; }

  std::vector<extremes> pending;
  pending.reserve(candidates.size());
  for (int col_id : candidates) {
    auto [lo, hi] = cudf::minmax(table.column(col_id), stream, mr);
    pending.push_back({col_id, std::move(lo), std::move(hi)});
  }
  for (auto const& e : pending) {
    auto const type_id = table.column(e.column).type().id();
    if (!e.min->is_valid(stream)) { continue; }
    auto const lo = unscaled_value(*e.min, type_id, stream);
    auto const hi = unscaled_value(*e.max, type_id, stream);
    auto const max_abs =
      static_cast<unsigned __int128>(std::max(lo < 0 ? -lo : lo, hi < 0 ? -hi : hi));
    auto const limit = type_id == cudf::type_id::DECIMAL32
                         ? static_cast<unsigned __int128>(std::numeric_limits<int32_t>::max())
                         : static_cast<unsigned __int128>(std::numeric_limits<int64_t>::max());
    if (num_rows * max_abs > limit) { widen.insert(e.column); }
  }
  return widen;
}

}  // namespace op
}  // namespace sirius
