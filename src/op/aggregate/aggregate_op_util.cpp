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

#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>

#include <sirius/exception.hpp>

#include <algorithm>
#include <cstdint>
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

/// Magnitude of an int64 as uint64, defined for INT64_MIN too (modular negation).
uint64_t magnitude(int64_t v)
{
  return v < 0 ? -static_cast<uint64_t>(v) : static_cast<uint64_t>(v);
}

}  // namespace

void throw_if_int64_sum_could_overflow(const cudf::column_view& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const type_id = input.type().id();
  if (type_id != cudf::type_id::INT64 && type_id != cudf::type_id::UINT64) { return; }
  auto const valid_rows =
    static_cast<uint64_t>(input.size()) - static_cast<uint64_t>(input.null_count());
  // A lone 64-bit value is its own sum; nothing can wrap.
  if (valid_rows < 2) { return; }
  auto const [min_scalar, max_scalar] = cudf::minmax(input, stream, mr);
  if (type_id == cudf::type_id::INT64) {
    auto const lo = static_cast<const cudf::numeric_scalar<int64_t>&>(*min_scalar).value(stream);
    auto const hi = static_cast<const cudf::numeric_scalar<int64_t>&>(*max_scalar).value(stream);
    // `mag > INT64_MAX / rows` (floor division) is exactly `rows * mag > INT64_MAX`.
    auto const mag = std::max(magnitude(lo), magnitude(hi));
    if (mag > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / valid_rows) {
      throw sirius::invalid_input_exception(
        "sum over {} BIGINT values in [{}, {}] could overflow int64: DuckDB computes this sum "
        "as HUGEINT, but cuDF has no INT128, so Sirius runs it as BIGINT and an overflow would "
        "wrap silently. Refusing to run instead of risking a wrong result (conservative "
        "rows * max(|min|,|max|) bound); cast the summed column to DOUBLE to accept lossy "
        "accumulation.",
        valid_rows,
        lo,
        hi);
    }
  } else {
    auto const hi = static_cast<const cudf::numeric_scalar<uint64_t>&>(*max_scalar).value(stream);
    if (hi > std::numeric_limits<uint64_t>::max() / valid_rows) {
      throw sirius::invalid_input_exception(
        "sum over {} UBIGINT values with max {} could overflow uint64: DuckDB computes this "
        "sum as UHUGEINT, but cuDF has no UINT128, so Sirius runs it as UBIGINT and an "
        "overflow would wrap silently. Refusing to run instead of risking a wrong result "
        "(conservative rows * max bound); cast the summed column to DOUBLE to accept lossy "
        "accumulation.",
        valid_rows,
        hi);
    }
  }
}

bool is_order_sensitive_sum(cudf::aggregation::Kind kind, cudf::data_type values_type)
{
  return kind == cudf::aggregation::Kind::SUM &&
         (values_type.id() == cudf::type_id::FLOAT32 || values_type.id() == cudf::type_id::FLOAT64);
}

std::unique_ptr<cudf::table> canonicalize_row_order(
  const cudf::table_view& input,
  const std::vector<cudf::size_type>& sort_col_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<cudf::column_view> sort_cols;
  sort_cols.reserve(sort_col_indices.size());
  for (auto col_idx : sort_col_indices) {
    sort_cols.push_back(input.column(col_idx));
  }
  auto sort_order =
    cudf::sorted_order(cudf::table_view(sort_cols),
                       std::vector<cudf::order>(sort_cols.size(), cudf::order::ASCENDING),
                       std::vector<cudf::null_order>(sort_cols.size(), cudf::null_order::AFTER),
                       stream,
                       mr);
  return cudf::gather(
    input, sort_order->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
}

}  // namespace op
}  // namespace sirius
