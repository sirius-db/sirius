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

#include "planner/decimal_sum_input_bounds.hpp"

#include "expression/aggregate_id.hpp"
#include "helper/logical_type.hpp"
#include "log/logging.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "planner/scan_column_origin.hpp"

#include <duckdb/common/types/value.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>

#include <algorithm>
#include <utility>

namespace sirius::planner {

namespace {

/// The magnitude of the unscaled integer a DECIMAL statistics bound holds.
std::optional<std::uint64_t> unscaled_magnitude(duckdb::Value const& value)
{
  if (value.IsNull()) { return std::nullopt; }
  switch (value.type().InternalType()) {
    case duckdb::PhysicalType::INT16: return op::decimal_magnitude(value.GetValueUnsafe<int16_t>());
    case duckdb::PhysicalType::INT32: return op::decimal_magnitude(value.GetValueUnsafe<int32_t>());
    case duckdb::PhysicalType::INT64: return op::decimal_magnitude(value.GetValueUnsafe<int64_t>());
    default: return std::nullopt;
  }
}

// Read optional magnitude evidence; unavailable statistics keep per-batch measurement.
std::optional<std::uint64_t> scan_column_max_abs(duckdb::ClientContext& context,
                                                 scan_column_origin const& origin,
                                                 duckdb::LogicalType const& type)
{
  auto const& get = *origin.get;
  if ((!get.function.statistics && !get.function.statistics_extended) || !get.bind_data) {
    return std::nullopt;
  }
  auto const& column_ids = get.GetColumnIds();
  auto const index       = get.projection_ids.empty()
                             ? origin.ordinal
                             : static_cast<std::size_t>(get.projection_ids[origin.ordinal]);
  if (index >= column_ids.size()) { return std::nullopt; }
  auto const& column = column_ids[index];
  if (column.IsRowIdColumn() || column.IsVirtualColumn()) { return std::nullopt; }

  duckdb::unique_ptr<duckdb::BaseStatistics> stats;
  try {
    if (get.function.statistics_extended) {
      duckdb::TableFunctionGetStatisticsInput input(get.bind_data.get(), column);
      stats = get.function.statistics_extended(context, input);
    } else {
      stats = get.function.statistics(context, get.bind_data.get(), column.GetPrimaryIndex());
    }
  } catch (...) {
    return std::nullopt;
  }
  if (!stats || stats->GetType() != type || !duckdb::NumericStats::HasMinMax(*stats)) {
    return std::nullopt;
  }
  auto const lo = unscaled_magnitude(duckdb::NumericStats::Min(*stats));
  auto const hi = unscaled_magnitude(duckdb::NumericStats::Max(*stats));
  if (!lo || !hi) { return std::nullopt; }
  return std::max(*lo, *hi);
}

}  // namespace

std::vector<std::optional<std::uint64_t>> resolve_decimal_sum_input_bounds(
  duckdb::ClientContext& context, duckdb::LogicalAggregate const& op)
{
  std::vector<std::optional<std::uint64_t>> bounds(op.expressions.size());
  if (op.children.size() != 1) { return bounds; }
  auto const& child = *op.children[0];
  // Each scan column is queried once: SUM and AVG over the same input share the lookup.
  std::vector<std::pair<scan_column_origin, std::optional<std::uint64_t>>> memo;
  for (std::size_t i = 0; i < op.expressions.size(); ++i) {
    auto const& expression = *op.expressions[i];
    if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) { continue; }
    auto const& aggregate = expression.Cast<duckdb::BoundAggregateExpression>();
    auto const id         = sirius::from_duckdb_aggregate_name(aggregate.function.name);
    if (!id || (*id != sirius::aggregate_id::sum && *id != sirius::aggregate_id::sum_no_overflow &&
                *id != sirius::aggregate_id::avg)) {
      continue;
    }
    if (aggregate.children.size() != 1) { continue; }
    auto const& input = *aggregate.children[0];
    auto const& type  = input.return_type;
    // Wider decimals already run as DECIMAL128 and are never widened.
    if (type.id() != duckdb::LogicalTypeId::DECIMAL ||
        duckdb::DecimalType::GetWidth(type) > sirius::logical_type::decimal_max_precision_int64) {
      continue;
    }
    if (input.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { continue; }
    auto const ordinal =
      static_cast<std::size_t>(input.Cast<duckdb::BoundReferenceExpression>().index);
    auto const origin = resolve_scan_column_origin(child, ordinal, origin_policy::value_preserving);
    if (!origin) { continue; }
    auto const known = std::find_if(memo.begin(), memo.end(), [&](auto const& entry) {
      return entry.first.get == origin->get && entry.first.ordinal == origin->ordinal;
    });
    if (known != memo.end()) {
      bounds[i] = known->second;
      continue;
    }
    bounds[i] = scan_column_max_abs(context, *origin, type);
    memo.emplace_back(*origin, bounds[i]);
    if (bounds[i]) {
      SIRIUS_LOG_DEBUG("decimal sum bound: aggregate {} ({}) has |unscaled value| <= {}",
                       i,
                       aggregate.function.name,
                       *bounds[i]);
    }
  }
  return bounds;
}

}  // namespace sirius::planner
