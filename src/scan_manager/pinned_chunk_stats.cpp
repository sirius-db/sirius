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

#include "scan_manager/pinned_chunk_stats.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "log/logging.hpp"

#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/enums/filter_propagate_result.hpp>
#include <duckdb/common/types/date.hpp>
#include <duckdb/common/types/timestamp.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/filter/table_filter_functions.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

namespace sirius::scan_manager {

namespace {
std::optional<cudf::type_id> expected_cudf_type(duckdb::LogicalType const& type)
{
  switch (type.id()) {
    case duckdb::LogicalTypeId::TINYINT: return cudf::type_id::INT8;
    case duckdb::LogicalTypeId::SMALLINT: return cudf::type_id::INT16;
    case duckdb::LogicalTypeId::INTEGER: return cudf::type_id::INT32;
    case duckdb::LogicalTypeId::BIGINT: return cudf::type_id::INT64;
    case duckdb::LogicalTypeId::UTINYINT: return cudf::type_id::UINT8;
    case duckdb::LogicalTypeId::USMALLINT: return cudf::type_id::UINT16;
    case duckdb::LogicalTypeId::UINTEGER: return cudf::type_id::UINT32;
    case duckdb::LogicalTypeId::UBIGINT: return cudf::type_id::UINT64;
    case duckdb::LogicalTypeId::DATE: return cudf::type_id::TIMESTAMP_DAYS;
    case duckdb::LogicalTypeId::TIMESTAMP: return cudf::type_id::TIMESTAMP_MICROSECONDS;
    default: return std::nullopt;
  }
}

duckdb::Value scalar_to_value(cudf::scalar const& s, ::cuda::stream_ref stream)
{
  switch (s.type().id()) {
    case cudf::type_id::INT8:
      return duckdb::Value::TINYINT(
        static_cast<cudf::numeric_scalar<std::int8_t> const&>(s).value(stream));
    case cudf::type_id::INT16:
      return duckdb::Value::SMALLINT(
        static_cast<cudf::numeric_scalar<std::int16_t> const&>(s).value(stream));
    case cudf::type_id::INT32:
      return duckdb::Value::INTEGER(
        static_cast<cudf::numeric_scalar<std::int32_t> const&>(s).value(stream));
    case cudf::type_id::INT64:
      return duckdb::Value::BIGINT(
        static_cast<cudf::numeric_scalar<std::int64_t> const&>(s).value(stream));
    case cudf::type_id::UINT8:
      return duckdb::Value::UTINYINT(
        static_cast<cudf::numeric_scalar<std::uint8_t> const&>(s).value(stream));
    case cudf::type_id::UINT16:
      return duckdb::Value::USMALLINT(
        static_cast<cudf::numeric_scalar<std::uint16_t> const&>(s).value(stream));
    case cudf::type_id::UINT32:
      return duckdb::Value::UINTEGER(
        static_cast<cudf::numeric_scalar<std::uint32_t> const&>(s).value(stream));
    case cudf::type_id::UINT64:
      return duckdb::Value::UBIGINT(
        static_cast<cudf::numeric_scalar<std::uint64_t> const&>(s).value(stream));
    case cudf::type_id::TIMESTAMP_DAYS: {
      auto const days = static_cast<cudf::timestamp_scalar<cudf::timestamp_D> const&>(s)
                          .value(stream)
                          .time_since_epoch()
                          .count();
      return duckdb::Value::DATE(duckdb::date_t{days});
    }
    case cudf::type_id::TIMESTAMP_MICROSECONDS: {
      auto const micros = static_cast<cudf::timestamp_scalar<cudf::timestamp_us> const&>(s)
                            .value(stream)
                            .time_since_epoch()
                            .count();
      return duckdb::Value::TIMESTAMP(duckdb::timestamp_t{micros});
    }
    default:
      SIRIUS_LOG_DEBUG("[pinned_chunk_stats] scalar type {} outside allowlist; dropping stats cell",
                       static_cast<std::int32_t>(s.type().id()));
      return duckdb::Value();
  }
}
}  // namespace

std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> compute_pinned_chunk_stats(
  cudf::table_view const& chunk,
  duckdb::vector<duckdb::LogicalType> const& column_types,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const n_columns = static_cast<std::size_t>(chunk.num_columns());
  // Null entries by default (-> no stats for that column, never prunes)
  std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> stats(n_columns);

  if (column_types.size() != n_columns) {
    SIRIUS_LOG_WARN(
      "[pinned_chunk_stats] column_types size ({}) != chunk column count ({}); capturing no "
      "statistics for this chunk",
      column_types.size(),
      n_columns);
    return stats;
  }

  for (std::size_t i = 0; i < n_columns; ++i) {
    auto const& col     = chunk.column(static_cast<cudf::size_type>(i));
    auto const& type    = column_types[i];
    auto const expected = expected_cudf_type(type);
    if (!expected || col.type().id() != *expected) { continue; }  // outside allowlist
    if (col.size() == 0 || col.null_count() == col.size()) { continue; }

    // CUDA failures propagate: after a device fault the stream/device state is suspect and the
    // pin's own materialization error handling must abort the pin.
    auto const [min_scalar, max_scalar] = cudf::minmax(col, stream, mr);
    if (!min_scalar || !max_scalar || !min_scalar->is_valid(stream) ||
        !max_scalar->is_valid(stream)) {
      continue;
    }

    auto const min_value = scalar_to_value(*min_scalar, stream);
    auto const max_value = scalar_to_value(*max_scalar, stream);
    if (min_value.IsNull() || max_value.IsNull()) { continue; }

    // CreateUnknown pre-sets both null flags ("may have nulls, may have valid rows"); tighten to
    // the exact chunk-level facts. The all-null case was gated out above.
    auto column_stats = duckdb::NumericStats::CreateUnknown(type);
    duckdb::NumericStats::SetMin(column_stats, min_value);
    duckdb::NumericStats::SetMax(column_stats, max_value);
    if (col.null_count() == 0) {
      column_stats.Set(duckdb::StatsInfo::CANNOT_HAVE_NULL_VALUES);
    } else {
      column_stats.SetHasNull();
    }
    stats[i] = column_stats.ToUnique();
  }
  return stats;
}

pinned_zone_maps pinned_zone_maps::from_capture(
  duckdb::vector<duckdb::LogicalType> column_types,
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats,
  std::size_t n_columns,
  std::size_t n_chunks)
{
  bool ok = !column_types.empty() && column_types.size() == n_columns && !chunk_stats.empty() &&
            chunk_stats.size() == n_chunks;
  if (ok) {
    for (auto const& per_chunk : chunk_stats) {
      if (per_chunk.size() != n_columns) {
        ok = false;
        break;
      }
    }
  }
  if (!ok) { return {}; }

  pinned_zone_maps out;
  out._column_types = std::move(column_types);
  out._column_stats.resize(n_columns);
  for (auto& column : out._column_stats) {
    column.reserve(n_chunks);
  }
  for (auto& per_chunk : chunk_stats) {
    for (std::size_t i = 0; i < n_columns; ++i) {
      out._column_stats[i].push_back(std::move(per_chunk[i]));
    }
  }
  return out;
}

void pinned_zone_maps::append_column_from(pinned_zone_maps& incoming, std::size_t incoming_pos)
{
  bool const compatible =
    has_stats() && incoming.has_stats() && incoming_pos < incoming.column_count() &&
    incoming._column_stats[incoming_pos].size() == _column_stats.front().size();
  if (!compatible) {
    _column_types.clear();
    _column_stats.clear();
    return;
  }
  _column_types.push_back(incoming._column_types[incoming_pos]);
  _column_stats.push_back(std::move(incoming._column_stats[incoming_pos]));
}

pinned_zone_maps pinned_zone_maps::remap(pinned_zone_maps incoming,
                                         std::vector<std::size_t> const& incoming_pos_by_pos)
{
  if (!incoming.has_stats() || incoming_pos_by_pos.empty()) { return {}; }
  pinned_zone_maps out;
  out._column_types.reserve(incoming_pos_by_pos.size());
  out._column_stats.reserve(incoming_pos_by_pos.size());
  for (auto pos : incoming_pos_by_pos) {
    if (pos >= incoming.column_count() || incoming._column_stats[pos].empty()) { return {}; }
    out._column_types.push_back(incoming._column_types[pos]);
    out._column_stats.push_back(std::move(incoming._column_stats[pos]));
  }
  return out;
}

bool filter_safe_for_stats(duckdb::TableFilter const& filter)
{
  if (filter.filter_type != duckdb::TableFilterType::EXPRESSION_FILTER) { return false; }
  auto const& expression = filter.Cast<duckdb::ExpressionFilter>();
  if (!expression.expr) { return false; }
  std::optional<duckdb::LogicalType> column_type;
  std::function<void(duckdb::Expression const&)> find_reference =
    [&](duckdb::Expression const& expr) {
      if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
        column_type = expr.GetReturnType();
      }
      if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_FUNCTION) {
        auto const& function = expr.Cast<duckdb::BoundFunctionExpression>();
        if (function.Function().GetName() == duckdb::OptionalFilterScalarFun::NAME &&
            function.BindInfo()) {
          auto const& data = function.BindInfo()->Cast<duckdb::OptionalFilterFunctionData>();
          if (data.child_filter_expr) { find_reference(*data.child_filter_expr); }
        }
      }
      duckdb::ExpressionIterator::EnumerateChildren(expr, find_reference);
    };
  find_reference(*expression.expr);
  return column_type && filter_safe_for_stats(filter, *column_type);
}

bool filter_safe_for_stats(duckdb::TableFilter const& filter, duckdb::LogicalType const& stats_type)
{
  if (filter.filter_type != duckdb::TableFilterType::EXPRESSION_FILTER) { return false; }
  auto const& expression_filter = filter.Cast<duckdb::ExpressionFilter>();
  if (!expression_filter.expr || !expression_filter.column_indexes.empty()) { return false; }
  auto reference = [&](duckdb::Expression const& expr) {
    return expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF &&
           expr.Cast<duckdb::BoundReferenceExpression>().Index() == 0 &&
           expr.GetReturnType() == stats_type;
  };
  auto constant = [&](duckdb::Expression const& expr) {
    return expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT &&
           expr.GetReturnType() == stats_type &&
           !expr.Cast<duckdb::BoundConstantExpression>().GetValue().IsNull();
  };
  std::function<bool(duckdb::Expression const&)> safe = [&](duckdb::Expression const& expr) {
    using duckdb::ExpressionClass;
    using duckdb::ExpressionType;
    if (duckdb::BoundComparisonExpression::IsComparison(expr)) {
      if (!duckdb::ExpressionFilter::CanPropagateExpressionStatistics(expr)) { return false; }
      auto const& cmp   = expr.Cast<duckdb::BoundFunctionExpression>();
      auto const& left  = duckdb::BoundComparisonExpression::Left(cmp);
      auto const& right = duckdb::BoundComparisonExpression::Right(cmp);
      return (reference(left) && constant(right)) || (constant(left) && reference(right));
    }
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
      auto const& children = expr.Cast<duckdb::BoundConjunctionExpression>().GetChildren();
      return !children.empty() &&
             std::all_of(children.begin(), children.end(), [&](auto const& child) {
               return child && safe(*child);
             });
    }
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
      auto const& children = expr.Cast<duckdb::BoundOperatorExpression>().GetChildren();
      if (children.empty() || !children[0] || !reference(*children[0])) { return false; }
      if (expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ||
          expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NOT_NULL) {
        return children.size() == 1;
      }
      if (expr.GetExpressionType() != ExpressionType::COMPARE_IN || children.size() < 2) {
        return false;
      }
      return std::all_of(children.begin() + 1, children.end(), [&](auto const& child) {
        return child && constant(*child);
      });
    }
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
      auto const& function = expr.Cast<duckdb::BoundFunctionExpression>();
      if (function.Function().GetName() == duckdb::OptionalFilterScalarFun::NAME &&
          function.BindInfo()) {
        auto const& data = function.BindInfo()->Cast<duckdb::OptionalFilterFunctionData>();
        return data.child_filter_expr && safe(*data.child_filter_expr);
      }
    }
    // Runtime filters carry mutable state and cannot certify a cached prune decision.
    return false;
  };
  return safe(*expression_filter.expr);
}

bool chunk_provably_empty(duckdb::TableFilter const& filter,
                          duckdb::BaseStatistics const& stats) noexcept
{
  try {
    if (!filter_safe_for_stats(filter, stats.GetType())) { return false; }
    // CheckStatistics needs a non-const reference to the stats object. Copy instead of const_cast
    // to keep CheckStatistics safe.
    auto local_stats = stats.Copy();
    return filter.Cast<duckdb::ExpressionFilter>().CheckStatistics(local_stats) ==
           duckdb::FilterPropagateResult::FILTER_ALWAYS_FALSE;
  } catch (std::exception const& e) {
    // Don't propagate exceptions here on statistics checks; otherwise, a cache miss is generated
    // and the scan falls back to disk reads in sirius_scan_manager::try_assign_cached_entries().
    SIRIUS_LOG_DEBUG("[pinned_chunk_stats] prune probe failed, keeping chunk: {}", e.what());
    return false;
  } catch (...) {
    SIRIUS_LOG_DEBUG("[pinned_chunk_stats] prune probe failed, keeping chunk: unknown error");
    return false;
  }
}

}  // namespace sirius::scan_manager
