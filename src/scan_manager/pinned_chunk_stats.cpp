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
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/in_filter.hpp>
#include <duckdb/planner/filter/optional_filter.hpp>
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

duckdb::Value scalar_to_value(cudf::scalar const& s, rmm::cuda_stream_view stream)
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
  rmm::cuda_stream_view stream,
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

bool filter_safe_for_stats(duckdb::TableFilter const& filter, duckdb::LogicalType const& stats_type)
{
  switch (filter.filter_type) {
    case duckdb::TableFilterType::CONSTANT_COMPARISON: {
      auto const& cf = filter.Cast<duckdb::ConstantFilter>();
      switch (cf.comparison_type) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO: break;
        default: return false;
      }
      // ConstantFilter's constructor rejects NULL constants, but the type-match
      // below is the release-mode safety line (DuckDB only D_ASSERTs it), so
      // stay defensive on both.
      return !cf.constant.IsNull() && cf.constant.type() == stats_type;
    }
    case duckdb::TableFilterType::IS_NULL:
    case duckdb::TableFilterType::IS_NOT_NULL: return true;
    case duckdb::TableFilterType::IN_FILTER: {
      auto const& in = filter.Cast<duckdb::InFilter>();
      return !in.values.empty() && std::ranges::all_of(in.values, [&](duckdb::Value const& v) {
        return !v.IsNull() && v.type() == stats_type;
      });
    }
    case duckdb::TableFilterType::CONJUNCTION_AND:
    case duckdb::TableFilterType::CONJUNCTION_OR: {
      auto const& children = filter.filter_type == duckdb::TableFilterType::CONJUNCTION_AND
                               ? filter.Cast<duckdb::ConjunctionAndFilter>().child_filters
                               : filter.Cast<duckdb::ConjunctionOrFilter>().child_filters;
      // A childless OR would propagate FILTER_ALWAYS_FALSE and prune unconditionally; a childless
      // AND is merely vacuous. Neither shape is produced by the binder, so reject both rather than
      // reason about them.
      return !children.empty() &&
             std::ranges::all_of(children, [&](duckdb::unique_ptr<duckdb::TableFilter> const& c) {
               return c && filter_safe_for_stats(*c, stats_type);
             });
    }
    case duckdb::TableFilterType::OPTIONAL_FILTER: {
      // OptionalFilter's child may legitimately be nullptr (its constructor defaults it); an empty
      // optional constrains nothing and cannot prune.
      auto const& opt = filter.Cast<duckdb::OptionalFilter>();
      return opt.child_filter && filter_safe_for_stats(*opt.child_filter, stats_type);
    }
    // DYNAMIC (mutable at run time), STRUCT_EXTRACT / EXPRESSION / BLOOM
    // (shapes CheckStatistics may misread against numeric stats), and any
    // future filter type default to "keep the chunk".
    default: return false;
  }
}

bool chunk_provably_empty(duckdb::TableFilter const& filter,
                          duckdb::BaseStatistics const& stats) noexcept
{
  try {
    if (!filter_safe_for_stats(filter, stats.GetType())) { return false; }
    // CheckStatistics needs a non-const reference to the stats object. Copy instead of const_cast
    // to keep CheckStatistics safe.
    auto local_stats = stats.Copy();
    return filter.CheckStatistics(local_stats) ==
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
