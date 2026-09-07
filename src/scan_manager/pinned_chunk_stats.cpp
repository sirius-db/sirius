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

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

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
#include <stdexcept>
#include <string>
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
/// Host-side element -> duckdb::Value, mirroring scalar_to_value's allowlist exactly. Used by
/// the per-group capture, which reads whole reduction-output columns back rather than scalars.
template <typename T>
duckdb::Value element_to_value(cudf::type_id id, T raw)
{
  switch (id) {
    case cudf::type_id::INT8: return duckdb::Value::TINYINT(static_cast<std::int8_t>(raw));
    case cudf::type_id::INT16: return duckdb::Value::SMALLINT(static_cast<std::int16_t>(raw));
    case cudf::type_id::INT32: return duckdb::Value::INTEGER(static_cast<std::int32_t>(raw));
    case cudf::type_id::INT64: return duckdb::Value::BIGINT(static_cast<std::int64_t>(raw));
    case cudf::type_id::UINT8: return duckdb::Value::UTINYINT(static_cast<std::uint8_t>(raw));
    case cudf::type_id::UINT16: return duckdb::Value::USMALLINT(static_cast<std::uint16_t>(raw));
    case cudf::type_id::UINT32: return duckdb::Value::UINTEGER(static_cast<std::uint32_t>(raw));
    case cudf::type_id::UINT64: return duckdb::Value::UBIGINT(static_cast<std::uint64_t>(raw));
    case cudf::type_id::TIMESTAMP_DAYS:
      return duckdb::Value::DATE(duckdb::date_t{static_cast<std::int32_t>(raw)});
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return duckdb::Value::TIMESTAMP(duckdb::timestamp_t{static_cast<std::int64_t>(raw)});
    default: return duckdb::Value();
  }
}

/// Copy a fixed-width reduction-output column to host as duckdb::Values, one per group.
/// A null output element (an all-null group) yields a NULL Value, which the caller drops.
std::vector<duckdb::Value> reduction_column_to_values(cudf::column_view const& col,
                                                      rmm::cuda_stream_view stream)
{
  auto const n = static_cast<std::size_t>(col.size());
  std::vector<duckdb::Value> out(n);
  std::vector<bool> valid(n, true);
  if (col.nullable() && col.null_count() > 0) {
    auto const host_mask = cudf::detail::make_std_vector(
      cudf::device_span<cudf::bitmask_type const>{col.null_mask(),
                                                  cudf::num_bitmask_words(col.size())},
      stream);
    for (std::size_t g = 0; g < n; ++g) {
      valid[g] = (host_mask[g / 32] >> (g % 32)) & 1u;
    }
  }

  auto const copy_as = [&](auto tag) {
    using T   = decltype(tag);
    auto host = cudf::detail::make_std_vector(
      cudf::device_span<T const>{col.data<T>(), static_cast<std::size_t>(col.size())}, stream);
    for (std::size_t g = 0; g < n; ++g) {
      out[g] = valid[g] ? element_to_value(col.type().id(), host[g]) : duckdb::Value();
    }
  };
  switch (col.type().id()) {
    case cudf::type_id::INT8: copy_as(std::int8_t{}); break;
    case cudf::type_id::INT16: copy_as(std::int16_t{}); break;
    case cudf::type_id::INT32: copy_as(std::int32_t{}); break;
    case cudf::type_id::INT64: copy_as(std::int64_t{}); break;
    case cudf::type_id::UINT8: copy_as(std::uint8_t{}); break;
    case cudf::type_id::UINT16: copy_as(std::uint16_t{}); break;
    case cudf::type_id::UINT32: copy_as(std::uint32_t{}); break;
    case cudf::type_id::UINT64: copy_as(std::uint64_t{}); break;
    case cudf::type_id::TIMESTAMP_DAYS: copy_as(std::int32_t{}); break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS: copy_as(std::int64_t{}); break;
    default: break;  // outside the allowlist -> all-null, caller drops the cells
  }
  return out;
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

chunk_group_stats compute_pinned_group_stats(
  cudf::table_view const& chunk,
  duckdb::vector<duckdb::LogicalType> const& column_types,
  std::size_t group_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  chunk_group_stats out;
  auto const n_columns = static_cast<std::size_t>(chunk.num_columns());
  auto const n_rows    = static_cast<std::size_t>(chunk.num_rows());
  if (group_rows == 0 || n_rows == 0 || n_columns == 0) { return out; }
  if (column_types.size() != n_columns) {
    SIRIUS_LOG_WARN(
      "[pinned_chunk_stats] column_types size ({}) != chunk column count ({}); capturing no "
      "group statistics for this chunk",
      column_types.size(),
      n_columns);
    return out;
  }

  auto const n_groups = (n_rows + group_rows - 1) / group_rows;
  out.group_rows      = group_rows;
  out.groups.resize(n_groups);
  for (auto& row : out.groups) {
    row.resize(n_columns);
  }  // unique_ptr rows are move-only

  // One offsets column shared by every reduction: fixed stride, clamped final group.
  std::vector<cudf::size_type> host_offsets(n_groups + 1);
  for (std::size_t g = 0; g <= n_groups; ++g) {
    host_offsets[g] = static_cast<cudf::size_type>(std::min(g * group_rows, n_rows));
  }
  rmm::device_uvector<cudf::size_type> offsets(host_offsets.size(), stream, mr);
  if (auto const rc = cudaMemcpyAsync(offsets.data(),
                                      host_offsets.data(),
                                      host_offsets.size() * sizeof(cudf::size_type),
                                      cudaMemcpyHostToDevice,
                                      stream.value());
      rc != cudaSuccess) {
    // Match the whole-chunk capture: a CUDA failure here leaves device state suspect, so let the
    // pin's own error handling abort rather than silently pinning without statistics.
    throw std::runtime_error(std::string("[pinned_chunk_stats] offsets upload failed: ") +
                             cudaGetErrorString(rc));
  }
  auto const offsets_span =
    cudf::device_span<cudf::size_type const>{offsets.data(), offsets.size()};

  auto const min_agg = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
  auto const max_agg = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();

  for (std::size_t i = 0; i < n_columns; ++i) {
    auto const& col     = chunk.column(static_cast<cudf::size_type>(i));
    auto const& type    = column_types[i];
    auto const expected = expected_cudf_type(type);
    if (!expected || col.type().id() != *expected) { continue; }  // outside allowlist
    if (col.size() == 0 || col.null_count() == col.size()) { continue; }

    // EXCLUDE so a group with some nulls still reduces over its valid rows; the output element
    // is null only for an all-null group, which leaves that cell absent (and so never pruning).
    // CUDA failures propagate, as in the whole-chunk capture.
    auto const mins = cudf::segmented_reduce(
      col, offsets_span, *min_agg, col.type(), cudf::null_policy::EXCLUDE, stream, mr);
    auto const maxs = cudf::segmented_reduce(
      col, offsets_span, *max_agg, col.type(), cudf::null_policy::EXCLUDE, stream, mr);
    if (!mins || !maxs) { continue; }

    auto const min_values = reduction_column_to_values(mins->view(), stream);
    auto const max_values = reduction_column_to_values(maxs->view(), stream);
    if (min_values.size() != n_groups || max_values.size() != n_groups) { continue; }

    // Deliberately the same precision as the coarse capture: a column-level fact, not a
    // per-group count. See the header.
    bool const column_has_no_nulls = col.null_count() == 0;
    for (std::size_t g = 0; g < n_groups; ++g) {
      if (min_values[g].IsNull() || max_values[g].IsNull()) { continue; }
      auto group_stats = duckdb::NumericStats::CreateUnknown(type);
      duckdb::NumericStats::SetMin(group_stats, min_values[g]);
      duckdb::NumericStats::SetMax(group_stats, max_values[g]);
      if (column_has_no_nulls) {
        group_stats.Set(duckdb::StatsInfo::CANNOT_HAVE_NULL_VALUES);
      } else {
        group_stats.SetHasNull();
      }
      out.groups[g][i] = group_stats.ToUnique();
    }
  }
  return out;
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
