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
#include <cstring>
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

namespace {

/// Raw 8-byte carrier of a Value whose type is in the zone-map allowlist. Unsigned values are
/// stored as their bit pattern and compared via is_unsigned; see packed_column_bounds.
std::optional<std::int64_t> value_carrier(duckdb::Value const& v)
{
  if (v.IsNull()) { return std::nullopt; }
  switch (v.type().id()) {
    case duckdb::LogicalTypeId::TINYINT: return v.GetValue<std::int8_t>();
    case duckdb::LogicalTypeId::SMALLINT: return v.GetValue<std::int16_t>();
    case duckdb::LogicalTypeId::INTEGER: return v.GetValue<std::int32_t>();
    case duckdb::LogicalTypeId::BIGINT: return v.GetValue<std::int64_t>();
    case duckdb::LogicalTypeId::UTINYINT:
      return static_cast<std::int64_t>(v.GetValue<std::uint8_t>());
    case duckdb::LogicalTypeId::USMALLINT:
      return static_cast<std::int64_t>(v.GetValue<std::uint16_t>());
    case duckdb::LogicalTypeId::UINTEGER:
      return static_cast<std::int64_t>(v.GetValue<std::uint32_t>());
    case duckdb::LogicalTypeId::UBIGINT:
      return static_cast<std::int64_t>(v.GetValue<std::uint64_t>());
    case duckdb::LogicalTypeId::DATE:
      return static_cast<std::int64_t>(v.GetValue<duckdb::date_t>().days);
    case duckdb::LogicalTypeId::TIMESTAMP:
      return static_cast<std::int64_t>(v.GetValue<duckdb::timestamp_t>().value);
    default: return std::nullopt;
  }
}

bool type_is_unsigned(duckdb::LogicalType const& t)
{
  switch (t.id()) {
    case duckdb::LogicalTypeId::UTINYINT:
    case duckdb::LogicalTypeId::USMALLINT:
    case duckdb::LogicalTypeId::UINTEGER:
    case duckdb::LogicalTypeId::UBIGINT: return true;
    default: return false;
  }
}

/// Three-way ordering on carriers, honoring the column's signedness.
int carrier_cmp(std::int64_t a, std::int64_t b, bool is_unsigned) noexcept
{
  if (is_unsigned) {
    auto const ua = static_cast<std::uint64_t>(a);
    auto const ub = static_cast<std::uint64_t>(b);
    return ua < ub ? -1 : (ua > ub ? 1 : 0);
  }
  return a < b ? -1 : (a > b ? 1 : 0);
}

}  // namespace

packed_column_bounds group_bounds_arena::cell(std::size_t column, std::size_t chunk) const noexcept
{
  packed_column_bounds out;
  if (column >= _n_columns || chunk >= _n_chunks) { return out; }
  auto const& sl = _slices[column * _n_chunks + chunk];
  if (sl.count == 0) { return out; }
  out.type                = _types[column];
  out.is_unsigned         = _is_unsigned[column];
  out.column_has_no_nulls = _column_has_no_nulls[column];
  out.mins                = std::span<std::int64_t const>{_storage.data() + sl.offset, sl.count};
  out.maxs  = std::span<std::int64_t const>{_storage.data() + sl.offset + sl.count, sl.count};
  out.valid = std::span<std::uint8_t const>{_valid.data() + sl.valid_offset, sl.count};
  return out;
}

group_bounds_arena group_bounds_arena::select_columns(
  std::span<const std::size_t> columns) const
{
  group_bounds_arena out;
  if (columns.empty() || _n_chunks == 0) { return out; }
  for (auto const c : columns) {
    if (c >= _n_columns) { return out; }
  }

  out._group_rows = _group_rows;
  out._n_columns  = columns.size();
  out._n_chunks   = _n_chunks;
  out._slices.resize(columns.size() * _n_chunks);
  out._types.reserve(columns.size());
  out._is_unsigned.reserve(columns.size());
  out._column_has_no_nulls.reserve(columns.size());

  for (std::size_t i = 0; i < columns.size(); ++i) {
    auto const c = columns[i];
    out._types.push_back(_types[c]);
    out._is_unsigned.push_back(_is_unsigned[c]);
    out._column_has_no_nulls.push_back(_column_has_no_nulls[c]);
    for (std::size_t chunk = 0; chunk < _n_chunks; ++chunk) {
      auto const& src = _slices[c * _n_chunks + chunk];
      auto& dst       = out._slices[i * _n_chunks + chunk];
      dst.count       = src.count;
      if (src.count == 0) { continue; }
      // Mins and maxs are adjacent in the source and stay adjacent here; the copy is what makes
      // the result standalone, so the subset outlives the arena it came from.
      dst.offset = out._storage.size();
      out._storage.insert(out._storage.end(),
                          _storage.begin() + static_cast<std::ptrdiff_t>(src.offset),
                          _storage.begin() + static_cast<std::ptrdiff_t>(src.offset + 2 * src.count));
      dst.valid_offset = out._valid.size();
      out._valid.insert(out._valid.end(),
                        _valid.begin() + static_cast<std::ptrdiff_t>(src.valid_offset),
                        _valid.begin() + static_cast<std::ptrdiff_t>(src.valid_offset + src.count));
    }
  }
  return out;
}

duckdb::LogicalType group_bounds_arena::column_type(std::size_t column) const
{
  if (column >= _types.size()) { return duckdb::LogicalType(duckdb::LogicalTypeId::SQLNULL); }
  return _types[column];
}

std::size_t group_bounds_arena::groups_in_chunk(std::size_t chunk) const noexcept
{
  if (chunk >= _n_chunks || _n_columns == 0) { return 0; }
  return _slices[chunk].count;  // column 0's slice for this chunk
}

namespace {

// Wire tags for the types compute_pinned_group_stats can capture. Stable: do not renumber.
// Anything else packs as `absent`, which prunes nothing rather than prunes wrongly.
enum class packed_type : std::uint8_t {
  absent = 0,
  i8,
  i16,
  i32,
  i64,
  u8,
  u16,
  u32,
  u64,
  date,
  timestamp
};

packed_type to_packed(duckdb::LogicalType const& t)
{
  switch (t.id()) {
    case duckdb::LogicalTypeId::TINYINT: return packed_type::i8;
    case duckdb::LogicalTypeId::SMALLINT: return packed_type::i16;
    case duckdb::LogicalTypeId::INTEGER: return packed_type::i32;
    case duckdb::LogicalTypeId::BIGINT: return packed_type::i64;
    case duckdb::LogicalTypeId::UTINYINT: return packed_type::u8;
    case duckdb::LogicalTypeId::USMALLINT: return packed_type::u16;
    case duckdb::LogicalTypeId::UINTEGER: return packed_type::u32;
    case duckdb::LogicalTypeId::UBIGINT: return packed_type::u64;
    case duckdb::LogicalTypeId::DATE: return packed_type::date;
    case duckdb::LogicalTypeId::TIMESTAMP: return packed_type::timestamp;
    default: return packed_type::absent;
  }
}

duckdb::LogicalType from_packed(packed_type p)
{
  switch (p) {
    case packed_type::i8: return duckdb::LogicalType(duckdb::LogicalTypeId::TINYINT);
    case packed_type::i16: return duckdb::LogicalType(duckdb::LogicalTypeId::SMALLINT);
    case packed_type::i32: return duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    case packed_type::i64: return duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT);
    case packed_type::u8: return duckdb::LogicalType(duckdb::LogicalTypeId::UTINYINT);
    case packed_type::u16: return duckdb::LogicalType(duckdb::LogicalTypeId::USMALLINT);
    case packed_type::u32: return duckdb::LogicalType(duckdb::LogicalTypeId::UINTEGER);
    case packed_type::u64: return duckdb::LogicalType(duckdb::LogicalTypeId::UBIGINT);
    case packed_type::date: return duckdb::LogicalType(duckdb::LogicalTypeId::DATE);
    case packed_type::timestamp: return duckdb::LogicalType(duckdb::LogicalTypeId::TIMESTAMP);
    default: return duckdb::LogicalType(duckdb::LogicalTypeId::SQLNULL);
  }
}

// Explicit little-endian, matching the .hpln header's own push_le/read_le convention. A memcpy of
// the host representation agrees on every target this runs on and silently disagrees on a
// big-endian one -- and it disagrees by producing wrong BOUNDS, which prune wrong rows rather
// than failing.
template <typename T>
void put(std::vector<std::uint8_t>& out, T v)
{
  auto u = static_cast<std::make_unsigned_t<T>>(v);
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    out.push_back(static_cast<std::uint8_t>((u >> (8 * i)) & 0xFF));
  }
}

template <typename T>
bool take(std::span<const std::uint8_t>& in, T& v)
{
  if (in.size() < sizeof(T)) return false;
  std::make_unsigned_t<T> u = 0;
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    u |= static_cast<std::make_unsigned_t<T>>(in[i]) << (8 * i);
  }
  v  = static_cast<T>(u);
  in = in.subspan(sizeof(T));
  return true;
}

constexpr std::uint16_t kZoneMapWireVersion = 1;

}  // namespace

std::vector<std::uint8_t> group_bounds_arena::pack() const
{
  std::vector<std::uint8_t> out;
  put(out, kZoneMapWireVersion);
  put(out, static_cast<std::uint32_t>(_group_rows));
  put(out, static_cast<std::uint16_t>(_n_columns));
  put(out, static_cast<std::uint32_t>(_n_chunks));
  for (std::size_t c = 0; c < _n_columns; ++c) {
    auto const t = c < _types.size() ? to_packed(_types[c]) : packed_type::absent;
    put(out, static_cast<std::uint8_t>(t));
    std::uint8_t flags = 0;
    if (c < _is_unsigned.size() && _is_unsigned[c]) flags |= 1u;
    if (c < _column_has_no_nulls.size() && _column_has_no_nulls[c]) flags |= 2u;
    put(out, flags);
    for (std::size_t k = 0; k < _n_chunks; ++k) {
      auto const& sl = _slices[c * _n_chunks + k];
      auto const n   = (t == packed_type::absent) ? std::size_t{0} : sl.count;
      put(out, static_cast<std::uint32_t>(n));
      if (n == 0) continue;
      auto const* mins = _storage.data() + sl.offset;
      for (std::size_t k2 = 0; k2 < 2 * n; ++k2) {
        put(out, mins[k2]);
      }  // mins then maxs
      out.insert(out.end(), _valid.data() + sl.valid_offset, _valid.data() + sl.valid_offset + n);
    }
  }
  return out;
}

group_bounds_arena group_bounds_arena::unpack(std::span<const std::uint8_t> bytes,
                                              std::string* error)
{
  auto fail = [&](char const* why) {
    if (error) *error = why;
    return group_bounds_arena{};
  };
  std::uint16_t version = 0, n_columns = 0;
  std::uint32_t group_rows = 0, n_chunks = 0;
  if (!take(bytes, version) || !take(bytes, group_rows) || !take(bytes, n_columns) ||
      !take(bytes, n_chunks)) {
    return fail("zone-map segment: truncated preamble");
  }
  if (version != kZoneMapWireVersion) return fail("zone-map segment: unsupported version");

  group_bounds_arena a;
  a._group_rows = group_rows;
  a._n_columns  = n_columns;
  a._n_chunks   = n_chunks;
  a._slices.assign(static_cast<std::size_t>(n_columns) * n_chunks, slice{});
  a._types.reserve(n_columns);
  for (std::size_t c = 0; c < n_columns; ++c) {
    std::uint8_t tag = 0, flags = 0;
    if (!take(bytes, tag) || !take(bytes, flags)) return fail("zone-map segment: truncated column");
    a._types.push_back(from_packed(static_cast<packed_type>(tag)));
    a._is_unsigned.push_back((flags & 1u) != 0);
    a._column_has_no_nulls.push_back((flags & 2u) != 0);
    for (std::size_t k = 0; k < n_chunks; ++k) {
      std::uint32_t n = 0;
      if (!take(bytes, n)) return fail("zone-map segment: truncated group count");
      if (n == 0) continue;
      if (bytes.size() < n * (2 * sizeof(std::int64_t) + 1)) {
        return fail("zone-map segment: truncated bounds");
      }
      auto& sl        = a._slices[c * n_chunks + k];
      sl.offset       = a._storage.size();
      sl.valid_offset = a._valid.size();
      sl.count        = n;
      a._storage.resize(sl.offset + 2 * n);
      for (std::size_t k2 = 0; k2 < 2 * n; ++k2) {
        if (!take(bytes, a._storage[sl.offset + k2])) {
          return fail("zone-map segment: truncated bounds");
        }
      }
      a._valid.insert(a._valid.end(), bytes.begin(), bytes.begin() + n);
      bytes = bytes.subspan(n);
    }
  }
  return a;
}

group_bounds_arena group_bounds_arena::from_capture(
  duckdb::vector<duckdb::LogicalType> const& column_types,
  std::vector<chunk_group_stats> const& per_chunk)
{
  group_bounds_arena out;
  if (column_types.empty() || per_chunk.empty()) { return out; }
  auto const n_columns = column_types.size();
  auto const n_chunks  = per_chunk.size();

  // One group_rows for the whole table, and every chunk's cells must agree with its own group
  // count. Anything inconsistent yields an empty arena: no sub-chunk pruning, never a wrong one.
  std::size_t const group_rows = per_chunk.front().group_rows;
  if (group_rows == 0) { return out; }
  std::vector<std::size_t> groups_per_chunk(n_chunks);
  std::size_t total_groups = 0;
  for (std::size_t c = 0; c < n_chunks; ++c) {
    auto const& cs = per_chunk[c];
    if (cs.group_rows != group_rows || cs.groups.empty()) { return out; }
    for (auto const& row : cs.groups) {
      if (row.size() != n_columns) { return out; }
    }
    groups_per_chunk[c] = cs.groups.size();
    total_groups += groups_per_chunk[c];
  }

  out._group_rows = group_rows;
  out._n_columns  = n_columns;
  out._n_chunks   = n_chunks;
  out._types      = column_types;
  out._is_unsigned.resize(n_columns, false);
  out._column_has_no_nulls.resize(n_columns, true);
  out._slices.resize(n_columns * n_chunks);
  // Every column stores a full set of groups, and each (column, chunk) reserves 2 * groups for
  // its mins and maxs back to back.
  out._storage.assign(2 * total_groups * n_columns, 0);
  out._valid.assign(total_groups * n_columns, 0);

  std::size_t cursor       = 0;  // into _storage, advancing by 2 * groups
  std::size_t valid_cursor = 0;  // into _valid, advancing by groups
  for (std::size_t col = 0; col < n_columns; ++col) {
    out._is_unsigned[col] = type_is_unsigned(column_types[col]);
    for (std::size_t chunk = 0; chunk < n_chunks; ++chunk) {
      auto const n_groups                 = groups_per_chunk[chunk];
      out._slices[col * n_chunks + chunk] = {cursor, valid_cursor, n_groups};
      auto const& groups                  = per_chunk[chunk].groups;
      for (std::size_t g = 0; g < n_groups; ++g) {
        auto const* cell = groups[g][col].get();
        if (cell == nullptr) { continue; }  // leaves valid = 0, which never prunes
        auto const lo = value_carrier(duckdb::NumericStats::Min(*cell));
        auto const hi = value_carrier(duckdb::NumericStats::Max(*cell));
        if (!lo || !hi) { continue; }
        out._storage[cursor + g]            = *lo;
        out._storage[cursor + n_groups + g] = *hi;
        out._valid[valid_cursor + g]        = 1;
        // A single cell that admits nulls makes the column's cells all "may have nulls", which is
        // exactly the precision compute_pinned_group_stats captures.
        if (cell->CanHaveNull()) { out._column_has_no_nulls[col] = false; }
      }
      cursor += 2 * n_groups;
      valid_cursor += n_groups;
    }
  }
  return out;
}

namespace {

/// Inverse of @ref value_carrier: the 8-byte carrier back to a Value of @p type. Mirrors that
/// function's allowlist exactly -- a type it cannot carry has no bound to reconstruct, and a
/// mismatched pair here would produce bounds that prune the wrong rows rather than none.
duckdb::Value carrier_to_value(duckdb::LogicalType const& type, std::int64_t raw)
{
  switch (type.id()) {
    case duckdb::LogicalTypeId::TINYINT:
      return duckdb::Value::TINYINT(static_cast<std::int8_t>(raw));
    case duckdb::LogicalTypeId::SMALLINT:
      return duckdb::Value::SMALLINT(static_cast<std::int16_t>(raw));
    case duckdb::LogicalTypeId::INTEGER:
      return duckdb::Value::INTEGER(static_cast<std::int32_t>(raw));
    case duckdb::LogicalTypeId::BIGINT: return duckdb::Value::BIGINT(raw);
    case duckdb::LogicalTypeId::UTINYINT:
      return duckdb::Value::UTINYINT(static_cast<std::uint8_t>(raw));
    case duckdb::LogicalTypeId::USMALLINT:
      return duckdb::Value::USMALLINT(static_cast<std::uint16_t>(raw));
    case duckdb::LogicalTypeId::UINTEGER:
      return duckdb::Value::UINTEGER(static_cast<std::uint32_t>(raw));
    case duckdb::LogicalTypeId::UBIGINT:
      return duckdb::Value::UBIGINT(static_cast<std::uint64_t>(raw));
    case duckdb::LogicalTypeId::DATE:
      return duckdb::Value::DATE(duckdb::date_t{static_cast<std::int32_t>(raw)});
    case duckdb::LogicalTypeId::TIMESTAMP:
      return duckdb::Value::TIMESTAMP(duckdb::timestamp_t{raw});
    default: return duckdb::Value();
  }
}

}  // namespace

void group_bounds_arena::mark_column_nullable(std::size_t column)
{
  if (column < _column_has_no_nulls.size()) { _column_has_no_nulls[column] = false; }
}

pinned_zone_maps chunk_zone_maps_from_group_bounds(group_bounds_arena const& bounds)
{
  auto const n_columns = bounds.column_count();
  auto const n_chunks  = bounds.chunk_count();
  if (bounds.empty() || n_columns == 0 || n_chunks == 0) { return {}; }

  duckdb::vector<duckdb::LogicalType> types;
  types.reserve(n_columns);
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats(n_chunks);
  for (auto& row : chunk_stats) {
    row.resize(n_columns);
  }

  for (std::size_t col = 0; col < n_columns; ++col) {
    auto const type = bounds.column_type(col);
    types.push_back(type);
    bool const uns = type_is_unsigned(type);
    for (std::size_t chunk = 0; chunk < n_chunks; ++chunk) {
      auto const cell = bounds.cell(col, chunk);
      if (cell.empty()) { continue; }
      bool have       = false;
      std::int64_t lo = 0;
      std::int64_t hi = 0;
      for (std::size_t g = 0; g < cell.size(); ++g) {
        if (cell.valid[g] == 0) { continue; }  // absent cell: contributes no bound
        if (!have) {
          lo   = cell.mins[g];
          hi   = cell.maxs[g];
          have = true;
          continue;
        }
        if (carrier_cmp(cell.mins[g], lo, uns) < 0) { lo = cell.mins[g]; }
        if (carrier_cmp(cell.maxs[g], hi, uns) > 0) { hi = cell.maxs[g]; }
      }
      if (!have) { continue; }
      auto const min_value = carrier_to_value(type, lo);
      auto const max_value = carrier_to_value(type, hi);
      if (min_value.IsNull() || max_value.IsNull()) { continue; }

      auto column_stats = duckdb::NumericStats::CreateUnknown(type);
      duckdb::NumericStats::SetMin(column_stats, min_value);
      duckdb::NumericStats::SetMax(column_stats, max_value);
      // The group capture only knows nulls at column granularity, so a chunk of a column that has
      // any null is "may have nulls" -- conservative, and the same precision the fine pass uses.
      if (cell.column_has_no_nulls) {
        column_stats.Set(duckdb::StatsInfo::CANNOT_HAVE_NULL_VALUES);
      } else {
        column_stats.SetHasNull();
      }
      chunk_stats[chunk][col] = column_stats.ToUnique();
    }
  }
  return pinned_zone_maps::from_capture(
    std::move(types), std::move(chunk_stats), n_columns, n_chunks);
}

std::optional<lowered_bound_filter> lowered_bound_filter::lower(
  duckdb::TableFilter const& filter, duckdb::LogicalType const& stats_type)
{
  // Gate on exactly the same allowlist as the BaseStatistics path, so the two can never disagree
  // about WHICH filters are evaluable — only (and verifiably not) about the answer.
  if (!filter_safe_for_stats(filter, stats_type)) { return std::nullopt; }

  lowered_bound_filter out;
  bool const uns = type_is_unsigned(stats_type);
  bool ok        = true;

  // Recursive build; returns the index of the node it appended. Children are appended first and
  // their indices recorded in _children, so evaluation never chases pointers.
  auto const build = [&](auto&& self, duckdb::TableFilter const& f) -> std::uint32_t {
    node n;
    n.is_unsigned = uns;
    switch (f.filter_type) {
      case duckdb::TableFilterType::CONSTANT_COMPARISON: {
        auto const& cf = f.Cast<duckdb::ConstantFilter>();
        auto const c   = value_carrier(cf.constant);
        if (!c) {
          ok = false;
          break;
        }
        n.constant = *c;
        switch (cf.comparison_type) {
          case duckdb::ExpressionType::COMPARE_EQUAL: n.kind = op::cmp_eq; break;
          case duckdb::ExpressionType::COMPARE_NOTEQUAL: n.kind = op::cmp_ne; break;
          case duckdb::ExpressionType::COMPARE_LESSTHAN: n.kind = op::cmp_lt; break;
          case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO: n.kind = op::cmp_le; break;
          case duckdb::ExpressionType::COMPARE_GREATERTHAN: n.kind = op::cmp_gt; break;
          case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO: n.kind = op::cmp_ge; break;
          default: ok = false; break;
        }
        break;
      }
      case duckdb::TableFilterType::IS_NULL: n.kind = op::is_null; break;
      case duckdb::TableFilterType::IS_NOT_NULL: n.kind = op::is_not_null; break;
      case duckdb::TableFilterType::IN_FILTER: {
        auto const& in = f.Cast<duckdb::InFilter>();
        n.kind         = op::in_list;
        n.begin        = static_cast<std::uint32_t>(out._values.size());
        for (auto const& v : in.values) {
          auto const c = value_carrier(v);
          if (!c) {
            ok = false;
            break;
          }
          out._values.push_back(*c);
        }
        n.end = static_cast<std::uint32_t>(out._values.size());
        break;
      }
      case duckdb::TableFilterType::CONJUNCTION_AND:
      case duckdb::TableFilterType::CONJUNCTION_OR: {
        auto const& children = f.filter_type == duckdb::TableFilterType::CONJUNCTION_AND
                                 ? f.Cast<duckdb::ConjunctionAndFilter>().child_filters
                                 : f.Cast<duckdb::ConjunctionOrFilter>().child_filters;
        n.kind = f.filter_type == duckdb::TableFilterType::CONJUNCTION_AND ? op::conj : op::disj;
        std::vector<std::uint32_t> kids;
        kids.reserve(children.size());
        for (auto const& c : children) {
          if (!c) {
            ok = false;
            break;
          }
          kids.push_back(self(self, *c));
        }
        n.begin = static_cast<std::uint32_t>(out._children.size());
        out._children.insert(out._children.end(), kids.begin(), kids.end());
        n.end = static_cast<std::uint32_t>(out._children.size());
        break;
      }
      case duckdb::TableFilterType::OPTIONAL_FILTER: {
        auto const& opt = f.Cast<duckdb::OptionalFilter>();
        if (!opt.child_filter) {
          ok = false;
          break;
        }
        // An optional wrapping one child behaves as a single-child conjunction.
        auto const kid = self(self, *opt.child_filter);
        n.kind         = op::conj;
        n.begin        = static_cast<std::uint32_t>(out._children.size());
        out._children.push_back(kid);
        n.end = static_cast<std::uint32_t>(out._children.size());
        break;
      }
      default: ok = false; break;
    }
    out._nodes.push_back(n);
    return static_cast<std::uint32_t>(out._nodes.size() - 1);
  };

  auto const root = build(build, filter);
  if (!ok) { return std::nullopt; }
  // Evaluation starts from the root, which the post-order build leaves last.
  if (root + 1 != out._nodes.size()) { return std::nullopt; }
  std::swap(out._nodes[0], out._nodes[root]);
  // Re-point any child references to the two swapped slots.
  for (auto& c : out._children) {
    if (c == 0) {
      c = static_cast<std::uint32_t>(root);
    } else if (c == root) {
      c = 0;
    }
  }
  return out;
}

bool lowered_bound_filter::eval(std::uint32_t node_index,
                                std::int64_t min,
                                std::int64_t max,
                                bool has_null,
                                bool all_null) const noexcept
{
  auto const& n      = _nodes[node_index];
  auto const cmp_min = [&](std::int64_t c) { return carrier_cmp(min, c, n.is_unsigned); };
  auto const cmp_max = [&](std::int64_t c) { return carrier_cmp(max, c, n.is_unsigned); };
  switch (n.kind) {
    // A value range proof says nothing about rows that are NULL, but a NULL never satisfies a
    // comparison either, so "no non-null row can match" is enough to prune.
    case op::cmp_eq: return cmp_min(n.constant) > 0 || cmp_max(n.constant) < 0;
    case op::cmp_ne: return cmp_min(n.constant) == 0 && cmp_max(n.constant) == 0;
    case op::cmp_lt: return cmp_min(n.constant) >= 0;
    case op::cmp_le: return cmp_min(n.constant) > 0;
    case op::cmp_gt: return cmp_max(n.constant) <= 0;
    case op::cmp_ge: return cmp_max(n.constant) < 0;
    case op::in_list: {
      for (std::uint32_t i = n.begin; i < n.end; ++i) {
        if (cmp_min(_values[i]) <= 0 && cmp_max(_values[i]) >= 0) { return false; }
      }
      return true;
    }
    case op::is_null: return !has_null;
    case op::is_not_null: return all_null;
    case op::conj: {
      for (std::uint32_t i = n.begin; i < n.end; ++i) {
        if (eval(_children[i], min, max, has_null, all_null)) { return true; }
      }
      return false;
    }
    case op::disj: {
      for (std::uint32_t i = n.begin; i < n.end; ++i) {
        if (!eval(_children[i], min, max, has_null, all_null)) { return false; }
      }
      return n.begin != n.end;
    }
  }
  return false;
}

bool lowered_bound_filter::provably_empty(std::int64_t min,
                                          std::int64_t max,
                                          bool has_null,
                                          bool all_null) const noexcept
{
  if (_nodes.empty()) { return false; }
  return eval(0, min, max, has_null, all_null);
}

void lowered_bound_filter::select_survivors(packed_column_bounds const& bounds,
                                            std::vector<std::uint32_t>& survivors) const
{
  auto const n = bounds.size();
  survivors.clear();
  survivors.reserve(n);
  bool const has_null = !bounds.column_has_no_nulls;
  for (std::size_t i = 0; i < n; ++i) {
    // An absent cell never prunes, exactly as a null BaseStatistics does not.
    if (bounds.valid[i] == 0 || !provably_empty(bounds.mins[i], bounds.maxs[i], has_null, false)) {
      survivors.push_back(static_cast<std::uint32_t>(i));
    }
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
