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

#include <cudf/version_config.hpp>
#define CUDF_VERSION_NUM (CUDF_VERSION_MAJOR * 100 + CUDF_VERSION_MINOR)

#include "helper/logical_type.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <duckdb/common/exception.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/common/vector.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sirius {

/**
 * @brief Apply a boolean retention mask using the cuDF 26.08 API.
 */
inline std::unique_ptr<cudf::table> apply_retention_mask(cudf::table_view const& input,
                                                         cudf::column_view const& retention_mask,
                                                         ::cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
#if CUDF_VERSION_NUM >= 2610
  return cudf::apply_retention_mask(input, retention_mask, stream, mr);
#else
  return cudf::apply_boolean_mask(input, retention_mask, stream, mr);
#endif
}

/**
 * @brief Return @p col if it has type @p type, otherwise a copy of it cast to @p type.
 *
 * @param col The column, consumed.
 * @param type The type of the returned column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the cast column.
 *
 * @return @p col itself if it already has type @p type, otherwise a new column cast with
 *         `cudf::cast`.
 */
inline std::unique_ptr<cudf::column> cast_if_needed(std::unique_ptr<cudf::column> col,
                                                    cudf::data_type type,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  if (col->type() == type) { return col; }
  return cudf::cast(col->view(), type, stream, mr);
}

/**
 * @brief Make a column of @p num_rows NULLs with the type of @p like.
 *
 * Works for every type, including strings, decimals and nested types, whose children are
 * reproduced from @p like.
 *
 * @param like The column whose type the result has. Its values are not read.
 * @param num_rows The number of rows of the result.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the result.
 *
 * @return A column of @p num_rows rows that are all NULL.
 */
inline std::unique_ptr<cudf::column> make_null_column(cudf::column_view const& like,
                                                      cudf::size_type num_rows,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  if (cudf::is_fixed_width(like.type())) {
    return cudf::make_fixed_width_column(
      like.type(), num_rows, cudf::mask_state::ALL_NULL, stream, mr);
  }
  // Every index of the gather map is out of bounds, so NULLIFY makes every row NULL.
  auto const out_of_bounds = cudf::numeric_scalar<cudf::size_type>(like.size(), true, stream, mr);
  auto gather_map          = cudf::make_column_from_scalar(out_of_bounds, num_rows, stream, mr);
  auto gathered            = cudf::gather(
    cudf::table_view({like}), gather_map->view(), cudf::out_of_bounds_policy::NULLIFY, stream, mr);
  return std::move(gathered->release()[0]);
}

/**
 * @brief Make a column of @p num_rows rows that all hold @p value.
 *
 * @tparam T A numeric type with a `cudf::numeric_scalar`, which sets the column type.
 * @param value The value of every row.
 * @param num_rows The number of rows of the result.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the result.
 *
 * @return A column of @p num_rows rows that all hold @p value, with no NULLs.
 */
template <typename T>
std::unique_ptr<cudf::column> make_constant_column(T value,
                                                   cudf::size_type num_rows,
                                                   ::cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto const scalar = cudf::numeric_scalar<T>(value, true, stream, mr);
  return cudf::make_column_from_scalar(scalar, num_rows, stream, mr);
}

/**
 * @brief Estimate the device-memory footprint of a subset of an input table's columns.
 *
 * cuDF exposes an exact allocation size only for an *owned* column/table
 * (`cudf::column::alloc_size`), not for a `column_view`. So when we only hold views into a
 * subset of a table (e.g. the zero-copy passthrough columns of a projection) we estimate:
 *   - Fixed-width columns (numeric/temporal/decimal/bool): counted exactly as
 *     `size * size_of(type)` plus the validity bitmask when nullable.
 *   - Variable-width columns (STRING/LIST/STRUCT/...): their true size is not derivable from a
 *     view, so each is attributed the *average* variable-width column size — computed by
 *     subtracting the exact fixed-width total from the whole-table @p total_input_bytes and
 *     dividing by the number of variable-width columns.
 *
 * Duplicate indices are counted once (the same device buffers back every reference to a column).
 *
 * @param input              The full input table_view (provides per-column type, nullability,
 * rows).
 * @param referenced_indices Indices into @p input of the columns to size (duplicates ignored).
 * @param total_input_bytes  Total allocated size of the whole input (e.g. `get_size_in_bytes()`).
 * @return Estimated bytes occupied by the distinct referenced columns.
 */
inline std::size_t estimate_referenced_column_bytes(
  const cudf::table_view& input,
  const std::vector<cudf::size_type>& referenced_indices,
  std::size_t total_input_bytes)
{
  auto const fixed_width_bytes = [](const cudf::column_view& col) -> std::size_t {
    std::size_t bytes = static_cast<std::size_t>(col.size()) * cudf::size_of(col.type());
    if (col.nullable()) { bytes += cudf::bitmask_allocation_size_bytes(col.size()); }
    return bytes;
  };

  // Split the whole input into fixed-width (exactly sizable) vs. variable-width columns.
  std::size_t fixed_total  = 0;
  std::size_t num_variable = 0;
  for (const auto& col : input) {
    if (cudf::is_fixed_width(col.type())) {
      fixed_total += fixed_width_bytes(col);
    } else {
      ++num_variable;
    }
  }

  // Whatever the fixed-width columns don't account for belongs to the variable-width columns;
  // spread it evenly across them as the per-column average.
  std::size_t const variable_total =
    (total_input_bytes > fixed_total) ? (total_input_bytes - fixed_total) : 0;
  std::size_t const avg_variable = (num_variable > 0) ? variable_total / num_variable : 0;

  // Count each distinct referenced column once.
  std::vector<cudf::size_type> distinct(referenced_indices.begin(), referenced_indices.end());
  std::sort(distinct.begin(), distinct.end());
  distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());

  std::size_t estimate = 0;
  for (auto const idx : distinct) {
    const auto& col = input.column(idx);
    estimate += cudf::is_fixed_width(col.type()) ? fixed_width_bytes(col) : avg_variable;
  }
  return estimate;
}

/**
 * @brief Map a Sirius logical type to cuDF, or return `std::nullopt` if unsupported.
 *
 * Nested types map only their top-level carrier, without child metadata.
 */
[[nodiscard]] inline std::optional<cudf::data_type> get_cudf_type(const logical_type& t)
{
  switch (t.id()) {
    case type_id::TINYINT: return cudf::data_type(cudf::type_id::INT8);
    case type_id::SMALLINT: return cudf::data_type(cudf::type_id::INT16);
    case type_id::INTEGER: return cudf::data_type(cudf::type_id::INT32);
    case type_id::BIGINT: return cudf::data_type(cudf::type_id::INT64);
    case type_id::HUGEINT:
      // FIXME: Unsafe 128-to-64-bit narrowing can corrupt values outside the INT64 range.
      return cudf::data_type(cudf::type_id::INT64);
    case type_id::UTINYINT: return cudf::data_type(cudf::type_id::UINT8);
    case type_id::USMALLINT: return cudf::data_type(cudf::type_id::UINT16);
    case type_id::UINTEGER: return cudf::data_type(cudf::type_id::UINT32);
    case type_id::UBIGINT: return cudf::data_type(cudf::type_id::UINT64);
    case type_id::UHUGEINT:
      // FIXME: Unsafe 128-to-64-bit narrowing can corrupt values outside the UINT64 range.
      return cudf::data_type(cudf::type_id::UINT64);
    case type_id::FLOAT: return cudf::data_type(cudf::type_id::FLOAT32);
    case type_id::DOUBLE: return cudf::data_type(cudf::type_id::FLOAT64);
    case type_id::BOOLEAN: return cudf::data_type(cudf::type_id::BOOL8);
    case type_id::DATE: return cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
    case type_id::TIMESTAMP_SEC: return cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS);
    case type_id::TIMESTAMP_MS: return cudf::data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
    case type_id::TIMESTAMP: return cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS);
    case type_id::TIMESTAMP_NS: return cudf::data_type(cudf::type_id::TIMESTAMP_NANOSECONDS);
    case type_id::VARCHAR: return cudf::data_type(cudf::type_id::STRING);
    case type_id::STRUCT: return cudf::data_type(cudf::type_id::STRUCT);
    case type_id::ARRAY: return cudf::data_type(cudf::type_id::LIST);
    // cudf::data_type has no child metadata; id-only is enough for these comparisons.
    // MAP rides the LIST placeholder.
    case type_id::LIST: return cudf::data_type(cudf::type_id::LIST);
    case type_id::SQLNULL:
    case type_id::INVALID: return std::nullopt;
    case type_id::DECIMAL: {
      // cuDF uses negative scale convention; precision determines the storage width.
      auto const neg_scale = static_cast<int32_t>(-t.decimal_scale());
      if (t.decimal_precision() <= logical_type::decimal_max_precision_int16) return std::nullopt;
      if (t.decimal_precision() <= logical_type::decimal_max_precision_int32)
        return cudf::data_type(cudf::type_id::DECIMAL32, neg_scale);
      if (t.decimal_precision() <= logical_type::decimal_max_precision_int64)
        return cudf::data_type(cudf::type_id::DECIMAL64, neg_scale);
      return cudf::data_type(cudf::type_id::DECIMAL128, neg_scale);
    }
    default: return std::nullopt;
  }
}

/**
 * @brief Map a Sirius logical type to cuDF, throwing `InvalidInputException` if unsupported.
 */
[[nodiscard]] inline cudf::data_type require_cudf_type(const logical_type& t)
{
  auto const type = get_cudf_type(t);
  if (!type) {
    throw duckdb::InvalidInputException("Type %s cannot be mapped to cuDF", t.to_string());
  }
  return *type;
}

/**
 * @brief Convert a non-null DuckDB value to a cuDF scalar.
 *
 * The value must have type @p t. DECIMAL128 and nested types return STRING using `val.ToString()`;
 * unmappable types throw `InvalidInputException`.
 */
[[nodiscard]] inline std::unique_ptr<cudf::scalar> value_to_cudf_scalar(duckdb::Value const& val,
                                                                        logical_type const& t,
                                                                        ::cuda::stream_ref stream)
{
  auto const cudf_type = require_cudf_type(t);

  auto make_numeric = [&](auto value) {
    return std::make_unique<cudf::numeric_scalar<decltype(value)>>(value, true, stream);
  };
  switch (cudf_type.id()) {
    case cudf::type_id::INT8: return make_numeric(val.GetValue<int8_t>());
    case cudf::type_id::INT16: return make_numeric(val.GetValue<int16_t>());
    case cudf::type_id::INT32:
    case cudf::type_id::TIMESTAMP_DAYS: return make_numeric(val.GetValue<int32_t>());
    case cudf::type_id::INT64:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return make_numeric(val.GetValue<int64_t>());
    case cudf::type_id::UINT8: return make_numeric(val.GetValue<uint8_t>());
    case cudf::type_id::UINT16: return make_numeric(val.GetValue<uint16_t>());
    case cudf::type_id::UINT32: return make_numeric(val.GetValue<uint32_t>());
    case cudf::type_id::UINT64: return make_numeric(val.GetValue<uint64_t>());
    case cudf::type_id::FLOAT32: return make_numeric(val.GetValue<float>());
    case cudf::type_id::FLOAT64: return make_numeric(val.GetValue<double>());
    case cudf::type_id::BOOL8: return make_numeric(val.GetValue<bool>());
    case cudf::type_id::STRING:
      return std::make_unique<cudf::string_scalar>(val.GetValue<std::string>(), true, stream);
    case cudf::type_id::DECIMAL32:
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
        val.GetValueUnsafe<typename numeric::decimal32::rep>(),
        numeric::scale_type{-static_cast<int32_t>(t.decimal_scale())},
        true,
        stream);
    case cudf::type_id::DECIMAL64:
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        val.GetValueUnsafe<typename numeric::decimal64::rep>(),
        numeric::scale_type{-static_cast<int32_t>(t.decimal_scale())},
        true,
        stream);
    default: return std::make_unique<cudf::string_scalar>(val.ToString(), true, stream);
  }
}

/** @brief Assemble an owned table from column pointers; `cudf::table` has no variadic constructor.
 */
template <typename... Columns>
  requires(sizeof...(Columns) > 0 &&
           (std::convertible_to<Columns &&, std::unique_ptr<cudf::column>> && ...))
[[nodiscard]] inline std::unique_ptr<cudf::table> make_table(Columns&&... columns)
{
  std::vector<std::unique_ptr<cudf::column>> owned;
  owned.reserve(sizeof...(Columns));
  (owned.push_back(std::forward<Columns>(columns)), ...);
  return std::make_unique<cudf::table>(std::move(owned));
}

/**
 * @brief Build a 0-row cudf table with one column per logical type.
 *
 * Column order and types mirror @p types (each via get_cudf_type). An ARRAY entry becomes an empty
 * cuDF LIST column of its fixed-width element type (`cudf::make_empty_column` rejects nested
 * types). Used wherever an empty but schema-bearing table is needed — e.g. an all-pruned GPU-values
 * source, or the synthesized missing side of a join against an empty table.
 */
inline std::unique_ptr<cudf::table> make_empty_table(const duckdb::vector<logical_type>& types)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(types.size());
  for (auto const& t : types) {
    columns.push_back(t.is_array()
                        ? cudf::make_empty_lists_column(require_cudf_type(t.array_child()))
                        : cudf::make_empty_column(require_cudf_type(t)));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Build a zero-row cuDF table with one column per cuDF type
 *
 * The carrier-exact counterpart of the logical-type overload, for callers holding an operator's
 * `physical_types` sidecar: the result reproduces those carriers instead of re-deriving native
 * ones. Every entry must be non-nested: an id-only `cudf::data_type` carries no child type, so a
 * nested carrier cannot be synthesized here.
 *
 * @throws duckdb::InvalidInputException on a nested entry — use the logical-type overload, which
 *         carries element types.
 */
inline std::unique_ptr<cudf::table> make_empty_table(const std::vector<cudf::data_type>& types)
{
  for (auto const& t : types) {
    if (cudf::is_nested(t)) {
      throw duckdb::InvalidInputException(
        "sirius::make_empty_table: carrier-exact make_empty_table cannot synthesize nested type "
        "%s; use the logical-type overload, which carries element types",
        cudf::type_to_name(t));
    }
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(types.size());
  for (auto const& t : types) {
    columns.push_back(cudf::make_empty_column(t));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

/** @brief Map a DuckDB logical type to cuDF, or return `std::nullopt` if unsupported. */
[[nodiscard]] inline std::optional<cudf::data_type> get_cudf_type(
  const duckdb::LogicalType& logical_type)
{
  using duckdb::DecimalType;
  using duckdb::LogicalTypeId;
  using duckdb::PhysicalType;
  switch (logical_type.id()) {
    case LogicalTypeId::TINYINT: return cudf::data_type(cudf::type_id::INT8);
    case LogicalTypeId::SMALLINT: return cudf::data_type(cudf::type_id::INT16);
    case LogicalTypeId::INTEGER: return cudf::data_type(cudf::type_id::INT32);
    case LogicalTypeId::BIGINT:
    case LogicalTypeId::HUGEINT:  // FIXME: unsafe conversion from duckdb HugeInt to cudf Int64,
                                  // since cudf does not support Int128.
      return cudf::data_type(cudf::type_id::INT64);
    case LogicalTypeId::UTINYINT: return cudf::data_type(cudf::type_id::UINT8);
    case LogicalTypeId::USMALLINT: return cudf::data_type(cudf::type_id::UINT16);
    case LogicalTypeId::UINTEGER: return cudf::data_type(cudf::type_id::UINT32);
    case LogicalTypeId::UBIGINT:
    case LogicalTypeId::UHUGEINT: return cudf::data_type(cudf::type_id::UINT64); ;
    case LogicalTypeId::FLOAT: return cudf::data_type(cudf::type_id::FLOAT32);
    case LogicalTypeId::DOUBLE: return cudf::data_type(cudf::type_id::FLOAT64);
    case LogicalTypeId::BOOLEAN: return cudf::data_type(cudf::type_id::BOOL8);
    case LogicalTypeId::DATE: return cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
    case LogicalTypeId::TIMESTAMP_SEC: return cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS);
    case LogicalTypeId::TIMESTAMP_MS: return cudf::data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
    case LogicalTypeId::TIMESTAMP: return cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS);
    case LogicalTypeId::TIMESTAMP_NS: return cudf::data_type(cudf::type_id::TIMESTAMP_NANOSECONDS);
    case LogicalTypeId::VARCHAR: return cudf::data_type(cudf::type_id::STRING);
    case LogicalTypeId::TUPLE:
    case LogicalTypeId::STRUCT: return cudf::data_type(cudf::type_id::STRUCT);
    case LogicalTypeId::ARRAY: return cudf::data_type(cudf::type_id::LIST);
    case LogicalTypeId::DECIMAL: {
      switch (logical_type.InternalType()) {
        case PhysicalType::INT32:
          // cudf decimal type uses negative scale, same for below
          return cudf::data_type(cudf::type_id::DECIMAL32, -DecimalType::GetScale(logical_type));
        case PhysicalType::INT64:
          return cudf::data_type(cudf::type_id::DECIMAL64, -DecimalType::GetScale(logical_type));
        case PhysicalType::INT128:
          return cudf::data_type(cudf::type_id::DECIMAL128, -DecimalType::GetScale(logical_type));
        default: return std::nullopt;
      }
    }
    default: return std::nullopt;
  }
}

}  // namespace sirius
