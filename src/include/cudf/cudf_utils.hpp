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

#include <cudf/table/table.hpp>
#if CUDF_VERSION_NUM > 2504
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#else
#include <cudf/join.hpp>
#endif
#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#if CUDF_VERSION_NUM >= 2604
#include <cudf/reduction/distinct_count.hpp>
#endif
#include "helper/logical_type.hpp"
#include "sirius/exception.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <duckdb/common/exception.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/types/value.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sirius {

inline bool IsCudfTypeDecimal(const cudf::data_type& type)
{
  return type.id() == cudf::type_id::DECIMAL32 || type.id() == cudf::type_id::DECIMAL64 ||
         type.id() == cudf::type_id::DECIMAL128;
}

inline int GetCudfDecimalTypeSize(const cudf::data_type& type)
{
  if (type.id() == cudf::type_id::DECIMAL32) { return sizeof(int32_t); }
  if (type.id() == cudf::type_id::DECIMAL64) { return sizeof(int64_t); }
  if (type.id() == cudf::type_id::DECIMAL128) { return sizeof(__int128_t); }
  throw internal_exception("Non decimal cudf type called in GetCudfDecimalTypeSize: " +
                           std::to_string(static_cast<int>(type.id())));
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
 * @brief Map a sirius::logical_type to its corresponding cudf::data_type.
 *
 * Mirrors duckdb::GetCudfType() but operates on the Sirius-native type system.
 * This overload is used by GPU operators that have been decoupled from DuckDB.
 *
 * @throws duckdb::InvalidInputException if the type is unsupported.
 */
inline cudf::data_type get_cudf_type(const logical_type& t)
{
  switch (t.id()) {
    case type_id::TINYINT: return cudf::data_type(cudf::type_id::INT8);
    case type_id::SMALLINT: return cudf::data_type(cudf::type_id::INT16);
    case type_id::INTEGER: return cudf::data_type(cudf::type_id::INT32);
    case type_id::BIGINT: return cudf::data_type(cudf::type_id::INT64);
    case type_id::HUGEINT:
      // FIXME: unsafe 128→64-bit narrowing: DuckDB HUGEINT is INT128 but cuDF has no INT128.
      // Values outside INT64 range are silently corrupted. Matches legacy duckdb::GetCudfType().
      return cudf::data_type(cudf::type_id::INT64);
    case type_id::UTINYINT: return cudf::data_type(cudf::type_id::UINT8);
    case type_id::USMALLINT: return cudf::data_type(cudf::type_id::UINT16);
    case type_id::UINTEGER: return cudf::data_type(cudf::type_id::UINT32);
    case type_id::UBIGINT: return cudf::data_type(cudf::type_id::UINT64);
    case type_id::UHUGEINT:
      // FIXME: unsafe 128→64-bit narrowing: DuckDB UHUGEINT is UINT128 but cuDF has no UINT128.
      // Values outside UINT64 range are silently corrupted. Matches legacy duckdb::GetCudfType().
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
    case type_id::INVALID:
      throw duckdb::InvalidInputException("sirius::get_cudf_type: Type %s cannot be mapped to cuDF",
                                          t.to_string());
    case type_id::DECIMAL: {
      // cuDF uses negative scale convention; precision determines the storage width.
      auto const neg_scale = static_cast<int32_t>(-t.decimal_scale());
      if (t.decimal_precision() <= logical_type::decimal_max_precision_int16)
        throw duckdb::InvalidInputException(
          "sirius::get_cudf_type: DECIMAL with precision <= %d is stored as INT16 in DuckDB "
          "and has no direct cuDF equivalent — use DuckDB CPU fallback",
          logical_type::decimal_max_precision_int16);
      if (t.decimal_precision() <= logical_type::decimal_max_precision_int32)
        return cudf::data_type(cudf::type_id::DECIMAL32, neg_scale);
      if (t.decimal_precision() <= logical_type::decimal_max_precision_int64)
        return cudf::data_type(cudf::type_id::DECIMAL64, neg_scale);
      return cudf::data_type(cudf::type_id::DECIMAL128, neg_scale);
    }
    default:
      throw duckdb::InvalidInputException("sirius::get_cudf_type: Unsupported type: %s",
                                          t.to_string());
  }
}

/**
 * @brief Native cuDF mapping of @p t, or `std::nullopt` when no cuDF carrier exists
 *
 * The optional-returning counterpart of `get_cudf_type` for callers that treat an unmappable type
 * as a fallback signal rather than an error. Only that signal is absorbed: `get_cudf_type` builds
 * its message by formatting the type name, so an allocation failure there still propagates rather
 * than being reported as an unmappable type.
 */
[[nodiscard]] inline std::optional<cudf::data_type> try_get_cudf_type(const logical_type& t)
{
  try {
    return get_cudf_type(t);
  } catch (duckdb::InvalidInputException const&) {
    return std::nullopt;
  }
}

/**
 * @brief Convert a DuckDB Value to a cudf scalar, typed by sirius::logical_type.
 *
 * Used by:
 *   - Partition column injection (constant columns from hive path values)
 *   - Expression translator (AST literal nodes) — can be refactored to use this
 *
 * The Value argument remains duckdb::Value because Sirius has no native Value
 * type yet; this sits at the DuckDB boundary and returns a cudf scalar keyed
 * off the sirius-native logical_type used by GPU operators.
 *
 * @param val     The DuckDB value to convert.
 * @param t       The Sirius logical type (determines cudf type via get_cudf_type).
 * @param stream  CUDA stream for device operations.
 * @return A cudf scalar holding the same value. Falls back to string_scalar
 *         for unsupported cudf types (DECIMAL128, STRUCT, etc.).
 */
inline std::unique_ptr<cudf::scalar> value_to_cudf_scalar(duckdb::Value const& val,
                                                          logical_type const& t,
                                                          rmm::cuda_stream_view stream)
{
  auto cudf_type = get_cudf_type(t);

  switch (cudf_type.id()) {
    case cudf::type_id::INT8:
      return std::make_unique<cudf::numeric_scalar<int8_t>>(val.GetValue<int8_t>(), true, stream);
    case cudf::type_id::INT16:
      return std::make_unique<cudf::numeric_scalar<int16_t>>(val.GetValue<int16_t>(), true, stream);
    case cudf::type_id::INT32:
      return std::make_unique<cudf::numeric_scalar<int32_t>>(val.GetValue<int32_t>(), true, stream);
    case cudf::type_id::INT64:
      return std::make_unique<cudf::numeric_scalar<int64_t>>(val.GetValue<int64_t>(), true, stream);
    case cudf::type_id::UINT8:
      return std::make_unique<cudf::numeric_scalar<uint8_t>>(val.GetValue<uint8_t>(), true, stream);
    case cudf::type_id::UINT16:
      return std::make_unique<cudf::numeric_scalar<uint16_t>>(
        val.GetValue<uint16_t>(), true, stream);
    case cudf::type_id::UINT32:
      return std::make_unique<cudf::numeric_scalar<uint32_t>>(
        val.GetValue<uint32_t>(), true, stream);
    case cudf::type_id::UINT64:
      return std::make_unique<cudf::numeric_scalar<uint64_t>>(
        val.GetValue<uint64_t>(), true, stream);
    case cudf::type_id::FLOAT32:
      return std::make_unique<cudf::numeric_scalar<float>>(val.GetValue<float>(), true, stream);
    case cudf::type_id::FLOAT64:
      return std::make_unique<cudf::numeric_scalar<double>>(val.GetValue<double>(), true, stream);
    case cudf::type_id::BOOL8:
      return std::make_unique<cudf::numeric_scalar<bool>>(val.GetValue<bool>(), true, stream);
    case cudf::type_id::STRING:
      return std::make_unique<cudf::string_scalar>(val.GetValue<std::string>(), true, stream);
    case cudf::type_id::TIMESTAMP_DAYS:
      return std::make_unique<cudf::numeric_scalar<int32_t>>(val.GetValue<int32_t>(), true, stream);
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return std::make_unique<cudf::numeric_scalar<int64_t>>(val.GetValue<int64_t>(), true, stream);
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
    // For unsupported types (DECIMAL128, STRUCT, etc.), fall back to string.
    // Better than crashing — the column will be STRING instead of native type.
    default: return std::make_unique<cudf::string_scalar>(val.ToString(), true, stream);
  }
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
    columns.push_back(t.is_array() ? cudf::make_empty_lists_column(get_cudf_type(t.array_child()))
                                   : cudf::make_empty_column(get_cudf_type(t)));
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

}  // namespace sirius

namespace duckdb {

/**
 * @brief Type switch from duckdb LogicalType to cudf data_type
 *
 * @param[in] logical_type The duckdb LogicalType
 * @return The corresponding cudf data_type
 * @throws InvalidInputException if the duckdb type is unsupported
 */
inline cudf::data_type GetCudfType(const LogicalType& logical_type)
{
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
        default:
          throw InvalidInputException("GetCudfType: Unsupported duckdb decimal physical type: %d",
                                      static_cast<int>(logical_type.InternalType()));
      }
    }
    default:
      throw InvalidInputException("GetCudfType: Unsupported duckdb type: %d",
                                  static_cast<int>(logical_type.id()));
  }
}

/**
 * @brief Convert a DuckDB Value to a cudf scalar.
 *
 * This is the shared type-conversion bridge between DuckDB's type system
 * and cudf's scalar types. Used by:
 *   - Partition column injection (constant columns from hive path values)
 *   - Expression translator (AST literal nodes) — can be refactored to use this
 *
 * @param val          The DuckDB value to convert.
 * @param logical_type The DuckDB logical type (determines cudf type via GetCudfType).
 * @param stream       CUDA stream for device operations.
 * @return A cudf scalar holding the same value.
 * @throws InvalidInputException if the type is not supported by GetCudfType.
 */
inline std::unique_ptr<cudf::scalar> DuckDBValueToCudfScalar(Value const& val,
                                                             LogicalType const& logical_type,
                                                             rmm::cuda_stream_view stream)
{
  auto cudf_type = GetCudfType(logical_type);

  switch (cudf_type.id()) {
    case cudf::type_id::INT8:
      return std::make_unique<cudf::numeric_scalar<int8_t>>(val.GetValue<int8_t>(), true, stream);
    case cudf::type_id::INT16:
      return std::make_unique<cudf::numeric_scalar<int16_t>>(val.GetValue<int16_t>(), true, stream);
    case cudf::type_id::INT32:
      return std::make_unique<cudf::numeric_scalar<int32_t>>(val.GetValue<int32_t>(), true, stream);
    case cudf::type_id::INT64:
      return std::make_unique<cudf::numeric_scalar<int64_t>>(val.GetValue<int64_t>(), true, stream);
    case cudf::type_id::UINT8:
      return std::make_unique<cudf::numeric_scalar<uint8_t>>(val.GetValue<uint8_t>(), true, stream);
    case cudf::type_id::UINT16:
      return std::make_unique<cudf::numeric_scalar<uint16_t>>(
        val.GetValue<uint16_t>(), true, stream);
    case cudf::type_id::UINT32:
      return std::make_unique<cudf::numeric_scalar<uint32_t>>(
        val.GetValue<uint32_t>(), true, stream);
    case cudf::type_id::UINT64:
      return std::make_unique<cudf::numeric_scalar<uint64_t>>(
        val.GetValue<uint64_t>(), true, stream);
    case cudf::type_id::FLOAT32:
      return std::make_unique<cudf::numeric_scalar<float>>(val.GetValue<float>(), true, stream);
    case cudf::type_id::FLOAT64:
      return std::make_unique<cudf::numeric_scalar<double>>(val.GetValue<double>(), true, stream);
    case cudf::type_id::BOOL8:
      return std::make_unique<cudf::numeric_scalar<bool>>(val.GetValue<bool>(), true, stream);
    case cudf::type_id::STRING:
      return std::make_unique<cudf::string_scalar>(val.GetValue<std::string>(), true, stream);
    case cudf::type_id::TIMESTAMP_DAYS:
      return std::make_unique<cudf::numeric_scalar<int32_t>>(val.GetValue<int32_t>(), true, stream);
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return std::make_unique<cudf::numeric_scalar<int64_t>>(val.GetValue<int64_t>(), true, stream);
    case cudf::type_id::DECIMAL32:
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
        val.GetValueUnsafe<typename numeric::decimal32::rep>(),
        numeric::scale_type{-DecimalType::GetScale(logical_type)},
        true,
        stream);
    case cudf::type_id::DECIMAL64:
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        val.GetValueUnsafe<typename numeric::decimal64::rep>(),
        numeric::scale_type{-DecimalType::GetScale(logical_type)},
        true,
        stream);
    // For unsupported types (DECIMAL128, STRUCT, etc.), fall back to string.
    // Better than crashing — the column will be STRING instead of native type.
    default: return std::make_unique<cudf::string_scalar>(val.ToString(), true, stream);
  }
}

/**
 * @brief Build a zero-row cuDF table with the same per-column types as @p input.
 *
 * Nested-safe via `cudf::empty_like`: nested columns (LIST/STRUCT) reproduce their full child
 * hierarchy, not just the top-level type id.
 */
inline std::unique_ptr<cudf::table> make_empty_like(cudf::table_view input)
{
  return cudf::empty_like(input);
}

}  // namespace duckdb
