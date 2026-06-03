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

#include "io/parquet_helpers.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace sirius::io::parquet_helpers {

namespace {

namespace pq = cudf::io::parquet;

// DuckDB's DECIMAL type tops out at 38 digits of precision.
constexpr int kMaxDecimalPrecision = 38;

bool is_decimal(pq::SchemaElement const& el)
{
  if (el.logical_type.has_value() && el.logical_type->type == pq::LogicalType::DECIMAL) {
    return true;
  }
  return el.converted_type.has_value() && *el.converted_type == pq::ConvertedType::DECIMAL;
}

duckdb::LogicalType map_decimal(pq::SchemaElement const& el)
{
  int precision = 0;
  int scale     = 0;
  if (el.logical_type.has_value() && el.logical_type->type == pq::LogicalType::DECIMAL &&
      el.logical_type->decimal_type.has_value()) {
    precision = el.logical_type->decimal_type->precision;
    scale     = el.logical_type->decimal_type->scale;
  }
  if (precision <= 0) {  // fall back to the deprecated SchemaElement fields
    precision = el.decimal_precision;
    scale     = el.decimal_scale;
  }
  if (precision <= 0 || precision > kMaxDecimalPrecision) {
    throw std::runtime_error("[parquet_helpers] unsupported DECIMAL precision for column '" +
                             el.name + "'");
  }
  return duckdb::LogicalType::DECIMAL(precision, scale);
}

duckdb::LogicalType int_from_width(int bit_width, bool is_signed, pq::SchemaElement const& el)
{
  switch (bit_width) {
    case 8: return is_signed ? duckdb::LogicalType::TINYINT : duckdb::LogicalType::UTINYINT;
    case 16: return is_signed ? duckdb::LogicalType::SMALLINT : duckdb::LogicalType::USMALLINT;
    case 32: return is_signed ? duckdb::LogicalType::INTEGER : duckdb::LogicalType::UINTEGER;
    case 64: return is_signed ? duckdb::LogicalType::BIGINT : duckdb::LogicalType::UBIGINT;
    default:
      throw std::runtime_error("[parquet_helpers] unsupported integer bit width for column '" +
                               el.name + "'");
  }
}

duckdb::LogicalType map_int32(pq::SchemaElement const& el)
{
  // The modern logical type takes precedence over the deprecated converted type.
  if (el.logical_type.has_value()) {
    switch (el.logical_type->type) {
      case pq::LogicalType::DATE: return duckdb::LogicalType::DATE;
      case pq::LogicalType::TIME: return duckdb::LogicalType::TIME;
      case pq::LogicalType::INTEGER:
        if (el.logical_type->int_type.has_value()) {
          return int_from_width(
            el.logical_type->int_type->bitWidth, el.logical_type->int_type->isSigned, el);
        }
        break;
      default: break;
    }
  }
  if (el.converted_type.has_value()) {
    switch (*el.converted_type) {
      case pq::ConvertedType::DATE: return duckdb::LogicalType::DATE;
      case pq::ConvertedType::TIME_MILLIS: return duckdb::LogicalType::TIME;
      case pq::ConvertedType::INT_8: return duckdb::LogicalType::TINYINT;
      case pq::ConvertedType::INT_16: return duckdb::LogicalType::SMALLINT;
      case pq::ConvertedType::INT_32: return duckdb::LogicalType::INTEGER;
      case pq::ConvertedType::UINT_8: return duckdb::LogicalType::UTINYINT;
      case pq::ConvertedType::UINT_16: return duckdb::LogicalType::USMALLINT;
      case pq::ConvertedType::UINT_32: return duckdb::LogicalType::UINTEGER;
      default: break;
    }
  }
  return duckdb::LogicalType::INTEGER;
}

duckdb::LogicalType map_int64(pq::SchemaElement const& el)
{
  if (el.logical_type.has_value()) {
    switch (el.logical_type->type) {
      case pq::LogicalType::TIMESTAMP: return duckdb::LogicalType::TIMESTAMP;
      case pq::LogicalType::TIME: return duckdb::LogicalType::TIME;
      case pq::LogicalType::INTEGER:
        if (el.logical_type->int_type.has_value()) {
          return int_from_width(
            el.logical_type->int_type->bitWidth, el.logical_type->int_type->isSigned, el);
        }
        break;
      default: break;
    }
  }
  if (el.converted_type.has_value()) {
    switch (*el.converted_type) {
      case pq::ConvertedType::TIMESTAMP_MILLIS:
      case pq::ConvertedType::TIMESTAMP_MICROS: return duckdb::LogicalType::TIMESTAMP;
      case pq::ConvertedType::TIME_MICROS: return duckdb::LogicalType::TIME;
      case pq::ConvertedType::INT_64: return duckdb::LogicalType::BIGINT;
      case pq::ConvertedType::UINT_64: return duckdb::LogicalType::UBIGINT;
      default: break;
    }
  }
  return duckdb::LogicalType::BIGINT;
}

duckdb::LogicalType map_byte_array(pq::SchemaElement const& el)
{
  if (el.logical_type.has_value()) {
    auto const type = el.logical_type->type;
    if (type == pq::LogicalType::STRING || type == pq::LogicalType::ENUM ||
        type == pq::LogicalType::JSON) {
      return duckdb::LogicalType::VARCHAR;
    }
  }
  if (el.converted_type.has_value()) {
    switch (*el.converted_type) {
      case pq::ConvertedType::UTF8:
      case pq::ConvertedType::ENUM:
      case pq::ConvertedType::JSON: return duckdb::LogicalType::VARCHAR;
      default: break;
    }
  }
  return duckdb::LogicalType::BLOB;
}

duckdb::LogicalType leaf_to_duckdb_type(pq::SchemaElement const& el)
{
  if (is_decimal(el)) { return map_decimal(el); }
  switch (el.type) {
    case pq::Type::BOOLEAN: return duckdb::LogicalType::BOOLEAN;
    case pq::Type::FLOAT: return duckdb::LogicalType::FLOAT;
    case pq::Type::DOUBLE: return duckdb::LogicalType::DOUBLE;
    case pq::Type::INT96: return duckdb::LogicalType::TIMESTAMP;  // deprecated nanosecond ts
    case pq::Type::INT32: return map_int32(el);
    case pq::Type::INT64: return map_int64(el);
    case pq::Type::BYTE_ARRAY:
    case pq::Type::FIXED_LEN_BYTE_ARRAY: return map_byte_array(el);
    case pq::Type::UNDEFINED: break;
  }
  throw std::runtime_error("[parquet_helpers] unsupported parquet physical type for column '" +
                           el.name + "'");
}

}  // namespace

schema_info extract_schema(cudf::io::parquet::FileMetaData const& meta)
{
  if (meta.schema.empty()) { throw std::runtime_error("[parquet_helpers] empty parquet schema"); }

  // schema[0] is the root group; its children are the top-level columns laid
  // out in preorder. A flat leaf column occupies exactly one schema element.
  auto const& root = meta.schema.front();
  schema_info out;
  std::size_t idx = 1;
  for (int col = 0; col < root.num_children; ++col) {
    if (idx >= meta.schema.size()) {
      throw std::runtime_error("[parquet_helpers] malformed parquet schema: truncated");
    }
    auto const& element = meta.schema[idx];
    if (element.num_children > 0) {
      throw std::runtime_error(
        "[parquet_helpers] nested parquet types not yet supported in sirius_read_parquet "
        "(column '" +
        element.name + "')");
    }
    out.names.push_back(element.name);
    out.types.push_back(leaf_to_duckdb_type(element));
    ++idx;
  }
  return out;
}

}  // namespace sirius::io::parquet_helpers
