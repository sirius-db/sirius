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
#include <utility>

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

// A LIST-annotated group wraps the standard 3-level encoding:
// `<col>(LIST) -> repeated group -> element`.
bool is_list_annotated(pq::SchemaElement const& el)
{
  if (el.logical_type.has_value() && el.logical_type->type == pq::LogicalType::LIST) {
    return true;
  }
  return el.converted_type.has_value() && *el.converted_type == pq::ConvertedType::LIST;
}

// A MAP-annotated group wraps `<col>(MAP) -> repeated key_value -> {key, value}`.
// MAP_KEY_VALUE marks the inner key_value group, not the outer column — not a map.
bool is_map_annotated(pq::SchemaElement const& el)
{
  if (el.logical_type.has_value() && el.logical_type->type == pq::LogicalType::MAP) { return true; }
  return el.converted_type.has_value() && *el.converted_type == pq::ConvertedType::MAP;
}

// One mapped subtree: the DuckDB type plus the index of the next sibling in the
// preorder-flattened schema array.
struct mapped_subtree {
  duckdb::LogicalType type;
  std::size_t next;
};

// Map the subtree rooted at `idx` (preorder) to a DuckDB LogicalType, advancing
// past it so the caller resumes at the next sibling. Throws on a truncated or
// malformed nested subtree.
mapped_subtree map_subtree(pq::FileMetaData const& meta, std::size_t idx)
{
  if (idx >= meta.schema.size()) {
    throw std::runtime_error("[parquet_helpers] malformed parquet schema: truncated");
  }
  auto const& el = meta.schema[idx];

  if (el.num_children == 0) { return {leaf_to_duckdb_type(el), idx + 1}; }

  if (is_map_annotated(el)) {
    // el -> key_value group (idx+1) -> key (idx+2), value (after key subtree)
    std::size_t const kv = idx + 1;
    if (kv >= meta.schema.size() || meta.schema[kv].num_children < 2) {
      throw std::runtime_error("[parquet_helpers] malformed parquet MAP schema for column '" +
                               el.name + "'");
    }
    auto key   = map_subtree(meta, kv + 1);
    auto value = map_subtree(meta, key.next);
    return {duckdb::LogicalType::MAP(std::move(key.type), std::move(value.type)), value.next};
  }

  if (is_list_annotated(el)) {
    // el -> repeated middle group (idx+1) -> element (idx+2)
    std::size_t const mid = idx + 1;
    if (mid >= meta.schema.size() || meta.schema[mid].num_children < 1) {
      throw std::runtime_error("[parquet_helpers] malformed parquet LIST schema for column '" +
                               el.name + "'");
    }
    auto element = map_subtree(meta, mid + 1);
    return {duckdb::LogicalType::LIST(std::move(element.type)), element.next};
  }

  // Plain group with no LIST/MAP annotation => STRUCT over its children.
  duckdb::child_list_t<duckdb::LogicalType> children;
  std::size_t cur = idx + 1;
  for (int c = 0; c < el.num_children; ++c) {
    if (cur >= meta.schema.size()) {
      throw std::runtime_error("[parquet_helpers] malformed parquet schema: truncated");
    }
    auto const child_name = meta.schema[cur].name;
    auto child            = map_subtree(meta, cur);
    children.emplace_back(child_name, std::move(child.type));
    cur = child.next;
  }
  return {duckdb::LogicalType::STRUCT(std::move(children)), cur};
}

}  // namespace

schema_info extract_schema(cudf::io::parquet::FileMetaData const& meta)
{
  if (meta.schema.empty()) { throw std::runtime_error("[parquet_helpers] empty parquet schema"); }

  // schema[0] is the root group; its children are the top-level columns laid
  // out in preorder. A flat leaf occupies one element; a nested column (STRUCT /
  // LIST / MAP) occupies a whole subtree that map_subtree consumes recursively,
  // advancing `idx` past it so the next top-level column is mapped correctly.
  auto const& root = meta.schema.front();
  schema_info out;
  std::size_t idx = 1;
  for (int col = 0; col < root.num_children; ++col) {
    if (idx >= meta.schema.size()) {
      throw std::runtime_error("[parquet_helpers] malformed parquet schema: truncated");
    }
    out.names.push_back(meta.schema[idx].name);
    auto mapped = map_subtree(meta, idx);
    out.types.push_back(std::move(mapped.type));
    idx = mapped.next;
  }
  return out;
}

}  // namespace sirius::io::parquet_helpers
