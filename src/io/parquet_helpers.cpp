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
      case pq::LogicalType::TIMESTAMP:
        if (el.logical_type->timestamp_type &&
            el.logical_type->timestamp_type->unit.type == pq::TimeUnit::NANOS)
          return duckdb::LogicalType::TIMESTAMP_NS;
        return duckdb::LogicalType::TIMESTAMP;
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

duckdb::LogicalType leaf_to_duckdb_type(pq::SchemaElement const& el,
                                        bool decoded                     = false,
                                        bool original_logical_annotation = true)
{
  if (decoded) {
    // The pinned cuDF reader synthesizes a UTC TIMESTAMP logical annotation
    // for converted-type-only legacy timestamps. DuckDB binds those as plain
    // TIMESTAMP; only a logicalType present in the raw footer means UTC here.
    if (!original_logical_annotation && el.converted_type) {
      switch (*el.converted_type) {
        case pq::ConvertedType::TIMESTAMP_MILLIS: return duckdb::LogicalType::TIMESTAMP_MS;
        case pq::ConvertedType::TIMESTAMP_MICROS: return duckdb::LogicalType::TIMESTAMP;
        default: break;
      }
    }
    if (el.logical_type && el.logical_type->type == pq::LogicalType::UNDEFINED)
      return duckdb::LogicalType::SQLNULL;  // annotation unsupported by the pinned decoder
    // Match the pinned cuDF decoder's temporal units. Duration columns have
    // no DuckDB export mapping, even when their physical storage is INT64.
    if (el.type == pq::Type::INT64 && el.arrow_type && !el.logical_type && !el.converted_type)
      return duckdb::LogicalType::SQLNULL;
    if (el.logical_type && el.logical_type->type == pq::LogicalType::TIME)
      return duckdb::LogicalType::SQLNULL;
    if (el.logical_type && el.logical_type->type == pq::LogicalType::TIMESTAMP &&
        el.logical_type->timestamp_type) {
      // cuDF drops the UTC annotation. Exporting that timezone-free column
      // cannot implement DuckDB's TIMESTAMPTZ conversion, so do not qualify it.
      if (el.logical_type->timestamp_type->isAdjustedToUTC && original_logical_annotation)
        return duckdb::LogicalType::SQLNULL;
      switch (el.logical_type->timestamp_type->unit.type) {
        case pq::TimeUnit::MILLIS: return duckdb::LogicalType::TIMESTAMP_MS;
        case pq::TimeUnit::MICROS: return duckdb::LogicalType::TIMESTAMP;
        case pq::TimeUnit::NANOS: return duckdb::LogicalType::TIMESTAMP_NS;
      }
    }
    if (!el.logical_type && el.converted_type) {
      switch (*el.converted_type) {
        case pq::ConvertedType::TIMESTAMP_MILLIS: return duckdb::LogicalType::TIMESTAMP_MS;
        case pq::ConvertedType::TIMESTAMP_MICROS: return duckdb::LogicalType::TIMESTAMP;
        case pq::ConvertedType::TIME_MILLIS:
        case pq::ConvertedType::TIME_MICROS: return duckdb::LogicalType::SQLNULL;
        default: break;
      }
    }
    if (el.type == pq::Type::INT96) return duckdb::LogicalType::TIMESTAMP_NS;
  }
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

// A normalized cuDF schema can append nodes to the original preorder array.
// Its children_idx links are authoritative. Hand-built test metadata has no
// links, so derive children from the original preorder layout in that case.
std::size_t preorder_end(pq::FileMetaData const& meta, std::size_t idx)
{
  if (idx >= meta.schema.size())
    throw std::runtime_error("[parquet_helpers] malformed parquet schema: truncated");
  auto next = idx + 1;
  for (int c = 0; c < meta.schema[idx].num_children; ++c)
    next = preorder_end(meta, next);
  return next;
}

std::vector<std::size_t> children(pq::FileMetaData const& meta, std::size_t idx)
{
  auto const& el = meta.schema.at(idx);
  std::vector<std::size_t> out;
  out.reserve(el.num_children);
  if (!el.children_idx.empty()) {
    for (auto child : el.children_idx) {
      if (child < 0 || static_cast<std::size_t>(child) >= meta.schema.size())
        throw std::runtime_error("[parquet_helpers] malformed parquet schema: child index");
      out.push_back(static_cast<std::size_t>(child));
    }
  } else {
    auto next = idx + 1;
    for (int c = 0; c < el.num_children; ++c) {
      if (next >= meta.schema.size())
        throw std::runtime_error("[parquet_helpers] malformed parquet schema: truncated");
      out.push_back(next);
      next = preorder_end(meta, next);
    }
  }
  if (out.size() != static_cast<std::size_t>(el.num_children))
    throw std::runtime_error("[parquet_helpers] malformed parquet schema: child count");
  return out;
}

bool has_original_logical(pq::SchemaElement const& el,
                          std::size_t idx,
                          std::span<uint8_t const> original_annotations)
{
  return original_annotations.empty() || idx >= original_annotations.size()
           ? el.logical_type.has_value()
           : original_annotations[idx] != 0;
}

duckdb::LogicalType map_subtree(pq::FileMetaData const& meta,
                                std::size_t idx,
                                bool decoded,
                                std::span<uint8_t const> original_annotations,
                                bool list_element = false)
{
  if (idx >= meta.schema.size())
    throw std::runtime_error("[parquet_helpers] malformed parquet schema: truncated");
  auto const& el = meta.schema[idx];

  if (el.num_children == 0) {
    auto type =
      leaf_to_duckdb_type(el, decoded, has_original_logical(el, idx, original_annotations));
    // Parquet's one-level repeated primitive is a LIST to DuckDB. A repeated
    // element inside an annotated LIST is already wrapped by its parent.
    if (el.repetition_type == pq::FieldRepetitionType::REPEATED && !list_element)
      return duckdb::LogicalType::LIST(std::move(type));
    return type;
  }

  auto const child = children(meta, idx);

  if (is_map_annotated(el)) {
    if (child.size() != 1) {
      throw std::runtime_error("[parquet_helpers] malformed parquet MAP schema for column '" +
                               el.name + "'");
    }
    auto const kv = children(meta, child.front());
    if (kv.size() != 2)
      throw std::runtime_error("[parquet_helpers] malformed parquet MAP key/value schema");
    return duckdb::LogicalType::MAP(map_subtree(meta, kv[0], decoded, original_annotations, true),
                                    map_subtree(meta, kv[1], decoded, original_annotations, true));
  }

  if (is_list_annotated(el)) {
    if (child.size() != 1) {
      throw std::runtime_error("[parquet_helpers] malformed parquet LIST schema for column '" +
                               el.name + "'");
    }
    auto const mid          = child.front();
    auto const& middle      = meta.schema[mid];
    auto const mid_children = children(meta, mid);
    if (middle.repetition_type == pq::FieldRepetitionType::REPEATED && middle.is_stub() &&
        mid_children.size() == 1)
      return duckdb::LogicalType::LIST(
        map_subtree(meta, mid_children.front(), decoded, original_annotations, true));
    return duckdb::LogicalType::LIST(map_subtree(meta, mid, decoded, original_annotations, true));
  }

  // Plain group with no LIST/MAP annotation => STRUCT over its children.
  duckdb::child_list_t<duckdb::LogicalType> children;
  for (auto position : child) {
    children.emplace_back(meta.schema[position].name,
                          map_subtree(meta, position, decoded, original_annotations));
  }
  auto type = duckdb::LogicalType::STRUCT(std::move(children));
  if (el.repetition_type == pq::FieldRepetitionType::REPEATED && !list_element)
    return duckdb::LogicalType::LIST(std::move(type));
  return type;
}

}  // namespace

duckdb::LogicalType leaf_schema_type(pq::SchemaElement const& element,
                                     bool original_logical_annotation)
{
  // The Iceberg comparison uses DuckDB's parquet_schema type spelling,
  // including UTC annotations that cuDF does not retain in its output type.
  if (element.logical_type && original_logical_annotation) {
    auto const& logical = *element.logical_type;
    if (logical.type == pq::LogicalType::TIMESTAMP && logical.timestamp_type &&
        logical.timestamp_type->isAdjustedToUTC)
      return duckdb::LogicalType::TIMESTAMP_TZ;
    if (logical.type == pq::LogicalType::TIME && logical.time_type &&
        logical.time_type->isAdjustedToUTC)
      return duckdb::LogicalType::TIME_TZ;
    if (logical.type == pq::LogicalType::UNKNOWN) return duckdb::LogicalType::SQLNULL;
  }
  return leaf_to_duckdb_type(element);
}

schema_info extract_schema(cudf::io::parquet::FileMetaData const& meta,
                           bool decoded,
                           std::span<uint8_t const> original_logical_annotations)
{
  if (meta.schema.empty()) { throw std::runtime_error("[parquet_helpers] empty parquet schema"); }

  // schema[0] is the root group; its children are the top-level columns laid
  // out in preorder. A flat leaf occupies one element; a nested column (STRUCT /
  // LIST / MAP) occupies a whole subtree that map_subtree consumes recursively,
  // advancing `idx` past it so the next top-level column is mapped correctly.
  schema_info out;
  for (auto idx : children(meta, 0)) {
    out.names.push_back(meta.schema[idx].name);
    out.types.push_back(map_subtree(meta, idx, decoded, original_logical_annotations));
  }
  return out;
}

}  // namespace sirius::io::parquet_helpers
