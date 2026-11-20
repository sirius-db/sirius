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

// sirius
#include <op/scan/parquet_schema_mapping.hpp>

#include <limits>

namespace sirius::op::scan::detail {

std::size_t carrier_decoded_width(cudf::io::parquet::SchemaElement const& column)
{
  using namespace cudf::io::parquet;
  if (column.parent_idx != 0 || column.num_children != 0 || column.max_repetition_level != 0 ||
      column.output_as_byte_array) {
    return 0;
  }
  constexpr auto string_width = std::numeric_limits<std::size_t>::max();
  // cuDF honors Arrow duration metadata only for unannotated INT64 storage.
  if (column.arrow_type && column.type == Type::INT64 && !column.logical_type &&
      !column.converted_type) {
    switch (*column.arrow_type) {
      case cudf::type_id::DURATION_DAYS: return 4;
      case cudf::type_id::DURATION_SECONDS:
      case cudf::type_id::DURATION_MILLISECONDS:
      case cudf::type_id::DURATION_MICROSECONDS:
      case cudf::type_id::DURATION_NANOSECONDS: return 8;
      default: return 0;
    }
  }
  if (column.logical_type) {
    auto const& logical = *column.logical_type;
    switch (logical.type) {
      case LogicalType::INTEGER:
        if (!logical.int_type) { return 0; }
        switch (logical.int_type->bitWidth) {
          case 8: return 1;
          case 16: return 2;
          case 32: return 4;
          case 64: return 8;
          default: return 0;
        }
      case LogicalType::DATE: return 4;
      case LogicalType::TIME:
        return logical.is_time_millis() || logical.is_time_micros() || logical.is_time_nanos() ? 8
                                                                                               : 0;
      case LogicalType::TIMESTAMP:
        return logical.is_timestamp_millis() || logical.is_timestamp_micros() ||
                   logical.is_timestamp_nanos()
                 ? 8
                 : 0;
      case LogicalType::STRING: return column.type == Type::BYTE_ARRAY ? string_width : 0;
      case LogicalType::UNDEFINED: break;
      default: return 0;
    }
  } else if (column.converted_type) {
    switch (*column.converted_type) {
      case ConvertedType::INT_8:
      case ConvertedType::UINT_8: return 1;
      case ConvertedType::INT_16:
      case ConvertedType::UINT_16: return 2;
      case ConvertedType::INT_32:
      case ConvertedType::UINT_32:
      case ConvertedType::DATE: return 4;
      case ConvertedType::INT_64:
      case ConvertedType::UINT_64:
      case ConvertedType::TIME_MILLIS:
      case ConvertedType::TIME_MICROS:
      case ConvertedType::TIMESTAMP_MILLIS:
      case ConvertedType::TIMESTAMP_MICROS: return 8;
      case ConvertedType::UTF8:
      case ConvertedType::ENUM: return column.type == Type::BYTE_ARRAY ? string_width : 0;
      default: return 0;
    }
  }
  switch (column.type) {
    case Type::BOOLEAN: return 1;
    case Type::INT32:
    case Type::FLOAT: return 4;
    case Type::INT64:
    case Type::DOUBLE: return 8;
    default: return 0;
  }
}

std::vector<std::size_t> leaf_indices_for_column(cudf::io::parquet::FileMetaData const& metadata,
                                                 std::string const& column_name)
{
  std::vector<std::size_t> result;
  if (metadata.row_groups.empty()) { return result; }

  auto const& columns = metadata.row_groups.front().columns;
  for (std::size_t i = 0; i < columns.size(); ++i) {
    auto const& path = columns[i].meta_data.path_in_schema;
    if (!path.empty() && path.front() == column_name) { result.push_back(i); }
  }
  return result;
}

}  // namespace sirius::op::scan::detail
