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

#include "helper/type_conversions.hpp"

#include "duckdb/common/optional_idx.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/types/decimal.hpp>

namespace sirius {

logical_type from_duckdb(const duckdb::LogicalType& t)
{
  using duckdb::LogicalTypeId;
  switch (t.id()) {
    case LogicalTypeId::SQLNULL: return logical_type::make(type_id::SQLNULL);
    case LogicalTypeId::BOOLEAN: return logical_type::make(type_id::BOOLEAN);
    case LogicalTypeId::TINYINT: return logical_type::make(type_id::TINYINT);
    case LogicalTypeId::SMALLINT: return logical_type::make(type_id::SMALLINT);
    case LogicalTypeId::INTEGER: return logical_type::make(type_id::INTEGER);
    case LogicalTypeId::BIGINT: return logical_type::make(type_id::BIGINT);
    case LogicalTypeId::HUGEINT: return logical_type::make(type_id::HUGEINT);
    case LogicalTypeId::UTINYINT: return logical_type::make(type_id::UTINYINT);
    case LogicalTypeId::USMALLINT: return logical_type::make(type_id::USMALLINT);
    case LogicalTypeId::UINTEGER: return logical_type::make(type_id::UINTEGER);
    case LogicalTypeId::UBIGINT: return logical_type::make(type_id::UBIGINT);
    case LogicalTypeId::UHUGEINT: return logical_type::make(type_id::UHUGEINT);
    case LogicalTypeId::FLOAT: return logical_type::make(type_id::FLOAT);
    case LogicalTypeId::DOUBLE: return logical_type::make(type_id::DOUBLE);
    case LogicalTypeId::DATE: return logical_type::make(type_id::DATE);
    case LogicalTypeId::TIMESTAMP_SEC: return logical_type::make(type_id::TIMESTAMP_SEC);
    case LogicalTypeId::TIMESTAMP_MS: return logical_type::make(type_id::TIMESTAMP_MS);
    case LogicalTypeId::TIMESTAMP: return logical_type::make(type_id::TIMESTAMP);
    case LogicalTypeId::TIMESTAMP_NS: return logical_type::make(type_id::TIMESTAMP_NS);
    case LogicalTypeId::VARCHAR: return logical_type::make(type_id::VARCHAR);
    // sirius::logical_type is flat/opaque for nested types here; the faithful nested type is
    // carried separately (the result collector threads the full DuckDB types). A DuckDB MAP is
    // physically a LIST<STRUCT<key,value>>, so it rides the LIST placeholder.
    case LogicalTypeId::STRUCT: return logical_type::make(type_id::STRUCT);
    case LogicalTypeId::LIST: return logical_type::make(type_id::LIST);
    case LogicalTypeId::MAP: return logical_type::make(type_id::LIST);
    case LogicalTypeId::ARRAY: {
      const auto& child_type = duckdb::ArrayType::GetChildType(t);
      const auto array_size  = duckdb::ArrayType::GetSize(t);
      return logical_type::make_array(from_duckdb(child_type), static_cast<uint32_t>(array_size));
    }
    case LogicalTypeId::DECIMAL:
      return logical_type::make_decimal(duckdb::DecimalType::GetWidth(t),
                                        duckdb::DecimalType::GetScale(t));
    default:
      throw duckdb::InvalidInputException("from_duckdb: Unsupported DuckDB type: %s", t.ToString());
  }
}

duckdb::LogicalType to_duckdb(const logical_type& t)
{
  using duckdb::LogicalType;
  switch (t.id()) {
    case type_id::SQLNULL: return LogicalType::SQLNULL;
    case type_id::BOOLEAN: return LogicalType::BOOLEAN;
    case type_id::TINYINT: return LogicalType::TINYINT;
    case type_id::SMALLINT: return LogicalType::SMALLINT;
    case type_id::INTEGER: return LogicalType::INTEGER;
    case type_id::BIGINT: return LogicalType::BIGINT;
    case type_id::HUGEINT: return LogicalType::HUGEINT;
    case type_id::UTINYINT: return LogicalType::UTINYINT;
    case type_id::USMALLINT: return LogicalType::USMALLINT;
    case type_id::UINTEGER: return LogicalType::UINTEGER;
    case type_id::UBIGINT: return LogicalType::UBIGINT;
    case type_id::UHUGEINT: return LogicalType::UHUGEINT;
    case type_id::FLOAT: return LogicalType::FLOAT;
    case type_id::DOUBLE: return LogicalType::DOUBLE;
    case type_id::DATE: return LogicalType::DATE;
    case type_id::TIMESTAMP_SEC: return LogicalType::TIMESTAMP_S;
    case type_id::TIMESTAMP_MS: return LogicalType::TIMESTAMP_MS;
    case type_id::TIMESTAMP: return LogicalType::TIMESTAMP;
    case type_id::TIMESTAMP_NS: return LogicalType::TIMESTAMP_NS;
    case type_id::VARCHAR: return LogicalType::VARCHAR;
    case type_id::ARRAY:
      if (!t.has_child()) {
        throw duckdb::InvalidInputException(
          "to_duckdb: ARRAY type missing child metadata (use logical_type::make_array)");
      }
      return LogicalType::ARRAY(to_duckdb(t.array_child()), duckdb::optional_idx(t.array_size()));
    case type_id::STRUCT: return LogicalType::STRUCT({});
    case type_id::LIST:
      throw duckdb::InvalidInputException(
        "to_duckdb: LIST conversion requires nested element metadata and is not supported");
    case type_id::INVALID:
      throw duckdb::InvalidInputException("to_duckdb: INVALID type has no DuckDB equivalent");
    case type_id::DECIMAL: return LogicalType::DECIMAL(t.decimal_precision(), t.decimal_scale());
    default:
      throw duckdb::InvalidInputException("to_duckdb: Unsupported sirius type: %s", t.to_string());
  }
}

duckdb::vector<logical_type> from_duckdb_vec(const duckdb::vector<duckdb::LogicalType>& types)
{
  duckdb::vector<logical_type> result;
  result.reserve(types.size());
  for (const auto& t : types) {
    result.push_back(from_duckdb(t));
  }
  return result;
}

duckdb::vector<duckdb::LogicalType> to_duckdb_vec(const duckdb::vector<logical_type>& types)
{
  duckdb::vector<duckdb::LogicalType> result;
  result.reserve(types.size());
  for (const auto& t : types) {
    result.push_back(to_duckdb(t));
  }
  return result;
}

}  // namespace sirius
