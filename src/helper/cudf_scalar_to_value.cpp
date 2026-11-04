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

#include "helper/cudf_scalar_to_value.hpp"

#include <cudf/wrappers/timestamps.hpp>

#include <duckdb/common/types/date.hpp>
#include <duckdb/common/types/timestamp.hpp>

#include <cstdint>

namespace sirius {

duckdb::Value cudf_scalar_to_value(cudf::scalar const& scalar, ::cuda::stream_ref stream)
{
  switch (scalar.type().id()) {
    case cudf::type_id::INT8:
      return duckdb::Value::TINYINT(
        static_cast<cudf::numeric_scalar<std::int8_t> const&>(scalar).value(stream));
    case cudf::type_id::INT16:
      return duckdb::Value::SMALLINT(
        static_cast<cudf::numeric_scalar<std::int16_t> const&>(scalar).value(stream));
    case cudf::type_id::INT32:
      return duckdb::Value::INTEGER(
        static_cast<cudf::numeric_scalar<std::int32_t> const&>(scalar).value(stream));
    case cudf::type_id::INT64:
      return duckdb::Value::BIGINT(
        static_cast<cudf::numeric_scalar<std::int64_t> const&>(scalar).value(stream));
    case cudf::type_id::UINT8:
      return duckdb::Value::UTINYINT(
        static_cast<cudf::numeric_scalar<std::uint8_t> const&>(scalar).value(stream));
    case cudf::type_id::UINT16:
      return duckdb::Value::USMALLINT(
        static_cast<cudf::numeric_scalar<std::uint16_t> const&>(scalar).value(stream));
    case cudf::type_id::UINT32:
      return duckdb::Value::UINTEGER(
        static_cast<cudf::numeric_scalar<std::uint32_t> const&>(scalar).value(stream));
    case cudf::type_id::UINT64:
      return duckdb::Value::UBIGINT(
        static_cast<cudf::numeric_scalar<std::uint64_t> const&>(scalar).value(stream));
    case cudf::type_id::TIMESTAMP_DAYS:
      return duckdb::Value::DATE(
        duckdb::date_t{static_cast<cudf::timestamp_scalar<cudf::timestamp_D> const&>(scalar)
                         .value(stream)
                         .time_since_epoch()
                         .count()});
    case cudf::type_id::TIMESTAMP_SECONDS:
      return duckdb::Value::TIMESTAMPSEC(duckdb::timestamp_sec_t{
        static_cast<cudf::timestamp_scalar<cudf::timestamp_s> const&>(scalar)
          .value(stream)
          .time_since_epoch()
          .count()});
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return duckdb::Value::TIMESTAMPMS(duckdb::timestamp_ms_t{
        static_cast<cudf::timestamp_scalar<cudf::timestamp_ms> const&>(scalar)
          .value(stream)
          .time_since_epoch()
          .count()});
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return duckdb::Value::TIMESTAMP(
        duckdb::timestamp_t{static_cast<cudf::timestamp_scalar<cudf::timestamp_us> const&>(scalar)
                              .value(stream)
                              .time_since_epoch()
                              .count()});
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return duckdb::Value::TIMESTAMPNS(duckdb::timestamp_ns_t{
        static_cast<cudf::timestamp_scalar<cudf::timestamp_ns> const&>(scalar)
          .value(stream)
          .time_since_epoch()
          .count()});
    case cudf::type_id::STRING:
      return duckdb::Value(static_cast<cudf::string_scalar const&>(scalar).to_string(stream));
    default: return duckdb::Value();
  }
}

}  // namespace sirius
