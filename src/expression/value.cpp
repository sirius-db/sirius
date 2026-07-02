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

#include "expression/value.hpp"

#include "helper/type_conversions.hpp"
#include "sirius/exception.hpp"

// duckdb
#include <duckdb/common/types/date.hpp>
#include <duckdb/common/types/decimal.hpp>
#include <duckdb/common/types/hugeint.hpp>
#include <duckdb/common/types/timestamp.hpp>
#include <duckdb/common/types/value.hpp>

// standard library
#include <cstdint>
#include <string>
#include <variant>

namespace sirius {

value from_duckdb(const duckdb::Value& v, const logical_type& type)
{
  // Typed NULL fidelity — every IsNull() input becomes null_value, regardless
  // of the value's logical type. The SQL type of the NULL is recovered later
  // via the supplied logical_type at the to_duckdb boundary.
  if (v.IsNull()) { return value{null_value{}}; }

  switch (type.id()) {
    case type_id::BOOLEAN: return value{v.GetValue<bool>()};
    case type_id::TINYINT: return value{v.GetValue<int8_t>()};
    case type_id::SMALLINT: return value{v.GetValue<int16_t>()};
    case type_id::INTEGER: return value{v.GetValue<int32_t>()};
    case type_id::BIGINT: return value{v.GetValue<int64_t>()};
    case type_id::HUGEINT: return value{v.GetValue<int64_t>()};  // narrowing
    case type_id::UTINYINT: return value{v.GetValue<uint8_t>()};
    case type_id::USMALLINT: return value{v.GetValue<uint16_t>()};
    case type_id::UINTEGER: return value{v.GetValue<uint32_t>()};
    case type_id::UBIGINT: return value{v.GetValue<uint64_t>()};
    case type_id::UHUGEINT: return value{v.GetValue<uint64_t>()};  // narrowing
    case type_id::FLOAT: return value{v.GetValue<float>()};
    case type_id::DOUBLE: return value{v.GetValue<double>()};
    case type_id::DATE: return value{date_value{v.GetValue<duckdb::date_t>().days}};
    case type_id::TIMESTAMP_SEC:
      return value{timestamp_sec_value{v.GetValue<duckdb::timestamp_sec_t>().value}};
    case type_id::TIMESTAMP_MS:
      return value{timestamp_ms_value{v.GetValue<duckdb::timestamp_ms_t>().value}};
    case type_id::TIMESTAMP:
      // constant.cpp uses duckdb::timestamp_tz_t for microsecond
      // TIMESTAMP — mirror that accessor exactly so the executor migration is
      // a structural rewrite, not a behavior change.
      return value{timestamp_us_value{v.GetValue<duckdb::timestamp_tz_t>().value}};
    case type_id::TIMESTAMP_NS:
      return value{timestamp_ns_value{v.GetValue<duckdb::timestamp_ns_t>().value}};
    case type_id::VARCHAR: return value{v.GetValue<std::string>()};
    case type_id::DECIMAL: {
      // Precision lives in the supplied logical_type, NOT in the alternative
      // struct. The alt carries only (rep, scale).
      auto const precision = type.decimal_precision();
      auto const scale     = type.decimal_scale();
      // Sirius has only decimal32/64/128, so DuckDB's narrower physical
      // widths are widened on read. GetValueUnsafe<T> reinterprets the
      // value union — it does NOT cast — so each branch must read at the
      // actual storage width (mirrors to_duckdb above and
      // constant.cpp:148-167).
      if (precision <= logical_type::decimal_max_precision_int16) {
        return value{decimal32{static_cast<int32_t>(v.GetValueUnsafe<int16_t>()), scale}};
      }
      if (precision <= logical_type::decimal_max_precision_int32) {
        return value{decimal32{v.GetValueUnsafe<int32_t>(), scale}};
      }
      if (precision <= logical_type::decimal_max_precision_int64) {
        return value{decimal64{v.GetValueUnsafe<int64_t>(), scale}};
      }
      // DECIMAL128 unpack — mirror constant.cpp:162-163 EXACTLY.
      auto const hi  = v.GetValueUnsafe<duckdb::hugeint_t>();
      auto const rep = (static_cast<__int128_t>(hi.upper) << 64) | hi.lower;
      return value{decimal128{rep, scale}};
    }
    default:
      throw not_implemented_exception("sirius::from_duckdb: unsupported type {}", type.to_string());
  }
}

duckdb::Value to_duckdb(const value& v, const logical_type& type)
{
  // Typed NULL recovery — null_value plus the supplied type yields a
  // typed-null duckdb::Value via the `Value(LogicalType)` ctor. Reuses the
  // sirius::to_duckdb(logical_type) overload from helper/type_conversions.hpp.
  if (std::holds_alternative<null_value>(v)) { return duckdb::Value(to_duckdb(type)); }

  switch (type.id()) {
    case type_id::BOOLEAN: return duckdb::Value::BOOLEAN(std::get<bool>(v));
    case type_id::TINYINT: return duckdb::Value::TINYINT(std::get<int8_t>(v));
    case type_id::SMALLINT: return duckdb::Value::SMALLINT(std::get<int16_t>(v));
    case type_id::INTEGER: return duckdb::Value::INTEGER(std::get<int32_t>(v));
    case type_id::BIGINT: return duckdb::Value::BIGINT(std::get<int64_t>(v));
    case type_id::HUGEINT:
      return duckdb::Value::HUGEINT(std::get<int64_t>(v));  // narrowed alt → widen via factory
    case type_id::UTINYINT: return duckdb::Value::UTINYINT(std::get<uint8_t>(v));
    case type_id::USMALLINT: return duckdb::Value::USMALLINT(std::get<uint16_t>(v));
    case type_id::UINTEGER: return duckdb::Value::UINTEGER(std::get<uint32_t>(v));
    case type_id::UBIGINT: return duckdb::Value::UBIGINT(std::get<uint64_t>(v));
    case type_id::UHUGEINT: return duckdb::Value::UHUGEINT(std::get<uint64_t>(v));
    case type_id::FLOAT: return duckdb::Value::FLOAT(std::get<float>(v));
    case type_id::DOUBLE: return duckdb::Value::DOUBLE(std::get<double>(v));
    case type_id::DATE: return duckdb::Value::DATE(duckdb::date_t{std::get<date_value>(v).days});
    case type_id::TIMESTAMP_SEC:
      return duckdb::Value::TIMESTAMPSEC(
        duckdb::timestamp_sec_t{std::get<timestamp_sec_value>(v).value});
    case type_id::TIMESTAMP_MS:
      return duckdb::Value::TIMESTAMPMS(
        duckdb::timestamp_ms_t{std::get<timestamp_ms_value>(v).value});
    case type_id::TIMESTAMP:
      return duckdb::Value::TIMESTAMP(duckdb::timestamp_t{std::get<timestamp_us_value>(v).value});
    case type_id::TIMESTAMP_NS:
      return duckdb::Value::TIMESTAMPNS(
        duckdb::timestamp_ns_t{std::get<timestamp_ns_value>(v).value});
    case type_id::VARCHAR: return duckdb::Value(std::get<std::string>(v));
    case type_id::DECIMAL: {
      auto const precision = type.decimal_precision();
      auto const scale     = type.decimal_scale();
      if (precision <= logical_type::decimal_max_precision_int16) {
        // DuckDB physical width int16_t — narrow decimal32 down.
        auto const& d = std::get<decimal32>(v);
        return duckdb::Value::DECIMAL(static_cast<int16_t>(d.value), precision, scale);
      }
      if (precision <= logical_type::decimal_max_precision_int32) {
        auto const& d = std::get<decimal32>(v);
        return duckdb::Value::DECIMAL(d.value, precision, scale);
      }
      if (precision <= logical_type::decimal_max_precision_int64) {
        auto const& d = std::get<decimal64>(v);
        return duckdb::Value::DECIMAL(d.value, precision, scale);
      }
      // DECIMAL128 — repack into duckdb::hugeint_t (reverse of from_duckdb).
      auto const& d = std::get<decimal128>(v);
      duckdb::hugeint_t hi{static_cast<int64_t>(d.value >> 64), static_cast<uint64_t>(d.value)};
      return duckdb::Value::DECIMAL(hi, precision, scale);
    }
    default:
      throw not_implemented_exception("sirius::to_duckdb: unsupported type {}", type.to_string());
  }
}

}  // namespace sirius
