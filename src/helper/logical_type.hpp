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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace sirius {

//===----------------------------------------------------------------------===//
// type_id
//===----------------------------------------------------------------------===//

/**
 * @brief Sirius SQL type identifiers — independent of DuckDB and cuDF.
 *
 * Covers all SQL types currently supported by the Sirius GPU engine.
 * For DECIMAL, precision and scale are carried separately in logical_type.
 */
enum class type_id : uint8_t {
  INVALID = 0,
  SQLNULL,
  BOOLEAN,
  TINYINT,
  SMALLINT,
  INTEGER,
  BIGINT,
  HUGEINT,
  UTINYINT,
  USMALLINT,
  UINTEGER,
  UBIGINT,
  UHUGEINT,
  FLOAT,
  DOUBLE,
  DATE,
  TIMESTAMP_SEC,
  TIMESTAMP_MS,
  TIMESTAMP,  ///< microsecond precision (default SQL TIMESTAMP)
  TIMESTAMP_NS,
  VARCHAR,
  STRUCT,
  LIST,
  ARRAY,  // fixed-sized list type
  DECIMAL,
};

//===----------------------------------------------------------------------===//
// logical_type
//===----------------------------------------------------------------------===//

/**
 * @brief Sirius-native SQL type descriptor.
 *
 * A lightweight, copyable value type that carries:
 * - The base SQL type via sirius::type_id
 * - For DECIMAL: precision (total significant digits) and scale (fractional digits)
 *
 * No DuckDB or cuDF headers are required. Use type_conversions.hpp at the
 * DuckDB boundaries (plan generator entry / result collector exit).
 */
class logical_type {
 public:
  //===--------------------------------------------------------------------===//
  // DECIMAL precision thresholds
  //
  // Maximum number of decimal digits representable in each signed integer width.
  // These mirror duckdb::Decimal::MAX_WIDTH_INT{16,32,64,128} and determine
  // the physical storage type (INT16/32/64/128) for a given DECIMAL precision.
  //===--------------------------------------------------------------------===//
  static constexpr uint8_t decimal_max_precision_int16 =
    4;  ///< INT16  (2B): up to 4 digits (max 9,999)
  static constexpr uint8_t decimal_max_precision_int32 =
    9;  ///< INT32  (4B): up to 9 digits (max 999,999,999)
  static constexpr uint8_t decimal_max_precision_int64  = 18;  ///< INT64  (8B): up to 18 digits
  static constexpr uint8_t decimal_max_precision_int128 = 38;  ///< INT128 (16B): up to 38 digits

  //===--------------------------------------------------------------------===//
  // Constructors / factories
  //===--------------------------------------------------------------------===//

  /// Default-constructs a SQLNULL type (used as a sentinel/placeholder).
  logical_type() : _id(type_id::SQLNULL), _precision(0), _scale(0), _array_size(0), _child(nullptr)
  {
  }

  /**
   * @brief Construct a non-DECIMAL logical type.
   * @param id  The SQL type identifier (must not be DECIMAL).
   */
  static logical_type make(type_id id) { return logical_type(id, 0, 0); }

  /**
   * @brief Construct a DECIMAL logical type with full precision and scale.
   * @param precision  Total significant digits (1–38).
   * @param scale      Fractional digits (0–precision).
   */
  static logical_type make_decimal(uint8_t precision, uint8_t scale)
  {
    return logical_type(type_id::DECIMAL, precision, scale);
  }

  /**
   * @brief Construct an ARRAY logical type with child element type and size.
   * @param child  The type of elements in the array.
   * @param size   Fixed array size (0 = any size).
   */
  static logical_type make_array(const logical_type& child, uint32_t size)
  {
    logical_type result(type_id::ARRAY, 0, 0);
    result._child      = std::make_shared<logical_type>(child);
    result._array_size = size;
    return result;
  }

  //===--------------------------------------------------------------------===//
  // Type inspection
  //===--------------------------------------------------------------------===//

  /// Returns the base SQL type identifier.
  type_id id() const noexcept { return _id; }

  /// Returns true if this is a DECIMAL type.
  bool is_decimal() const noexcept { return _id == type_id::DECIMAL; }

  /// Returns true if this is a VARCHAR (variable-length string) type.
  bool is_varchar() const noexcept { return _id == type_id::VARCHAR; }

  /// Returns true if this is a fixed-size ARRAY type
  bool is_array() const noexcept { return _id == type_id::ARRAY; }

  /// Returns true if this is an integer type (signed or unsigned, including HUGEINT variants).
  bool is_integer() const noexcept
  {
    switch (_id) {
      case type_id::TINYINT:
      case type_id::SMALLINT:
      case type_id::INTEGER:
      case type_id::BIGINT:
      case type_id::HUGEINT:
      case type_id::UTINYINT:
      case type_id::USMALLINT:
      case type_id::UINTEGER:
      case type_id::UBIGINT:
      case type_id::UHUGEINT: return true;
      default: return false;
    }
  }

  /// Returns true if this is a numeric type (integer, float, double, or decimal).
  bool is_numeric() const noexcept
  {
    return is_integer() || _id == type_id::FLOAT || _id == type_id::DOUBLE ||
           _id == type_id::DECIMAL;
  }

  /// Returns true if this is a date or timestamp type (any precision).
  bool is_temporal() const noexcept
  {
    switch (_id) {
      case type_id::DATE:
      case type_id::TIMESTAMP_SEC:
      case type_id::TIMESTAMP_MS:
      case type_id::TIMESTAMP:
      case type_id::TIMESTAMP_NS: return true;
      default: return false;
    }
  }

  /**
   * @brief Returns true for types with a fixed, known storage size.
   *
   * All types except VARCHAR (variable-length), STRUCT (nested), SQLNULL, and INVALID.
   * Unlike fixed_width_byte_size(), this does NOT throw — safe to use in predicates.
   */
  bool is_fixed_width() const noexcept
  {
    return _id != type_id::VARCHAR && _id != type_id::STRUCT && _id != type_id::LIST &&
           _id != type_id::ARRAY && _id != type_id::SQLNULL && _id != type_id::INVALID;
  }

  /**
   * @brief Returns the total number of significant digits for a DECIMAL type.
   * @note Only meaningful when is_decimal() == true.
   */
  uint8_t decimal_precision() const noexcept { return _precision; }

  /**
   * @brief Returns the number of fractional digits for a DECIMAL type.
   * @note Only meaningful when is_decimal() == true.
   */
  uint8_t decimal_scale() const noexcept { return _scale; }

  /**
   * @brief Returns the child element type for ARRAY types.
   * @note Only valid when id() == type_id::ARRAY and has_child() == true.
   */
  const logical_type& array_child() const
  {
    if (!_child) { throw std::runtime_error("array_child: ARRAY type has no child metadata"); }
    return *_child;
  }

  /**
   * @brief Returns the fixed array size, or 0 if any size is allowed.
   * @note Only meaningful when id() == type_id::ARRAY.
   */
  uint32_t array_size() const noexcept { return _array_size; }

  /**
   * @brief Returns true if this type has child type metadata (ARRAY/LIST).
   */
  bool has_child() const noexcept { return _child != nullptr; }

  /**
   * @brief Returns the storage size in bytes for fixed-width types.
   *
   * Replaces `duckdb::GetTypeIdSize(type.InternalType())` for fixed-width dispatch.
   * Returns 0 for VARCHAR (variable-length). Throws for STRUCT, INVALID, and SQLNULL.
   */
  std::size_t fixed_width_byte_size() const
  {
    switch (_id) {
      case type_id::BOOLEAN:
      case type_id::TINYINT:
      case type_id::UTINYINT: return 1;
      case type_id::SMALLINT:
      case type_id::USMALLINT: return 2;
      case type_id::INTEGER:
      case type_id::UINTEGER:
      case type_id::FLOAT:
      case type_id::DATE: return 4;
      case type_id::BIGINT:
      case type_id::UBIGINT:
      case type_id::HUGEINT:   // cuDF maps HUGEINT → INT64
      case type_id::UHUGEINT:  // cuDF maps UHUGEINT → UINT64
      case type_id::DOUBLE:
      case type_id::TIMESTAMP_SEC:
      case type_id::TIMESTAMP_MS:
      case type_id::TIMESTAMP:
      case type_id::TIMESTAMP_NS: return 8;
      case type_id::DECIMAL:
        if (_precision <= decimal_max_precision_int32) return 4;  // DECIMAL32
        if (_precision <= decimal_max_precision_int64) return 8;  // DECIMAL64
        return 16;                                                // DECIMAL128
      case type_id::VARCHAR: return 0;                            // variable-length
      case type_id::LIST:
      case type_id::ARRAY:
      case type_id::STRUCT:
      case type_id::SQLNULL:
      case type_id::INVALID:
      default:
        throw std::runtime_error("fixed_width_byte_size: not applicable for type " + to_string());
    }
  }

  /// Returns a human-readable name for the type (for error messages / logging).
  std::string to_string() const
  {
    switch (_id) {
      case type_id::INVALID: return "INVALID";
      case type_id::SQLNULL: return "NULL";
      case type_id::BOOLEAN: return "BOOLEAN";
      case type_id::TINYINT: return "TINYINT";
      case type_id::SMALLINT: return "SMALLINT";
      case type_id::INTEGER: return "INTEGER";
      case type_id::BIGINT: return "BIGINT";
      case type_id::HUGEINT: return "HUGEINT";
      case type_id::UTINYINT: return "UTINYINT";
      case type_id::USMALLINT: return "USMALLINT";
      case type_id::UINTEGER: return "UINTEGER";
      case type_id::UBIGINT: return "UBIGINT";
      case type_id::UHUGEINT: return "UHUGEINT";
      case type_id::FLOAT: return "FLOAT";
      case type_id::DOUBLE: return "DOUBLE";
      case type_id::DATE: return "DATE";
      case type_id::TIMESTAMP_SEC: return "TIMESTAMP_S";
      case type_id::TIMESTAMP_MS: return "TIMESTAMP_MS";
      case type_id::TIMESTAMP: return "TIMESTAMP";
      case type_id::TIMESTAMP_NS: return "TIMESTAMP_NS";
      case type_id::VARCHAR: return "VARCHAR";
      case type_id::STRUCT: return "STRUCT";
      case type_id::LIST: return "LIST";
      case type_id::ARRAY: {
        std::string child = _child ? _child->to_string() : "?";
        return _array_size == 0 ? child + "[ANY]" : child + "[" + std::to_string(_array_size) + "]";
      }
      case type_id::DECIMAL:
        return "DECIMAL(" + std::to_string(_precision) + "," + std::to_string(_scale) + ")";
      default: return "UNKNOWN";
    }
  }

  //===--------------------------------------------------------------------===//
  // Comparison
  //===--------------------------------------------------------------------===//

  bool operator==(const logical_type& other) const noexcept
  {
    // base field
    bool base_eq = _id == other._id && _precision == other._precision && _scale == other._scale &&
                   _array_size == other._array_size;
    if (!base_eq) return false;

    // recursively compare types for child
    if (_child && other._child) return *_child == *other._child;
    return !_child && !other._child;
  }

  bool operator!=(const logical_type& other) const noexcept { return !(*this == other); }

 private:
  explicit logical_type(type_id id, uint8_t precision, uint8_t scale)
    : _id(id), _precision(precision), _scale(scale), _array_size(0), _child(nullptr)
  {
  }

  type_id _id;
  uint8_t _precision{0};  ///< Meaningful only for DECIMAL: total significant digits (1–38)
  uint8_t _scale{0};      ///< Meaningful only for DECIMAL: fractional digits (0–precision)

  // array
  uint32_t _array_size{0};               ///< Meaningful only for ARRAY: fixed size (0 = any size)
  std::shared_ptr<logical_type> _child;  ///< Child type for ARRAY/LIST (nullptr if not nested)
};

}  // namespace sirius
