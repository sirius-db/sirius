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

#include <cudf/types.hpp>

#include <duckdb/common/types.hpp>

#include <cstdint>
#include <vector>

namespace sirius::test::operator_utils {

template <typename T>
struct gpu_type_traits;

template <>
struct gpu_type_traits<int32_t> {
  using type = int32_t;
  static duckdb::LogicalType logical_type()
  {
    return duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
  }
  static constexpr cudf::type_id cudf_type = cudf::type_id::INT32;
  static constexpr bool is_decimal         = false;
  static constexpr bool is_string          = false;
  static constexpr bool is_ts              = false;
  static std::vector<type> sample_values() { return {1, 3, 5, 7}; }
  static type threshold() { return 3; }
};

template <>
struct gpu_type_traits<int64_t> {
  using type = int64_t;
  static duckdb::LogicalType logical_type()
  {
    return duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT);
  }
  static constexpr cudf::type_id cudf_type = cudf::type_id::INT64;
  static constexpr bool is_decimal         = false;
  static constexpr bool is_string          = false;
  static constexpr bool is_ts              = false;
  static std::vector<type> sample_values() { return {2, 4, 6, 8}; }
  static type threshold() { return 4; }
};

template <>
struct gpu_type_traits<float> {
  using type = float;
  static duckdb::LogicalType logical_type()
  {
    return duckdb::LogicalType(duckdb::LogicalTypeId::FLOAT);
  }
  static constexpr cudf::type_id cudf_type = cudf::type_id::FLOAT32;
  static constexpr bool is_decimal         = false;
  static constexpr bool is_string          = false;
  static constexpr bool is_ts              = false;
  static std::vector<type> sample_values() { return {1.0F, 2.5F, 4.5F}; }
  static type threshold() { return 2.0F; }
};

template <>
struct gpu_type_traits<double> {
  using type = double;
  static duckdb::LogicalType logical_type()
  {
    return duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE);
  }
  static constexpr cudf::type_id cudf_type = cudf::type_id::FLOAT64;
  static constexpr bool is_decimal         = false;
  static constexpr bool is_string          = false;
  static constexpr bool is_ts              = false;
  static std::vector<type> sample_values() { return {1.5, 2.5, 3.5}; }
  static type threshold() { return 2.0; }
};

template <>
struct gpu_type_traits<int16_t> {
  using type = int16_t;
  static duckdb::LogicalType logical_type()
  {
    return duckdb::LogicalType(duckdb::LogicalTypeId::SMALLINT);
  }
  static constexpr cudf::type_id cudf_type = cudf::type_id::INT16;
  static constexpr bool is_decimal         = false;
  static constexpr bool is_string          = false;
  static constexpr bool is_ts              = false;
  static std::vector<type> sample_values() { return {static_cast<type>(1), static_cast<type>(2)}; }
  static type threshold() { return static_cast<type>(1); }
};

template <>
struct gpu_type_traits<bool> {
  using type = bool;
  static duckdb::LogicalType logical_type()
  {
    return duckdb::LogicalType(duckdb::LogicalTypeId::BOOLEAN);
  }
  static constexpr cudf::type_id cudf_type = cudf::type_id::BOOL8;
  static constexpr bool is_decimal         = false;
  static constexpr bool is_string          = false;
  static constexpr bool is_ts              = false;
  static std::vector<type> sample_values() { return {false, true, true}; }
  static type threshold() { return false; }  // filter keeps true values
};

struct decimal64_tag {};
template <>
struct gpu_type_traits<decimal64_tag> {
  using type = int64_t;  // underlying storage
  static duckdb::LogicalType logical_type() { return duckdb::LogicalType::DECIMAL(18, 2); }
  static constexpr cudf::type_id cudf_type = cudf::type_id::DECIMAL64;
  static constexpr int32_t scale           = -2;
  static constexpr bool is_decimal         = true;
  static constexpr bool is_string          = false;
  static constexpr bool is_ts              = false;
  static std::vector<type> sample_values() { return {100, 250, 350}; }  // 1.00, 2.50, 3.50
  static type threshold() { return 200; }                               // 2.00
};

struct string_tag {};
template <>
struct gpu_type_traits<string_tag> {
  using type = std::string;
  static duckdb::LogicalType logical_type()
  {
    return duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
  }
  static constexpr cudf::type_id cudf_type = cudf::type_id::STRING;
  static constexpr bool is_decimal         = false;
  static constexpr bool is_string          = true;
  static constexpr bool is_ts              = false;
  static std::vector<type> sample_values() { return {"apple", "banana", "cherry"}; }
  static type threshold() { return std::string("banana"); }  // lex compare
};

struct timestamp_us_tag {};
template <>
struct gpu_type_traits<timestamp_us_tag> {
  using type = int64_t;  // microseconds since epoch
  static duckdb::LogicalType logical_type()
  {
    return duckdb::LogicalType(duckdb::LogicalTypeId::TIMESTAMP);
  }
  static constexpr cudf::type_id cudf_type = cudf::type_id::TIMESTAMP_MICROSECONDS;
  static constexpr bool is_decimal         = false;
  static constexpr bool is_string          = false;
  static constexpr bool is_ts              = true;
  static std::vector<type> sample_values() { return {1'000'000, 2'000'000, 3'000'000}; }
  static type threshold() { return 1'500'000; }
};

struct date32_tag {};
template <>
struct gpu_type_traits<date32_tag> {
  using type = int32_t;  // days since epoch
  static duckdb::LogicalType logical_type()
  {
    return duckdb::LogicalType(duckdb::LogicalTypeId::DATE);
  }
  static constexpr cudf::type_id cudf_type = cudf::type_id::TIMESTAMP_DAYS;
  static constexpr bool is_decimal         = false;
  static constexpr bool is_string          = false;
  static constexpr bool is_ts              = true;
  static std::vector<type> sample_values() { return {0, 1, 5}; }
  static type threshold() { return 1; }
};

}  // namespace sirius::test::operator_utils
