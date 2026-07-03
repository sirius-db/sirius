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

// Tests for sirius::estimate_referenced_column_bytes (cudf/cudf_utils.hpp).
//
// The estimator only reads column metadata (type, size, nullability) and never dereferences
// device buffers, so these tests fabricate metadata-only column_views over dummy non-null
// sentinel pointers — no GPU memory is allocated or touched.

#include "catch.hpp"
#include "cudf/cudf_utils.hpp"
#include "helper/logical_type.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <duckdb/common/optional_idx.hpp>
#include <duckdb/common/types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

using sirius::estimate_referenced_column_bytes;
using sirius::logical_type;
using sirius::type_id;

namespace {

// Sentinels: the estimator never reads through these, it only checks for non-null.
std::int64_t g_data_sentinel          = 0;
cudf::bitmask_type g_mask_sentinel    = 0;
const void* const kData               = &g_data_sentinel;
const cudf::bitmask_type* const kMask = &g_mask_sentinel;

cudf::column_view meta_view(cudf::type_id id, cudf::size_type size, bool nullable)
{
  // STRING is exempt from the "compound columns cannot have data" check, and fixed-width columns
  // require non-null data, so a non-null sentinel data pointer is valid for every type used here.
  return cudf::column_view(
    cudf::data_type{id}, size, kData, nullable ? kMask : nullptr, nullable ? 1 : 0);
}

std::size_t fixed_bytes(cudf::type_id id, cudf::size_type n, bool nullable)
{
  std::size_t b = static_cast<std::size_t>(n) * cudf::size_of(cudf::data_type{id});
  if (nullable) { b += cudf::bitmask_allocation_size_bytes(n); }
  return b;
}

}  // namespace

TEST_CASE("estimate_referenced_column_bytes: fixed vs. single variable-width column",
          "[cudf_utils][estimate]")
{
  constexpr cudf::size_type N = 100;

  // col0: INT64 (not nullable), col1: STRING (variable), col2: INT32 (nullable).
  std::vector<cudf::column_view> cols{meta_view(cudf::type_id::INT64, N, false),
                                      meta_view(cudf::type_id::STRING, N, false),
                                      meta_view(cudf::type_id::INT32, N, true)};
  cudf::table_view input(cols);

  auto const i64_bytes   = fixed_bytes(cudf::type_id::INT64, N, false);
  auto const i32_bytes   = fixed_bytes(cudf::type_id::INT32, N, true);
  auto const fixed_total = i64_bytes + i32_bytes;
  // One variable column, so its attributed average is exactly the whole non-fixed remainder.
  std::size_t const string_bytes = 5000;
  std::size_t const total        = fixed_total + string_bytes;

  SECTION("fixed-width columns are sized exactly")
  {
    REQUIRE(estimate_referenced_column_bytes(input, {0}, total) == i64_bytes);
    REQUIRE(estimate_referenced_column_bytes(input, {2}, total) == i32_bytes);
    REQUIRE(estimate_referenced_column_bytes(input, {0, 2}, total) == fixed_total);
  }
  SECTION("variable-width column gets the non-fixed remainder")
  {
    REQUIRE(estimate_referenced_column_bytes(input, {1}, total) == string_bytes);
  }
  SECTION("referencing every column reconstructs the total")
  {
    REQUIRE(estimate_referenced_column_bytes(input, {0, 1, 2}, total) == total);
  }
  SECTION("duplicate indices are counted once")
  {
    REQUIRE(estimate_referenced_column_bytes(input, {0, 0, 0}, total) == i64_bytes);
    REQUIRE(estimate_referenced_column_bytes(input, {1, 1}, total) == string_bytes);
  }
  SECTION("total smaller than the fixed total clamps the variable remainder to zero")
  {
    REQUIRE(estimate_referenced_column_bytes(input, {1}, 0) == 0);
    REQUIRE(estimate_referenced_column_bytes(input, {0}, 0) == i64_bytes);
  }
}

TEST_CASE("estimate_referenced_column_bytes: averages across multiple variable-width columns",
          "[cudf_utils][estimate]")
{
  constexpr cudf::size_type N = 50;

  // Two STRING columns share the non-fixed remainder; one INT64 column is sized exactly.
  std::vector<cudf::column_view> cols{meta_view(cudf::type_id::STRING, N, false),
                                      meta_view(cudf::type_id::STRING, N, false),
                                      meta_view(cudf::type_id::INT64, N, false)};
  cudf::table_view input(cols);

  auto const i64_bytes        = fixed_bytes(cudf::type_id::INT64, N, false);
  std::size_t const var_total = 4000;  // shared by the two string columns
  std::size_t const total     = i64_bytes + var_total;
  std::size_t const avg_var   = var_total / 2;

  REQUIRE(estimate_referenced_column_bytes(input, {0}, total) == avg_var);
  REQUIRE(estimate_referenced_column_bytes(input, {0, 1}, total) == 2 * avg_var);
  REQUIRE(estimate_referenced_column_bytes(input, {2}, total) == i64_bytes);
  REQUIRE(estimate_referenced_column_bytes(input, {0, 1, 2}, total) == total);
}

// Tests for the cuDF type-mapping helpers in cudf/cudf_utils.hpp:
//   - sirius::get_cudf_type(const sirius::logical_type&)
//   - duckdb::GetCudfType(const duckdb::LogicalType&)
//
// These focus on the ARRAY (fixed-size list) mapping added alongside the
// ARRAY data type, both helpers must lower ARRAY to a cuDF LIST column.

// ============================================================================
// sirius::get_cudf_type — ARRAY mapping
// ============================================================================

TEST_CASE("get_cudf_type - ARRAY maps to cuDF LIST", "[cudf_utils]")
{
  auto arr = logical_type::make_array(logical_type::make(type_id::INTEGER), 3);
  REQUIRE(sirius::get_cudf_type(arr).id() == cudf::type_id::LIST);
}

TEST_CASE("get_cudf_type - ARRAY mapping ignores child type and size", "[cudf_utils]")
{
  // The cuDF type is always LIST regardless of element type or fixed size;
  // the element type is carried separately on the child column
  REQUIRE(
    sirius::get_cudf_type(logical_type::make_array(logical_type::make(type_id::DOUBLE), 1)).id() ==
    cudf::type_id::LIST);
  REQUIRE(
    sirius::get_cudf_type(logical_type::make_array(logical_type::make(type_id::BIGINT), 0)).id() ==
    cudf::type_id::LIST);
  // Nested ARRAY of ARRAYs still maps to a single top-level LIST
  auto nested =
    logical_type::make_array(logical_type::make_array(logical_type::make(type_id::INTEGER), 2), 4);
  REQUIRE(sirius::get_cudf_type(nested).id() == cudf::type_id::LIST);
}

// ============================================================================
// duckdb::GetCudfType — ARRAY mapping
// ============================================================================

TEST_CASE("GetCudfType - duckdb ARRAY maps to cuDF LIST", "[cudf_utils]")
{
  using duckdb::LogicalType;

  auto arr = LogicalType::ARRAY(LogicalType::INTEGER, duckdb::optional_idx(3));
  REQUIRE(duckdb::GetCudfType(arr).id() == cudf::type_id::LIST);
}

TEST_CASE("GetCudfType - duckdb ARRAY mapping is independent of child and size", "[cudf_utils]")
{
  using duckdb::LogicalType;

  REQUIRE(
    duckdb::GetCudfType(LogicalType::ARRAY(LogicalType::DOUBLE, duckdb::optional_idx(1))).id() ==
    cudf::type_id::LIST);
  REQUIRE(
    duckdb::GetCudfType(LogicalType::ARRAY(LogicalType::BIGINT, duckdb::optional_idx(16))).id() ==
    cudf::type_id::LIST);
}
