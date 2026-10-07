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

// Tests for the type-mapping and schema helpers in cudf/cudf_utils.hpp:
// sirius::estimate_referenced_column_bytes, the get_cudf_type / duckdb::GetCudfType mappings, and
// the two sirius::make_empty_table overloads.
//
// The estimator only reads column metadata (type, size, nullability) and never dereferences
// device buffers, so its tests fabricate metadata-only column_views over dummy non-null
// sentinel pointers. The make_empty_table tests build real zero-row cuDF columns.

#include "catch.hpp"
#include "cudf/cudf_utils.hpp"
#include "helper/logical_type.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <duckdb/common/optional_idx.hpp>
#include <duckdb/common/types.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
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

// ============================================================================
// sirius::make_empty_table — the two schema sources
// ============================================================================

TEST_CASE("make_empty_table - the cudf-type overload reproduces the given carriers", "[cudf_utils]")
{
  // The carrier-exact overload is what lets a caller holding a physical_types sidecar reproduce
  // narrow carriers verbatim; the hash join builds an absent side's empty batch this way.
  std::vector<cudf::data_type> const carriers{cudf::data_type{cudf::type_id::INT16},
                                              cudf::data_type{cudf::type_id::DECIMAL32, -2},
                                              cudf::data_type{cudf::type_id::STRING}};

  auto const table = sirius::make_empty_table(carriers);
  REQUIRE(table != nullptr);
  REQUIRE(table->num_columns() == static_cast<cudf::size_type>(carriers.size()));
  REQUIRE(table->num_rows() == 0);
  for (std::size_t column_idx = 0; column_idx < carriers.size(); column_idx++) {
    REQUIRE(table->get_column(static_cast<cudf::size_type>(column_idx)).type() ==
            carriers[column_idx]);
  }
}

TEST_CASE("make_empty_table - the logical-type overload derives native carriers", "[cudf_utils]")
{
  // Columns whose narrow carriers the overload above reproduces verbatim come out at their wider
  // native carriers here, which is what distinguishes the two schema sources.
  duckdb::vector<logical_type> const types{logical_type::make(type_id::BIGINT),
                                           logical_type::make_decimal(18, 2)};

  auto const table = sirius::make_empty_table(types);
  REQUIRE(table != nullptr);
  REQUIRE(table->num_columns() == 2);
  REQUIRE(table->num_rows() == 0);
  REQUIRE(table->get_column(0).type() == cudf::data_type{cudf::type_id::INT64});
  REQUIRE(table->get_column(1).type() == cudf::data_type{cudf::type_id::DECIMAL64, -2});
}

TEST_CASE("make_empty_table - the logical-type overload builds nested-safe ARRAY columns",
          "[cudf_utils]")
{
  // An ARRAY entry needs cuDF's LIST factory (make_empty_column rejects nested types) and must
  // carry its fixed-width element type on the child column, so empty synthesized tables
  // concatenate/join cleanly against decoded batches.
  duckdb::vector<logical_type> const types{
    logical_type::make(type_id::INTEGER),
    logical_type::make_array(logical_type::make(type_id::INTEGER), 3),
    logical_type::make(type_id::VARCHAR)};

  auto const table = sirius::make_empty_table(types);
  REQUIRE(table != nullptr);
  REQUIRE(table->num_columns() == 3);
  REQUIRE(table->num_rows() == 0);
  REQUIRE(table->get_column(0).type().id() == cudf::type_id::INT32);
  REQUIRE(table->get_column(1).type().id() == cudf::type_id::LIST);
  cudf::lists_column_view const lists(table->get_column(1).view());
  REQUIRE(lists.child().type().id() == cudf::type_id::INT32);
  REQUIRE(table->get_column(2).type().id() == cudf::type_id::STRING);
}

TEST_CASE("make_empty_table - the cudf-type overload refuses nested carriers", "[cudf_utils]")
{
  // An id-only cudf::data_type carries no element type, so the carrier-exact overload cannot
  // synthesize a nested column; it must direct the caller to the logical-type overload instead of
  // surfacing an opaque cudf::logic_error.
  std::vector<cudf::data_type> const carriers{cudf::data_type{cudf::type_id::INT32},
                                              cudf::data_type{cudf::type_id::LIST}};
  REQUIRE_THROWS_AS(sirius::make_empty_table(carriers), duckdb::InvalidInputException);
  REQUIRE_THROWS_WITH(sirius::make_empty_table(carriers),
                      Catch::Matchers::ContainsSubstring("logical-type overload"));
}

TEST_CASE("make_empty_like reproduces nested LIST columns", "[cudf_utils]")
{
  // The empty rebuild must mirror the input's full per-column type hierarchy, not just the
  // top-level type ids -- a fully pruned ARRAY scan hands TOP-N a 0-row LIST view to emulate.
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}));
  cols.push_back(cudf::make_empty_lists_column(cudf::data_type{cudf::type_id::FLOAT64}));
  cudf::table const input(std::move(cols));

  auto const empty = duckdb::make_empty_like(input.view());
  REQUIRE(empty != nullptr);
  REQUIRE(empty->num_columns() == 2);
  REQUIRE(empty->num_rows() == 0);
  REQUIRE(empty->get_column(0).type().id() == cudf::type_id::INT32);
  REQUIRE(empty->get_column(1).type().id() == cudf::type_id::LIST);
  cudf::lists_column_view const lists(empty->get_column(1).view());
  REQUIRE(lists.child().type().id() == cudf::type_id::FLOAT64);
}

TEST_CASE("shared column comparison ignores unused storage and checks row validity",
          "[cudf_utils][column_comparison]")
{
  using sirius::test::columns_logically_equal;
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const make_column          = [stream](std::vector<int32_t> const& values) {
    auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                            static_cast<cudf::size_type>(values.size()),
                                            cudf::mask_state::UNALLOCATED,
                                            stream);
    REQUIRE(cudaMemcpyAsync(column->mutable_view().data<int32_t>(),
                            values.data(),
                            values.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice,
                            stream.get()) == cudaSuccess);
    stream.sync();
    return column;
  };
  auto const null_at = [stream](cudf::column& column, cudf::size_type row) {
    auto mask = cudf::create_null_mask(column.size(), cudf::mask_state::ALL_VALID, stream);
    cudf::set_null_mask(
      reinterpret_cast<cudf::bitmask_type*>(mask.data()), row, row + 1, false, stream);
    column.set_null_mask(std::move(mask), 1);
  };

  SECTION("null payloads are ignored, but a moved null is detected")
  {
    auto lhs = make_column({10, 111, 30});
    auto rhs = make_column({10, 999, 30});
    null_at(*lhs, 1);
    null_at(*rhs, 1);
    REQUIRE(columns_logically_equal(lhs->view(), rhs->view(), stream));
    REQUIRE(sirius::test::expect_tables_equivalent_impl(
      cudf::table_view{{lhs->view()}}, cudf::table_view{{rhs->view()}}, stream));

    null_at(*rhs, 2);
    REQUIRE_FALSE(columns_logically_equal(lhs->view(), rhs->view(), stream));
  }

  SECTION("all-valid masks are optional and mask padding is ignored")
  {
    auto lhs  = make_column({10, 20, 30});
    auto rhs  = make_column({10, 20, 30});
    auto mask = cudf::create_null_mask(3, cudf::mask_state::ALL_VALID, stream);
    rhs->set_null_mask(std::move(mask), 0);
    REQUIRE(columns_logically_equal(lhs->view(), rhs->view(), stream));

    auto other_mask = cudf::create_null_mask(3, cudf::mask_state::ALL_NULL, stream);
    cudf::set_null_mask(
      reinterpret_cast<cudf::bitmask_type*>(other_mask.data()), 0, 3, true, stream);
    lhs->set_null_mask(std::move(other_mask), 0);
    REQUIRE(columns_logically_equal(lhs->view(), rhs->view(), stream));
  }

  SECTION("sliced lists compare referenced elements rather than offsets or unused children")
  {
    auto lhs =
      cudf::make_lists_column(3,
                              make_column({0, 1, 3, 4}),
                              make_column({999, 10, 20, 888}),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream));
    auto rhs =
      cudf::make_lists_column(1,
                              make_column({0, 2}),
                              make_column({10, 20}),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream));
    auto const slice = cudf::slice(lhs->view(), {1, 2}, stream).front();
    REQUIRE(columns_logically_equal(slice, rhs->view(), stream));

    auto different =
      cudf::make_lists_column(1,
                              make_column({0, 2}),
                              make_column({10, 21}),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream));
    REQUIRE_FALSE(columns_logically_equal(slice, different->view(), stream));
  }

  SECTION("string values are checked and sliced views compare against compact storage")
  {
    auto const make_strings = [stream, &make_column](std::string const& text) {
      rmm::device_buffer chars(text.size(), stream);
      REQUIRE(cudaMemcpyAsync(
                chars.data(), text.data(), text.size(), cudaMemcpyHostToDevice, stream.get()) ==
              cudaSuccess);
      stream.sync();
      return cudf::make_strings_column(
        3,
        make_column({0, 1, 2, 3}),
        std::move(chars),
        0,
        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream));
    };
    auto lhs = make_strings("abc");
    auto rhs = make_strings("xbc");
    REQUIRE_FALSE(columns_logically_equal(lhs->view(), rhs->view(), stream));
    REQUIRE(columns_logically_equal(cudf::slice(lhs->view(), {1, 3}, stream).front(),
                                    cudf::slice(rhs->view(), {1, 3}, stream).front(),
                                    stream));
    null_at(*lhs, 0);
    null_at(*rhs, 0);
    REQUIRE(columns_logically_equal(lhs->view(), rhs->view(), stream));
  }

  SECTION("empty columns still check exact types and reject unsupported payloads")
  {
    auto const empty =
      cudf::column_view{cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0};
    REQUIRE(columns_logically_equal(empty, empty, stream));
    auto const unsupported =
      cudf::column_view{cudf::data_type{cudf::type_id::DICTIONARY32}, 0, nullptr, nullptr, 0};
    REQUIRE_FALSE(columns_logically_equal(unsupported, unsupported, stream));
    auto const nested_unsupported = cudf::column_view{
      cudf::data_type{cudf::type_id::LIST}, 0, nullptr, nullptr, 0, 0, {empty, unsupported}};
    REQUIRE_FALSE(columns_logically_equal(nested_unsupported, nested_unsupported, stream));
    auto const decimal_lhs =
      cudf::column_view{cudf::data_type{cudf::type_id::DECIMAL32, -2}, 0, nullptr, nullptr, 0};
    auto const decimal_rhs =
      cudf::column_view{cudf::data_type{cudf::type_id::DECIMAL32, -3}, 0, nullptr, nullptr, 0};
    REQUIRE_FALSE(columns_logically_equal(decimal_lhs, decimal_rhs, stream));
  }
}
