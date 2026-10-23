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

/**
 * @file test_decimal_sum_widening.cpp
 * @brief Direct tests of sums_needing_widening, the exact overflow proof that decides
 * whether a DECIMAL32/DECIMAL64 SUM input or an INT64 wide integer SUM input must be widened, and
 * of the merge-side checks for wide integer SUMs.
 *
 * The proof widens a column when rows * max|value| exceeds the storage width's maximum
 * (2^31 - 1 or 2^63 - 1). The extremes below sit exactly on that boundary, including the most
 * negative values (-2^31, -2^63), whose magnitude is one more than the maximum.
 */

#include "op/aggregate/aggregate_op_util.hpp"
#include "op/aggregate/wide_integer_sum.hpp"
#include "operator/operator_test_utils.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <bit>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

namespace {

using sirius::op::sums_needing_widening;

/// A column of type @p type whose rows hold @p values (the storage representation), all valid.
template <typename Rep>
std::unique_ptr<cudf::column> make_column(cudf::data_type type, std::vector<Rep> const& values)
{
  auto column = cudf::make_fixed_width_column(type,
                                              static_cast<cudf::size_type>(values.size()),
                                              cudf::mask_state::UNALLOCATED,
                                              cudf::get_default_stream(),
                                              cudf::get_current_device_resource_ref());
  REQUIRE(cudaMemcpy(column->mutable_view().data<Rep>(),
                     values.data(),
                     values.size() * sizeof(Rep),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  return column;
}

std::unique_ptr<cudf::column> d64(std::vector<int64_t> const& values)
{
  return make_column(cudf::data_type{cudf::type_id::DECIMAL64, -2}, values);
}

std::unique_ptr<cudf::column> d32(std::vector<int32_t> const& values)
{
  return make_column(cudf::data_type{cudf::type_id::DECIMAL32, -2}, values);
}

/// Marks every row of @p column NULL.
void set_all_null(cudf::column& column)
{
  column.set_null_mask(cudf::create_null_mask(column.size(),
                                              cudf::mask_state::ALL_NULL,
                                              cudf::get_default_stream(),
                                              cudf::get_current_device_resource_ref()),
                       column.size());
}

/// Gives @p column (at most 32 rows) the validity @p bits: row i is valid when bit i is set.
void set_validity(cudf::column& column, uint32_t bits)
{
  column.set_null_mask(cudf::create_null_mask(column.size(),
                                              cudf::mask_state::ALL_VALID,
                                              cudf::get_default_stream(),
                                              cudf::get_current_device_resource_ref()),
                       0);
  REQUIRE(
    cudaMemcpy(column.mutable_view().null_mask(), &bits, sizeof(bits), cudaMemcpyHostToDevice) ==
    cudaSuccess);
  column.set_null_count(column.size() - std::popcount(bits));
}

/// The set of columns (by index) the proof says must be widened.
std::unordered_set<int> widened(std::vector<cudf::column_view> const& columns,
                                std::vector<int> const& candidates)
{
  return sums_needing_widening(cudf::table_view(columns),
                               candidates,
                               cudf::get_default_stream(),
                               cudf::get_current_device_resource_ref());
}

bool widens(std::unique_ptr<cudf::column> const& column)
{
  return widened({column->view()}, {0}).contains(0);
}

constexpr int64_t kMin64 = std::numeric_limits<int64_t>::min();
constexpr int64_t kMax64 = std::numeric_limits<int64_t>::max();
constexpr int32_t kMin32 = std::numeric_limits<int32_t>::min();
constexpr int32_t kMax32 = std::numeric_limits<int32_t>::max();

}  // namespace

TEST_CASE("decimal sum widening - DECIMAL64 extremes on the 2^63 - 1 boundary",
          "[aggregate][decimal_sum_overflow]")
{
  // One row of the most negative value has magnitude 2^63, one more than the limit.
  CHECK(widens(d64({kMin64})));
  CHECK_FALSE(widens(d64({kMin64 + 1})));  // magnitude 2^63 - 1 == limit
  CHECK_FALSE(widens(d64({kMax64})));      // 1 * (2^63 - 1) == limit
  CHECK(widens(d64({kMax64, 0})));         // 2 rows * (2^63 - 1) > limit
  // A mixed batch is bounded by its largest magnitude, whichever sign it has.
  CHECK(widens(d64({5, kMin64, 7})));
  CHECK_FALSE(widens(d64({kMin64 + 1})));
}

TEST_CASE("decimal sum widening - DECIMAL64 rows times magnitude around 2^63",
          "[aggregate][decimal_sum_overflow]")
{
  constexpr int64_t half = kMax64 / 2;       // 4611686018427387903
  CHECK_FALSE(widens(d64({half, half})));    // 2 * half = 2^63 - 2 < limit
  CHECK(widens(d64({half + 1, half + 1})));  // 2 * 2^62 = 2^63 > limit
  CHECK_FALSE(widens(d64({-half, -half})));  // negatives are bounded the same way
  CHECK(widens(d64({-(half + 1), -(half + 1)})));
}

TEST_CASE("decimal sum widening - DECIMAL32 extremes on the 2^31 - 1 boundary",
          "[aggregate][decimal_sum_overflow]")
{
  CHECK(widens(d32({kMin32})));            // magnitude 2^31 > limit
  CHECK_FALSE(widens(d32({kMin32 + 1})));  // magnitude 2^31 - 1 == limit
  CHECK_FALSE(widens(d32({kMax32})));
  CHECK(widens(d32({kMax32, 0})));
  constexpr int32_t half = kMax32 / 2;  // 1073741823
  CHECK_FALSE(widens(d32({half, half})));
  CHECK(widens(d32({half + 1, half + 1})));  // 2 * 2^30 = 2^31 > limit
  CHECK(widens(d32({-(half + 1), -(half + 1)})));
}

TEST_CASE("decimal sum widening - several candidates are decided independently",
          "[aggregate][decimal_sum_overflow]")
{
  auto narrow32 = d32({1, 2, 3});
  auto wide64   = d64({kMax64, 1, 1});
  auto narrow64 = d64({9999999999999999, 1, 1});  // 3 * 1e16 << 2^63
  auto result   = widened({narrow32->view(), wide64->view(), narrow64->view()}, {0, 1, 2});
  CHECK(result == std::unordered_set<int>{1});
  // Only the listed candidates are examined.
  CHECK(widened({narrow32->view(), wide64->view(), narrow64->view()}, {0, 2}).empty());
}

TEST_CASE("decimal sum widening - nulls, empty input and no candidates never widen",
          "[aggregate][decimal_sum_overflow]")
{
  // A column whose only large value is null has no valid extreme: nothing can overflow.
  auto all_null = d64({kMin64, kMin64});
  set_all_null(*all_null);
  CHECK_FALSE(widens(all_null));

  // The large value is null, so the valid ones decide: 3 rows * 5 is far below the limit.
  auto some_null = d64({kMin64, 5, 5});
  set_validity(*some_null, 0b110);  // row 0 invalid
  CHECK_FALSE(widens(some_null));

  CHECK_FALSE(widens(d64({})));  // zero rows
  auto column = d64({kMax64, kMax64});
  CHECK(widened({column->view()}, {}).empty());  // no candidates
}

TEST_CASE("decimal sum widening - a non-decimal candidate is rejected",
          "[aggregate][decimal_sum_overflow]")
{
  auto ints = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                        4,
                                        cudf::mask_state::UNALLOCATED,
                                        cudf::get_default_stream(),
                                        cudf::get_current_device_resource_ref());
  CHECK_THROWS_AS(widened({ints->view()}, {0}), sirius::internal_exception);
}

namespace {

using sirius::op::sum_candidate;
using sirius::op::sum_may_overflow;
using sirius::op::sums_to_widen;

std::unordered_set<int> to_widen(std::vector<cudf::column_view> const& columns,
                                 std::vector<sum_candidate> const& candidates)
{
  return sums_to_widen(cudf::table_view(columns),
                       candidates,
                       cudf::get_default_stream(),
                       cudf::get_current_device_resource_ref());
}

}  // namespace

TEST_CASE("decimal sum widening - a plan-time bound decides from the row count alone",
          "[aggregate][decimal_sum_overflow]")
{
  auto const d64t = cudf::data_type{cudf::type_id::DECIMAL64, -2};
  auto const d32t = cudf::data_type{cudf::type_id::DECIMAL32, -2};
  CHECK_FALSE(sum_may_overflow(d64t, 0, std::numeric_limits<uint64_t>::max()));
  CHECK_FALSE(sum_may_overflow(d64t, 1, static_cast<uint64_t>(kMax64)));  // == limit
  CHECK(sum_may_overflow(d64t, 2, static_cast<uint64_t>(kMax64)));
  CHECK(sum_may_overflow(d64t, 1, uint64_t{1} << 63));  // |INT64_MIN|
  // A batch has fewer than 2^31 rows, so a magnitude up to 2^32 never overflows DECIMAL64.
  CHECK_FALSE(sum_may_overflow(d64t, kMax32, uint64_t{1} << 32));
  CHECK_FALSE(sum_may_overflow(d32t, 2, static_cast<uint64_t>(kMax32) / 2));  // 2^31 - 2
  CHECK(sum_may_overflow(d32t, 3, static_cast<uint64_t>(kMax32) / 2));
  // INT64 (a wide integer SUM) has the DECIMAL64 limit.
  auto const i64t = cudf::data_type{cudf::type_id::INT64};
  CHECK_FALSE(sum_may_overflow(i64t, 1, static_cast<uint64_t>(kMax64)));
  CHECK(sum_may_overflow(i64t, 2, static_cast<uint64_t>(kMax64)));
  CHECK_THROWS_AS(sum_may_overflow(cudf::data_type{cudf::type_id::INT32}, 1, 1),
                  sirius::internal_exception);
}

TEST_CASE("decimal sum widening - bounded columns are decided on the host, the rest measured",
          "[aggregate][decimal_sum_overflow]")
{
  auto const big   = d64({kMax64, 0});  // 2 rows * (2^63 - 1) > limit when measured
  auto const small = d64({5, 7});
  auto const ids   = [](std::initializer_list<int> columns) {
    return std::unordered_set<int>(columns);
  };
  // A bound is the planner's promise about the values; the column itself is not read.
  CHECK(to_widen({big->view()}, {{0, static_cast<uint64_t>(kMax64)}}) == ids({0}));
  CHECK(to_widen({small->view()}, {{0, 7}}).empty());
  // Without a bound the batch is measured.
  CHECK(to_widen({big->view()}, {{0, std::nullopt}}) == ids({0}));
  CHECK(to_widen({small->view()}, {{0, std::nullopt}}).empty());
  // One unproven candidate sends its whole column to measurement.
  CHECK(to_widen({big->view()}, {{0, 1}, {0, std::nullopt}}) == ids({0}));
  // Several bounded candidates on one column take the largest bound.
  CHECK(to_widen({small->view()}, {{0, 1}, {0, static_cast<uint64_t>(kMax64)}}) == ids({0}));
  // Columns are independent: a measured column beside a bounded one.
  CHECK(to_widen({big->view(), small->view()}, {{0, std::nullopt}, {1, 7}}) == ids({0}));
}

namespace {

using sirius::op::decimal_sums_to_narrow;
using sirius::op::narrowed_decimal_sum_type;

std::unique_ptr<cudf::column> narrow_d128(std::vector<__int128> const& values)
{
  return make_column(cudf::data_type{cudf::type_id::DECIMAL128, -2}, values);
}

std::unordered_set<int> to_narrow(std::vector<cudf::column_view> const& columns,
                                  std::vector<sum_candidate> const& candidates)
{
  return decimal_sums_to_narrow(cudf::table_view(columns),
                                candidates,
                                cudf::get_default_stream(),
                                cudf::get_current_device_resource_ref());
}

/// Whether the measured values of @p column let its SUM run over DECIMAL64.
bool narrows(std::unique_ptr<cudf::column> const& column)
{
  return to_narrow({column->view()}, {{0, std::nullopt}}).contains(0);
}

using sirius::op::is_wide_integer_sum;
using sirius::op::narrow_wide_integer_sum;
using sirius::op::partial_sums_may_overflow;
using sirius::op::wide_integer_sum_type;

std::unique_ptr<cudf::column> i64(std::vector<int64_t> const& values)
{
  return make_column(cudf::data_type{cudf::type_id::INT64}, values);
}

std::unique_ptr<cudf::column> i32(std::vector<int32_t> const& values)
{
  return make_column(cudf::data_type{cudf::type_id::INT32}, values);
}

std::unique_ptr<cudf::column> d128(std::vector<__int128_t> const& values)
{
  return make_column(wide_integer_sum_type(), values);
}

std::vector<int64_t> to_host(cudf::column_view const& column)
{
  return sirius::test::operator_utils::copy_column_to_host<int64_t>(column,
                                                                    cudf::get_default_stream());
}

bool partials_may_overflow(std::unique_ptr<cudf::column> const& column)
{
  return partial_sums_may_overflow(
    column->view(), cudf::get_default_stream(), cudf::get_current_device_resource_ref());
}

std::unique_ptr<cudf::column> narrow(std::unique_ptr<cudf::column> column)
{
  return narrow_wide_integer_sum(
    std::move(column), cudf::get_default_stream(), cudf::get_current_device_resource_ref());
}

}  // namespace

TEST_CASE("decimal sum narrowing - only DECIMAL128 narrows, to DECIMAL64 at the same scale",
          "[aggregate][decimal_sum_overflow]")
{
  CHECK(narrowed_decimal_sum_type(cudf::data_type{cudf::type_id::DECIMAL128, -4}) ==
        cudf::data_type{cudf::type_id::DECIMAL64, -4});
  CHECK_FALSE(narrowed_decimal_sum_type(cudf::data_type{cudf::type_id::DECIMAL64, -2}));
  CHECK_FALSE(narrowed_decimal_sum_type(cudf::data_type{cudf::type_id::INT64}));
  auto const narrow64 = d64({1, 2});
  CHECK_THROWS_AS(to_narrow({narrow64->view()}, {{0, std::nullopt}}), sirius::internal_exception);
}

TEST_CASE("decimal sum narrowing - measured values on the 2^63 - 1 boundary",
          "[aggregate][decimal_sum_overflow]")
{
  __int128 const max64 = kMax64;
  CHECK(narrows(narrow_d128({max64})));             // 1 * (2^63 - 1) == limit
  CHECK_FALSE(narrows(narrow_d128({max64, 0})));    // 2 rows * (2^63 - 1) > limit
  CHECK_FALSE(narrows(narrow_d128({max64 + 1})));   // a value beyond 64 bits
  CHECK(narrows(narrow_d128({-max64})));            // magnitude 2^63 - 1
  CHECK_FALSE(narrows(narrow_d128({-max64 - 1})));  // INT64_MIN: magnitude 2^63
  CHECK_FALSE(narrows(narrow_d128({__int128{1} << 100})));
  CHECK_FALSE(narrows(narrow_d128({-(__int128{1} << 100), 1})));
  constexpr int64_t half = kMax64 / 2;
  CHECK(narrows(narrow_d128({half, -half})));               // 2 * (2^62 - 1) < limit
  CHECK_FALSE(narrows(narrow_d128({half + 1, half + 1})));  // 2 * 2^62 = 2^63 > limit
  // TPC-H Q9 profits: four-decimal amounts of a few hundred thousand over many rows.
  CHECK(narrows(narrow_d128({1049999900, -1000000000, 7, 0})));
}

TEST_CASE("decimal sum narrowing - nulls, empty input and no candidates",
          "[aggregate][decimal_sum_overflow]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Only nulls: the sum is NULL in every group at any width.
  auto all_null = narrow_d128({__int128{1} << 100});
  all_null->set_null_mask(
    cudf::create_null_mask(all_null->size(), cudf::mask_state::ALL_NULL, stream, mr),
    all_null->size());
  CHECK(narrows(all_null));

  // The huge value is null, so the valid ones decide.
  auto some_null = narrow_d128({__int128{1} << 100, 5, 5});
  some_null->set_null_mask(
    cudf::create_null_mask(some_null->size(), cudf::mask_state::ALL_VALID, stream, mr), 0);
  std::vector<uint32_t> bits{0b110};  // row 0 invalid
  REQUIRE(cudaMemcpy(some_null->mutable_view().null_mask(),
                     bits.data(),
                     sizeof(uint32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  some_null->set_null_count(1);
  CHECK(narrows(some_null));

  CHECK_FALSE(narrows(narrow_d128({})));  // zero rows: nothing to gain
  auto column = narrow_d128({1, 2});
  CHECK(to_narrow({column->view()}, {}).empty());
}

TEST_CASE("decimal sum narrowing - plan-time bounds are decided on the host, the rest measured",
          "[aggregate][decimal_sum_overflow]")
{
  auto const big   = narrow_d128({__int128{1} << 100, 0});
  auto const small = narrow_d128({5, 7});
  auto const ids   = [](std::initializer_list<int> columns) {
    return std::unordered_set<int>(columns);
  };
  // A bound is the planner's promise about the values; the column itself is not read.
  CHECK(to_narrow({big->view()}, {{0, 7}}) == ids({0}));
  CHECK(to_narrow({small->view()}, {{0, static_cast<uint64_t>(kMax64)}}).empty());  // 2 rows
  CHECK(to_narrow({small->view()}, {{0, uint64_t{1} << 63}}).empty());
  // Without a bound the batch is measured.
  CHECK(to_narrow({big->view()}, {{0, std::nullopt}}).empty());
  CHECK(to_narrow({small->view()}, {{0, std::nullopt}}) == ids({0}));
  // One unproven candidate sends its whole column to measurement.
  CHECK(to_narrow({big->view()}, {{0, 7}, {0, std::nullopt}}).empty());
  // Columns are independent.
  CHECK(to_narrow({big->view(), small->view()}, {{0, std::nullopt}, {1, std::nullopt}}) ==
        ids({1}));
}

TEST_CASE("wide integer sum - which aggregates DuckDB accumulates in HUGEINT",
          "[aggregate][wide_integer_sum]")
{
  using sirius::aggregate_id;
  using sirius::logical_type;
  using sirius::type_id;
  for (auto id : {type_id::INTEGER, type_id::BIGINT, type_id::HUGEINT}) {
    CHECK(is_wide_integer_sum(aggregate_id::sum, logical_type::make(id)));
    CHECK(is_wide_integer_sum(aggregate_id::avg, logical_type::make(id)));
    // DuckDB's statistics proved the total fits in INT64.
    CHECK_FALSE(is_wide_integer_sum(aggregate_id::sum_no_overflow, logical_type::make(id)));
    CHECK_FALSE(is_wide_integer_sum(aggregate_id::min, logical_type::make(id)));
  }
  // DuckDB itself sums these in INT64; decimals have their own widening.
  for (auto id : {type_id::TINYINT, type_id::SMALLINT, type_id::BOOLEAN, type_id::DOUBLE}) {
    CHECK_FALSE(is_wide_integer_sum(aggregate_id::sum, logical_type::make(id)));
  }
  CHECK_FALSE(is_wide_integer_sum(aggregate_id::sum, logical_type::make_decimal(18, 2)));
}

TEST_CASE("wide integer sum - INT64 inputs are measured like DECIMAL64",
          "[aggregate][wide_integer_sum]")
{
  CHECK(widens(i64({kMin64})));
  CHECK_FALSE(widens(i64({kMin64 + 1})));
  CHECK_FALSE(widens(i64({kMax64})));
  CHECK(widens(i64({kMax64, 0})));
  // The ClickBench Q03 shape: ten values of 4e18 sum past 2^63 - 1.
  CHECK(widens(i64(std::vector<int64_t>(10, 4000000000000000000))));
  CHECK_FALSE(widens(i64(std::vector<int64_t>(2, 4000000000000000000))));
  CHECK(sirius::op::widened_sum_type(cudf::data_type{cudf::type_id::INT64}) ==
        wide_integer_sum_type());
}

TEST_CASE("wide integer sum - INT64 partials are merged in INT64 only below 2^62 in total",
          "[aggregate][wide_integer_sum]")
{
  constexpr int64_t two_pow_60 = int64_t{1} << 60;
  constexpr int64_t two_pow_61 = int64_t{1} << 61;
  CHECK_FALSE(partials_may_overflow(i64({})));
  CHECK_FALSE(partials_may_overflow(i64({1, -2, 3})));
  CHECK_FALSE(partials_may_overflow(i64({two_pow_61, -two_pow_60})));  // 3 * 2^60 < 2^62
  CHECK(partials_may_overflow(i64({two_pow_61, -two_pow_61})));        // magnitudes reach 2^62
  // Magnitudes count, so a total that cancels to zero is still guarded: its running sum may not.
  CHECK(partials_may_overflow(i64({kMax64, kMin64 + 1})));
  CHECK(partials_may_overflow(i64({kMin64})));
  // Nulls are skipped: an all-null column cannot overflow.
  auto all_null = i64({kMax64, kMax64});
  set_all_null(*all_null);
  CHECK_FALSE(partials_may_overflow(all_null));
  CHECK_THROWS_AS(partials_may_overflow(d64({1})), sirius::internal_exception);
}

TEST_CASE("wide integer sum - a DECIMAL128 total narrows to INT64 or falls back",
          "[aggregate][wide_integer_sum]")
{
  // INT64 passes through untouched.
  auto int64_result = narrow(i64({5, -7}));
  CHECK(int64_result->type().id() == cudf::type_id::INT64);
  CHECK(to_host(int64_result->view()) == std::vector<int64_t>{5, -7});

  // Totals inside the INT64 range, including both extremes, narrow exactly.
  auto narrowed = narrow(d128({kMax64, kMin64, 0}));
  CHECK(narrowed->type().id() == cudf::type_id::INT64);
  CHECK(to_host(narrowed->view()) == std::vector<int64_t>{kMax64, kMin64, 0});

  // One past either end is a HUGEINT result: the query must fall back to DuckDB.
  CHECK_THROWS_AS(narrow(d128({__int128_t{kMax64} + 1})), sirius::not_implemented_exception);
  CHECK_THROWS_AS(narrow(d128({0, __int128_t{kMin64} - 1})), sirius::not_implemented_exception);

  // An all-null total (an all-null group) narrows to an all-null INT64 column.
  auto all_null = d128({__int128_t{kMax64} * 4});
  set_all_null(*all_null);
  auto narrowed_null = narrow(std::move(all_null));
  CHECK(narrowed_null->type().id() == cudf::type_id::INT64);
  CHECK(narrowed_null->null_count() == 1);
}

namespace {

/// The one value of a one-row DECIMAL128 column, or nullopt if it is NULL.
std::optional<__int128_t> single_value(cudf::column_view const& column)
{
  REQUIRE(column.size() == 1);
  REQUIRE(column.type() == wide_integer_sum_type());
  if (column.null_count() == 1) { return std::nullopt; }
  return sirius::test::operator_utils::copy_column_to_host<__int128_t>(
    column, cudf::get_default_stream())[0];
}

std::optional<__int128_t> sum128(cudf::column_view const& column)
{
  auto result = sirius::op::wide_integer_sum(
    column, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  return single_value(result->view());
}

}  // namespace

TEST_CASE("wide integer sum - one-pass 128-bit reduction of INT64 and INT32",
          "[aggregate][wide_integer_sum]")
{
  // Ten rows of 4e18: the repro's 4e19 total, far past INT64.
  CHECK(sum128(i64(std::vector<int64_t>(10, 4000000000000000000))->view()) ==
        __int128_t{4000000000000000000} * 10);
  // Both extremes, repeated, and a total that cancels.
  CHECK(sum128(i64({kMax64, kMax64, kMax64})->view()) == __int128_t{kMax64} * 3);
  CHECK(sum128(i64({kMin64, kMin64})->view()) == __int128_t{kMin64} * 2);
  CHECK(sum128(i64({kMax64, kMin64, 1})->view()) == 0);

  CHECK(sum128(i32({kMax32, kMax32, kMin32})->view()) == __int128_t{kMax32} * 2 + kMin32);

  // NULL rows are skipped.
  auto some_null = i64({kMax64, kMax64, 5});
  set_validity(*some_null, 0b101);  // row 1 invalid
  CHECK(sum128(some_null->view()) == __int128_t{kMax64} + 5);

  // No valid value: the sum is NULL, as SQL's SUM of nothing.
  CHECK_FALSE(sum128(i64({})->view()).has_value());
  auto all_null = i64({1, 2});
  set_all_null(*all_null);
  CHECK_FALSE(sum128(all_null->view()).has_value());

  // A slice keeps its offset.
  auto column = i64({kMax64, 1, 2, 3});
  auto sliced = cudf::slice(column->view(), {1, 4})[0];
  CHECK(sum128(sliced) == 6);

  CHECK_THROWS_AS(sum128(d64({1})->view()), sirius::internal_exception);
}
