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
 * @brief Direct tests of decimal_sums_needing_widening, the exact overflow proof that decides
 * whether a DECIMAL32/DECIMAL64 SUM input must be widened.
 *
 * The proof widens a column when rows * max|value| exceeds the storage width's maximum
 * (2^31 - 1 or 2^63 - 1). The extremes below sit exactly on that boundary, including the most
 * negative values (-2^31, -2^63), whose magnitude is one more than the maximum.
 */

#include "op/aggregate/aggregate_op_util.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <unordered_set>
#include <vector>

namespace {

using sirius::op::decimal_sums_needing_widening;

template <typename Rep>
std::unique_ptr<cudf::column> make_decimal_column(cudf::type_id id, std::vector<Rep> const& values)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto column       = cudf::make_fixed_point_column(cudf::data_type{id, -2},
                                              static_cast<cudf::size_type>(values.size()),
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              mr);
  REQUIRE(cudaMemcpy(column->mutable_view().data<Rep>(),
                     values.data(),
                     values.size() * sizeof(Rep),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  return column;
}

std::unique_ptr<cudf::column> d64(std::vector<int64_t> const& values)
{
  return make_decimal_column<int64_t>(cudf::type_id::DECIMAL64, values);
}

std::unique_ptr<cudf::column> d32(std::vector<int32_t> const& values)
{
  return make_decimal_column<int32_t>(cudf::type_id::DECIMAL32, values);
}

/// The set of columns (by index) the proof says must be widened.
std::unordered_set<int> widened(std::vector<cudf::column_view> const& columns,
                                std::vector<int> const& candidates)
{
  return decimal_sums_needing_widening(cudf::table_view(columns),
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
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // A column whose only large value is null has no valid extreme: nothing can overflow.
  auto all_null = d64({kMin64, kMin64});
  all_null->set_null_mask(
    cudf::create_null_mask(all_null->size(), cudf::mask_state::ALL_NULL, stream, mr),
    all_null->size());
  CHECK_FALSE(widens(all_null));

  // The large value is null, so the valid ones decide: 3 rows * 5 is far below the limit.
  auto some_null = d64({kMin64, 5, 5});
  auto mask = cudf::create_null_mask(some_null->size(), cudf::mask_state::ALL_VALID, stream, mr);
  some_null->set_null_mask(std::move(mask), 0);
  std::vector<uint32_t> bits{0b110};  // row 0 invalid
  REQUIRE(cudaMemcpy(some_null->mutable_view().null_mask(),
                     bits.data(),
                     sizeof(uint32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  some_null->set_null_count(1);
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

using sirius::op::decimal_sum_candidate;
using sirius::op::decimal_sum_may_overflow;
using sirius::op::decimal_sums_to_widen;

std::unordered_set<int> to_widen(std::vector<cudf::column_view> const& columns,
                                 std::vector<decimal_sum_candidate> const& candidates)
{
  return decimal_sums_to_widen(cudf::table_view(columns),
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
  CHECK_FALSE(decimal_sum_may_overflow(d64t, 0, std::numeric_limits<uint64_t>::max()));
  CHECK_FALSE(decimal_sum_may_overflow(d64t, 1, static_cast<uint64_t>(kMax64)));  // == limit
  CHECK(decimal_sum_may_overflow(d64t, 2, static_cast<uint64_t>(kMax64)));
  CHECK(decimal_sum_may_overflow(d64t, 1, uint64_t{1} << 63));  // |INT64_MIN|
  // A batch has fewer than 2^31 rows, so a magnitude up to 2^32 never overflows DECIMAL64.
  CHECK_FALSE(decimal_sum_may_overflow(d64t, kMax32, uint64_t{1} << 32));
  CHECK_FALSE(decimal_sum_may_overflow(d32t, 2, static_cast<uint64_t>(kMax32) / 2));  // 2^31 - 2
  CHECK(decimal_sum_may_overflow(d32t, 3, static_cast<uint64_t>(kMax32) / 2));
  CHECK_THROWS_AS(decimal_sum_may_overflow(cudf::data_type{cudf::type_id::INT64}, 1, 1),
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
