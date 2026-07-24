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

#include "catch.hpp"
#include "helper/numeric_narrowing.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void require_type(std::optional<cudf::data_type> const& actual, cudf::type_id expected)
{
  REQUIRE(actual.has_value());
  REQUIRE(actual->id() == expected);
}

void require_decimal_type(std::optional<cudf::data_type> const& actual,
                          cudf::type_id expected,
                          std::int32_t expected_scale)
{
  require_type(actual, expected);
  REQUIRE(actual->scale() == expected_scale);
}

template <typename T>
std::unique_ptr<cudf::column> make_test_column(cudf::data_type type,
                                               std::vector<T> const& values,
                                               std::vector<bool> const& valid = {})
{
  if (!valid.empty() && valid.size() != values.size()) {
    throw std::invalid_argument("validity width must match values");
  }

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const size   = static_cast<cudf::size_type>(values.size());
  std::unique_ptr<cudf::column> column;
  if (valid.empty()) {
    column = cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);
  } else {
    auto null_mask = cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
    auto* mask     = static_cast<cudf::bitmask_type*>(null_mask.data());
    cudf::size_type null_count = 0;
    for (cudf::size_type row = 0; row < size; ++row) {
      if (!valid[static_cast<std::size_t>(row)]) {
        cudf::set_null_mask(mask, row, row + 1, false, stream);
        ++null_count;
      }
    }
    column =
      cudf::make_fixed_width_column(type, size, std::move(null_mask), null_count, stream, mr);
  }

  if (!values.empty() && cudaMemcpy(column->mutable_view().data<T>(),
                                    values.data(),
                                    values.size() * sizeof(T),
                                    cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("failed to populate numeric narrowing test column");
  }
  return column;
}

}  // namespace

TEST_CASE("numeric narrowing selects the narrowest exact signed carrier", "[numeric_narrowing]")
{
  using sirius::logical_type;
  using sirius::signed_integer_range;
  using sirius::type_id;

  SECTION("BIGINT traverses every narrower carrier boundary")
  {
    auto const logical = logical_type::make(type_id::BIGINT);

    require_type(sirius::choose_narrow_physical_type(
                   logical,
                   signed_integer_range(std::numeric_limits<std::int8_t>::lowest(),
                                        std::numeric_limits<std::int8_t>::max())),
                 cudf::type_id::INT8);
    require_type(sirius::choose_narrow_physical_type(
                   logical,
                   signed_integer_range(std::numeric_limits<std::int8_t>::lowest() - 1LL,
                                        std::numeric_limits<std::int16_t>::max())),
                 cudf::type_id::INT16);
    require_type(sirius::choose_narrow_physical_type(
                   logical,
                   signed_integer_range(std::numeric_limits<std::int16_t>::lowest() - 1LL,
                                        std::numeric_limits<std::int32_t>::max())),
                 cudf::type_id::INT32);

    REQUIRE_FALSE(sirius::choose_narrow_physical_type(
      logical,
      signed_integer_range(
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::lowest()) - 1,
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1)));
  }

  SECTION("INTEGER never selects its native INT32 carrier")
  {
    auto const logical = logical_type::make(type_id::INTEGER);
    require_type(sirius::choose_narrow_physical_type(logical, signed_integer_range(-128, 127)),
                 cudf::type_id::INT8);
    require_type(sirius::choose_narrow_physical_type(logical, signed_integer_range(-129, 128)),
                 cudf::type_id::INT16);
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(
      logical,
      signed_integer_range(std::numeric_limits<std::int16_t>::lowest() - 1LL,
                           std::numeric_limits<std::int16_t>::max() + 1LL)));
  }

  SECTION("SMALLINT narrows only to INT8")
  {
    auto const logical = logical_type::make(type_id::SMALLINT);
    require_type(sirius::choose_narrow_physical_type(logical, signed_integer_range(-128, 127)),
                 cudf::type_id::INT8);
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical, signed_integer_range(-129, 128)));
  }
}

TEST_CASE("numeric narrowing selects the narrowest exact unsigned carrier", "[numeric_narrowing]")
{
  using sirius::logical_type;
  using sirius::type_id;
  using sirius::unsigned_integer_range;

  SECTION("UBIGINT traverses every narrower carrier boundary")
  {
    auto const logical = logical_type::make(type_id::UBIGINT);
    require_type(sirius::choose_narrow_physical_type(
                   logical, unsigned_integer_range(0, std::numeric_limits<std::uint8_t>::max())),
                 cudf::type_id::UINT8);
    require_type(sirius::choose_narrow_physical_type(
                   logical,
                   unsigned_integer_range(
                     0, static_cast<std::uint64_t>(std::numeric_limits<std::uint8_t>::max()) + 1)),
                 cudf::type_id::UINT16);
    require_type(sirius::choose_narrow_physical_type(
                   logical,
                   unsigned_integer_range(
                     0, static_cast<std::uint64_t>(std::numeric_limits<std::uint16_t>::max()) + 1)),
                 cudf::type_id::UINT32);
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(
      logical,
      unsigned_integer_range(
        0, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1)));
  }

  SECTION("UINTEGER and USMALLINT require a strict reduction")
  {
    auto const uint32_type = logical_type::make(type_id::UINTEGER);
    require_type(sirius::choose_narrow_physical_type(uint32_type, unsigned_integer_range(0, 255)),
                 cudf::type_id::UINT8);
    require_type(sirius::choose_narrow_physical_type(uint32_type, unsigned_integer_range(0, 256)),
                 cudf::type_id::UINT16);
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(
      uint32_type, unsigned_integer_range(0, std::numeric_limits<std::uint16_t>::max() + 1ULL)));

    auto const uint16_type = logical_type::make(type_id::USMALLINT);
    require_type(sirius::choose_narrow_physical_type(uint16_type, unsigned_integer_range(0, 255)),
                 cudf::type_id::UINT8);
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(uint16_type, unsigned_integer_range(0, 256)));
  }
}

TEST_CASE("numeric narrowing preserves decimal semantics while reducing carrier width",
          "[numeric_narrowing]")
{
  using sirius::decimal_range;
  using sirius::logical_type;

  SECTION("DECIMAL32 is already the narrowest supported decimal carrier")
  {
    auto const logical = logical_type::make_decimal(9, 2);
    REQUIRE_FALSE(sirius::is_narrowable_numeric_type(logical));
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical, decimal_range(-1, 1, 2)));
  }

  SECTION("DECIMAL64 narrows to DECIMAL32 only when raw values fit")
  {
    auto const logical = logical_type::make_decimal(18, 4);
    require_decimal_type(
      sirius::choose_narrow_physical_type(logical,
                                          decimal_range(std::numeric_limits<std::int32_t>::lowest(),
                                                        std::numeric_limits<std::int32_t>::max(),
                                                        4)),
      cudf::type_id::DECIMAL32,
      -4);
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(
      logical,
      decimal_range(static_cast<__int128_t>(std::numeric_limits<std::int32_t>::lowest()) - 1,
                    static_cast<__int128_t>(std::numeric_limits<std::int32_t>::max()) + 1,
                    4)));
  }

  SECTION("DECIMAL128 chooses DECIMAL32, DECIMAL64, or no reduction from raw bounds")
  {
    auto const logical = logical_type::make_decimal(38, 7);
    require_decimal_type(sirius::choose_narrow_physical_type(logical, decimal_range(-42, 42, 7)),
                         cudf::type_id::DECIMAL32,
                         -7);
    require_decimal_type(
      sirius::choose_narrow_physical_type(
        logical,
        decimal_range(static_cast<__int128_t>(std::numeric_limits<std::int32_t>::lowest()) - 1,
                      static_cast<__int128_t>(std::numeric_limits<std::int32_t>::max()) + 1,
                      7)),
      cudf::type_id::DECIMAL64,
      -7);
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(
      logical,
      decimal_range(static_cast<__int128_t>(std::numeric_limits<std::int64_t>::lowest()) - 1,
                    static_cast<__int128_t>(std::numeric_limits<std::int64_t>::max()) + 1,
                    7)));
  }
}

TEST_CASE("numeric narrowing rejects incompatible or malformed requests", "[numeric_narrowing]")
{
  using sirius::decimal_range;
  using sirius::logical_type;
  using sirius::numeric_range;
  using sirius::numeric_range_domain;
  using sirius::signed_integer_range;
  using sirius::type_id;
  using sirius::unsigned_integer_range;

  SECTION("invalid ranges")
  {
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical_type::make(type_id::BIGINT),
                                                      signed_integer_range(2, 1)));
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical_type::make(type_id::UBIGINT),
                                                      unsigned_integer_range(2, 1)));
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical_type::make_decimal(18, 2),
                                                      decimal_range(2, 1, 2)));
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(
      logical_type::make(type_id::UBIGINT),
      numeric_range{numeric_range_domain::UNSIGNED_INTEGER, -1, 1, 0}));
  }

  SECTION("range domains must match the logical family")
  {
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical_type::make(type_id::BIGINT),
                                                      unsigned_integer_range(0, 1)));
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical_type::make(type_id::UBIGINT),
                                                      signed_integer_range(0, 1)));
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical_type::make_decimal(18, 2),
                                                      signed_integer_range(0, 1)));
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical_type::make(type_id::BIGINT),
                                                      decimal_range(0, 1, 0)));
  }

  SECTION("decimal scale must match exactly")
  {
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical_type::make_decimal(18, 2),
                                                      decimal_range(-100, 100, 3)));
  }

  SECTION("excluded and already-minimal logical types never narrow")
  {
    for (auto const id : {type_id::TINYINT,
                          type_id::UTINYINT,
                          type_id::HUGEINT,
                          type_id::UHUGEINT,
                          type_id::FLOAT,
                          type_id::DOUBLE,
                          type_id::BOOLEAN,
                          type_id::DATE,
                          type_id::TIMESTAMP}) {
      auto const logical = logical_type::make(id);
      REQUIRE_FALSE(sirius::is_narrowable_numeric_type(logical));
      REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical, signed_integer_range(0, 1)));
    }
  }
}

TEST_CASE("numeric carrier conversions are family, scale, and direction explicit",
          "[numeric_narrowing]")
{
  auto const int8  = cudf::data_type{cudf::type_id::INT8};
  auto const int64 = cudf::data_type{cudf::type_id::INT64};
  auto const uint8 = cudf::data_type{cudf::type_id::UINT8};

  REQUIRE(sirius::can_narrow_to(int64, int8));
  REQUIRE(sirius::can_restore_to(int8, int64));
  REQUIRE_FALSE(sirius::can_narrow_to(int8, int64));
  REQUIRE_FALSE(sirius::can_restore_to(int64, int8));
  REQUIRE_FALSE(sirius::can_narrow_to(int64, int64));
  // Crossing the signedness family forbids conversion in either direction, even at otherwise
  // valid widths.
  auto const uint64 = cudf::data_type{cudf::type_id::UINT64};
  REQUIRE_FALSE(sirius::can_narrow_to(int64, uint8));
  REQUIRE_FALSE(sirius::can_narrow_to(uint64, int8));
  REQUIRE_FALSE(sirius::can_restore_to(int8, uint64));
  REQUIRE_FALSE(sirius::can_restore_to(uint8, int64));

  auto const decimal32_scale2 = cudf::data_type{cudf::type_id::DECIMAL32, -2};
  auto const decimal64_scale2 = cudf::data_type{cudf::type_id::DECIMAL64, -2};
  auto const decimal64_scale3 = cudf::data_type{cudf::type_id::DECIMAL64, -3};
  REQUIRE(sirius::can_narrow_to(decimal64_scale2, decimal32_scale2));
  REQUIRE(sirius::can_restore_to(decimal32_scale2, decimal64_scale2));
  // Crossing the fixed-point scale family forbids conversion in either direction.
  REQUIRE_FALSE(sirius::can_narrow_to(decimal64_scale3, decimal32_scale2));
  REQUIRE_FALSE(
    sirius::can_narrow_to(decimal64_scale2, cudf::data_type{cudf::type_id::DECIMAL32, -3}));
  REQUIRE_FALSE(sirius::can_restore_to(decimal32_scale2, decimal64_scale3));
  REQUIRE_FALSE(
    sirius::can_restore_to(cudf::data_type{cudf::type_id::DECIMAL32, -3}, decimal64_scale2));
  REQUIRE_FALSE(sirius::can_narrow_to(cudf::data_type{cudf::type_id::FLOAT64},
                                      cudf::data_type{cudf::type_id::FLOAT32}));

  auto verify_selected_direction = [](sirius::logical_type const& logical,
                                      sirius::numeric_range const& range,
                                      cudf::data_type native) {
    auto target = sirius::choose_narrow_physical_type(logical, range);
    REQUIRE(target);
    REQUIRE(sirius::can_narrow_to(native, *target));
    REQUIRE(sirius::can_restore_to(*target, native));
  };
  verify_selected_direction(sirius::logical_type::make(sirius::type_id::BIGINT),
                            sirius::signed_integer_range(-42, 42),
                            cudf::data_type{cudf::type_id::INT64});
  verify_selected_direction(sirius::logical_type::make(sirius::type_id::UBIGINT),
                            sirius::unsigned_integer_range(0, 42),
                            cudf::data_type{cudf::type_id::UINT64});
  verify_selected_direction(
    sirius::logical_type::make_decimal(18, 2), sirius::decimal_range(-42, 42, 2), decimal64_scale2);
  verify_selected_direction(sirius::logical_type::make_decimal(38, 7),
                            sirius::decimal_range(-42, 42, 7),
                            cudf::data_type{cudf::type_id::DECIMAL128, -7});
}

TEST_CASE("decimal narrowing eligibility follows cuDF carrier boundaries", "[numeric_narrowing]")
{
  for (auto const precision : {uint8_t{4}, uint8_t{5}, uint8_t{9}}) {
    REQUIRE_FALSE(
      sirius::is_narrowable_numeric_type(sirius::logical_type::make_decimal(precision, 2)));
  }
  for (auto const precision : {uint8_t{10}, uint8_t{18}, uint8_t{19}, uint8_t{38}}) {
    REQUIRE(sirius::is_narrowable_numeric_type(sirius::logical_type::make_decimal(precision, 2)));
  }
}

TEST_CASE("exact numeric ranges validate target carriers", "[numeric_narrowing]")
{
  using sirius::decimal_range;
  using sirius::numeric_range;
  using sirius::numeric_range_domain;
  using sirius::signed_integer_range;
  using sirius::unsigned_integer_range;

  REQUIRE(sirius::numeric_range_fits(cudf::data_type{cudf::type_id::INT8},
                                     signed_integer_range(-128, 127)));
  REQUIRE_FALSE(sirius::numeric_range_fits(cudf::data_type{cudf::type_id::INT8},
                                           signed_integer_range(-129, 127)));
  REQUIRE(sirius::numeric_range_fits(cudf::data_type{cudf::type_id::UINT8},
                                     unsigned_integer_range(0, 255)));
  REQUIRE_FALSE(
    sirius::numeric_range_fits(cudf::data_type{cudf::type_id::UINT8},
                               numeric_range{numeric_range_domain::UNSIGNED_INTEGER, -1, 42, 0}));
  REQUIRE(sirius::numeric_range_fits(cudf::data_type{cudf::type_id::DECIMAL32, -2},
                                     decimal_range(-100, 100, 2)));
  REQUIRE_FALSE(sirius::numeric_range_fits(cudf::data_type{cudf::type_id::DECIMAL32, -3},
                                           decimal_range(-100, 100, 2)));
  REQUIRE_FALSE(sirius::numeric_range_fits(
    cudf::data_type{cudf::type_id::DECIMAL32, -2},
    decimal_range(static_cast<__int128_t>(std::numeric_limits<int32_t>::lowest()) - 1,
                  std::numeric_limits<int32_t>::max(),
                  2)));
}

TEST_CASE("exact numeric minmax handles nulls and wide decimals", "[numeric_narrowing]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  SECTION("signed values ignore interleaved nulls")
  {
    auto column = make_test_column<int64_t>(
      cudf::data_type{cudf::type_id::INT64}, {-300, -9999, 42, 1000}, {true, false, true, true});
    auto range = sirius::compute_exact_numeric_range(
      column->view(), sirius::logical_type::make(sirius::type_id::BIGINT), stream, mr);
    REQUIRE(range);
    REQUIRE(range->domain == sirius::numeric_range_domain::SIGNED_INTEGER);
    REQUIRE(range->minimum == -300);
    REQUIRE(range->maximum == 1000);
  }

  SECTION("empty and all-null columns have no exact bounds")
  {
    auto empty =
      make_test_column<int64_t>(cudf::data_type{cudf::type_id::INT64}, std::vector<int64_t>{});
    REQUIRE_FALSE(sirius::compute_exact_numeric_range(empty->view(), stream, mr));

    auto all_null =
      make_test_column<int64_t>(cudf::data_type{cudf::type_id::INT64}, {1, 2}, {false, false});
    REQUIRE_FALSE(sirius::compute_exact_numeric_range(all_null->view(), stream, mr));
  }

  SECTION("logical overload rejects a non-native carrier")
  {
    auto column = make_test_column<int32_t>(cudf::data_type{cudf::type_id::INT32}, {-5, 7});
    REQUIRE_FALSE(sirius::compute_exact_numeric_range(
      column->view(), sirius::logical_type::make(sirius::type_id::BIGINT), stream, mr));

    auto carrier_range = sirius::compute_exact_numeric_range(column->view(), stream, mr);
    REQUIRE(carrier_range);
    REQUIRE(carrier_range->minimum == -5);
    REQUIRE(carrier_range->maximum == 7);
  }

  SECTION("DECIMAL64 retains raw values and scale")
  {
    auto column = make_test_column<int64_t>(
      cudf::data_type{cudf::type_id::DECIMAL64, -2}, {-12345, 999999, 42}, {true, false, true});
    auto range = sirius::compute_exact_numeric_range(
      column->view(), sirius::logical_type::make_decimal(18, 2), stream, mr);
    REQUIRE(range);
    REQUIRE(range->domain == sirius::numeric_range_domain::DECIMAL);
    REQUIRE(range->decimal_scale == 2);
    REQUIRE(range->minimum == -12345);
    REQUIRE(range->maximum == 42);
  }

  SECTION("DECIMAL128 retains values outside INT64")
  {
    auto const minimum = -(static_cast<__int128_t>(1) << 80);
    auto const maximum = static_cast<__int128_t>(1) << 75;
    auto column = make_test_column<__int128_t>(cudf::data_type{cudf::type_id::DECIMAL128, -7},
                                               {minimum, 0, maximum});
    auto range  = sirius::compute_exact_numeric_range(
      column->view(), sirius::logical_type::make_decimal(38, 7), stream, mr);
    REQUIRE(range);
    REQUIRE(range->domain == sirius::numeric_range_domain::DECIMAL);
    REQUIRE(range->decimal_scale == 7);
    REQUIRE(range->minimum == minimum);
    REQUIRE(range->maximum == maximum);
  }
}
