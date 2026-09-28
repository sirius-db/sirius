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
#include "cudf/cudf_utils.hpp"
#include "helper/numeric_narrowing.hpp"
#include "helper/type_conversions.hpp"
#include "pin_table.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>
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

/// Read @p column back as @p T, which for a temporal carrier is its device storage type.
template <typename T>
std::vector<T> copy_values_to_host(cudf::column_view const& column)
{
  std::vector<T> values(static_cast<std::size_t>(column.size()));
  cuda::stream_ref const stream = cudf::get_default_stream();
  stream.sync();
  if (!values.empty() && cudaMemcpy(values.data(),
                                    column.data<T>(),
                                    values.size() * sizeof(T),
                                    cudaMemcpyDeviceToHost) != cudaSuccess) {
    throw std::runtime_error("failed to read numeric narrowing test column");
  }
  return values;
}

std::vector<bool> copy_valids_to_host(cudf::column_view const& column)
{
  std::vector<bool> valids(static_cast<std::size_t>(column.size()), true);
  if (!column.nullable() || column.null_count() == 0) { return valids; }

  auto const words = cudf::num_bitmask_words(column.offset() + column.size());
  std::vector<cudf::bitmask_type> mask(static_cast<std::size_t>(words));
  cuda::stream_ref const stream = cudf::get_default_stream();
  stream.sync();
  if (cudaMemcpy(mask.data(),
                 column.null_mask(),
                 mask.size() * sizeof(cudf::bitmask_type),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    throw std::runtime_error("failed to read numeric narrowing test validity");
  }

  constexpr auto bits_per_word = sizeof(cudf::bitmask_type) * 8;
  for (cudf::size_type row = 0; row < column.size(); ++row) {
    auto const bit = static_cast<std::size_t>(column.offset() + row);
    valids[static_cast<std::size_t>(row)] =
      ((mask[bit / bits_per_word] >> (bit % bits_per_word)) & 1U) != 0U;
  }
  return valids;
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

  SECTION("excluded, sub-day temporal, and already-minimal logical types never narrow")
  {
    for (auto const id : {type_id::TINYINT,
                          type_id::UTINYINT,
                          type_id::HUGEINT,
                          type_id::UHUGEINT,
                          type_id::FLOAT,
                          type_id::DOUBLE,
                          type_id::BOOLEAN,
                          type_id::TIMESTAMP_SEC,
                          type_id::TIMESTAMP_MS,
                          type_id::TIMESTAMP,
                          type_id::TIMESTAMP_NS}) {
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

TEST_CASE("numeric narrowing admits DATE through its int32 representation", "[numeric_narrowing]")
{
  using sirius::logical_type;
  using sirius::signed_integer_range;
  using sirius::type_id;

  auto const date  = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
  auto const int8  = cudf::data_type{cudf::type_id::INT8};
  auto const int16 = cudf::data_type{cudf::type_id::INT16};
  auto const int32 = cudf::data_type{cudf::type_id::INT32};

  SECTION("narrowing_rep_type maps only TIMESTAMP_DAYS")
  {
    REQUIRE(sirius::narrowing_rep_type(date) == int32);
    for (auto const id : {cudf::type_id::INT8,
                          cudf::type_id::INT16,
                          cudf::type_id::INT32,
                          cudf::type_id::INT64,
                          cudf::type_id::UINT8,
                          cudf::type_id::UINT16,
                          cudf::type_id::UINT32,
                          cudf::type_id::UINT64,
                          cudf::type_id::FLOAT64,
                          cudf::type_id::BOOL8,
                          cudf::type_id::TIMESTAMP_SECONDS,
                          cudf::type_id::TIMESTAMP_MICROSECONDS,
                          cudf::type_id::DURATION_DAYS,
                          cudf::type_id::STRING}) {
      auto const type = cudf::data_type{id};
      REQUIRE(sirius::narrowing_rep_type(type) == type);
    }
    for (auto const id : {cudf::type_id::DECIMAL32, cudf::type_id::DECIMAL64}) {
      auto const type = cudf::data_type{id, -2};
      REQUIRE(sirius::narrowing_rep_type(type) == type);
    }
  }

  SECTION("DATE is narrowable and every other temporal type is not")
  {
    REQUIRE(sirius::is_narrowable_numeric_type(logical_type::make(type_id::DATE)));
    for (auto const id : {type_id::TIMESTAMP_SEC,
                          type_id::TIMESTAMP_MS,
                          type_id::TIMESTAMP,
                          type_id::TIMESTAMP_NS}) {
      REQUIRE_FALSE(sirius::is_narrowable_numeric_type(logical_type::make(id)));
    }
  }

  SECTION("DATE picks the narrowest signed carrier that holds its epoch days")
  {
    auto const logical = logical_type::make(type_id::DATE);
    // 1992-01-01 through 1998-12-31, the span every TPC-H date column lives in.
    require_type(sirius::choose_narrow_physical_type(logical, signed_integer_range(8035, 10591)),
                 cudf::type_id::INT16);
    require_type(sirius::choose_narrow_physical_type(logical, signed_integer_range(0, 100)),
                 cudf::type_id::INT8);
    require_type(sirius::choose_narrow_physical_type(
                   logical,
                   signed_integer_range(std::numeric_limits<std::int16_t>::lowest(),
                                        std::numeric_limits<std::int16_t>::max())),
                 cudf::type_id::INT16);
    // A range needing four bytes has no carrier at all: INT32 is the width DATE already occupies,
    // so it is never a reduction.
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(
      logical,
      signed_integer_range(std::numeric_limits<std::int16_t>::lowest() - 1LL,
                           std::numeric_limits<std::int16_t>::max())));
    REQUIRE_FALSE(
      sirius::choose_narrow_physical_type(logical, sirius::unsigned_integer_range(0, 1)));
    REQUIRE_FALSE(sirius::choose_narrow_physical_type(logical, sirius::decimal_range(0, 1, 0)));
  }

  SECTION("DATE carriers convert as int32 in both directions")
  {
    REQUIRE(sirius::can_narrow_to(date, int16));
    REQUIRE(sirius::can_narrow_to(date, int8));
    REQUIRE_FALSE(sirius::can_narrow_to(date, int32));
    REQUIRE(sirius::can_restore_to(int16, date));
    REQUIRE_FALSE(sirius::can_restore_to(int32, date));
    REQUIRE_FALSE(sirius::can_narrow_to(date, cudf::data_type{cudf::type_id::UINT16}));
    // Only TIMESTAMP_DAYS has a representation; a sub-day carrier is in no family.
    REQUIRE_FALSE(sirius::can_narrow_to(cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, int32));

    // A carrier does not identify the domain it carries: INT16 restores equally to a plain int and
    // to epoch days, which is why keeping the two apart is the caller's job, not this layer's.
    REQUIRE(sirius::can_restore_to(int16, int32));
    REQUIRE(sirius::can_restore_to(int16, date));

    // The representation stands in for TIMESTAMP_DAYS only where the other side is a carrier it
    // could be narrowed to. INT64 is wider than the int32 representation, so it is in no family
    // with epoch days in either direction and cannot pass a validator whose job is rejecting a
    // physical type that contradicts the declared one.
    auto const int64 = cudf::data_type{cudf::type_id::INT64};
    REQUIRE_FALSE(sirius::can_narrow_to(int64, date));
    REQUIRE_FALSE(sirius::can_restore_to(date, int64));
  }
}

TEST_CASE("narrowing domains keep the narrowable types apart", "[numeric_narrowing]")
{
  using sirius::logical_type;
  using sirius::narrow_domain;
  using sirius::type_id;

  REQUIRE(sirius::narrow_domain_of(logical_type::make(type_id::SMALLINT)) ==
          narrow_domain::SIGNED_INTEGER);
  REQUIRE(sirius::narrow_domain_of(logical_type::make(type_id::BIGINT)) ==
          narrow_domain::SIGNED_INTEGER);
  REQUIRE(sirius::narrow_domain_of(logical_type::make(type_id::UBIGINT)) ==
          narrow_domain::UNSIGNED_INTEGER);
  REQUIRE(sirius::narrow_domain_of(logical_type::make_decimal(18, 2)) == narrow_domain::DECIMAL);
  REQUIRE(sirius::narrow_domain_of(logical_type::make(type_id::DATE)) == narrow_domain::DATE);

  // The distinction the shared comparison predicates rely on: epoch days and plain integers share
  // every signed carrier but never a domain.
  REQUIRE(sirius::narrow_domain_of(logical_type::make(type_id::DATE)) !=
          sirius::narrow_domain_of(logical_type::make(type_id::BIGINT)));

  // A type whose bits carry no narrowable numeric meaning has no domain to be compared against,
  // which is what keeps a newly admitted type from inheriting one by omission.
  for (auto const id : {type_id::HUGEINT, type_id::DOUBLE, type_id::BOOLEAN, type_id::TIMESTAMP}) {
    REQUIRE(sirius::narrow_domain_of(logical_type::make(id)) == narrow_domain::NONE);
  }

  // Having a domain is not the same as being worth narrowing. The already-minimal carriers mean
  // something narrowable and still have nowhere narrower to go, so they keep their domain while
  // declining narrowing -- which is what lets a literal of one fold into a wider column's narrow
  // carrier instead of forcing that column to restore.
  struct minimal_case {
    logical_type type;
    narrow_domain domain;
  };
  for (auto const& c :
       {minimal_case{logical_type::make(type_id::TINYINT), narrow_domain::SIGNED_INTEGER},
        minimal_case{logical_type::make(type_id::UTINYINT), narrow_domain::UNSIGNED_INTEGER},
        minimal_case{logical_type::make_decimal(9, 2), narrow_domain::DECIMAL}}) {
    REQUIRE(sirius::narrow_domain_of(c.type) == c.domain);
    REQUIRE_FALSE(sirius::is_narrowable_numeric_type(c.type));
  }
}

TEST_CASE("narrowing_rep_view aliases the source buffers", "[numeric_narrowing]")
{
  auto const date = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
  auto column =
    make_test_column<std::int32_t>(date, {8035, 9000, 10591, 0}, {true, false, true, true});
  auto const view = column->view();

  SECTION("a DATE view is retagged in place")
  {
    auto const rep = sirius::narrowing_rep_view(view);
    REQUIRE(rep.type().id() == cudf::type_id::INT32);
    REQUIRE(rep.head<void>() == view.head<void>());
    REQUIRE(rep.null_mask() == view.null_mask());
    REQUIRE(rep.size() == view.size());
    REQUIRE(rep.null_count() == view.null_count());
    REQUIRE(rep.offset() == view.offset());
  }

  SECTION("a sliced DATE view keeps its offset")
  {
    auto const sliced = cudf::slice(view, {1, 4}).front();
    auto const rep    = sirius::narrowing_rep_view(sliced);
    REQUIRE(rep.type().id() == cudf::type_id::INT32);
    REQUIRE(rep.size() == 3);
    REQUIRE(rep.offset() == sliced.offset());
    REQUIRE(rep.head<void>() == sliced.head<void>());
    REQUIRE(rep.data<std::int32_t>() == sliced.data<std::int32_t>());
  }

  SECTION("carriers that are already their own representation are returned unchanged")
  {
    auto integers = make_test_column<std::int32_t>(cudf::data_type{cudf::type_id::INT32}, {-5, 7});
    auto const integer_rep = sirius::narrowing_rep_view(integers->view());
    REQUIRE(integer_rep.type() == integers->type());
    REQUIRE(integer_rep.head<void>() == integers->view().head<void>());

    auto decimals =
      make_test_column<std::int32_t>(cudf::data_type{cudf::type_id::DECIMAL32, -2}, {-5, 7});
    auto const decimal_rep = sirius::narrowing_rep_view(decimals->view());
    REQUIRE(decimal_rep.type() == decimals->type());
    REQUIRE(decimal_rep.head<void>() == decimals->view().head<void>());
  }
}

TEST_CASE("cast_through_rep converts carriers cuDF refuses and defers every other cast",
          "[numeric_narrowing]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto const date  = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
  auto const int16 = cudf::data_type{cudf::type_id::INT16};

  SECTION("cuDF refuses the direct temporal conversion")
  {
    auto days     = make_test_column<std::int32_t>(date, {8035, 10591});
    auto narrowed = make_test_column<std::int16_t>(int16, {8035, 10591});
    REQUIRE_THROWS(cudf::cast(days->view(), int16, stream, mr));
    REQUIRE_THROWS(cudf::cast(narrowed->view(), date, stream, mr));
  }

  SECTION("DATE narrows to INT16 preserving values and nulls")
  {
    auto column =
      make_test_column<std::int32_t>(date, {8035, -1, 10591, 0}, {true, false, true, true});
    auto result = sirius::cast_through_rep(column->view(), int16, stream, mr);
    REQUIRE(result->type() == int16);
    REQUIRE(result->size() == 4);
    REQUIRE(result->null_count() == 1);
    REQUIRE(copy_values_to_host<std::int16_t>(result->view()) ==
            std::vector<std::int16_t>{8035, -1, 10591, 0});
    REQUIRE(copy_valids_to_host(result->view()) == std::vector<bool>{true, false, true, true});
  }

  SECTION("INT16 restores to DATE preserving values and nulls")
  {
    auto column =
      make_test_column<std::int16_t>(int16, {8035, -1, 10591, 0}, {true, false, true, true});
    auto result = sirius::cast_through_rep(column->view(), date, stream, mr);
    REQUIRE(result->type() == date);
    REQUIRE(result->size() == 4);
    REQUIRE(result->null_count() == 1);
    REQUIRE(copy_values_to_host<std::int32_t>(result->view()) ==
            std::vector<std::int32_t>{8035, -1, 10591, 0});
    REQUIRE(copy_valids_to_host(result->view()) == std::vector<bool>{true, false, true, true});
  }

  SECTION("an all-null DATE column keeps its null count in both directions")
  {
    auto column   = make_test_column<std::int32_t>(date, {1, 2}, {false, false});
    auto narrowed = sirius::cast_through_rep(column->view(), int16, stream, mr);
    REQUIRE(narrowed->type() == int16);
    REQUIRE(narrowed->null_count() == 2);

    auto restored = sirius::cast_through_rep(narrowed->view(), date, stream, mr);
    REQUIRE(restored->type() == date);
    REQUIRE(restored->null_count() == 2);
  }

  SECTION("an empty column converts to an empty column of the target")
  {
    auto empty    = make_test_column<std::int32_t>(date, std::vector<std::int32_t>{});
    auto narrowed = sirius::cast_through_rep(empty->view(), int16, stream, mr);
    REQUIRE(narrowed->type() == int16);
    REQUIRE(narrowed->size() == 0);

    auto restored = sirius::cast_through_rep(narrowed->view(), date, stream, mr);
    REQUIRE(restored->type() == date);
    REQUIRE(restored->size() == 0);
  }

  SECTION("conversions needing no representation match cudf::cast exactly")
  {
    auto integers = make_test_column<std::int64_t>(
      cudf::data_type{cudf::type_id::INT64}, {-300, 42, 1000}, {true, false, true});
    auto tunnelled = sirius::cast_through_rep(integers->view(), int16, stream, mr);
    auto direct    = cudf::cast(integers->view(), int16, stream, mr);
    REQUIRE(tunnelled->type() == direct->type());
    REQUIRE(tunnelled->null_count() == direct->null_count());
    REQUIRE(copy_values_to_host<std::int16_t>(tunnelled->view()) ==
            copy_values_to_host<std::int16_t>(direct->view()));

    auto const decimal32 = cudf::data_type{cudf::type_id::DECIMAL32, -2};
    auto decimals        = make_test_column<std::int64_t>(
      cudf::data_type{cudf::type_id::DECIMAL64, -2}, {-12345, 42}, {true, true});
    auto tunnelled_decimal = sirius::cast_through_rep(decimals->view(), decimal32, stream, mr);
    auto direct_decimal    = cudf::cast(decimals->view(), decimal32, stream, mr);
    REQUIRE(tunnelled_decimal->type() == direct_decimal->type());
    REQUIRE(copy_values_to_host<std::int32_t>(tunnelled_decimal->view()) ==
            copy_values_to_host<std::int32_t>(direct_decimal->view()));
  }

  SECTION("a conversion outside the carrier families keeps cuDF's own semantics")
  {
    // DATE<->TIMESTAMP is a real unit conversion cuDF performs directly. Tunnelling it would
    // reinterpret epoch days as a tick count instead.
    constexpr std::int64_t micros_per_day = 86'400'000'000;
    auto const timestamp                  = cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};

    auto days    = make_test_column<std::int32_t>(date, {8035, 10591});
    auto widened = sirius::cast_through_rep(days->view(), timestamp, stream, mr);
    REQUIRE(widened->type() == timestamp);
    REQUIRE(copy_values_to_host<std::int64_t>(widened->view()) ==
            std::vector<std::int64_t>{8035 * micros_per_day, 10591 * micros_per_day});

    auto narrowed = sirius::cast_through_rep(widened->view(), date, stream, mr);
    REQUIRE(narrowed->type() == date);
    REQUIRE(copy_values_to_host<std::int32_t>(narrowed->view()) ==
            std::vector<std::int32_t>{8035, 10591});

    auto identity = sirius::cast_through_rep(days->view(), date, stream, mr);
    REQUIRE(identity->type() == date);
    REQUIRE(copy_values_to_host<std::int32_t>(identity->view()) ==
            std::vector<std::int32_t>{8035, 10591});

    // A carrier wider than int32 is never a DATE carrier, so neither direction is a carrier
    // conversion and cuDF refuses the pair as it refuses any timestamp<->numeric cast.
    auto const int64 = cudf::data_type{cudf::type_id::INT64};
    auto wide        = make_test_column<std::int64_t>(int64, {8035, 10591});
    REQUIRE_THROWS(sirius::cast_through_rep(wide->view(), date, stream, mr));
    REQUIRE_THROWS(sirius::cast_through_rep(days->view(), int64, stream, mr));
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

  SECTION("DATE reduces over its int32 representation")
  {
    auto const date    = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
    auto const logical = sirius::logical_type::make(sirius::type_id::DATE);

    // cudf::minmax on a TIMESTAMP_DAYS column yields timestamp scalars, which carry no numeric
    // range: reducing over the representation is the whole reason DATE narrowing ever engages.
    auto column =
      make_test_column<int32_t>(date, {8035, -9999, 10591, 9000}, {true, false, true, true});
    auto range = sirius::compute_exact_numeric_range(column->view(), logical, stream, mr);
    REQUIRE(range);
    REQUIRE(range->domain == sirius::numeric_range_domain::SIGNED_INTEGER);
    REQUIRE(range->minimum == 8035);
    REQUIRE(range->maximum == 10591);

    auto carrier_range = sirius::compute_exact_numeric_range(column->view(), stream, mr);
    REQUIRE(carrier_range);
    REQUIRE(carrier_range->domain == sirius::numeric_range_domain::SIGNED_INTEGER);
    REQUIRE(carrier_range->minimum == 8035);
    REQUIRE(carrier_range->maximum == 10591);

    // The logical overload still demands the native carrier, so a DATE declared over raw int32
    // days has no range.
    auto integers = make_test_column<int32_t>(cudf::data_type{cudf::type_id::INT32}, {8035, 10591});
    REQUIRE_FALSE(sirius::compute_exact_numeric_range(integers->view(), logical, stream, mr));

    auto empty = make_test_column<int32_t>(date, std::vector<int32_t>{});
    REQUIRE_FALSE(sirius::compute_exact_numeric_range(empty->view(), stream, mr));
    auto all_null = make_test_column<int32_t>(date, {1, 2}, {false, false});
    REQUIRE_FALSE(sirius::compute_exact_numeric_range(all_null->view(), stream, mr));

    auto seconds =
      make_test_column<int64_t>(cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, {1, 2});
    REQUIRE_FALSE(sirius::compute_exact_numeric_range(seconds->view(), stream, mr));

    auto const target = sirius::choose_narrow_physical_type(logical, *range);
    require_type(target, cudf::type_id::INT16);
    REQUIRE(sirius::can_narrow_to(date, *target));
    REQUIRE(sirius::can_restore_to(*target, date));
  }
}

TEST_CASE("pin-time native type records the declared mapping, not the decoded carrier",
          "[numeric_narrowing]")
{
  // Parquet fixed-length-byte-array decimals with precision <= 18 decode as DECIMAL128, while the
  // declared DECIMAL(15,2) maps to DECIMAL64 at plan time. Recording the decoded type would make
  // the scan manager's strict native-identity comparisons fail on every query, permanently
  // disabling the pin's narrow serving.
  auto const declared = duckdb::LogicalType::DECIMAL(15, 2);
  auto const decoded  = cudf::data_type{cudf::type_id::DECIMAL128, -2};

  auto const plan_time_native = sirius::try_get_cudf_type(sirius::from_duckdb(declared));
  require_decimal_type(plan_time_native, cudf::type_id::DECIMAL64, -2);

  REQUIRE(sirius::pin_native_type(decoded, &declared) == *plan_time_native);

  // A cleared declared vector (zone-map stats and compressed pin both off) is the only case the
  // pin drivers pass no declared type; only then does the decoded type stand in.
  REQUIRE(sirius::pin_native_type(decoded, nullptr) == decoded);

  // When declared and decoded already agree the two rules coincide.
  auto const declared_int = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
  auto const decoded_int  = cudf::data_type{cudf::type_id::INT32};
  REQUIRE(sirius::pin_native_type(decoded_int, &declared_int) == decoded_int);
}
