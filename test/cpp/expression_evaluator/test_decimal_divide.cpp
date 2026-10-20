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

// sirius::decimal_divide(), the GPU kernel behind sirius_decimal_divide(), on cuDF columns built
// here: sliced inputs whose offsets differ and are not multiples of 32 (the kernel reads both null
// masks at an offset and writes its own from bit 0), a tail of a partial mask word, NULLs and zero
// divisors. Values are checked against decimal_divide_round(), whose arithmetic
// test/cpp/helper/test_decimal_divide.cpp checks against Python's decimal module.

#include "catch.hpp"

#include <expression_evaluator/decimal_divide.hpp>
#include <helper/decimal_divide.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// rmm
#include <rmm/device_buffer.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

// standard library
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace {

/// A column of @p type holding @p values, NULL where @p valid is false.
template <typename T>
std::unique_ptr<cudf::column> make_column(cudf::data_type type,
                                          std::vector<T> const& values,
                                          std::vector<bool> const& valid)
{
  cuda::stream_ref const stream = cudf::get_default_stream();
  auto mr                       = cudf::get_current_device_resource_ref();
  auto const n                  = static_cast<cudf::size_type>(values.size());
  auto column =
    cudf::is_fixed_point(type)
      ? cudf::make_fixed_point_column(type, n, cudf::mask_state::UNALLOCATED, stream, mr)
      : cudf::make_numeric_column(type, n, cudf::mask_state::UNALLOCATED, stream, mr);
  REQUIRE(cudaMemcpyAsync(column->mutable_view().data<T>(),
                          values.data(),
                          values.size() * sizeof(T),
                          cudaMemcpyHostToDevice,
                          stream.get()) == cudaSuccess);
  std::vector<cudf::bitmask_type> words(
    cudf::bitmask_allocation_size_bytes(n) / sizeof(cudf::bitmask_type), 0);
  cudf::size_type nulls = 0;
  for (cudf::size_type i = 0; i < n; ++i) {
    if (valid[i]) {
      words[i / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++nulls;
    }
  }
  column->set_null_mask(
    rmm::device_buffer(words.data(), words.size() * sizeof(cudf::bitmask_type), stream, mr), nulls);
  stream.sync();
  return column;
}

/// The result as host values, std::nullopt for NULL.
std::vector<std::optional<__int128_t>> to_host(cudf::column_view const& column)
{
  auto const n = static_cast<std::size_t>(column.size());
  std::vector<__int128_t> values(n);
  REQUIRE(cudaMemcpy(values.data(),
                     column.data<__int128_t>(),
                     n * sizeof(__int128_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  std::vector<cudf::bitmask_type> words(
    cudf::bitmask_allocation_size_bytes(column.size()) / sizeof(cudf::bitmask_type), ~0U);
  if (column.nullable()) {
    REQUIRE(cudaMemcpy(words.data(),
                       column.null_mask(),
                       words.size() * sizeof(cudf::bitmask_type),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  std::vector<std::optional<__int128_t>> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (words[i / 32] >> (i % 32) & 1U) { out[i] = values[i]; }
  }
  return out;
}

}  // namespace

TEST_CASE("decimal_divide reads sliced, NULL-masked inputs at their own offsets",
          "[decimal_divide]")
{
  cuda::stream_ref const stream = cudf::get_default_stream();
  auto mr                       = cudf::get_current_device_resource_ref();

  // DECIMAL64(scale 2) dividends with every 7th row NULL; INT32 divisors with every 11th row 0
  // and every 13th NULL.
  constexpr int rows = 1000;
  std::vector<int64_t> dividends(rows);
  std::vector<int32_t> divisors(rows);
  std::vector<bool> dividend_valid(rows);
  std::vector<bool> divisor_valid(rows);
  for (int i = 0; i < rows; ++i) {
    dividends[i]      = (static_cast<int64_t>(i) * 7919 % 2000003) - 1000000;
    divisors[i]       = i % 11 == 0 ? 0 : (i * 31 % 1999) - 999;
    dividend_valid[i] = i % 7 != 0;
    divisor_valid[i]  = i % 13 != 0;
  }
  auto const dividend =
    make_column(cudf::data_type{cudf::type_id::DECIMAL64, -2}, dividends, dividend_valid);
  auto const divisor = make_column(cudf::data_type{cudf::type_id::INT32}, divisors, divisor_valid);

  // 333 rows (10 whole mask words and a 13-row tail), at offsets 37 and 5.
  constexpr cudf::size_type length = 333;
  auto const dividend_slice        = cudf::slice(dividend->view(), {37, 37 + length})[0];
  auto const divisor_slice         = cudf::slice(divisor->view(), {5, 5 + length})[0];
  auto const result = sirius::decimal_divide(dividend_slice, divisor_slice, 8, stream, mr);
  REQUIRE(result->type() == cudf::data_type{cudf::type_id::DECIMAL128, -8});
  REQUIRE(result->size() == length);

  auto const got        = to_host(result->view());
  cudf::size_type nulls = 0;
  for (cudf::size_type i = 0; i < length; ++i) {
    CAPTURE(i);
    auto const a = 37 + i;
    auto const b = 5 + i;
    if (!dividend_valid[a] || !divisor_valid[b] || divisors[b] == 0) {
      ++nulls;
      CHECK_FALSE(got[i].has_value());
      continue;
    }
    __int128_t expected = 0;
    REQUIRE(sirius::decimal_divide_round(dividends[a], divisors[b], 8 + 0 - 2, &expected) ==
            sirius::decimal_divide_status::ok);
    REQUIRE(got[i].has_value());
    CHECK(*got[i] == expected);
  }
  CHECK(result->null_count() == nulls);
  CHECK(nulls > 0);
}

TEST_CASE("decimal_divide raises invalid_input_exception for an overflow or an unsupported type",
          "[decimal_divide]")
{
  cuda::stream_ref const stream = cudf::get_default_stream();
  auto mr                       = cudf::get_current_device_resource_ref();
  auto const ones               = make_column(cudf::data_type{cudf::type_id::INT64},
                                std::vector<int64_t>(40, 1),
                                std::vector<bool>(40, true));
  // 1 / 1 at scale 38 is 10^38 units: 39 digits.
  REQUIRE_THROWS_AS(sirius::decimal_divide(ones->view(), ones->view(), 38, stream, mr),
                    sirius::invalid_input_exception);
  REQUIRE_NOTHROW(sirius::decimal_divide(ones->view(), ones->view(), 37, stream, mr));
  auto const doubles = make_column(cudf::data_type{cudf::type_id::FLOAT64},
                                   std::vector<double>(40, 1.0),
                                   std::vector<bool>(40, true));
  REQUIRE_THROWS_AS(sirius::decimal_divide(doubles->view(), ones->view(), 2, stream, mr),
                    sirius::invalid_input_exception);
}
