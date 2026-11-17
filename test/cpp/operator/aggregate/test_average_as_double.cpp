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
 * @file test_average_as_double.cpp
 * @brief Direct tests of average_as_double at the extremes of its DECIMAL sum and INT64 count
 * inputs, which SQL cannot reach: sums whose DOUBLE rounds to +-2^127 and counts beyond 2^53.
 */

#include "op/aggregate/average.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace {

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

/// average_as_double over a scale-0 DECIMAL128 sum column and an INT64 count column.
std::vector<double> averages(std::vector<__int128_t> const& sums,
                             std::vector<int64_t> const& counts)
{
  auto const sum    = make_column(cudf::data_type{cudf::type_id::DECIMAL128, 0}, sums);
  auto const count  = make_column(cudf::data_type{cudf::type_id::INT64}, counts);
  auto const result = sirius::op::average_as_double(sum->view(),
                                                    count->view(),
                                                    cudf::get_default_stream(),
                                                    cudf::get_current_device_resource_ref());
  std::vector<double> host(sums.size());
  REQUIRE(cudaMemcpy(host.data(),
                     result->view().data<double>(),
                     host.size() * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return host;
}

}  // namespace

TEST_CASE("average_as_double - sums whose DOUBLE rounds to +-2^127 keep their residual",
          "[operator][aggregate][average_as_double]")
{
  // 5 * value is below 2^127 but rounds to 2^127 as a DOUBLE.
  __int128_t const value =
    __int128_t{3402823669209384540} * 10000000000000000000ULL + 1864164169247778406;
  __int128_t const max128 = std::numeric_limits<__int128_t>::max();
  __int128_t const min128 = std::numeric_limits<__int128_t>::min();
  auto const result =
    averages({value, 5 * value, -value, -5 * value, max128, min128}, {1, 5, 1, 5, 1, 1});
  CHECK(result[0] == static_cast<double>(value));
  CHECK(result[1] == result[0]);
  CHECK(result[2] == -result[0]);
  CHECK(result[3] == result[2]);
  CHECK(result[4] == 0x1p127);
  CHECK(result[5] == -0x1p127);
}

TEST_CASE("average_as_double - counts beyond 2^53 keep their residual",
          "[operator][aggregate][average_as_double]")
{
  int64_t const big = (int64_t{1} << 53) + 3;
  int64_t const max = std::numeric_limits<int64_t>::max();
  auto const result =
    averages({big, 3 * __int128_t{big}, max, -__int128_t{max}}, {big, big, max, max});
  CHECK(result[0] == 1.0);
  CHECK(result[1] == 3.0);
  CHECK(result[2] == 1.0);
  CHECK(result[3] == -1.0);
}
