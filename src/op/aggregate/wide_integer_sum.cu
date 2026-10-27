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

#include "op/aggregate/wide_integer_sum.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sirius::op {

namespace {

/// Row i's value widened to 128 bits; a NULL row adds nothing.
template <typename T>
struct widened_value {
  cudf::column_device_view input;

  __device__ __int128_t operator()(cudf::size_type i) const
  {
    return input.is_valid(i) ? static_cast<__int128_t>(input.element<T>(i)) : __int128_t{0};
  }
};

/// |row i's value| as a double; a NULL row adds nothing. The magnitude is taken after the
/// conversion, so |INT64_MIN| does not overflow.
struct value_magnitude {
  cudf::column_device_view input;

  __device__ double operator()(cudf::size_type i) const
  {
    return input.is_valid(i) ? fabs(static_cast<double>(input.element<int64_t>(i))) : 0.0;
  }
};

/// Writes the sum of @p row_value(i) over the rows of @p input to the device address @p result.
template <typename Out, typename RowValue>
void sum_rows_into(cudf::column_view const& input,
                   RowValue row_value,
                   Out* result,
                   ::cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr)
{
  auto const values =
    thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0), row_value);
  std::size_t temp_bytes = 0;
  CUDF_CUDA_TRY(
    cub::DeviceReduce::Sum(nullptr, temp_bytes, values, result, input.size(), stream.get()));
  rmm::device_buffer temp(temp_bytes, stream, mr);
  CUDF_CUDA_TRY(
    cub::DeviceReduce::Sum(temp.data(), temp_bytes, values, result, input.size(), stream.get()));
}

}  // namespace

std::unique_ptr<cudf::column> wide_integer_sum(cudf::column_view const& input,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  auto const type_id = input.type().id();
  if (type_id != cudf::type_id::INT32 && type_id != cudf::type_id::INT64) {
    throw sirius::internal_exception("wide_integer_sum: expected an INT32 or INT64 column, got {}",
                                     cudf::type_to_name(input.type()));
  }
  if (input.size() == 0 || input.null_count() == input.size()) {
    return cudf::make_fixed_point_column(
      wide_integer_sum_type(), 1, cudf::mask_state::ALL_NULL, stream, mr);
  }
  auto result = cudf::make_fixed_point_column(
    wide_integer_sum_type(), 1, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* sum         = result->mutable_view().data<__int128_t>();
  auto const device = cudf::column_device_view::create(input, stream);
  if (type_id == cudf::type_id::INT32) {
    sum_rows_into(input, widened_value<int32_t>{*device}, sum, stream, mr);
  } else {
    sum_rows_into(input, widened_value<int64_t>{*device}, sum, stream, mr);
  }
  return result;
}

double sum_of_magnitudes(cudf::column_view const& input,
                         ::cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
{
  if (input.type().id() != cudf::type_id::INT64) {
    throw sirius::internal_exception("sum_of_magnitudes: expected an INT64 column, got {}",
                                     cudf::type_to_name(input.type()));
  }
  if (input.size() == 0) { return 0.0; }
  rmm::device_scalar<double> total(stream, mr);
  auto const device = cudf::column_device_view::create(input, stream);
  sum_rows_into(input, value_magnitude{*device}, total.data(), stream, mr);
  return total.value(stream);
}

}  // namespace sirius::op
