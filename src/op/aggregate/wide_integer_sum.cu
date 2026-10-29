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
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

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

constexpr int halves_block_size = 256;

/// The grid that covers @p rows with halves_block_size threads per block, capped so very large
/// columns loop instead.
unsigned int halves_grid_size(cudf::size_type rows)
{
  auto const blocks = (static_cast<std::int64_t>(rows) + halves_block_size - 1) / halves_block_size;
  return static_cast<unsigned int>(std::min<std::int64_t>(blocks, 65535));
}

/// high[i] = value >> 32 (arithmetic) and low[i] = value & 0xFFFFFFFF. NULL rows are written too;
/// the output null masks hide them.
__global__ void split_halves_kernel(std::int64_t const* input,
                                    std::int64_t* high,
                                    std::int64_t* low,
                                    cudf::size_type rows)
{
  auto const stride = static_cast<cudf::size_type>(blockDim.x * gridDim.x);
  for (auto i = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x); i < rows;
       i += stride) {
    auto const value = input[i];
    high[i]          = value >> 32;
    low[i]           = value & 0xFFFFFFFFLL;
  }
}

/// out[i] = high[i] * 2^32 + low[i] in 128 bits; |high| < 2^63, so the shift cannot overflow.
__global__ void combine_halves_kernel(std::int64_t const* high,
                                      std::int64_t const* low,
                                      __int128_t* out,
                                      cudf::size_type rows)
{
  auto const stride = static_cast<cudf::size_type>(blockDim.x * gridDim.x);
  for (auto i = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x); i < rows;
       i += stride) {
    out[i] = (static_cast<__int128_t>(high[i]) << 32) + low[i];
  }
}

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

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> split_int64_halves(
  cudf::column_view const& input, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  if (input.type().id() != cudf::type_id::INT64) {
    throw sirius::internal_exception("split_int64_halves: expected an INT64 column, got {}",
                                     cudf::type_to_name(input.type()));
  }
  auto const int64 = cudf::data_type{cudf::type_id::INT64};
  auto high        = cudf::make_numeric_column(
    int64, input.size(), cudf::copy_bitmask(input, stream, mr), input.null_count(), stream, mr);
  auto low = cudf::make_numeric_column(
    int64, input.size(), cudf::copy_bitmask(input, stream, mr), input.null_count(), stream, mr);
  if (input.size() > 0) {
    split_halves_kernel<<<halves_grid_size(input.size()), halves_block_size, 0, stream.get()>>>(
      input.data<std::int64_t>(),
      high->mutable_view().data<std::int64_t>(),
      low->mutable_view().data<std::int64_t>(),
      input.size());
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  return {std::move(high), std::move(low)};
}

std::unique_ptr<cudf::column> combine_int64_halves(cudf::column_view const& high,
                                                   cudf::column_view const& low,
                                                   ::cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  if (high.type().id() != cudf::type_id::INT64 || low.type().id() != cudf::type_id::INT64 ||
      high.size() != low.size()) {
    throw sirius::internal_exception(
      "combine_int64_halves: expected two INT64 columns of one size, got {} ({} rows) and {} ({} "
      "rows)",
      cudf::type_to_name(high.type()),
      high.size(),
      cudf::type_to_name(low.type()),
      low.size());
  }
  auto result = cudf::make_fixed_point_column(wide_integer_sum_type(),
                                              high.size(),
                                              cudf::copy_bitmask(high, stream, mr),
                                              high.null_count(),
                                              stream,
                                              mr);
  if (high.size() > 0) {
    combine_halves_kernel<<<halves_grid_size(high.size()), halves_block_size, 0, stream.get()>>>(
      high.data<std::int64_t>(),
      low.data<std::int64_t>(),
      result->mutable_view().data<__int128_t>(),
      high.size());
    CUDF_CUDA_TRY(cudaGetLastError());
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
