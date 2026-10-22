// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "gpu.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <memory>
#include <stdexcept>

namespace {
__global__ void magnitude(double const* input,
                          cudf::bitmask_type const* validity,
                          cudf::size_type offset,
                          double* output,
                          cudf::size_type size)
{
  auto const row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= size) { return; }
  constexpr auto bits_per_word = sizeof(cudf::bitmask_type) * 8;
  auto const bit               = row + offset;
  bool const valid = !validity || ((validity[bit / bits_per_word] >> (bit % bits_per_word)) & 1);
  output[row]      = valid ? fabs(input[row]) : 0.0;
}
}  // namespace

namespace sirius_udf_example {
std::shared_ptr<sirius::gpu_scalar_function const> make_gpu_function()
{
  return std::make_shared<sirius::gpu_scalar_function const>(
    sirius::gpu_scalar_function{[](std::span<cudf::column_view const> arguments,
                                   cudf::size_type rows,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr) {
      auto result = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::FLOAT64}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
      if (rows > 0) {
        constexpr int threads = 256;
        magnitude<<<(rows - 1) / threads + 1, threads, 0, stream.get()>>>(
          arguments[0].data<double>(),
          arguments[0].null_mask(),
          arguments[0].offset(),
          result->mutable_view().data<double>(),
          rows);
        auto const status = cudaGetLastError();
        if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
      }
      // DEFAULT_NULL_HANDLING is applied by Sirius after the callback.
      return result;
    }});
}
}  // namespace sirius_udf_example
