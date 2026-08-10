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

#include "op/scan/pinned_block_gather.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sirius::op::scan {

namespace {

/// One aligned load per element. Buffers are 8-byte aligned in the virtual offset space and
/// block sizes are large powers-of-two multiples in practice, so an element of width <= 8 never
/// straddles a block boundary. 16-byte elements may only be 8-aligned, so they are gathered as
/// two independently translated 8-byte halves.
template <typename T>
__global__ void gather_kernel(std::byte const* const* __restrict__ blocks,
                              std::size_t block_size,
                              std::size_t data_offset,
                              cudf::size_type const* __restrict__ row_indices,
                              cudf::size_type num_rows,
                              T* __restrict__ out)
{
  auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < static_cast<std::size_t>(num_rows);
       i += stride) {
    auto const off = data_offset + static_cast<std::size_t>(row_indices[i]) * sizeof(T);
    out[i]         = *reinterpret_cast<T const*>(blocks[off / block_size] + off % block_size);
  }
}

__global__ void gather_kernel_16(std::byte const* const* __restrict__ blocks,
                                 std::size_t block_size,
                                 std::size_t data_offset,
                                 cudf::size_type const* __restrict__ row_indices,
                                 cudf::size_type num_rows,
                                 std::uint64_t* __restrict__ out)
{
  auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < static_cast<std::size_t>(num_rows);
       i += stride) {
    auto const off = data_offset + static_cast<std::size_t>(row_indices[i]) * 16;
    for (int half = 0; half < 2; ++half) {
      auto const h = off + static_cast<std::size_t>(half) * 8;
      out[2 * i + half] =
        *reinterpret_cast<std::uint64_t const*>(blocks[h / block_size] + h % block_size);
    }
  }
}

}  // namespace

void gather_fixed_width_from_host_blocks(std::byte const* const* device_block_ptrs,
                                         std::size_t block_size,
                                         std::size_t data_offset,
                                         std::size_t element_size,
                                         cudf::size_type const* device_row_indices,
                                         cudf::size_type num_rows,
                                         std::byte* device_out,
                                         rmm::cuda_stream_view stream)
{
  if (num_rows == 0) { return; }
  constexpr int block_dim = 256;
  auto const grid_dim     = static_cast<int>(
    std::min<std::size_t>((static_cast<std::size_t>(num_rows) + block_dim - 1) / block_dim, 65535));
  switch (element_size) {
    case 1:
      gather_kernel<std::uint8_t>
        <<<grid_dim, block_dim, 0, stream.value()>>>(device_block_ptrs,
                                                     block_size,
                                                     data_offset,
                                                     device_row_indices,
                                                     num_rows,
                                                     reinterpret_cast<std::uint8_t*>(device_out));
      break;
    case 2:
      gather_kernel<std::uint16_t>
        <<<grid_dim, block_dim, 0, stream.value()>>>(device_block_ptrs,
                                                     block_size,
                                                     data_offset,
                                                     device_row_indices,
                                                     num_rows,
                                                     reinterpret_cast<std::uint16_t*>(device_out));
      break;
    case 4:
      gather_kernel<std::uint32_t>
        <<<grid_dim, block_dim, 0, stream.value()>>>(device_block_ptrs,
                                                     block_size,
                                                     data_offset,
                                                     device_row_indices,
                                                     num_rows,
                                                     reinterpret_cast<std::uint32_t*>(device_out));
      break;
    case 8:
      gather_kernel<std::uint64_t>
        <<<grid_dim, block_dim, 0, stream.value()>>>(device_block_ptrs,
                                                     block_size,
                                                     data_offset,
                                                     device_row_indices,
                                                     num_rows,
                                                     reinterpret_cast<std::uint64_t*>(device_out));
      break;
    case 16:
      gather_kernel_16<<<grid_dim, block_dim, 0, stream.value()>>>(
        device_block_ptrs,
        block_size,
        data_offset,
        device_row_indices,
        num_rows,
        reinterpret_cast<std::uint64_t*>(device_out));
      break;
    default:
      throw std::invalid_argument(
        "[gather_fixed_width_from_host_blocks] unsupported element size " +
        std::to_string(element_size));
  }
  auto const status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string("[gather_fixed_width_from_host_blocks] kernel launch failed: ") +
      cudaGetErrorString(status));
  }
}

}  // namespace sirius::op::scan
