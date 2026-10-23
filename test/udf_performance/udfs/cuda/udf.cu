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

#include "gpu.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace {
__global__ void mix_words(uint64_t const* input,
                          cudf::bitmask_type const* validity,
                          cudf::size_type offset,
                          uint64_t* output,
                          cudf::size_type size)
{
  auto const row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= size) { return; }
  constexpr auto bits_per_word = sizeof(cudf::bitmask_type) * 8;
  auto const bit               = row + offset;
  bool const valid = !validity || ((validity[bit / bits_per_word] >> (bit % bits_per_word)) & 1);
  if (!valid) {
    output[row] = 0;
    return;
  }
  auto word = input[row] ^ (input[row] >> 17);
  word *= UINT64_C(0x9E3779B185EBCA87);
  auto const rotated = (word << 23) | (word >> 41);
  output[row]        = rotated ^ UINT64_C(0xD1B54A32D192ED03);
}
}  // namespace

namespace sirius_mix_word {
std::shared_ptr<sirius::gpu_scalar_function const> make_gpu_function()
{
  return std::make_shared<sirius::gpu_scalar_function const>(
    sirius::gpu_scalar_function{[](std::span<cudf::column_view const> arguments,
                                   cudf::size_type rows,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr) {
      if (arguments.size() != 1 || arguments[0].size() != rows ||
          arguments[0].type().id() != cudf::type_id::UINT64) {
        throw std::invalid_argument("mix_word_cuda requires one UINT64 column of row_count rows");
      }
      auto const& input = arguments[0];
      auto result       = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT64},
                                              rows,
                                              cudf::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);
      if (rows > 0) {
        constexpr int threads = 256;
        mix_words<<<(rows - 1) / threads + 1, threads, 0, stream.get()>>>(
          input.data<uint64_t>(),  // data() already includes the slice offset.
          input.null_mask(),       // The mask still uses the original base offset.
          input.offset(),
          result->mutable_view().data<uint64_t>(),
          rows);
        auto const status = cudaGetLastError();
        if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
      }
      // Copy validity for direct callback users; Sirius also applies default propagation.
      return result;
    }});
}
}  // namespace sirius_mix_word
