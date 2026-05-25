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

#include "operator/empty_str_check.cuh"

#include <cudf/column/column.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>

namespace duckdb {
namespace sirius {

static constexpr int BLOCK_SIZE = 256;

__global__ void empty_str_check_kernel(const uint64_t* __restrict__ offsets,
                                       bool* __restrict__ output,
                                       size_t num_rows)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_rows) { output[tid] = (offsets[tid + 1] - offsets[tid]) > 0; }
}

__global__ void empty_str_check_with_rowids_kernel(const uint64_t* __restrict__ offsets,
                                                   const uint64_t* __restrict__ row_ids,
                                                   bool* __restrict__ output,
                                                   size_t num_rows)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_rows) {
    uint64_t rid = row_ids[tid];
    output[tid]  = (offsets[rid + 1] - offsets[rid]) > 0;
  }
}

std::unique_ptr<cudf::column> EmptyStrCheck(const uint64_t* offsets,
                                            const uint64_t* row_ids,
                                            size_t num_rows,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<bool> output(num_rows, stream, mr);

  if (num_rows > 0) {
    int num_blocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (row_ids != nullptr) {
      empty_str_check_with_rowids_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
        offsets, row_ids, output.data(), num_rows);
    } else {
      empty_str_check_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
        offsets, output.data(), num_rows);
    }
  }

  return std::make_unique<cudf::column>(std::move(output), rmm::device_buffer(0, stream, mr), 0);
}

}  // namespace sirius
}  // namespace duckdb
