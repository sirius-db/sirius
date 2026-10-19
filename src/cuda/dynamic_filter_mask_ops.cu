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

#include <cudf/utilities/bit.hpp>

#include <rmm/error.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

#include <op/dynamic_filter/dynamic_filter_mask_ops.hpp>

#include <stdexcept>

namespace sirius::op {
namespace {
constexpr int block_size = 256;

__global__ void fold_mask_kernel(bool* running,
                                 bool const* incoming,
                                 cudf::bitmask_type const* validity,
                                 cudf::size_type offset,
                                 cudf::size_type rows,
                                 cudf::size_type* count,
                                 bool initialize)
{
  using reduction = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename reduction::TempStorage scratch;
  auto const row       = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  cudf::size_type kept = 0;
  if (row < static_cast<std::size_t>(rows)) {
    bool const valid =
      !validity || cudf::bit_is_set(validity, static_cast<cudf::size_type>(row) + offset);
    bool const value = (initialize || running[row]) && (!incoming || (valid && incoming[row]));
    running[row]     = value;
    kept             = value;
  }
  if (count == nullptr) { return; }
  auto const total = reduction(scratch).Sum(kept);
  if (threadIdx.x == 0) { atomicAdd(count, total); }
}
}  // namespace

void fold_dynamic_filter_mask(cudf::mutable_column_view running,
                              std::optional<cudf::column_view> incoming,
                              cudf::size_type* count,
                              bool initialize,
                              ::cuda::stream_ref stream)
{
  if (running.type().id() != cudf::type_id::BOOL8 || running.nullable() ||
      (incoming &&
       (incoming->type().id() != cudf::type_id::BOOL8 || incoming->size() != running.size()))) {
    throw std::invalid_argument("invalid dynamic-filter fold buffers");
  }
  if (running.size() == 0) { return; }
  auto const blocks = (static_cast<std::size_t>(running.size()) + block_size - 1) / block_size;
  fold_mask_kernel<<<blocks, block_size, 0, stream.get()>>>(
    running.data<bool>(),
    incoming ? incoming->data<bool>() : nullptr,
    incoming ? incoming->null_mask() : nullptr,
    incoming ? incoming->offset() : 0,
    running.size(),
    count,
    initialize);
  RMM_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace sirius::op
