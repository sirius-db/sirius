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

//! @file persistent_kernel.cu
//! @brief Fused scan + filter + aggregate kernel for simple queries.
//!
//! For queries identified as "single-scan + single-aggregate" (e.g.
//! `SELECT count(*) FROM lineitem WHERE l_shipdate < '1995-01-01'`),
//! this kernel performs all three operations in a single kernel launch,
//! eliminating the overhead of multiple kernel launches (~5-10μs each
//! on AMD ROCm) and the intermediate materialization of filtered rows.
//!
//! The kernel reads column data, applies a simple comparison filter,
//! and atomically updates the aggregate (count, sum, min, max).

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstddef>

namespace sirius::cuda {

namespace cg = cooperative_groups;

/// Aggregate type enumeration.
enum class agg_type : uint8_t {
  COUNT,  // count matching rows
  SUM,    // sum of a column's values for matching rows
  MIN,    // min of a column's values for matching rows
  MAX,    // max of a column's values for matching rows
};

/// Filter type enumeration (simple predicates only).
enum class filter_op : uint8_t {
  NONE,     // no filter (all rows match)
  LT,       // column < value
  LE,       // column <= value
  GT,       // column > value
  GE,       // column >= value
  EQ,       // column == value
  NE,       // column != value
};

/// Descriptor for the fused kernel.
struct persistent_kernel_desc {
  void const* column_data;      // pointer to the column's device data
  uint32_t column_stride;       // bytes per element (1, 2, 4, 8)
  uint32_t num_rows;            // total rows in the column
  agg_type agg;                 // which aggregate to compute
  filter_op filter;             // which filter to apply
  int64_t filter_value;         // the filter comparison value (cast to column type)
  int64_t* result;              // device pointer to the result (atomic)
};

/// Device function: apply the filter to a single value.
__device__ __forceinline__ bool apply_filter(int64_t val, filter_op op, int64_t target) {
  switch (op) {
    case filter_op::NONE: return true;
    case filter_op::LT:   return val <  target;
    case filter_op::LE:   return val <= target;
    case filter_op::GT:   return val >  target;
    case filter_op::GE:   return val >= target;
    case filter_op::EQ:   return val == target;
    case filter_op::NE:   return val != target;
    default:              return true;
  }
}

/// Device function: read a value from the column at row index.
__device__ __forceinline__ int64_t read_value(void const* data, uint32_t stride, uint32_t row) {
  auto const* base = static_cast<char const*>(data);
  switch (stride) {
    case 1: return static_cast<int64_t>(*reinterpret_cast<int8_t const*>(base + row));
    case 2: return static_cast<int64_t>(*reinterpret_cast<int16_t const*>(base + row * 2));
    case 4: return static_cast<int64_t>(*reinterpret_cast<int32_t const*>(base + row * 4));
    case 8: return *reinterpret_cast<int64_t const*>(base + row * 8);
    default: return 0;
  }
}

/// The fused kernel: each block processes a range of rows, applies the
/// filter, and atomically updates the result.
__global__ void persistent_scan_filter_agg(persistent_kernel_desc desc) {
  auto block = cg::this_thread_block();

  // Each block processes [blockIdx.x * blockDim.x, (blockIdx.x+1) * blockDim.x)
  uint32_t const start = blockIdx.x * blockDim.x;
  uint32_t const end   = min(start + blockDim.x, desc.num_rows);

  // Local accumulators (per-block, then reduced atomically)
  int64_t local_count = 0;
  int64_t local_sum   = 0;
  int64_t local_min   = INT64_MAX;
  int64_t local_max   = INT64_MIN;

  for (uint32_t row = start + threadIdx.x; row < end; row += blockDim.x) {
    int64_t val = read_value(desc.column_data, desc.column_stride, row);
    if (apply_filter(val, desc.filter, desc.filter_value)) {
      local_count++;
      local_sum += val;
      if (val < local_min) local_min = val;
      if (val > local_max) local_max = val;
    }
  }

  // Warp-level reduction using shuffle
  auto warp = cg::tiled_partition<64>(block);  // AMD wavefront = 64
  for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
    local_count += warp.shfl_down(local_count, offset);
    local_sum   += warp.shfl_down(local_sum, offset);
    local_min    = min(local_min, warp.shfl_down(local_min, offset));
    local_max    = max(local_max, warp.shfl_down(local_max, offset));
  }

  // First thread in each warp writes to shared memory
  __shared__ int64_t sm_count[16];  // max 16 warps per block (1024/64)
  __shared__ int64_t sm_sum[16];
  __shared__ int64_t sm_min[16];
  __shared__ int64_t sm_max[16];
  uint32_t const warp_id = threadIdx.x / 64;
  if (warp.thread_rank() == 0) {
    sm_count[warp_id] = local_count;
    sm_sum[warp_id]   = local_sum;
    sm_min[warp_id]   = local_min;
    sm_max[warp_id]   = local_max;
  }
  __syncthreads();

  // First warp reduces across warps
  if (warp_id == 0) {
    int num_warps = (blockDim.x + 63) / 64;
    local_count = (threadIdx.x < num_warps) ? sm_count[threadIdx.x] : 0;
    local_sum   = (threadIdx.x < num_warps) ? sm_sum[threadIdx.x]   : 0;
    local_min   = (threadIdx.x < num_warps) ? sm_min[threadIdx.x]   : INT64_MAX;
    local_max   = (threadIdx.x < num_warps) ? sm_max[threadIdx.x]   : INT64_MIN;

    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
      local_count += warp.shfl_down(local_count, offset);
      local_sum   += warp.shfl_down(local_sum, offset);
      local_min    = min(local_min, warp.shfl_down(local_min, offset));
      local_max    = max(local_max, warp.shfl_down(local_max, offset));
    }

    // Thread 0 of block 0 writes the final atomic update
    if (threadIdx.x == 0) {
      switch (desc.agg) {
        case agg_type::COUNT:
          atomicAdd(reinterpret_cast<unsigned long long*>(desc.result),
                    static_cast<unsigned long long>(local_count));
          break;
        case agg_type::SUM:
          atomicAdd(reinterpret_cast<unsigned long long*>(desc.result),
                    static_cast<unsigned long long>(local_sum));
          break;
        case agg_type::MIN:
          atomicMin(desc.result, local_min);
          break;
        case agg_type::MAX:
          atomicMax(desc.result, local_max);
          break;
      }
    }
  }
}

/// Launch the persistent kernel.
/// @param column_data  Device pointer to the column's data
/// @param stride        Bytes per element (1, 2, 4, 8)
/// @param num_rows      Total rows
/// @param agg           Aggregate type
/// @param filter        Filter operation
/// @param filter_value  Comparison value
/// @param result        Device pointer to initialized result (0 for count/sum,
///                       INT64_MAX for min, INT64_MIN for max)
/// @param stream        CUDA/HIP stream
void launch_persistent_scan_filter_agg(
    void const* column_data, uint32_t stride, uint32_t num_rows,
    agg_type agg, filter_op filter, int64_t filter_value,
    int64_t* result, cudaStream_t stream) {
  if (num_rows == 0) return;

  uint32_t const block_dim = 256;
  uint32_t const grid_dim = (num_rows + block_dim - 1) / block_dim;

  persistent_kernel_desc desc;
  desc.column_data  = column_data;
  desc.column_stride = stride;
  desc.num_rows     = num_rows;
  desc.agg          = agg;
  desc.filter       = filter;
  desc.filter_value = filter_value;
  desc.result       = result;

  persistent_scan_filter_agg<<<grid_dim, block_dim, 0, stream>>>(desc);
}

}  // namespace sirius::cuda
