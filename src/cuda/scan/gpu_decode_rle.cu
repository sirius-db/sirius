/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "cuda/scan/gpu_decode.cuh"
#include "log/logging.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace sirius::cuda::scan {

// DuckDB RLE header size: uint64_t storing the offset to the counts array.
static constexpr uint32_t RLE_HEADER_SIZE = 8;

//===----------------------------------------------------------------------===//
// Device: binary search (upper_bound) in inclusive prefix-sum array
//===----------------------------------------------------------------------===//

/// Find the index of the first element in `cumsum` that is strictly greater
/// than `key`.  Equivalent to std::upper_bound on the cumsum array.
///
/// cumsum[i] = sum(counts[0..i])  (inclusive prefix sum of run lengths).
/// For output row `r`, upper_bound gives the RLE entry that owns that row.
__device__ __forceinline__ uint32_t
rle_upper_bound(const uint32_t* __restrict__ cumsum, uint32_t n, uint32_t key)
{
  uint32_t lo = 0, hi = n;
  while (lo < hi) {
    uint32_t mid = lo + ((hi - lo) >> 1);
    if (cumsum[mid] <= key)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

//===----------------------------------------------------------------------===//
// Device: expand kernel — one thread per output row
// Loads cumsum into shared memory when it fits (entry_count <= 4096),
// eliminating L2 cache pressure from 256 threads all binary-searching
// the same global memory array.
//===----------------------------------------------------------------------===//

/// Max RLE entries that fit in shared memory (4096 × 4B = 16KB).
static constexpr uint32_t RLE_SMEM_MAX_ENTRIES = 4096;

template <typename T>
__global__ void kernel_rle_expand(
    const T* __restrict__       values,
    const uint32_t* __restrict__ cumsum,
    uint32_t                     entry_count,
    T* __restrict__              output,
    uint32_t                     row_count)
{
  // Load cumsum into shared memory for cache-friendly binary search
  __shared__ uint32_t s_cumsum[RLE_SMEM_MAX_ENTRIES];
  const bool use_smem = (entry_count <= RLE_SMEM_MAX_ENTRIES);
  if (use_smem) {
    for (uint32_t i = threadIdx.x; i < entry_count; i += blockDim.x) {
      s_cumsum[i] = cumsum[i];
    }
    __syncthreads();
  }

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= row_count) return;

  const uint32_t* search = use_smem ? s_cumsum : cumsum;
  uint32_t entry = rle_upper_bound(search, entry_count, idx);
  output[idx] = values[entry];
}

//===----------------------------------------------------------------------===//
// Host API
//===----------------------------------------------------------------------===//

void gpu_decode_rle(
    const uint8_t* segment_data,
    size_t         segment_size,
    uint32_t       block_offset,
    uint32_t       row_count,
    uint32_t       type_size,
    void*          d_output,
    rmm::cuda_stream_view stream,
    void*          d_scratch,
    bool           skip_block_copy,
    uint32_t*      d_cumsum_scratch,
    size_t         cumsum_scratch_capacity)
{
  if (row_count == 0) return;

  const uint8_t* seg_base = segment_data + block_offset;

  // ---- CPU: parse header ------------------------------------------------
  uint64_t rle_count_offset;
  std::memcpy(&rle_count_offset, seg_base, sizeof(uint64_t));

  // Scan counts to find entry_count and build inclusive prefix sums.
  // Counts are uint16_t, stored at seg_base + rle_count_offset.
  const uint16_t* h_counts =
      reinterpret_cast<const uint16_t*>(seg_base + rle_count_offset);

  std::vector<uint32_t> h_cumsum;
  h_cumsum.reserve(256);  // typical entry counts are small
  uint32_t total = 0;
  while (total < row_count) {
    total += h_counts[h_cumsum.size()];
    h_cumsum.push_back(total);
  }
  uint32_t entry_count = static_cast<uint32_t>(h_cumsum.size());

  // ---- GPU: H2D segment block data --------------------------------------
  uint8_t* d_block;
  bool owns_block = false;
  if (d_scratch) {
    d_block = static_cast<uint8_t*>(d_scratch);
    if (!skip_block_copy) {
      cudaMemcpyAsync(d_block, segment_data, segment_size,
                       cudaMemcpyHostToDevice, stream.value());
    }
  } else {
    cudaMallocAsync(&d_block, segment_size, stream.value());
    cudaMemcpyAsync(d_block, segment_data, segment_size,
                     cudaMemcpyHostToDevice, stream.value());
    owns_block = true;
  }

  // ---- GPU: H2D prefix sums (reuse scratch if provided) -----------------
  uint32_t* d_cumsum;
  size_t cumsum_bytes = entry_count * sizeof(uint32_t);
  bool owns_cumsum = false;
  if (d_cumsum_scratch && cumsum_bytes <= cumsum_scratch_capacity) {
    d_cumsum = d_cumsum_scratch;
  } else {
    cudaMallocAsync(&d_cumsum, cumsum_bytes, stream.value());
    owns_cumsum = true;
  }
  cudaMemcpyAsync(d_cumsum, h_cumsum.data(), cumsum_bytes,
                   cudaMemcpyHostToDevice, stream.value());

  // Device pointer to values within the block
  const uint8_t* d_seg    = d_block + block_offset;
  const void*    d_values = d_seg + RLE_HEADER_SIZE;

  // ---- GPU: launch expand kernel ----------------------------------------
  constexpr uint32_t kThreads = 256;
  uint32_t blocks = (row_count + kThreads - 1) / kThreads;

  switch (type_size) {
    case 1:
      kernel_rle_expand<uint8_t><<<blocks, kThreads, 0, stream.value()>>>(
          static_cast<const uint8_t*>(d_values), d_cumsum, entry_count,
          static_cast<uint8_t*>(d_output), row_count);
      break;
    case 2:
      kernel_rle_expand<uint16_t><<<blocks, kThreads, 0, stream.value()>>>(
          static_cast<const uint16_t*>(d_values), d_cumsum, entry_count,
          static_cast<uint16_t*>(d_output), row_count);
      break;
    case 4:
      kernel_rle_expand<uint32_t><<<blocks, kThreads, 0, stream.value()>>>(
          static_cast<const uint32_t*>(d_values), d_cumsum, entry_count,
          static_cast<uint32_t*>(d_output), row_count);
      break;
    case 8:
      kernel_rle_expand<uint64_t><<<blocks, kThreads, 0, stream.value()>>>(
          static_cast<const uint64_t*>(d_values), d_cumsum, entry_count,
          static_cast<uint64_t*>(d_output), row_count);
      break;
    case 16:
      kernel_rle_expand<ulonglong2><<<blocks, kThreads, 0, stream.value()>>>(
          static_cast<const ulonglong2*>(d_values), d_cumsum, entry_count,
          static_cast<ulonglong2*>(d_output), row_count);
      break;
    default:
      if (owns_cumsum) cudaFreeAsync(d_cumsum, stream.value());
      if (owns_block) cudaFreeAsync(d_block, stream.value());
      throw std::runtime_error(
          "gpu_decode_rle: unsupported type_size " + std::to_string(type_size));
  }

  // ---- Cleanup ----------------------------------------------------------
  if (owns_cumsum) cudaFreeAsync(d_cumsum, stream.value());
  if (owns_block) cudaFreeAsync(d_block, stream.value());
}

}  // namespace sirius::cuda::scan
