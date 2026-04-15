/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "cuda/scan/gpu_decode.cuh"
#include "log/logging.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <vector>

namespace sirius::cuda::scan {

//===----------------------------------------------------------------------===//
// Unified decode kernel — one CTA per metadata group, mode dispatched per-block
//===----------------------------------------------------------------------===//

/// Per-group metadata uploaded to GPU as a struct-of-arrays.
struct gpu_bp_meta {
  uint8_t mode;          // BitpackingMode
  uint32_t data_offset;  // Byte offset in segment to packed data
  uint32_t width;        // Bitpacking width
  int64_t frame_of_ref;  // FOR/CONSTANT_DELTA frame
  int64_t aux;           // CONSTANT value, CONSTANT_DELTA delta, DELTA_FOR delta_offset
  uint32_t row_count;    // Rows in this group
  uint32_t out_offset;   // Row offset in output buffer
};

/// Single unified kernel: each block decodes one 2048-value metadata group.
/// Mode is dispatched per-block via the metadata array.
template <typename T>
__global__ void kernel_decode_segment(
    const uint8_t* __restrict__ d_segment,
    T* __restrict__ d_output,
    const gpu_bp_meta* __restrict__ d_meta,
    uint32_t num_groups)
{
  uint32_t gid = blockIdx.x;
  if (gid >= num_groups) return;

  const gpu_bp_meta& meta = d_meta[gid];
  T* out = d_output + meta.out_offset;
  uint32_t row_count = meta.row_count;
  auto mode = static_cast<BitpackingMode>(meta.mode);

  //--- CONSTANT: fill with constant value ---
  if (mode == BitpackingMode::CONSTANT) {
    T val = static_cast<T>(meta.aux);
    for (uint32_t i = threadIdx.x; i < row_count; i += blockDim.x) {
      out[i] = val;
    }
    return;
  }

  //--- CONSTANT_DELTA: values[i] = frame + i * delta ---
  if (mode == BitpackingMode::CONSTANT_DELTA) {
    T frame = static_cast<T>(meta.frame_of_ref);
    T delta = static_cast<T>(meta.aux);
    for (uint32_t i = threadIdx.x; i < row_count; i += blockDim.x) {
      out[i] = frame + static_cast<T>(i) * delta;
    }
    return;
  }

  //--- FOR or DELTA_FOR: unpack bitpacked data ---
  uint32_t width = meta.width;
  T frame = static_cast<T>(meta.frame_of_ref);

  // Load packed data into shared memory. Use memcpy to handle potentially
  // misaligned packed data (e.g. int16 segments with 2-byte alignment).
  extern __shared__ uint32_t shmem[];
  const uint8_t* packed_bytes = d_segment + meta.data_offset;
  uint32_t packed_words = (row_count * width + 31) / 32;

  for (uint32_t i = threadIdx.x; i < packed_words; i += blockDim.x) {
    memcpy(&shmem[i], packed_bytes + i * sizeof(uint32_t), sizeof(uint32_t));
  }
  __syncthreads();

  if (mode == BitpackingMode::FOR) {
    // Simple FOR: unpack + add frame_of_reference
    constexpr uint32_t VPT = 8;  // values per thread
    for (uint32_t v = 0; v < VPT; ++v) {
      uint32_t idx = threadIdx.x * VPT + v;
      if (idx >= row_count) break;
      out[idx] = frame + unpack_value<T>(shmem, idx, width);
    }
    return;
  }

  //--- DELTA_FOR: unpack deltas, add frame, then prefix sum ---
  // CUB BlockScan for inclusive prefix sum within the block
  constexpr uint32_t BLOCK_DIM = 256;
  constexpr uint32_t VPT = 8;  // 256 * 8 = 2048 = BP_META_GROUP_SIZE
  typedef cub::BlockScan<T, BLOCK_DIM> BlockScanT;

  // BlockScan temp storage goes after packed data in shared memory
  // We need to be careful about alignment
  __shared__ typename BlockScanT::TempStorage scan_temp;

  T thread_data[VPT];
  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t idx = threadIdx.x * VPT + v;
    if (idx < row_count) {
      thread_data[v] = frame + unpack_value<T>(shmem, idx, width);
    } else {
      thread_data[v] = T(0);
    }
  }

  // Intra-thread prefix sum
  for (uint32_t v = 1; v < VPT; ++v) {
    thread_data[v] += thread_data[v - 1];
  }

  // Block-wide inclusive scan on thread aggregates
  T thread_agg = thread_data[VPT - 1];
  T scanned_agg;
  BlockScanT(scan_temp).InclusiveSum(thread_agg, scanned_agg);

  // Each thread's prefix = scanned_agg - thread_agg (sum of all prior threads)
  T delta_offset = static_cast<T>(meta.aux);
  T prefix = (scanned_agg - thread_agg) + delta_offset;

  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t idx = threadIdx.x * VPT + v;
    if (idx < row_count) {
      out[idx] = thread_data[v] + prefix;
    }
  }
}

//===----------------------------------------------------------------------===//
// Fused parse-and-decode kernel — metadata parsed on GPU, zero H2D for metadata
//===----------------------------------------------------------------------===//

/// Fused kernel: each CTA parses one metadata group directly from the block
/// data already on GPU, then all threads decode cooperatively.
///
/// Eliminates the per-segment metadata H2D transfer entirely (was 27K calls
/// across TPC-H SF10).  Zero warp divergence: thread 0 parses into shared
/// memory, then all 256 threads branch on the same mode value.
template <typename T>
__global__ void kernel_parse_and_decode_segment_fused(
    const uint8_t* __restrict__ d_segment,
    T* __restrict__ d_output,
    uint32_t block_offset,
    uint32_t row_count,
    uint32_t num_groups)
{
  uint32_t gid = blockIdx.x;
  if (gid >= num_groups) return;

  // Shared metadata — written by thread 0, read by all after barrier
  __shared__ uint8_t   sm_mode;
  __shared__ uint32_t  sm_data_offset;
  __shared__ uint32_t  sm_width;
  __shared__ uint32_t  sm_row_count;
  __shared__ T         sm_frame;
  __shared__ T         sm_aux;

  // --- Thread 0: parse this group's metadata from block data on GPU ---
  if (threadIdx.x == 0) {
    const uint8_t* base = d_segment + block_offset;

    // First 8 bytes of segment = end-of-metadata offset
    uint64_t metadata_end;
    memcpy(&metadata_end, base, sizeof(uint64_t));

    // Metadata entries are consecutive uint32s stored backwards from metadata_end
    const uint8_t* entry_addr =
        base + metadata_end - (gid + 1) * sizeof(uint32_t);
    uint32_t encoded;
    memcpy(&encoded, entry_addr, sizeof(uint32_t));

    sm_mode = (encoded >> 24) & 0xFF;
    uint32_t data_off = encoded & 0x00FFFFFF;

    uint32_t out_offset = gid * BP_META_GROUP_SIZE;
    sm_row_count = min(row_count - out_offset, BP_META_GROUP_SIZE);

    // Data pointer for this group's mode-specific fields
    const uint8_t* dp = base + data_off;

    // Read up to 2 values unconditionally (safe: always within the 256KB block).
    // The third value (DELTA_FOR only) is read conditionally below.
    T v0{}, v1{};
    memcpy(&v0, dp, sizeof(T));
    memcpy(&v1, dp + sizeof(T), sizeof(T));

    auto mode = static_cast<BitpackingMode>(sm_mode);
    switch (mode) {
      case BitpackingMode::CONSTANT:
        sm_frame = v0;
        sm_aux   = v0;
        sm_width = 0;
        sm_data_offset = 0;
        break;
      case BitpackingMode::CONSTANT_DELTA:
        sm_frame = v0;
        sm_aux   = v1;
        sm_width = 0;
        sm_data_offset = 0;
        break;
      case BitpackingMode::FOR:
        sm_frame = v0;
        sm_width = static_cast<uint32_t>(v1);
        sm_aux   = T(0);
        // Packed data starts after frame (sizeof T) + width (sizeof T)
        sm_data_offset = block_offset + data_off + 2 * sizeof(T);
        break;
      case BitpackingMode::DELTA_FOR: {
        T v2{};
        memcpy(&v2, dp + 2 * sizeof(T), sizeof(T));
        sm_frame = v0;
        sm_width = static_cast<uint32_t>(v1);
        sm_aux   = v2;
        // Packed data starts after frame + width + delta (3 × sizeof T)
        sm_data_offset = block_offset + data_off + 3 * sizeof(T);
        break;
      }
      default:
        sm_mode  = static_cast<uint8_t>(BitpackingMode::INVALID);
        sm_width = 0;
        sm_frame = T(0);
        sm_aux   = T(0);
        sm_data_offset = 0;
        break;
    }
  }
  __syncthreads();

  // --- All threads: decode using shared metadata (zero divergence) ---
  uint32_t rc = sm_row_count;
  T* out = d_output + gid * BP_META_GROUP_SIZE;
  auto mode = static_cast<BitpackingMode>(sm_mode);

  //--- CONSTANT ---
  if (mode == BitpackingMode::CONSTANT) {
    T val = sm_aux;
    for (uint32_t i = threadIdx.x; i < rc; i += blockDim.x) {
      out[i] = val;
    }
    return;
  }

  //--- CONSTANT_DELTA ---
  if (mode == BitpackingMode::CONSTANT_DELTA) {
    T frame = sm_frame;
    T delta = sm_aux;
    for (uint32_t i = threadIdx.x; i < rc; i += blockDim.x) {
      out[i] = frame + static_cast<T>(i) * delta;
    }
    return;
  }

  //--- FOR or DELTA_FOR: load packed data into shared memory ---
  uint32_t width = sm_width;
  T frame = sm_frame;

  extern __shared__ uint32_t shmem[];
  const uint8_t* packed_bytes = d_segment + sm_data_offset;
  uint32_t packed_words = (rc * width + 31) / 32;

  // Use memcpy to handle potentially misaligned packed data (e.g. int16 segments
  // where the packed data start is only 2-byte aligned, not 4-byte aligned).
  for (uint32_t i = threadIdx.x; i < packed_words; i += blockDim.x) {
    memcpy(&shmem[i], packed_bytes + i * sizeof(uint32_t), sizeof(uint32_t));
  }
  __syncthreads();

  if (mode == BitpackingMode::FOR) {
    constexpr uint32_t VPT = 8;
    for (uint32_t v = 0; v < VPT; ++v) {
      uint32_t idx = threadIdx.x * VPT + v;
      if (idx >= rc) break;
      out[idx] = frame + unpack_value<T>(shmem, idx, width);
    }
    return;
  }

  //--- DELTA_FOR: unpack, prefix sum ---
  constexpr uint32_t BLOCK_DIM = 256;
  constexpr uint32_t VPT = 8;
  typedef cub::BlockScan<T, BLOCK_DIM> BlockScanT;
  __shared__ typename BlockScanT::TempStorage scan_temp;

  T thread_data[VPT];
  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t idx = threadIdx.x * VPT + v;
    thread_data[v] = (idx < rc)
        ? frame + unpack_value<T>(shmem, idx, width)
        : T(0);
  }

  for (uint32_t v = 1; v < VPT; ++v) {
    thread_data[v] += thread_data[v - 1];
  }

  T thread_agg = thread_data[VPT - 1];
  T scanned_agg;
  BlockScanT(scan_temp).InclusiveSum(thread_agg, scanned_agg);

  T delta_offset = sm_aux;
  T prefix = (scanned_agg - thread_agg) + delta_offset;

  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t idx = threadIdx.x * VPT + v;
    if (idx < rc) {
      out[idx] = thread_data[v] + prefix;
    }
  }
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

template <typename T>
static void decode_typed(
    const uint8_t* segment_data,
    size_t segment_size,
    uint32_t block_offset,
    uint32_t row_count,
    void* d_output,
    rmm::cuda_stream_view stream,
    void* d_scratch,
    void* /* d_meta_scratch */,
    size_t /* meta_scratch_size */,
    bool skip_block_copy)
{
  if (row_count == 0) return;

  // 1. Copy block data to GPU (reuse scratch buffer if available)
  uint8_t* d_segment = nullptr;
  bool own_segment = false;
  if (d_scratch && segment_size <= UINT32_MAX) {
    d_segment = static_cast<uint8_t*>(d_scratch);
  } else {
    cudaMallocAsync(&d_segment, segment_size, stream.value());
    own_segment = true;
  }
  if (!skip_block_copy) {
    cudaMemcpyAsync(d_segment, segment_data, segment_size,
                    cudaMemcpyHostToDevice, stream.value());
  }

  // 2. Launch fused parse-and-decode kernel.
  //    Metadata is parsed on GPU directly from the block data that was just
  //    transferred, eliminating the separate per-segment metadata H2D.
  uint32_t num_groups = (row_count + BP_META_GROUP_SIZE - 1) / BP_META_GROUP_SIZE;
  constexpr uint32_t BLOCK_DIM = 256;
  // Conservative upper bound: max bitpacking width is sizeof(T)*8 bits.
  constexpr uint32_t max_width = sizeof(T) * 8;
  constexpr uint32_t max_packed_words = (BP_META_GROUP_SIZE * max_width + 31) / 32;
  constexpr size_t shmem_bytes = max_packed_words * sizeof(uint32_t);

  kernel_parse_and_decode_segment_fused<T>
      <<<num_groups, BLOCK_DIM, shmem_bytes, stream.value()>>>(
      d_segment, static_cast<T*>(d_output),
      block_offset, row_count, num_groups);

  // 3. Cleanup only if we allocated
  if (own_segment) cudaFreeAsync(d_segment, stream.value());
}


void gpu_decode_bitpacking(
    const uint8_t* segment_data,
    size_t segment_size,
    uint32_t block_offset,
    uint32_t row_count,
    uint32_t type_size,
    bool is_signed,
    void* d_output,
    rmm::cuda_stream_view stream,
    void* d_scratch,
    void* d_meta_scratch,
    size_t meta_scratch_size,
    bool skip_block_copy)
{
  // Fully async — caller is responsible for stream synchronization.
  switch (type_size) {
    case 1:
      is_signed
          ? decode_typed<int8_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size, skip_block_copy)
          : decode_typed<uint8_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size, skip_block_copy);
      break;
    case 2:
      is_signed
          ? decode_typed<int16_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size, skip_block_copy)
          : decode_typed<uint16_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size, skip_block_copy);
      break;
    case 4:
      is_signed
          ? decode_typed<int32_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size, skip_block_copy)
          : decode_typed<uint32_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size, skip_block_copy);
      break;
    case 8:
      is_signed
          ? decode_typed<int64_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size, skip_block_copy)
          : decode_typed<uint64_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size, skip_block_copy);
      break;
    default:
      SIRIUS_LOG_ERROR("[gpu_decode] unsupported type_size={}", type_size);
      return;
  }
}

}  // namespace sirius::cuda::scan
