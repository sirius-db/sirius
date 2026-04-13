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

  // Load packed data into shared memory (GPU-FOR insight: shared mem extraction)
  extern __shared__ uint32_t shmem[];
  const uint32_t* packed = reinterpret_cast<const uint32_t*>(d_segment + meta.data_offset);
  uint32_t packed_words = (row_count * width + 31) / 32;

  for (uint32_t i = threadIdx.x; i < packed_words; i += blockDim.x) {
    shmem[i] = packed[i];
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
// Host-side metadata parsing
//===----------------------------------------------------------------------===//

static std::vector<gpu_bp_meta> parse_metadata(
    const uint8_t* segment_data,
    uint32_t block_offset,
    uint32_t row_count,
    uint32_t type_size)
{
  const uint8_t* base = segment_data + block_offset;

  // Header: first 8 bytes = end of metadata region (metadata_offset + metadata_size).
  // DuckDB stores metadata entries backwards from that end point.
  // The first group's encoded metadata is at (base + header_value - 4).
  uint64_t metadata_end;
  std::memcpy(&metadata_end, base, sizeof(uint64_t));

  const uint32_t* meta_ptr = reinterpret_cast<const uint32_t*>(
      base + metadata_end - sizeof(uint32_t));

  uint32_t num_groups = (row_count + BP_META_GROUP_SIZE - 1) / BP_META_GROUP_SIZE;
  std::vector<gpu_bp_meta> groups(num_groups);

  uint32_t rows_left = row_count;
  uint32_t out_row = 0;

  for (uint32_t g = 0; g < num_groups; ++g) {
    auto& gm = groups[g];

    // Decode metadata entry
    uint32_t encoded = *meta_ptr;
    meta_ptr--;
    if (g == 0) {
      fprintf(stderr, "[gpu_bp_parse] group0: encoded=0x%08x mode=%u data_off=%u\n",
              encoded, (encoded >> 24) & 0xFF, encoded & 0x00FFFFFF);
    }

    gm.mode = (encoded >> 24) & 0xFF;
    uint32_t data_off = encoded & 0x00FFFFFF;
    gm.row_count = std::min(rows_left, BP_META_GROUP_SIZE);
    gm.out_offset = out_row;
    rows_left -= gm.row_count;
    out_row += gm.row_count;

    // Absolute address of group data
    const uint8_t* dp = base + data_off;
    auto mode = static_cast<BitpackingMode>(gm.mode);

    switch (mode) {
      case BitpackingMode::CONSTANT: {
        int64_t v = 0;
        std::memcpy(&v, dp, type_size);
        gm.frame_of_ref = v;
        gm.aux = v;
        gm.width = 0;
        gm.data_offset = 0;  // unused
        break;
      }
      case BitpackingMode::CONSTANT_DELTA: {
        int64_t f = 0, d = 0;
        std::memcpy(&f, dp, type_size);
        std::memcpy(&d, dp + type_size, type_size);
        gm.frame_of_ref = f;
        gm.aux = d;
        gm.width = 0;
        gm.data_offset = 0;  // unused
        break;
      }
      case BitpackingMode::FOR: {
        int64_t f = 0;
        std::memcpy(&f, dp, type_size);
        dp += type_size;
        int64_t w = 0;
        std::memcpy(&w, dp, type_size);
        gm.width = static_cast<uint32_t>(w);
        dp += std::max<size_t>(type_size, sizeof(uint8_t));
        gm.frame_of_ref = f;
        gm.aux = 0;
        // data_offset: absolute offset from start of segment_data
        gm.data_offset = static_cast<uint32_t>(dp - segment_data);
        break;
      }
      case BitpackingMode::DELTA_FOR: {
        int64_t f = 0;
        std::memcpy(&f, dp, type_size);
        dp += type_size;
        int64_t w = 0;
        std::memcpy(&w, dp, type_size);
        gm.width = static_cast<uint32_t>(w);
        dp += std::max<size_t>(type_size, sizeof(uint8_t));
        int64_t d = 0;
        std::memcpy(&d, dp, type_size);
        dp += type_size;
        gm.frame_of_ref = f;
        gm.aux = d;
        gm.data_offset = static_cast<uint32_t>(dp - segment_data);
        break;
      }
      default:
        gm.mode = static_cast<uint8_t>(BitpackingMode::INVALID);
        gm.width = 0;
        gm.frame_of_ref = 0;
        gm.aux = 0;
        gm.data_offset = 0;
        break;
    }
  }

  return groups;
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
    void* d_meta_scratch,
    size_t meta_scratch_size)
{
  // 1. Parse metadata on host
  auto groups = parse_metadata(segment_data, block_offset, row_count, sizeof(T));
  if (groups.empty()) return;

  // 2. Copy segment data to GPU �� use scratch buffer if provided
  uint8_t* d_segment = nullptr;
  bool own_segment = false;
  if (d_scratch && segment_size <= UINT32_MAX) {
    d_segment = static_cast<uint8_t*>(d_scratch);
  } else {
    cudaMallocAsync(&d_segment, segment_size, stream.value());
    own_segment = true;
  }
  cudaMemcpyAsync(d_segment, segment_data, segment_size,
                  cudaMemcpyHostToDevice, stream.value());

  // 3. Copy metadata array to GPU — use scratch buffer if provided
  gpu_bp_meta* d_meta = nullptr;
  bool own_meta = false;
  size_t meta_bytes = groups.size() * sizeof(gpu_bp_meta);
  if (d_meta_scratch && meta_bytes <= meta_scratch_size) {
    d_meta = static_cast<gpu_bp_meta*>(d_meta_scratch);
  } else {
    cudaMallocAsync(&d_meta, meta_bytes, stream.value());
    own_meta = true;
  }
  cudaMemcpyAsync(d_meta, groups.data(), meta_bytes,
                  cudaMemcpyHostToDevice, stream.value());

  // 4. Compute max shared memory needed (largest FOR/DELTA_FOR group)
  uint32_t max_shmem = 0;
  for (auto& g : groups) {
    auto mode = static_cast<BitpackingMode>(g.mode);
    if (mode == BitpackingMode::FOR || mode == BitpackingMode::DELTA_FOR) {
      uint32_t words = (g.row_count * g.width + 31) / 32;
      max_shmem = std::max(max_shmem, words * 4);
    }
  }

  // 5. Launch unified kernel: one CTA per group
  constexpr uint32_t BLOCK_DIM = 256;
  uint32_t num_groups = static_cast<uint32_t>(groups.size());
  kernel_decode_segment<T><<<num_groups, BLOCK_DIM, max_shmem, stream.value()>>>(
      d_segment, static_cast<T*>(d_output), d_meta, num_groups);

  // 6. Cleanup only if we allocated
  if (own_segment) cudaFreeAsync(d_segment, stream.value());
  if (own_meta) cudaFreeAsync(d_meta, stream.value());
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
    size_t meta_scratch_size)
{
  // Fully async — caller is responsible for stream synchronization.
  switch (type_size) {
    case 4:
      is_signed
          ? decode_typed<int32_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size)
          : decode_typed<uint32_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size);
      break;
    case 8:
      is_signed
          ? decode_typed<int64_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size)
          : decode_typed<uint64_t>(segment_data, segment_size, block_offset, row_count, d_output, stream, d_scratch, d_meta_scratch, meta_scratch_size);
      break;
    default:
      SIRIUS_LOG_ERROR("[gpu_decode] unsupported type_size={}", type_size);
      return;
  }
}

}  // namespace sirius::cuda::scan
