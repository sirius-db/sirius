/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <rmm/cuda_stream_view.hpp>

namespace sirius::cuda::scan {

//===----------------------------------------------------------------------===//
// DuckDB compression format constants (must match duckdb internals)
//===----------------------------------------------------------------------===//

/// Bitpacking algorithm group size — 32 values per group (warp-aligned).
static constexpr uint32_t BP_GROUP_SIZE = 32;

/// Metadata group size — 2048 values per metadata entry.
static constexpr uint32_t BP_META_GROUP_SIZE = 2048;

/// Number of algorithm groups per metadata group.
static constexpr uint32_t BP_ALGO_GROUPS_PER_META = BP_META_GROUP_SIZE / BP_GROUP_SIZE;  // 64

/// DuckDB BitpackingMode enum values (matches duckdb::BitpackingMode).
enum class BitpackingMode : uint8_t {
  INVALID = 0,
  AUTO = 1,
  CONSTANT = 2,
  CONSTANT_DELTA = 3,
  DELTA_FOR = 4,
  FOR = 5
};

/// Decoded metadata for one 2048-value group.
struct bp_group_meta {
  BitpackingMode mode;
  uint32_t data_offset;     ///< Byte offset from segment base to compressed data
  uint32_t width;           ///< Bitpacking width (only for FOR/DELTA_FOR)
  int64_t frame_of_ref;     ///< Frame of reference value
  int64_t constant_or_delta;///< Constant value (CONSTANT) or delta (CONSTANT_DELTA/DELTA_FOR)
  uint32_t row_count;       ///< Number of rows in this group (last group may be < 2048)
};

//===----------------------------------------------------------------------===//
// Host-side API: decode a full segment's bitpacked data on GPU
//===----------------------------------------------------------------------===//

/// @brief Decode a bitpacked numeric segment on GPU.
///
/// Copies raw segment data to GPU, parses metadata on host, launches decode
/// kernels per metadata group.  Fully async — caller must sync the stream.
///
/// @param segment_data   Host pointer to pinned segment block data
/// @param segment_size   Size of segment data in bytes
/// @param block_offset   Offset within the block to the segment start
/// @param row_count      Total rows in the segment
/// @param type_size      Size of the output type in bytes (4 for int32, 8 for int64)
/// @param is_signed      Whether the type is signed (for sign extension)
/// @param d_output       Pre-allocated device buffer (row_count * type_size bytes)
/// @param stream         CUDA stream
/// @param d_scratch      Optional pre-allocated device buffer (>= segment_size bytes).
///                       If non-null, used instead of cudaMallocAsync for the block copy.
/// @param d_meta_scratch Optional pre-allocated device buffer for metadata.
///                       If non-null, used instead of cudaMallocAsync.
/// @param meta_scratch_size Size of d_meta_scratch in bytes (0 if not provided).
void gpu_decode_bitpacking(
    const uint8_t* segment_data,
    size_t segment_size,
    uint32_t block_offset,
    uint32_t row_count,
    uint32_t type_size,
    bool is_signed,
    void* d_output,
    rmm::cuda_stream_view stream,
    void* d_scratch = nullptr,
    void* d_meta_scratch = nullptr,
    size_t meta_scratch_size = 0);

//===----------------------------------------------------------------------===//
// Host-side API: decode a dictionary string segment on GPU
//===----------------------------------------------------------------------===//

/// Result of dictionary string decode — raw buffers for cuDF string column.
struct dict_decode_result {
  int32_t* d_offsets;     ///< Device: int32 offsets array (num_rows + 1 elements)
  uint8_t* d_chars;       ///< Device: contiguous char buffer
  size_t total_chars;     ///< Total bytes in char buffer
};

/// @brief Decode a dictionary-compressed string segment on GPU.
///
/// Has one mid-stream sync to read total_chars for char buffer allocation.
/// Otherwise async — caller must sync the stream.
///
/// @param segment_data   Host pointer to pinned segment block data
/// @param segment_size   Size of segment data in bytes
/// @param block_offset   Offset within the block to the segment start
/// @param block_size     Total block size (usually 262144 = 256KB)
/// @param row_count      Total rows in the segment
/// @param d_offsets      Pre-allocated device buffer ((row_count+1) * sizeof(int32_t))
/// @param d_chars_out    Output: device pointer to allocated char buffer (caller must free)
/// @param total_chars_out Output: total bytes in char buffer
/// @param stream         CUDA stream
/// @param d_scratch      Optional pre-allocated device buffer (>= segment_size bytes).
void gpu_decode_dictionary(
    const uint8_t* segment_data,
    size_t segment_size,
    uint32_t block_offset,
    uint32_t block_size,
    uint32_t row_count,
    int32_t* d_offsets,
    uint8_t** d_chars_out,
    size_t* total_chars_out,
    rmm::cuda_stream_view stream,
    void* d_scratch = nullptr);

//===----------------------------------------------------------------------===//
// Device functions: inline bitpacking extraction (for future operator fusion)
//===----------------------------------------------------------------------===//

/// @brief Extract one bitpacked value from a packed buffer.
/// GPU-FOR style: horizontal layout, 32-value groups.
/// Each value occupies bits [idx*width, (idx+1)*width) in the packed stream.
template <typename T>
__device__ __forceinline__ T unpack_value(
    const uint32_t* packed,
    uint32_t idx,
    uint32_t width)
{
  if (width == 0) return T(0);

  uint64_t bit_pos = static_cast<uint64_t>(idx) * width;
  uint32_t word_idx = static_cast<uint32_t>(bit_pos / 32);
  uint32_t bit_off = static_cast<uint32_t>(bit_pos & 31);

  // Load two consecutive 32-bit words and combine into 64-bit
  uint64_t combined = static_cast<uint64_t>(packed[word_idx]);
  if (bit_off + width > 32) {
    combined |= static_cast<uint64_t>(packed[word_idx + 1]) << 32;
  }

  uint64_t mask = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);
  return static_cast<T>((combined >> bit_off) & mask);
}

}  // namespace sirius::cuda::scan
