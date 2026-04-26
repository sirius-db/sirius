/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

//===----------------------------------------------------------------------===//
// Shared format constants + bitpacking decode API for the GPU-native scan
// path.  This header collects the DuckDB on-disk constants that every codec
// agrees on (block size, bitpacking group sizes / mode enum / group meta)
// plus the batched-bitpacking descriptor and host entrypoint.  Codec-specific
// .cu files live alongside src/cuda/scan/.
//
// Caller contracts:
//   * `unpack_value<T>` is a __device__ helper — callers must pre-stage the
//     packed buffer with one extra guard word beyond the live data so 64-bit
//     decode reads can span 3 words without going OOB.
//   * `gpu_decode_bitpacking_batched` is fully async — the caller owns
//     stream sync.  Descriptors must reference device pointers (or GH200
//     unified host pointers).
//
// As A1 ships, only batched bitpacking is wired up here.  String / RLE /
// FSST / ALP entrypoints will be added back as their kernels arrive in
// later PRs in the split.
//===----------------------------------------------------------------------===//

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

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

/// DuckDB storage block size (256KB).
static constexpr size_t DUCKDB_BLOCK_SIZE = 262144;

/// DuckDB BitpackingMode enum values (matches duckdb::BitpackingMode).
enum class BitpackingMode : uint8_t {
  INVALID        = 0,
  AUTO           = 1,
  CONSTANT       = 2,
  CONSTANT_DELTA = 3,
  DELTA_FOR      = 4,
  FOR            = 5
};

/// Decoded metadata for one 2048-value group.
struct bp_group_meta {
  BitpackingMode mode;
  uint32_t data_offset;       ///< Byte offset from segment base to compressed data
  uint32_t width;             ///< Bitpacking width (only for FOR/DELTA_FOR)
  int64_t frame_of_ref;       ///< Frame of reference value
  int64_t constant_or_delta;  ///< Constant value (CONSTANT) or delta (CONSTANT_DELTA/DELTA_FOR)
  uint32_t row_count;         ///< Number of rows in this group (last group may be < 2048)
};

//===----------------------------------------------------------------------===//
// Batched bitpacking decode
//===----------------------------------------------------------------------===//

/// Descriptor for batched bitpacking decode — one entry per metadata group.
/// The batched kernel launches one CTA per group (matching the original
/// per-segment kernel's parallelism) while eliminating launch overhead.
struct alignas(8) batched_bp_seg_desc {
  const uint8_t* d_block;      ///< Device pointer to 256KB block data
  uint32_t block_offset;       ///< Offset within block to segment start
  uint32_t group_idx;          ///< Metadata group index within the segment (0-based)
  uint32_t group_row_count;    ///< Rows in this group (last group may be < 2048)
  uint32_t global_row_offset;  ///< Output offset in elements for this group
};

/// @brief Decode multiple bitpacked segments in a single kernel launch.
///
/// Replaces N per-segment gpu_decode_bitpacking() calls with one batched launch.
/// Descriptor d_block pointers must be valid device (or GH200 host) pointers.
///
/// @param descs         Host array of segment descriptors (uploaded internally)
/// @param num_segments  Number of segments to decode
/// @param d_output      Pre-allocated device buffer (total_rows * type_size bytes)
/// @param type_size     Size of the output type in bytes (1, 2, 4, or 8)
/// @param is_signed     Whether the type is signed
/// @param stream        CUDA stream
void gpu_decode_bitpacking_batched(const batched_bp_seg_desc* descs,
                                   uint32_t num_segments,
                                   void* d_output,
                                   uint32_t type_size,
                                   bool is_signed,
                                   rmm::cuda_stream_view stream);

//===----------------------------------------------------------------------===//
// Device functions: inline bitpacking extraction (for future operator fusion)
//===----------------------------------------------------------------------===//

/// @brief Extract one bitpacked value from a packed buffer.
/// GPU-FOR style: horizontal layout, 32-value groups.
/// Each value occupies bits [idx*width, (idx+1)*width) in the packed stream.
///
/// For types wider than 32 bits (int64/uint64), a value can span 3 uint32
/// words when bit_off > 0 and width > 32.  E.g. width=50, bit_off=20 needs
/// bits 20..69, spanning words [word_idx, word_idx+1, word_idx+2].
/// Callers must ensure packed[] has one extra guard word beyond the packed data.
template <typename T>
__device__ __forceinline__ T unpack_value(const uint32_t* packed, uint32_t idx, uint32_t width)
{
  if (width == 0) return T(0);

  uint64_t bit_pos  = static_cast<uint64_t>(idx) * width;
  uint32_t word_idx = static_cast<uint32_t>(bit_pos / 32);
  uint32_t bit_off  = static_cast<uint32_t>(bit_pos & 31);

  // Load two consecutive 32-bit words and combine into 64-bit
  uint64_t combined = static_cast<uint64_t>(packed[word_idx]);
  if (bit_off + width > 32) { combined |= static_cast<uint64_t>(packed[word_idx + 1]) << 32; }

  uint64_t result = combined >> bit_off;

  // For 64-bit types, the value can span a third word when bit_off + width > 64
  if constexpr (sizeof(T) > 4) {
    if (bit_off > 0 && bit_off + width > 64) {
      result |= static_cast<uint64_t>(packed[word_idx + 2]) << (64 - bit_off);
    }
  }

  uint64_t mask = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);
  return static_cast<T>(result & mask);
}

}  // namespace sirius::cuda::scan
