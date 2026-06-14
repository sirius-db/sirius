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

//! @file
//! Shared core for the string codecs (UNCOMPRESSED, DICTIONARY, FSST,
//! DICT_FSST): the on-device FSST decoder type, the per-kernel work
//! descriptors, the per-codec prepared-data structs the orchestrator
//! aggregates, the tuning constants, and the host chunking helpers. Each
//! codec lives in its own translation unit (strings/<codec>.cu) and shares
//! these definitions; gpu_decode_strings.cu owns the orchestrator.

#pragma once

#include "cuda/scan/detail/warp.cuh"

#include <rmm/detail/error.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sirius::cuda::scan {

using detail::FULL_MASK;

//----- FSST decoder types (shared: FSST + DICT_FSST modes 1+2) --------------//
/// 255 codes (0..254); byte 255 is the escape sentinel.
constexpr uint32_t FSST_SIZE             = 256;
constexpr uint32_t FSST_NUM_SYMBOLS      = 255;
constexpr uint8_t FSST_ESC               = 255;
constexpr uint32_t FSST_SYMTAB_MAX_BYTES = 8192;  // opaque serialized blob

/// Trimmed `duckdb_fsst_decoder_t`: just the `len` + `symbol` arrays the device
/// decode path populates and reads (drops `version` + `zeroTerminated`).
struct fsst_decoder_compact {
  uint8_t len[FSST_NUM_SYMBOLS];
  unsigned long long symbol[FSST_NUM_SYMBOLS];
};

//----- Per-kernel descriptors -----------------------------------------------//
/// Descriptor for DICTIONARY chunks and FSST length-pass segments.
/// FSST length pass can't chunk: the in-CTA prefix sum is per-segment.
struct alignas(8) string_chunk_desc {
  uint8_t const* d_bytes;
  uint32_t bytes_size;
  uint32_t row_count;
  uint32_t global_row_start;  ///< output position
  uint32_t seg_row_start;     ///< rows skipped at segment head when chunked
};

struct alignas(8) fsst_chunk_desc {
  uint8_t const* d_bytes;
  uint32_t bytes_size;
  uint32_t row_count;
  uint32_t global_row_start;
  uint32_t fsst_row_start;  ///< offset into d_comp_offsets
  uint32_t seg_decoder_idx;
  uint8_t is_first_chunk;  ///< gates the my_comp[-1] read
  uint8_t _pad[3];
};

struct alignas(8) dict_fsst_desc {
  uint8_t const* d_bytes;
  uint32_t bytes_size;
  uint32_t row_count;
  uint32_t global_row_start;
  uint32_t seg_row_start;
  uint32_t dict_data_offset;      ///< raw/FSST-compressed dict bytes
  uint32_t dict_indices_offset;   ///< bitpacked dictionary_indices
  uint32_t seg_dict_offset_base;  ///< base into d_byte_offsets / d_decoded_offsets
  uint32_t seg_decoder_idx;       ///< unused for mode 0
  uint32_t dict_count;            ///< includes reserved idx 0
  uint32_t predecode_seg_offset;  ///< mode-1 only
  uint8_t dict_indices_width;
  uint8_t mode;
  uint8_t _pad[6];
};

//----- Tuning constants -----------------------------------------------------//
constexpr uint32_t BLOCK_DIM = 256;  // see FSST_WARPS_PER_CTA static_assert
constexpr uint32_t MIN_ROWS_PER_CHUNK =
  64;  ///< Minimum rows per segment chunk; BLOCK_DIM=256 threads -> 8 warps
       ///< per chunk -> 8 rows per warp at this minimum.
constexpr uint32_t MAX_BITPACKING_WIDTH = 32;

/// Above this, take the exact-total sync rather than trust the host upper
/// bound — a pathological max_string_length could otherwise force a GB-class
/// over-allocation.
constexpr size_t HOST_UPPER_BOUND_LIMIT = size_t{512} * 1024u * 1024u;

/// Per-segment threshold for switching DICTIONARY gather from thread-per-row
/// (launch-bound at short rows) to warp-cooperative (bandwidth-bound at long
/// rows). Mirrors cuDF's strings-gather split (32B) with headroom.
constexpr uint32_t DICT_WARP_COOP_MIN_LEN = 64u;

//----- Per-codec prepared-data struct types ---------------------------------//
// Each codec's prepare_* returns one of these; gpu_decode_strings_column
// aggregates them across runs before launching kernels.

struct prepared_uncomp {
  std::vector<string_chunk_desc> descs;
};

struct prepared_dict {
  std::vector<string_chunk_desc> descs_short;  ///< max_string_length < DICT_WARP_COOP_MIN_LEN
  std::vector<string_chunk_desc> descs_long;   ///< max_string_length >= DICT_WARP_COOP_MIN_LEN
};

struct prepared_fsst {
  std::vector<string_chunk_desc> length_descs;  ///< pass-1 A+B (per segment)
  std::vector<fsst_chunk_desc> gather_chunks;   ///< pass-1 phase-C + pass-2 (per chunk)
  std::vector<fsst_decoder_compact> decoders;   ///< symbol tables (per segment)
  std::vector<uint32_t> row_starts;             ///< prefix sum of FSST row counts
  uint32_t total_fsst_row_count;
};

struct prepared_dict_fsst {
  std::vector<dict_fsst_desc> descs;
  std::vector<fsst_decoder_compact> decoders;
  std::vector<uint32_t> byte_offsets;     ///< per-segment, dict_count+1 entries each
  std::vector<uint32_t> decoded_offsets;  ///< per-segment, dict_count+1 entries each
  bool any_inline_nulls;
  uint32_t total_predecode_bytes;  ///< sum of mode-1 dict-decoded bytes
};

//----- Host chunking helpers ------------------------------------------------//
//! @brief Align a value to the next 8-byte boundary.
//!
//! Mirror of DuckDB's AlignValue<idx_t> for 64-bit idx_t.
constexpr uint32_t align_up8(uint32_t n) { return (n + 7u) & ~7u; }

//! @brief Target CTA count for chunking segments: two full device waves at
//! BLOCK_DIM threads. Cached per device.
inline uint32_t get_target_ctas()
{
  int device = 0;
  RMM_CUDA_TRY(cudaGetDevice(&device));
  static int cached_device = -1;
  static uint32_t cached   = 0;
  if (cached_device == device) return cached;
  cudaDeviceProp prop;
  RMM_CUDA_TRY(cudaGetDeviceProperties(&prop, device));
  int occupancy_blocks = prop.maxThreadsPerMultiProcessor / BLOCK_DIM;
  cached_device        = device;
  cached               = static_cast<uint32_t>(prop.multiProcessorCount * occupancy_blocks * 2);
  return cached;
}

//! @brief Expand segment descriptors into smaller chunks.
//!
//! Used for per-row work (compute_lengths, gather); per-dict work (predecode, mark_nulls) keeps
//! one-CTA-per-segment so dictionaries are not decoded per chunk.
template <typename Desc>
std::vector<Desc> expand_chunks(std::vector<Desc> const& descs, uint32_t target_ctas)
{
  uint32_t total_rows = 0;
  for (auto const& d : descs)
    total_rows += d.row_count;
  if (descs.size() >= target_ctas || total_rows == 0) return descs;

  uint32_t chunk_size = total_rows / target_ctas;
  chunk_size          = std::max(chunk_size, MIN_ROWS_PER_CHUNK);
  // Round down to warp size for store coalescing within a chunk.
  chunk_size = (chunk_size / cub::detail::warp_threads) * cub::detail::warp_threads;
  if (chunk_size == 0) chunk_size = cub::detail::warp_threads;

  std::vector<Desc> out;
  out.reserve(target_ctas + descs.size());
  for (auto const& seg : descs) {
    uint32_t remaining = seg.row_count;
    uint32_t off       = 0;
    while (remaining > 0) {
      uint32_t n             = std::min(remaining, chunk_size);
      Desc chunk             = seg;
      chunk.row_count        = n;
      chunk.global_row_start = seg.global_row_start + off;
      chunk.seg_row_start    = seg.seg_row_start + off;
      out.push_back(chunk);
      off += n;
      remaining -= n;
    }
  }
  return out;
}

}  // namespace sirius::cuda::scan
