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

#include "cuda/scan/detail/byte_copy.cuh"
#include "cuda/scan/detail/warp.cuh"
#include "cuda/scan/gpu_decode_strings.cuh"
#include "cuda/scan/strings/common.cuh"
#include "cuda/scan/strings/dictionary.cuh"
#include "cuda/scan/strings/uncompressed.cuh"
#include "cuda/scan/unpack_value.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::cuda::scan {

namespace {

//===----------------------------------------------------------------------===//
// 1. Shared core
//
// The descriptors, prepared-data structs, tuning constants, and host chunking
// helpers shared across codecs live in strings/common.cuh. What remains here is
// the on-disk header layouts + their device parsers, kept beside the codecs
// that read them. References to the DuckDB sources for the on-disk layouts:
//   duckdb/src/storage/compression/dictionary.cpp     (DICTIONARY)
//   duckdb/src/storage/compression/fsst.cpp           (FSST)
//   duckdb/src/storage/compression/dict_fsst/{compression,decompression}.cpp
//                                                     (DICT_FSST)
//===----------------------------------------------------------------------===//

//----- On-disk headers ------------------------------------------------------//
struct fsst_header_t {
  uint32_t dict_size;
  uint32_t dict_end;
  uint32_t bitpacking_width;
  uint32_t fsst_symbol_table_offset;
};

struct dict_fsst_header_t {
  uint32_t dict_size;
  uint32_t dict_count;  ///< includes reserved idx 0 (NULL, length 0)
  uint8_t mode;         ///< 0=DICTIONARY, 1=DICT_FSST, 2=FSST_ONLY
  uint8_t string_lengths_width;
  uint8_t dictionary_indices_width;
  uint8_t _pad;
  uint32_t symbol_table_size;
};

enum : uint8_t {
  DICT_FSST_MODE_DICTIONARY = 0,
  DICT_FSST_MODE_DICT_FSST  = 1,
  DICT_FSST_MODE_FSST_ONLY  = 2,
};

//----- FSST decoder types (shared `fsst_decoder_compact` in strings/common.cuh) ----//
/// `duckdb_fsst_decoder_t` layout — opaque outside the import → memcpy step.
struct fsst_decoder_full {
  unsigned long long version;
  unsigned char zeroTerminated;
  unsigned char len[FSST_NUM_SYMBOLS];
  unsigned long long symbol[FSST_NUM_SYMBOLS];
};
static_assert(sizeof(fsst_decoder_compact::len) == sizeof(fsst_decoder_full::len));
static_assert(sizeof(fsst_decoder_compact::symbol) == sizeof(fsst_decoder_full::symbol));

// Declared locally to avoid pulling the libduckdb FSST header into the CUDA TU.
extern "C" unsigned int duckdb_fsst_import(void* decoder, unsigned char* buf);

//----- On-disk header parsers -----------------------------------------------//
/**
 * @brief Copy FSST header @p hdr into @p base, bounded by the buffer size @p limit.
 * @return true if the header was successfully copied (i.e. the header fits within the buffer), and
 * if the header metadata is valid; false otherwise.
 */
__device__ __forceinline__ bool parse_fsst_header(uint8_t const* base,
                                                  uint32_t limit,
                                                  fsst_header_t* hdr)
{
  if (limit < sizeof(fsst_header_t)) return false;
  memcpy(hdr, base, sizeof(fsst_header_t));
  return hdr->dict_end <= limit && hdr->fsst_symbol_table_offset < hdr->dict_end &&
         hdr->bitpacking_width <= MAX_BITPACKING_WIDTH;
}

//===----------------------------------------------------------------------===//
// 4. FSST codec
//
//   offset 0                                                       seg_end
//   +--------+--------------------+---------------+------------------------+
//   | header | compressed_lengths |  symbol table |  FSST-compressed bytes |
//   |  16B   |  bitpacked per row |  opaque blob  |  rows packed in        |
//   |        |  -> comp_len       |  imported on  |  REVERSE order ending  |
//   |        |                    |  device       |  at dict_end           |
//   +--------+--------------------+---------------+------------------------+
//   0        16                   ^                                        ^
//                                 hdr.fsst_symbol_table_offset             hdr.dict_end
//
// Decode pipeline (in launch order):
//   kernel_build_fsst_decoders               — parse opaque symtab into
//                                              fsst_decoder_compact on device
//   kernel_compute_compressed_offsets_fsst   — Pass-1 A+B: per-segment
//                                              prefix sum of compressed
//                                              lengths -> d_comp_offsets
//   kernel_compute_decompressed_lengths_fsst — Pass-1 C: per-row byte-walk
//                                              to compute decoded length
//   kernel_gather_fsst_chunked               — Pass-2: per-row decode +
//                                              emit to d_chars
//
// Pass-1 A+B owns per-segment prefix-sum state (one CTA per segment);
// Pass-1 C and Pass-2 partition each segment across CTAs into chunks.
// The warp-decode helper `warp_decode_fsst` is also used by DICT_FSST
// mode 2 (inline-decompress per row).
//===----------------------------------------------------------------------===//

//----- On-device FSST symbol-table import -----------------------------------//
/**
 * @brief Import FSST symbol table on device from the opaque on-disk blob format.
 *
 * Device port of duckdb_fsst_import (libfsst.cpp:429). Same opaque blob layout (8B version | 1B
 * zeroTerminated | 8B lenHisto | packed symbols by length group 1..8,1). Used by
 * kernel_build_fsst_decoders to populate per-segment decoders without round-tripping through host.
 */
__device__ __forceinline__ bool device_fsst_import(uint8_t const* buf, fsst_decoder_compact* out)
{
  constexpr uint32_t FSST_VERSION_LO = 20190218;
  constexpr uint64_t FSST_CORRUPT    = 32774747032022883;  // "corrupt" in LE

  // Zero-fill so a bad header / corrupted symtab leaves a deterministic (all-zero-length) decoder.
  for (uint32_t i = 0; i < FSST_NUM_SYMBOLS; ++i) {
    out->len[i]    = 0;
    out->symbol[i] = 0;
  }

  uint64_t version;
  memcpy(&version, buf, sizeof(uint64_t));
  if ((version >> 32) != FSST_VERSION_LO) { return false; }

  uint32_t pos      = 17;
  uint8_t zero_term = buf[8] & 1u;
  uint8_t length_histo[8];
  for (uint32_t i = 0; i < 8u; ++i) {
    length_histo[i] = buf[9 + i];
  }

  out->len[0]    = 1;
  out->symbol[0] = 0;
  uint32_t code  = zero_term;
  if (zero_term) length_histo[0]--;

  for (uint32_t l = 1; l <= 8u; ++l) {
    uint32_t hist_idx = l & 7u;         // 1,2,3,4,5,6,7,0
    uint32_t sym_len  = (l & 7u) + 1u;  // 2,3,4,5,6,7,8,1
    for (uint32_t i = 0; i < length_histo[hist_idx]; ++i, ++code) {
      out->len[code] = static_cast<uint8_t>(sym_len);
      uint64_t sym   = 0;
      for (uint32_t j = 0; j < sym_len; ++j) {
        sym |= uint64_t(buf[pos + j]) << (8u * j);
      }
      out->symbol[code] = sym;
      pos += sym_len;
    }
  }
  while (code < FSST_NUM_SYMBOLS) {
    out->symbol[code] = FSST_CORRUPT;
    out->len[code++]  = 8u;
  }
  return true;
}

/**
 * @brief Build per-segment FSST decoders from the on-disk symbol table blob.
 *
 * Coalesced load of the symbol table blob for the segment into shared memory, then single-threaded
 * parse into the fsst_decoder_compact format in global memory. One CTA per segment.
 */
__global__ __launch_bounds__(BLOCK_DIM, 4) void kernel_build_fsst_decoders(
  string_chunk_desc const* __restrict__ descs,
  uint32_t num_segments,
  fsst_decoder_compact* __restrict__ d_decoders)
{
  auto const seg_idx = blockIdx.x;
  auto const desc    = descs[seg_idx];

  __shared__ bool sm_ok;
  __shared__ fsst_header_t sm_hdr;
  __shared__ uint8_t sm_symtab[FSST_SYMTAB_MAX_BYTES];

  if (threadIdx.x == 0) { sm_ok = parse_fsst_header(desc.d_bytes, desc.bytes_size, &sm_hdr); }
  __syncthreads();

  if (!sm_ok) {
    for (uint32_t i = threadIdx.x; i < FSST_NUM_SYMBOLS; i += blockDim.x) {
      d_decoders[seg_idx].len[i]    = 0;
      d_decoders[seg_idx].symbol[i] = 0;
    }
    return;
  }

  auto const symtab_off = sm_hdr.fsst_symbol_table_offset;
  auto const symtab_size =
    ::cuda::std::min(desc.bytes_size - symtab_off, uint32_t{FSST_SYMTAB_MAX_BYTES});
  auto const* sym_src = desc.d_bytes + symtab_off;
  for (uint32_t i = threadIdx.x; i < symtab_size; i += blockDim.x) {
    sm_symtab[i] = sym_src[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) { device_fsst_import(sm_symtab, &d_decoders[seg_idx]); }
}

//----- Pass 1 (A+B): per-row compressed-byte offsets ------------------------//
/**
 * @brief Compute the per-row offsets into the FSST compressed byte stream for an FSST segment.
 *
 * One CTA per segment; in-CTA cub::BlockScan over per-thread chunk sums produces the inclusive
 * prefix sum the gather kernel reads.
 */
__global__ void kernel_compute_compressed_offsets_fsst(
  uint32_t* __restrict__ d_comp_offsets,
  string_chunk_desc const* __restrict__ descs,
  uint32_t const* __restrict__ d_fsst_row_starts,
  uint32_t num_segments)
{
  using BlockScanT = cub::BlockScan<uint32_t, BLOCK_DIM>;

  __shared__ typename BlockScanT::TempStorage scan_temp;
  __shared__ bool sm_ok;
  __shared__ fsst_header_t sm_hdr;

  auto const seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc  = descs[seg_idx];
  auto const* base = desc.d_bytes;

  if (threadIdx.x == 0) { sm_ok = parse_fsst_header(base, desc.bytes_size, &sm_hdr); }
  __syncthreads();

  auto const segment_base      = d_fsst_row_starts[seg_idx];
  auto const segment_row_count = desc.row_count;
  auto* segment_comp_offsets   = d_comp_offsets + segment_base;

  if (!sm_ok) {
    // Zero the offsets for the segment so phase-C emits empty strings.
    for (uint32_t i = threadIdx.x; i < segment_row_count; i += blockDim.x)
      segment_comp_offsets[i] = 0;
    return;
  }

  // Phase A: unpack compressed lengths from the bitpacked length stream after the header.
  auto const* packed = reinterpret_cast<uint32_t const*>(base + sizeof(fsst_header_t));
  for (uint32_t i = threadIdx.x; i < segment_row_count; i += blockDim.x) {
    segment_comp_offsets[i] = unpack_value<uint32_t>(packed, i, sm_hdr.bitpacking_width);
  }
  __syncthreads();

  // Phase B: per-thread sequential scan + BlockScan over per-thread totals.
  auto const max_rows_per_thread = ::cuda::ceil_div(segment_row_count, blockDim.x);
  auto const start               = threadIdx.x * max_rows_per_thread;
  auto const end                 = ::cuda::std::min(start + max_rows_per_thread, segment_row_count);
  uint32_t thread_sum            = 0;
  for (int i = start; i < end; ++i) {
    /// WARNING: uncoalesced GMEM access pattern
    thread_sum += segment_comp_offsets[i];
    segment_comp_offsets[i] = thread_sum;
  }
  uint32_t exclusive_sum = 0;
  BlockScanT(scan_temp).ExclusiveSum(thread_sum, exclusive_sum);
  if (exclusive_sum > 0) {
    for (int i = start; i < end; ++i) {
      /// WARNING: uncoalesced GMEM access pattern
      segment_comp_offsets[i] += exclusive_sum;
    }
  }
}

//----- Pass 1 (C): per-row decompressed lengths -----------------------------//
/**
 * @brief Compute the compressed length of a row in the FSST compressed stream, where @p comp_ptr
 * points to the start of the compressed bytes for that row, and @p sm_len is the symbol length
 * table from the FSST header.
 */
__device__ __forceinline__ uint32_t warp_compute_decomp_len(uint8_t const* __restrict__ comp_ptr,
                                                            uint32_t comp_len,
                                                            uint8_t const* __restrict__ sm_len,
                                                            uint32_t lane)
{
  uint32_t total             = 0;
  int prev_chunk_last_is_esc = 0;
  for (uint32_t off = 0; off < comp_len; off += WARP_THREADS) {
    auto const bytes_in_chunk = ::cuda::std::min(comp_len - off, uint32_t{WARP_THREADS});
    auto const is_active      = lane < bytes_in_chunk;
    uint8_t const my_byte     = is_active ? comp_ptr[off + lane] : 0;
    int const is_esc          = is_active && my_byte == FSST_ESC ? 1 : 0;

    auto const neighbor_is_esc = __shfl_up_sync(FULL_MASK, is_esc, 1);
    auto const prev_is_esc     = (lane == 0) ? prev_chunk_last_is_esc : neighbor_is_esc;

    // Per-lane contribution: 0 for inactive / escape-sentinel;
    //                        1 for the literal byte that follows an escape;
    //                        sm_len[code] otherwise.
    uint32_t my_len = 0;
    if (is_active) {
      if (prev_is_esc) {
        my_len = 1;
      } else if (!is_esc) {
        my_len = sm_len[my_byte];
      }
    }

    // Warp-reduce per-lane lengths.
#pragma unroll
    for (uint32_t s = WARP_THREADS / 2; s > 0; s /= 2) {
      my_len += __shfl_xor_sync(FULL_MASK, my_len, s);
    }
    total += my_len;

    // Broadcast whether the last byte in this 32B stripe was an escape for the next stripe.
    auto const last_active_lane = bytes_in_chunk - 1;
    prev_chunk_last_is_esc      = __shfl_sync(FULL_MASK, is_esc, last_active_lane);
  }
  return total;
}

/**
 * @brief Compute the decompressed lengths for each row in the FSST compressed stream.
 *
 * Adaptive per CTA: thread-per-row for short rows (lower warp-coord tax dominates), warp-per-row
 * for longer rows (coalesced LDG dominates). 2 × WARP_THREADS = 64 B / row crossover matches the
 * empirical sweet spot.
 */
__global__ __launch_bounds__(BLOCK_DIM, 8) void kernel_compute_decompressed_lengths_fsst(
  fsst_chunk_desc const* __restrict__ descs,
  uint32_t* __restrict__ d_lengths,
  uint32_t const* __restrict__ d_comp_offsets,
  fsst_decoder_compact const* __restrict__ d_decoders,
  uint32_t num_chunks)
{
  auto const chunk_id = blockIdx.x;
  if (chunk_id >= num_chunks) return;
  auto const desc  = descs[chunk_id];
  auto const* base = desc.d_bytes;

  __shared__ bool sm_ok;
  __shared__ fsst_header_t sm_hdr;
  __shared__ uint8_t sm_len[FSST_NUM_SYMBOLS];
  if (threadIdx.x == 0) { sm_ok = parse_fsst_header(base, desc.bytes_size, &sm_hdr); }
  __syncthreads();

  // Zero-fill lengths for the chunk if the metadata is malformed.
  if (!sm_ok) {
    for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
      d_lengths[desc.global_row_start + i] = 0;
    }
    return;
  }

  fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
  for (uint32_t i = threadIdx.x; i < FSST_NUM_SYMBOLS; i += blockDim.x) {
    sm_len[i] = dec.len[i];
  }
  __syncthreads();

  auto const* dict_end_ptr          = base + sm_hdr.dict_end;
  auto const* compressed_cumsum_ptr = d_comp_offsets + desc.fsst_row_start;

  // Compute the chunk's average compressed-byte/row to dispatch between
  // thread-per-row and warp-per-row strategies.
  auto const start           = desc.is_first_chunk ? 0 : *(compressed_cumsum_ptr - 1);
  auto const end             = compressed_cumsum_ptr[desc.row_count - 1];
  auto const avg_comp_length = (end - start) / desc.row_count;

  if (avg_comp_length >= 2 * WARP_THREADS) {
    // Warp-per-row: 32 lanes coalesce LDG over a row's compressed bytes.
    auto const lane          = threadIdx.x % WARP_THREADS;
    auto const warp_id       = threadIdx.x / WARP_THREADS;
    auto const warps_per_cta = blockDim.x / WARP_THREADS;
    for (uint32_t i = warp_id; i < desc.row_count; i += warps_per_cta) {
      auto const cumsum      = compressed_cumsum_ptr[i];
      auto const prev_cumsum = (i > 0) ? compressed_cumsum_ptr[i - 1] : start;
      if (cumsum > sm_hdr.dict_end || prev_cumsum > cumsum) {
        if (lane == 0) { d_lengths[desc.global_row_start + i] = 0u; }
        continue;
      }
      auto const comp_len = cumsum - prev_cumsum;
      if (comp_len == 0) {
        if (lane == 0) { d_lengths[desc.global_row_start + i] = 0; }
        continue;
      }
      auto const* comp_length_ptr = dict_end_ptr - cumsum;
      uint32_t decomp_len = warp_compute_decomp_len(comp_length_ptr, comp_len, sm_len, lane);
      if (lane == 0) { d_lengths[desc.global_row_start + i] = decomp_len; }
    }
  } else {
    // Thread-per-row: short rows; warp-coord tax of cooperative scan dominates.
    for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
      auto const cumsum      = compressed_cumsum_ptr[i];
      auto const prev_cumsum = (i > 0) ? compressed_cumsum_ptr[i - 1] : start;
      if (cumsum > sm_hdr.dict_end || prev_cumsum > cumsum) {
        d_lengths[desc.global_row_start + i] = 0u;
        continue;
      }
      auto const comp_len = cumsum - prev_cumsum;
      if (comp_len == 0) {
        d_lengths[desc.global_row_start + i] = 0u;
        continue;
      }
      auto const* comp_ptr = dict_end_ptr - cumsum;
      int decomp_len       = 0;
      int pos              = 0;
      while (pos < comp_len) {
        auto const code = comp_ptr[pos++];
        if (code < FSST_ESC) {
          decomp_len += sm_len[code];
        } else if (pos < comp_len) {
          // Trailing escape (corrupt input) is dropped — stay in sync with gather.
          ++pos;
          ++decomp_len;
        }
      }
      d_lengths[desc.global_row_start + i] = decomp_len;
    }
  }
}

//----- Pass 2: gather (decode + emit) ---------------------------------------//
constexpr uint32_t FSST_SCRATCH_U32_PER_WARP   = 256;
constexpr uint32_t FSST_SCRATCH_BYTES_PER_WARP = FSST_SCRATCH_U32_PER_WARP * sizeof(uint32_t);
constexpr uint32_t FSST_MAX_CHUNK_EMIT =
  WARP_THREADS * 8;  ///< worst-case bytes emitted per 32B input chunk (8B max symbol)
constexpr uint32_t FSST_WARPS_PER_CTA = BLOCK_DIM / WARP_THREADS;

/**
 * @brief Decode a chunk of compressed bytes using the FSST algorithm.
 *
 * One warp decodes `comp_len` compressed bytes in 32-byte input chunks. Decoded symbols accumulate
 * in a per-warp scratch slab; we flush to `dst` only when the next chunk's worst-case emit (256 B)
 * would overflow scratch, or once at the end of the row.
 * `sm_sym` is split lo/hi so the common short-symbol load is one 4 B bank read.
 * `sm_sym_hi` is only read when len[code] > 4.
 *
 * @note Reused by DICT_FSST mode-2 inline decompress.
 * @note Direct lane-to-global stores were measured 30-68 % slower than the shmem-staged drain
 * (small byte stores generated many partial L2 sectors).
 */
__device__ __forceinline__ void warp_decode_fsst(uint8_t const* __restrict__ comp_ptr,
                                                 uint32_t comp_len,
                                                 uint8_t* __restrict__ dst,
                                                 uint8_t const* __restrict__ sm_len,
                                                 uint32_t const* __restrict__ sm_sym_lo,
                                                 uint32_t const* __restrict__ sm_sym_hi,
                                                 uint32_t* __restrict__ warp_scratch_u32,
                                                 uint32_t lane)
{
  auto* warp_scratch_u8           = reinterpret_cast<uint8_t*>(warp_scratch_u32);
  uint32_t cumulative_out_offset  = 0;  ///< bytes already flushed to dst
  uint32_t scratch_used           = 0;  ///< bytes pending in scratch
  uint32_t prev_chunk_last_is_esc = 0;

  for (uint32_t off = 0; off < comp_len; off += WARP_THREADS) {
    auto const bytes_in_chunk = ::cuda::std::min(comp_len - off, uint32_t{WARP_THREADS});
    auto const active         = lane < bytes_in_chunk;
    uint8_t const my_byte     = active ? comp_ptr[off + lane] : 0;
    int const is_esc          = (active && my_byte == FSST_ESC) ? 1 : 0;
    auto const neighbor_esc   = __shfl_up_sync(FULL_MASK, is_esc, 1);
    auto const prev_was_esc   = (lane == 0) ? prev_chunk_last_is_esc : neighbor_esc;

    // Decode the symbol for this lane, if active and not an escape byte.
    // my_len == 0 → no write (inactive or unresolved escape sentinel).
    // my_sym_lo holds `my_byte` directly for escape-literal lanes.
    uint32_t my_len    = 0;
    uint32_t my_sym_lo = 0;
    uint32_t my_sym_hi = 0;
    if (active) {
      if (prev_was_esc) {
        my_len    = 1;
        my_sym_lo = my_byte;
      } else if (!is_esc) {
        auto const code = my_byte;
        my_len          = sm_len[code];
        my_sym_lo       = sm_sym_lo[code];
        if (my_len > 4) { my_sym_hi = sm_sym_hi[code]; }
      }
    }

    // Determine the write offsets into the scratch pad for this lane's decoded symbol via an
    // inclusive scan of the decoded lengths.
    auto scan = my_len;
#pragma unroll
    for (uint32_t step = 1; step < WARP_THREADS; step <<= 1) {
      auto const add = __shfl_up_sync(FULL_MASK, scan, step);
      if (lane >= step) { scan += add; }
    }
    auto const exclusive   = scan - my_len;
    auto const base        = scratch_used + exclusive;
    auto const chunk_total = __shfl_sync(FULL_MASK, scan, WARP_THREADS - 1);

    // Copy the decoded symbol into the scratch pad at the offset computed from the scan.
    if (my_len > 0) {
      switch (my_len) {
        case 1: warp_scratch_u8[base] = static_cast<uint8_t>(my_sym_lo); break;
        case 2: memcpy(warp_scratch_u8 + base, &my_sym_lo, 2); break;
        case 3: memcpy(warp_scratch_u8 + base, &my_sym_lo, 3); break;
        case 4: memcpy(warp_scratch_u8 + base, &my_sym_lo, 4); break;
        default: {
          memcpy(warp_scratch_u8 + base, &my_sym_lo, 4);
          auto const tail_bytes = my_len - 4;
          memcpy(warp_scratch_u8 + base + 4, &my_sym_hi, tail_bytes);
          break;
        }
      }
    }

    scratch_used += chunk_total;
    auto const last_active_lane = bytes_in_chunk - 1;
    prev_chunk_last_is_esc      = __shfl_sync(FULL_MASK, is_esc, last_active_lane);

    // Lazy flush: only when the next chunk might overflow scratch.
    auto const more_chunks = (off + WARP_THREADS) < comp_len;
    if (more_chunks && scratch_used + FSST_MAX_CHUNK_EMIT > FSST_SCRATCH_BYTES_PER_WARP) {
      __syncwarp();
      detail::warp_copy_bytes(dst + cumulative_out_offset, warp_scratch_u8, scratch_used, lane);
      __syncwarp();
      cumulative_out_offset += scratch_used;
      scratch_used = 0;
    }
  }

  // Final flush at end of row — always required, scratch holds the tail.
  if (scratch_used > 0) {
    __syncwarp();
    detail::warp_copy_bytes(dst + cumulative_out_offset, warp_scratch_u8, scratch_used, lane);
    __syncwarp();
  }
}

/**
 * @brief Gather kernel for FSST-compressed segments, with the same chunking as phase-C. Each warp
 * walks the compressed byte stream for its row, decodes on the fly into a per-warp scratch buffer,
 * and flushes to global when the next chunk's worst-case emit would overflow scratch.
 */
__global__ __launch_bounds__(BLOCK_DIM, 8) void kernel_gather_fsst_chunked(
  fsst_chunk_desc const* __restrict__ descs,
  int32_t const* __restrict__ d_offsets,
  uint8_t* __restrict__ d_chars,
  uint32_t const* __restrict__ d_comp_offsets,
  fsst_decoder_compact const* __restrict__ d_decoders,
  uint32_t num_chunks)
{
  __shared__ bool sm_ok;
  __shared__ fsst_header_t sm_hdr;
  __shared__ uint8_t sm_len[FSST_SIZE];
  __shared__ uint32_t sm_sym_lo[FSST_SIZE];  ///< The lower 4B of the symbol
  __shared__ uint32_t sm_sym_hi[FSST_SIZE];  ///< The upper 4B of the symbol (only used if len > 4)
  __shared__ uint32_t
    sm_scratch_u32[FSST_WARPS_PER_CTA]
                  [FSST_SCRATCH_U32_PER_WARP];  ///< Per-warp scratch for decoding symbols before
                                                ///< flushing to global

  auto const chunk_id = blockIdx.x;
  if (chunk_id >= num_chunks) return;
  auto const desc          = descs[chunk_id];
  auto const* segment_base = desc.d_bytes;

  // Parse the segment header.
  if (threadIdx.x == 0) { sm_ok = parse_fsst_header(segment_base, desc.bytes_size, &sm_hdr); }
  __syncthreads();
  if (!sm_ok) return;  // pass-1 emitted zero lengths → nothing to gather

  // Load the symbol table into SMEM.
  fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
  for (uint32_t i = threadIdx.x; i < FSST_SIZE; i += blockDim.x) {
    sm_len[i]          = (i < FSST_NUM_SYMBOLS) ? dec.len[i] : 0;
    uint64_t const sym = (i < FSST_NUM_SYMBOLS) ? dec.symbol[i] : 0;
    sm_sym_lo[i]       = static_cast<uint32_t>(sym);
    sm_sym_hi[i]       = static_cast<uint32_t>(sym >> 32);
  }
  __syncthreads();

  // Warp-per-row gather + decode.
  auto const* dict_end_ptr  = segment_base + sm_hdr.dict_end;
  auto const* my_cumsum_ptr = d_comp_offsets + desc.fsst_row_start;
  auto const lane           = threadIdx.x % WARP_THREADS;
  auto const warp_id        = threadIdx.x / WARP_THREADS;
  auto const warps_per_cta  = blockDim.x / WARP_THREADS;

  for (uint32_t i = warp_id; i < desc.row_count; i += warps_per_cta) {
    auto const my_cumsum = my_cumsum_ptr[i];
    auto const prev_cumsum =
      (i > 0) ? my_cumsum_ptr[i - 1] : (desc.is_first_chunk ? 0 : *(my_cumsum_ptr - 1));
    auto const comp_len = my_cumsum - prev_cumsum;
    if (comp_len == 0) continue;

    auto const* comp_ptr       = dict_end_ptr - my_cumsum;
    auto const out_base_offset = static_cast<uint32_t>(d_offsets[desc.global_row_start + i]);
    warp_decode_fsst(comp_ptr,
                     comp_len,
                     d_chars + out_base_offset,
                     sm_len,
                     sm_sym_lo,
                     sm_sym_hi,
                     &sm_scratch_u32[warp_id][0],
                     lane);
  }
}

//----- Host helpers + prepare_fsst ------------------------------------------//

prepared_fsst prepare_fsst(gpu_string_codec_run const& run)
{
  prepared_fsst out;
  out.total_fsst_row_count = 0;
  out.length_descs.reserve(run.segments.size());
  out.row_starts.reserve(run.segments.size());

  // Segment descriptors for pass-1 A+B.
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    out.row_starts.push_back(out.total_fsst_row_count);
    out.total_fsst_row_count += seg.row_count;
    out.length_descs.push_back(
      {seg.d_bytes, seg.bytes_size, seg.row_count, seg.row_offset, seg.seg_row_start});
  }
  auto const segment_count = out.length_descs.size();
  // decoders are populated on-device by kernel_build_fsst_decoders; here we
  // just allocate the slots so the upload + kernel see the right size.
  out.decoders.resize(segment_count);

  // Chunked descriptors for phase-C + gather. Split per-segment only when
  // total segments < target_ctas (else one-chunk-per-segment fills SMs already).
  auto const target_ctas         = get_target_ctas();
  uint32_t target_rows_per_chunk = 0;
  if (segment_count < target_ctas && out.total_fsst_row_count > 0) {
    target_rows_per_chunk = std::max(out.total_fsst_row_count / target_ctas, MIN_ROWS_PER_CHUNK);
    target_rows_per_chunk = (target_rows_per_chunk / 32) * 32;  // multiple of 32 for coalescing
    if (target_rows_per_chunk == 0) target_rows_per_chunk = 32;
  }
  for (uint32_t segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
    auto const& seg          = out.length_descs[segment_idx];
    auto const fsst_base_row = out.row_starts[segment_idx];
    if (target_rows_per_chunk == 0) {
      // No chunking: one CTA per segment.
      out.gather_chunks.push_back({seg.d_bytes,
                                   seg.bytes_size,
                                   seg.row_count,
                                   seg.global_row_start,
                                   fsst_base_row,
                                   segment_idx,
                                   1,
                                   {0, 0, 0}});
    } else {
      // Chunking: one CTA per chunk (slice of a segment).
      auto remaining                    = seg.row_count;
      uint32_t offset                   = 0;
      uint8_t is_first_chunk_in_segment = 1;
      while (remaining > 0) {
        auto const chunk_row_count = std::min(remaining, target_rows_per_chunk);
        out.gather_chunks.push_back({seg.d_bytes,
                                     seg.bytes_size,
                                     chunk_row_count,
                                     seg.global_row_start + offset,
                                     fsst_base_row + offset,
                                     segment_idx,
                                     is_first_chunk_in_segment,
                                     {0, 0, 0}});
        offset += chunk_row_count;
        remaining -= chunk_row_count;
        is_first_chunk_in_segment = 0;
      }
    }
  }
  return out;
}

//===----------------------------------------------------------------------===//
// 5. DICT_FSST codec
//
//   +-----+------------+--------+----------------+--------------------+
//   | hdr | dict bytes | symtab | string_lengths |    dict_indices    |
//   | 16B |            |        |                |  (absent: mode 2)  |
//   +-----+------------+--------+----------------+--------------------+
//   0     off_dict     off_symtab               off_slens         off_didx
//
// All regions 8-byte aligned (DuckDB AlignValue).
//
// Modes (hdr.mode):
//   0 DICTIONARY  — dict bytes raw; gather is memcpy.
//   1 DICT_FSST   — dict bytes FSST-compressed; predecode kernel decompresses
//                   each dict entry once into a per-segment buffer; gather
//                   memcpys from there.
//   2 FSST_ONLY   — all rows unique; row i → dict entry i+1; no dict_indices
//                   region; gather inline-decompresses each row.
//
// dict_idx == 0 = NULL. DuckDB ships COMPRESSION_EMPTY validity for these,
// which the overlay path skips — mark_nulls folds them in.
//===----------------------------------------------------------------------===//

/**
 * @brief Parse DICT_FSST header @p hdr from @p base, bounded by the buffer size @p limit.
 * @return true if the header fits within the buffer and metadata is valid; false otherwise.
 */
__device__ __forceinline__ bool parse_dict_fsst_header(uint8_t const* base,
                                                       uint32_t limit,
                                                       dict_fsst_header_t* hdr)
{
  if (limit < sizeof(dict_fsst_header_t)) return false;
  memcpy(hdr, base, sizeof(dict_fsst_header_t));
  return hdr->mode <= DICT_FSST_MODE_FSST_ONLY;
}

/**
 * @brief Per-segment scratch input for `kernel_build_dict_fsst_data`. Filled
 * host-side after a single batched header D2H so the kernel has all metadata
 * it needs without re-reading the header on device. `base_off` is the
 * cumulative `(dict_count + 1)` over prior valid segments — it indexes into
 * the global d_byte_offsets / d_decoded_offsets arrays.
 */
struct dict_fsst_pre_desc {
  uint8_t const* d_bytes;
  uint32_t bytes_size;
  uint32_t off_dict;
  uint32_t off_symtab;
  uint32_t off_slens;
  uint32_t base_off;
  uint32_t dict_count;
  uint8_t mode;
  uint8_t string_lengths_width;
  uint8_t valid;  ///< 0 = host-validated as a stub (kernel writes zeros and returns)
  uint8_t _pad;
};
static_assert(sizeof(dict_fsst_pre_desc) % 4 == 0);

/**
 * @brief One CTA per segment: parse the symbol table on device, unpack
 * string_lengths into byte_offsets and inclusive-scan it, then for FSST modes
 * walk each dict entry to fill decoded_offsets. Writes per-segment outputs
 * (total decoded bytes + inline-null flag) for the host to aggregate.
 * Same pattern as `kernel_build_fsst_decoders`.
 */
__global__ __launch_bounds__(BLOCK_DIM, 4) void kernel_build_dict_fsst_data(
  dict_fsst_pre_desc const* __restrict__ pre,
  uint32_t num_segments,
  fsst_decoder_compact* __restrict__ d_decoders,
  uint32_t* __restrict__ d_byte_offsets,
  uint32_t* __restrict__ d_decoded_offsets,
  uint32_t* __restrict__ d_per_seg_decoded_total,
  uint8_t* __restrict__ d_per_seg_inline_null)
{
  using BlockScanT = cub::BlockScan<uint32_t, BLOCK_DIM>;

  __shared__ typename BlockScanT::TempStorage scan_temp;
  __shared__ uint8_t sm_symtab[FSST_SYMTAB_MAX_BYTES];
  __shared__ uint8_t sm_len[FSST_NUM_SYMBOLS];

  auto const seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const d = pre[seg_idx];

  if (!d.valid) {
    if (threadIdx.x == 0) {
      d_per_seg_decoded_total[seg_idx] = 0;
      d_per_seg_inline_null[seg_idx]   = 0;
      for (uint32_t i = 0; i < FSST_NUM_SYMBOLS; ++i) {
        d_decoders[seg_idx].len[i]    = 0;
        d_decoders[seg_idx].symbol[i] = 0;
      }
    }
    return;
  }

  auto* my_byte_off   = d_byte_offsets + d.base_off;
  auto* my_dec_off    = d_decoded_offsets + d.base_off;
  bool const has_fsst = (d.mode != DICT_FSST_MODE_DICTIONARY) && (d.dict_count > 1);

  // Phase 1: symbol-table import on device (FSST modes only).
  if (has_fsst) {
    auto const symtab_size =
      ::cuda::std::min(d.bytes_size - d.off_symtab, uint32_t{FSST_SYMTAB_MAX_BYTES});
    for (uint32_t i = threadIdx.x; i < symtab_size; i += blockDim.x) {
      sm_symtab[i] = d.d_bytes[d.off_symtab + i];
    }
    __syncthreads();
    if (threadIdx.x == 0) { device_fsst_import(sm_symtab, &d_decoders[seg_idx]); }
    __syncthreads();
    // Cache sm_len[] for the per-entry walks below.
    for (uint32_t i = threadIdx.x; i < FSST_NUM_SYMBOLS; i += blockDim.x) {
      sm_len[i] = d_decoders[seg_idx].len[i];
    }
  } else {
    for (uint32_t i = threadIdx.x; i < FSST_NUM_SYMBOLS; i += blockDim.x) {
      d_decoders[seg_idx].len[i]    = 0;
      d_decoders[seg_idx].symbol[i] = 0;
    }
  }
  __syncthreads();

  // Phase 2: unpack string_lengths into byte_offsets[1..dict_count+1).
  // byte_offsets[0] = 0; the inclusive scan below yields the exclusive prefix
  // sum (decoded_offsets[k] = sum of entry_lens[0..k-1]).
  if (threadIdx.x == 0) my_byte_off[0] = 0;
  auto const* slens_packed = reinterpret_cast<uint32_t const*>(d.d_bytes + d.off_slens);
  for (uint32_t k = threadIdx.x; k < d.dict_count; k += blockDim.x) {
    my_byte_off[k + 1] = unpack_value<uint32_t>(slens_packed, k, d.string_lengths_width);
  }
  __syncthreads();

  // In-place inclusive scan over byte_offsets[0..dict_count].
  {
    auto const N              = d.dict_count + 1;
    auto const max_per_thread = ::cuda::ceil_div(N, blockDim.x);
    auto const start          = threadIdx.x * max_per_thread;
    auto const end            = ::cuda::std::min(start + max_per_thread, N);
    uint32_t thread_sum       = 0;
    for (int i = start; i < end; ++i) {
      thread_sum += my_byte_off[i];
      my_byte_off[i] = thread_sum;
    }
    uint32_t exclusive_sum = 0;
    BlockScanT(scan_temp).ExclusiveSum(thread_sum, exclusive_sum);
    if (exclusive_sum > 0) {
      for (int i = start; i < end; ++i) {
        my_byte_off[i] += exclusive_sum;
      }
    }
  }
  __syncthreads();

  // Phase 3: decoded_offsets.
  if (!has_fsst) {
    // Mode 0 (raw DICTIONARY): decoded_offsets = byte_offsets.
    for (uint32_t k = threadIdx.x; k <= d.dict_count; k += blockDim.x) {
      my_dec_off[k] = my_byte_off[k];
    }
    __syncthreads();
  } else {
    // FSST modes: walk each dict entry's compressed bytes, sum decoded lengths.
    // Entry k = 0 is the reserved NULL slot (decoded length = 0); per host
    // contract decoded_offsets[base+1] = decoded_offsets[base] regardless of
    // entry_lens[0].
    if (threadIdx.x == 0) my_dec_off[0] = 0;
    if (threadIdx.x == 0 && d.dict_count >= 1) my_dec_off[1] = 0;  // entry 0 is NULL
    auto const* dict_bytes_base = d.d_bytes + d.off_dict;
    for (uint32_t k = threadIdx.x + 1; k < d.dict_count; k += blockDim.x) {
      auto const comp_start = my_byte_off[k];
      auto const comp_len   = my_byte_off[k + 1] - comp_start;
      auto const* cp        = dict_bytes_base + comp_start;
      uint32_t decomp_len   = 0;
      uint32_t pos          = 0;
      while (pos < comp_len) {
        uint8_t code = cp[pos++];
        if (code < FSST_ESC) {
          decomp_len += sm_len[code];
        } else {
          ++pos;
          ++decomp_len;
        }
      }
      my_dec_off[k + 1] = decomp_len;
    }
    __syncthreads();

    // In-place inclusive scan over decoded_offsets[0..dict_count].
    auto const N              = d.dict_count + 1u;
    auto const max_per_thread = ::cuda::ceil_div(N, blockDim.x);
    auto const start          = threadIdx.x * max_per_thread;
    auto const end            = ::cuda::std::min(start + max_per_thread, N);
    uint32_t thread_sum       = 0;
    for (uint32_t i = start; i < end; ++i) {
      thread_sum += my_dec_off[i];
      my_dec_off[i] = thread_sum;
    }
    uint32_t exclusive_sum = 0;
    BlockScanT(scan_temp).ExclusiveSum(thread_sum, exclusive_sum);
    if (exclusive_sum > 0) {
      for (uint32_t i = start; i < end; ++i) {
        my_dec_off[i] += exclusive_sum;
      }
    }
    __syncthreads();
  }

  // Phase 4: per-segment scalars.
  if (threadIdx.x == 0) {
    d_per_seg_decoded_total[seg_idx] = my_dec_off[d.dict_count];
    // entry_lens[0] = byte_offsets[1] - byte_offsets[0] = byte_offsets[1].
    bool const any_null =
      (d.mode != DICT_FSST_MODE_FSST_ONLY) && (d.dict_count > 1) && (my_byte_off[1] == 0);
    d_per_seg_inline_null[seg_idx] = any_null ? 1 : 0;
  }
}

__global__ void kernel_compute_lengths_dict_fsst(dict_fsst_desc const* __restrict__ descs,
                                                 uint32_t* __restrict__ d_lengths,
                                                 uint32_t const* __restrict__ d_decoded_offsets,
                                                 uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc         = descs[seg_idx];
  uint32_t const* dec_off = d_decoded_offsets + desc.seg_dict_offset_base;
  uint32_t const* d_idx =
    reinterpret_cast<uint32_t const*>(desc.d_bytes + desc.dict_indices_offset);
  bool const fsst_only = (desc.mode == DICT_FSST_MODE_FSST_ONLY);

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t idx =
      fsst_only ? (seg_i + 1u) : unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
    uint32_t len = 0u;
    if (idx != 0u && idx < desc.dict_count) { len = dec_off[idx + 1] - dec_off[idx]; }
    d_lengths[desc.global_row_start + i] = len;
  }
}

/// Mode-1 only: one thread per dict entry decompresses once into predecode_buf;
/// per-row gather is then a memcpy. Avoids row_count×dict redundant FSST work.
__global__ void kernel_predecode_dict_fsst(dict_fsst_desc const* __restrict__ descs,
                                           uint32_t const* __restrict__ d_byte_offsets,
                                           uint32_t const* __restrict__ d_decoded_offsets,
                                           fsst_decoder_compact const* __restrict__ d_decoders,
                                           uint8_t* __restrict__ predecode_buf,
                                           uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc = descs[seg_idx];
  if (desc.mode != DICT_FSST_MODE_DICT_FSST) return;
  if (desc.dict_count <= 1u) return;  // only entry 0 = NULL, nothing to decode

  __shared__ uint8_t sm_len[FSST_NUM_SYMBOLS];
  __shared__ unsigned long long sm_sym[FSST_NUM_SYMBOLS];

  fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
  for (uint32_t i = threadIdx.x; i < FSST_NUM_SYMBOLS; i += blockDim.x) {
    sm_len[i] = dec.len[i];
    sm_sym[i] = dec.symbol[i];
  }
  __syncthreads();

  uint8_t const* dict_data = desc.d_bytes + desc.dict_data_offset;
  uint32_t const* byte_off = d_byte_offsets + desc.seg_dict_offset_base;
  uint32_t const* dec_off  = d_decoded_offsets + desc.seg_dict_offset_base;
  uint8_t* out_base        = predecode_buf + desc.predecode_seg_offset;

  // Skip k=0 (reserved NULL slot, length 0). One thread per dict entry.
  for (uint32_t k = threadIdx.x + 1u; k < desc.dict_count; k += blockDim.x) {
    uint32_t comp_start = byte_off[k];
    uint32_t comp_end   = byte_off[k + 1];
    uint32_t comp_len   = comp_end - comp_start;
    uint32_t out_pos    = dec_off[k];

    uint8_t const* comp_ptr = dict_data + comp_start;
    uint8_t* out_ptr        = out_base + out_pos;
    uint32_t pos            = 0;
    uint32_t op             = 0;
    while (pos < comp_len) {
      uint8_t code = comp_ptr[pos++];
      if (code < FSST_ESC) {
        unsigned long long sym = sm_sym[code];
        uint8_t sym_len        = sm_len[code];
        switch (sym_len) {
          case 1: out_ptr[op] = static_cast<uint8_t>(sym); break;
          case 2: memcpy(out_ptr + op, &sym, 2); break;
          case 3: memcpy(out_ptr + op, &sym, 3); break;
          case 4: memcpy(out_ptr + op, &sym, 4); break;
          default: memcpy(out_ptr + op, &sym, sym_len); break;
        }
        op += sym_len;
      } else if (pos < comp_len) {
        out_ptr[op++] = comp_ptr[pos++];
      }
    }
  }
}

/// Per-row gather for all three DICT_FSST modes (see codec banner above).
__global__ void kernel_gather_dict_fsst(dict_fsst_desc const* __restrict__ descs,
                                        int32_t const* __restrict__ d_offsets,
                                        uint8_t* __restrict__ d_chars,
                                        uint32_t const* __restrict__ d_byte_offsets,
                                        uint32_t const* __restrict__ d_decoded_offsets,
                                        uint8_t const* __restrict__ predecode_buf,
                                        fsst_decoder_compact const* __restrict__ d_decoders,
                                        uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc               = descs[seg_idx];
  uint8_t const* base           = desc.d_bytes;
  uint32_t const* dict_byte_off = d_byte_offsets + desc.seg_dict_offset_base;
  uint32_t const* dict_dec_off  = d_decoded_offsets + desc.seg_dict_offset_base;
  uint32_t const* d_idx     = reinterpret_cast<uint32_t const*>(base + desc.dict_indices_offset);
  bool const fsst_only      = (desc.mode == DICT_FSST_MODE_FSST_ONLY);
  bool const mode_dict_fsst = (desc.mode == DICT_FSST_MODE_DICT_FSST);

  __shared__ uint8_t sm_len[256];
  __shared__ uint32_t sm_sym_lo[256];
  __shared__ uint32_t sm_sym_hi[256];
  __shared__ uint32_t sm_scratch_u32[FSST_WARPS_PER_CTA][FSST_SCRATCH_U32_PER_WARP];

  if (fsst_only) {  // mode 2 inline decompresses; needs the symtab in shmem
    fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) {
      sm_len[i]    = (i < FSST_NUM_SYMBOLS) ? dec.len[i] : uint8_t{0};
      uint64_t sym = (i < FSST_NUM_SYMBOLS) ? dec.symbol[i] : 0ull;
      sm_sym_lo[i] = static_cast<uint32_t>(sym);
      sm_sym_hi[i] = static_cast<uint32_t>(sym >> 32);
    }
    __syncthreads();
  }

  // Modes 0/1 share the same warp-cooperative memcpy; only (offsets, source) differ.
  uint32_t const* memcpy_off = mode_dict_fsst ? dict_dec_off : dict_byte_off;
  uint8_t const* memcpy_src =
    mode_dict_fsst ? (predecode_buf + desc.predecode_seg_offset) : (base + desc.dict_data_offset);

  uint32_t const lane          = threadIdx.x & (WARP_THREADS - 1u);
  uint32_t const warp_id       = threadIdx.x / WARP_THREADS;
  uint32_t const warps_per_cta = blockDim.x / WARP_THREADS;

  for (uint32_t i = warp_id; i < desc.row_count; i += warps_per_cta) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t idx =
      fsst_only ? (seg_i + 1u) : unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
    if (idx == 0u) continue;  // NULL — pass-1 emitted length 0

    uint32_t op = static_cast<uint32_t>(d_offsets[desc.global_row_start + i]);

    if (!fsst_only) {
      uint32_t entry_start = memcpy_off[idx];
      uint32_t entry_len   = memcpy_off[idx + 1] - entry_start;
      uint8_t const* src   = memcpy_src + entry_start;
      detail::warp_copy_bytes(d_chars + op, src, entry_len, lane);
      continue;
    }

    // Mode 2: inline decompress via the FSST gather helper.
    uint32_t byte_start = dict_byte_off[idx];
    uint32_t comp_len   = dict_byte_off[idx + 1] - byte_start;
    if (comp_len == 0) continue;
    warp_decode_fsst(memcpy_src + byte_start,
                     comp_len,
                     d_chars + op,
                     sm_len,
                     sm_sym_lo,
                     sm_sym_hi,
                     &sm_scratch_u32[warp_id][0],
                     lane);
  }
}

/// Fold inline NULLs (idx==0) into the column mask — DuckDB ships
/// COMPRESSION_EMPTY validity for these, which the overlay path skips.
__global__ void kernel_dict_fsst_mark_nulls(dict_fsst_desc const* __restrict__ descs,
                                            uint8_t* __restrict__ d_mask,
                                            uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc = descs[seg_idx];
  uint32_t const* d_idx =
    reinterpret_cast<uint32_t const*>(desc.d_bytes + desc.dict_indices_offset);

  // FSST_ONLY can't encode NULL via idx==0 (row i → entry i+1 always non-zero).
  if (desc.mode == DICT_FSST_MODE_FSST_ONLY) return;

  auto* d_mask_words = reinterpret_cast<unsigned int*>(d_mask);
  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t idx   = unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
    if (idx != 0u) continue;
    uint32_t row = desc.global_row_start + i;
    atomicAnd(d_mask_words + (row >> 5), ~(1u << (row & 31u)));
  }
}

/// Stub descriptor for malformed segments — kernel zero-fills these rows.
dict_fsst_desc make_stub_dict_fsst_desc(gpu_string_segment_desc const& seg)
{
  return {seg.d_bytes,
          seg.bytes_size,
          seg.row_count,
          seg.row_offset,
          seg.seg_row_start,
          0u,
          0u,
          0u,
          0u,
          0u,
          0u,
          0u,
          0u,
          {0, 0, 0, 0, 0, 0}};
}

/**
 * @brief Resizable pinned-host scratch pool. cudaMallocHost is expensive
 * (~ms per call for MB-class allocations) so we keep a single buffer per
 * usage site and grow it on demand — once warm, subsequent calls pay zero
 * allocation cost.
 */
class pinned_host_pool {
  void* ptr_  = nullptr;
  size_t cap_ = 0;

 public:
  void* get(size_t bytes)
  {
    if (bytes > cap_) {
      if (ptr_) cudaFreeHost(ptr_);
      ptr_ = nullptr;
      cap_ = 0;
      if (bytes > 0) {
        RMM_CUDA_TRY(cudaMallocHost(&ptr_, bytes));
        cap_ = bytes;
      }
    }
    return ptr_;
  }
  ~pinned_host_pool()
  {
    if (ptr_) cudaFreeHost(ptr_);
  }
};

/**
 * @brief Build per-segment DICT_FSST predecode state on device.
 *
 * Pipeline:
 *   1. Batched async D2H of all headers (one stream-sync, pinned host pool).
 *   2. Validate headers + compute per-segment region offsets and a cumulative
 *      `base_off` into the flat byte_offsets / decoded_offsets arrays.
 *   3. Launch `kernel_build_dict_fsst_data` — one CTA per segment does symbol
 *      table import, string_lengths unpack + scan, FSST per-entry decoded
 *      length walks, and per-segment scalar outputs.
 *   4. Batched async D2H of the small result buffers (one stream-sync).
 *   5. Host-side build of `dict_fsst_desc[]` with the cross-segment
 *      `predecode_seg_offset` prefix sum.
 */
prepared_dict_fsst prepare_dict_fsst(gpu_string_codec_run const& run,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  prepared_dict_fsst out;
  out.any_inline_nulls      = false;
  out.total_predecode_bytes = 0;
  out.descs.reserve(run.segments.size());
  out.decoders.reserve(run.segments.size());

  uint32_t const num_segs = static_cast<uint32_t>(run.segments.size());
  if (num_segs == 0) return out;

  // Phase 1: batched async D2H of all headers into pinned host memory.
  // Replaces N synchronous cudaMemcpy(header) calls with a single sync. The
  // pinned pool amortizes cudaMallocHost across calls — first call pays, the
  // rest are zero-cost.
  static thread_local pinned_host_pool headers_pool;
  auto* headers =
    static_cast<dict_fsst_header_t*>(headers_pool.get(sizeof(dict_fsst_header_t) * num_segs));
  for (uint32_t i = 0; i < num_segs; ++i) {
    auto const& seg = run.segments[i];
    if (seg.row_count == 0 || seg.bytes_size < sizeof(dict_fsst_header_t)) {
      headers[i].mode = 0xFFu;  // out-of-range marker, host-validated below
      continue;
    }
    RMM_CUDA_TRY(cudaMemcpyAsync(&headers[i],
                                 seg.d_bytes,
                                 sizeof(dict_fsst_header_t),
                                 cudaMemcpyDeviceToHost,
                                 stream.value()));
  }
  RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  // Phase 2: validate, compute per-segment region offsets + cumulative
  // base_off into the global byte/decoded_offsets arrays.
  std::vector<dict_fsst_pre_desc> pre(num_segs);
  uint32_t total_dict_entries = 0;  // sum of (dict_count + 1) for valid segs
  for (uint32_t i = 0; i < num_segs; ++i) {
    auto const& seg = run.segments[i];
    auto& p         = pre[i];
    p.valid         = 0;
    p.d_bytes       = seg.d_bytes;
    p.bytes_size    = seg.bytes_size;
    if (seg.row_count == 0 || seg.bytes_size < sizeof(dict_fsst_header_t)) continue;
    auto const& hdr = headers[i];
    if (hdr.mode > DICT_FSST_MODE_FSST_ONLY) continue;

    uint32_t off_dict   = align_up8(static_cast<uint32_t>(sizeof(hdr)));
    uint32_t off_symtab = align_up8(off_dict + hdr.dict_size);
    uint32_t off_slens  = (hdr.mode == DICT_FSST_MODE_DICTIONARY)
                            ? off_dict + align_up8(hdr.dict_size)
                            : align_up8(off_symtab + hdr.symbol_table_size);
    uint32_t slens_bits = hdr.dict_count * hdr.string_lengths_width;
    uint32_t off_didx   = align_up8(off_slens + (slens_bits + 7u) / 8u);
    if (off_didx > seg.bytes_size && hdr.mode != DICT_FSST_MODE_FSST_ONLY) continue;

    p.off_dict             = off_dict;
    p.off_symtab           = off_symtab;
    p.off_slens            = off_slens;
    p.dict_count           = hdr.dict_count;
    p.mode                 = hdr.mode;
    p.string_lengths_width = hdr.string_lengths_width;
    p.base_off             = total_dict_entries;
    p.valid                = 1;
    total_dict_entries += hdr.dict_count + 1u;
  }

  if (total_dict_entries == 0) {
    for (auto const& seg : run.segments) {
      if (seg.row_count > 0) out.descs.push_back(make_stub_dict_fsst_desc(seg));
    }
    return out;
  }

  // Phase 3: launch the on-device prep kernel. Replaces the per-segment host
  // pipeline (D2H prefix + duckdb_fsst_import + host walks).
  rmm::device_buffer d_pre_buf(pre.size() * sizeof(dict_fsst_pre_desc), stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_pre_buf.data(),
                               pre.data(),
                               pre.size() * sizeof(dict_fsst_pre_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));
  rmm::device_buffer d_decoders_buf(num_segs * sizeof(fsst_decoder_compact), stream, mr);
  rmm::device_buffer d_byte_off_buf(total_dict_entries * sizeof(uint32_t), stream, mr);
  rmm::device_buffer d_dec_off_buf(total_dict_entries * sizeof(uint32_t), stream, mr);
  rmm::device_buffer d_per_seg_total_buf(num_segs * sizeof(uint32_t), stream, mr);
  rmm::device_buffer d_per_seg_inline_null_buf(num_segs * sizeof(uint8_t), stream, mr);

  kernel_build_dict_fsst_data<<<num_segs, BLOCK_DIM, 0, stream.value()>>>(
    static_cast<dict_fsst_pre_desc const*>(d_pre_buf.data()),
    num_segs,
    static_cast<fsst_decoder_compact*>(d_decoders_buf.data()),
    static_cast<uint32_t*>(d_byte_off_buf.data()),
    static_cast<uint32_t*>(d_dec_off_buf.data()),
    static_cast<uint32_t*>(d_per_seg_total_buf.data()),
    static_cast<uint8_t*>(d_per_seg_inline_null_buf.data()));

  // Phase 4: pull results back into host vectors (small — single ~5 MB D2H
  // for typical TPC-H multi-segment workloads).
  out.decoders.resize(num_segs);
  out.byte_offsets.resize(total_dict_entries);
  out.decoded_offsets.resize(total_dict_entries);
  std::vector<uint32_t> per_seg_total(num_segs);
  std::vector<uint8_t> per_seg_inline_null(num_segs);

  static thread_local pinned_host_pool d2h_pool;
  size_t const d2h_bytes = num_segs * sizeof(fsst_decoder_compact) +
                           2 * total_dict_entries * sizeof(uint32_t) + num_segs * sizeof(uint32_t) +
                           num_segs * sizeof(uint8_t);
  auto* staging      = static_cast<uint8_t*>(d2h_pool.get(d2h_bytes));
  size_t off         = 0;
  auto* pin_decoders = reinterpret_cast<fsst_decoder_compact*>(staging + off);
  off += num_segs * sizeof(fsst_decoder_compact);
  auto* pin_byte_off = reinterpret_cast<uint32_t*>(staging + off);
  off += total_dict_entries * sizeof(uint32_t);
  auto* pin_dec_off = reinterpret_cast<uint32_t*>(staging + off);
  off += total_dict_entries * sizeof(uint32_t);
  auto* pin_per_seg_total = reinterpret_cast<uint32_t*>(staging + off);
  off += num_segs * sizeof(uint32_t);
  auto* pin_per_seg_inline_null = reinterpret_cast<uint8_t*>(staging + off);

  RMM_CUDA_TRY(cudaMemcpyAsync(pin_decoders,
                               d_decoders_buf.data(),
                               num_segs * sizeof(fsst_decoder_compact),
                               cudaMemcpyDeviceToHost,
                               stream.value()));
  RMM_CUDA_TRY(cudaMemcpyAsync(pin_byte_off,
                               d_byte_off_buf.data(),
                               total_dict_entries * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost,
                               stream.value()));
  RMM_CUDA_TRY(cudaMemcpyAsync(pin_dec_off,
                               d_dec_off_buf.data(),
                               total_dict_entries * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost,
                               stream.value()));
  RMM_CUDA_TRY(cudaMemcpyAsync(pin_per_seg_total,
                               d_per_seg_total_buf.data(),
                               num_segs * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost,
                               stream.value()));
  RMM_CUDA_TRY(cudaMemcpyAsync(pin_per_seg_inline_null,
                               d_per_seg_inline_null_buf.data(),
                               num_segs * sizeof(uint8_t),
                               cudaMemcpyDeviceToHost,
                               stream.value()));
  RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  std::memcpy(out.decoders.data(), pin_decoders, num_segs * sizeof(fsst_decoder_compact));
  std::memcpy(out.byte_offsets.data(), pin_byte_off, total_dict_entries * sizeof(uint32_t));
  std::memcpy(out.decoded_offsets.data(), pin_dec_off, total_dict_entries * sizeof(uint32_t));
  std::memcpy(per_seg_total.data(), pin_per_seg_total, num_segs * sizeof(uint32_t));
  std::memcpy(per_seg_inline_null.data(), pin_per_seg_inline_null, num_segs * sizeof(uint8_t));

  // Phase 5: build dict_fsst_desc[] on host. predecode_seg_offset is a
  // cross-segment cumulative scan over per_seg_total (mode-1 segs only).
  uint32_t predecode_cursor = 0;
  for (uint32_t i = 0; i < num_segs; ++i) {
    auto const& seg = run.segments[i];
    if (seg.row_count == 0) continue;
    auto const& p = pre[i];
    if (!p.valid) {
      out.descs.push_back(make_stub_dict_fsst_desc(seg));
      continue;
    }
    auto const& hdr        = headers[i];
    uint32_t predecode_off = 0;
    if (p.mode == DICT_FSST_MODE_DICT_FSST) {
      predecode_off = predecode_cursor;
      predecode_cursor += per_seg_total[i];
    }
    out.descs.push_back(
      {seg.d_bytes,
       seg.bytes_size,
       seg.row_count,
       seg.row_offset,
       seg.seg_row_start,
       p.off_dict,
       (p.mode == DICT_FSST_MODE_FSST_ONLY)
         ? 0u
         : align_up8(p.off_slens + (p.dict_count * p.string_lengths_width + 7u) / 8u),
       p.base_off,
       i,  // seg_decoder_idx — 1:1 with seg_idx in the new layout
       p.dict_count,
       predecode_off,
       hdr.dictionary_indices_width,
       p.mode,
       {0, 0, 0, 0, 0, 0}});
    if (per_seg_inline_null[i]) out.any_inline_nulls = true;
  }
  out.total_predecode_bytes = predecode_cursor;

  return out;
}

//===----------------------------------------------------------------------===//
// 6. Orchestrator
//
// `gpu_decode_strings_column` aggregates per-codec `prepared_*` state across
// all runs in a column, uploads descriptors, dispatches the codec kernels in
// the correct order (lengths → prefix sum → gather), and assembles the cudf
// strings column.
//===----------------------------------------------------------------------===//

/// Sibling to `dispatch_validity_run` in gpu_native_decode.cu.
void overlay_validity_run(gpu_codec_run const& run, uint8_t* d_mask, rmm::cuda_stream_view stream)
{
  if (run.codec != duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
    throw std::runtime_error(
      "gpu_decode_strings_column: viability invariant violated — "
      "validity codec " +
      std::to_string(static_cast<int>(run.codec)) + " not implemented");
  }
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    if (seg.row_offset % 8 != 0) {
      throw std::runtime_error("gpu_decode_strings_column: validity row_offset (" +
                               std::to_string(seg.row_offset) + ") not byte-aligned");
    }
    auto const bytes  = ::cuda::ceil_div(seg.row_count, 8);
    auto const offset = seg.row_offset / 8;
    if (seg.bytes_size < bytes) {
      throw std::runtime_error("gpu_decode_strings_column: validity segment bytes_size (" +
                               std::to_string(seg.bytes_size) + ") < required " +
                               std::to_string(bytes));
    }
    RMM_CUDA_TRY(cudaMemcpyAsync(
      d_mask + offset, seg.d_bytes, bytes, cudaMemcpyDeviceToDevice, stream.value()));
  }
}

}  // namespace

std::unique_ptr<cudf::column> gpu_decode_strings_column(gpu_string_column_decode_input const& col,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  uint32_t const total_rows = col.total_rows;
  if (total_rows == 0) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}); }
  if (total_rows > static_cast<uint32_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::runtime_error("gpu_decode_strings_column: total_rows (" +
                             std::to_string(total_rows) + ") > cudf::size_type max");
  }

  prepared_uncomp prep_uncomp;
  prepared_dict prep_dict;
  prepared_fsst prep_fsst;
  prepared_dict_fsst prep_dict_fsst;
  prep_dict_fsst.any_inline_nulls      = false;
  prep_dict_fsst.total_predecode_bytes = 0;
  size_t cum_chars_upper               = 0;
  bool needs_exact_total               = false;
  for (auto const& run : col.data) {
    switch (run.codec) {
      case duckdb::CompressionType::COMPRESSION_DICTIONARY: {
        auto p = prepare_dict(run);
        prep_dict.descs_short.insert(
          prep_dict.descs_short.end(), p.descs_short.begin(), p.descs_short.end());
        prep_dict.descs_long.insert(
          prep_dict.descs_long.end(), p.descs_long.begin(), p.descs_long.end());
        break;
      }
      case duckdb::CompressionType::COMPRESSION_FSST: {
        auto p = prepare_fsst(run);
        // Rebase row_starts + decoder indices into the merged FSST set.
        auto const row_count_base     = prep_fsst.total_fsst_row_count;
        auto const decoder_count_base = static_cast<uint32_t>(prep_fsst.decoders.size());
        for (auto& s : p.row_starts) {
          s += row_count_base;
        }
        for (auto& c : p.gather_chunks) {
          c.fsst_row_start += row_count_base;
          c.seg_decoder_idx += decoder_count_base;
        }
        prep_fsst.length_descs.insert(
          prep_fsst.length_descs.end(), p.length_descs.begin(), p.length_descs.end());
        prep_fsst.row_starts.insert(
          prep_fsst.row_starts.end(), p.row_starts.begin(), p.row_starts.end());
        prep_fsst.decoders.insert(prep_fsst.decoders.end(), p.decoders.begin(), p.decoders.end());
        prep_fsst.gather_chunks.insert(
          prep_fsst.gather_chunks.end(), p.gather_chunks.begin(), p.gather_chunks.end());
        prep_fsst.total_fsst_row_count += p.total_fsst_row_count;
        break;
      }
      case duckdb::CompressionType::COMPRESSION_DICT_FSST: {
        auto p                    = prepare_dict_fsst(run, stream, mr);
        auto const bo_base        = static_cast<uint32_t>(prep_dict_fsst.byte_offsets.size());
        auto const dec_base       = static_cast<uint32_t>(prep_dict_fsst.decoders.size());
        auto const predecode_base = prep_dict_fsst.total_predecode_bytes;
        for (auto& d : p.descs) {
          d.seg_dict_offset_base += bo_base;
          d.seg_decoder_idx += dec_base;
          if (d.mode == DICT_FSST_MODE_DICT_FSST) { d.predecode_seg_offset += predecode_base; }
        }
        prep_dict_fsst.byte_offsets.insert(
          prep_dict_fsst.byte_offsets.end(), p.byte_offsets.begin(), p.byte_offsets.end());
        prep_dict_fsst.decoded_offsets.insert(
          prep_dict_fsst.decoded_offsets.end(), p.decoded_offsets.begin(), p.decoded_offsets.end());
        prep_dict_fsst.decoders.insert(
          prep_dict_fsst.decoders.end(), p.decoders.begin(), p.decoders.end());
        prep_dict_fsst.descs.insert(prep_dict_fsst.descs.end(), p.descs.begin(), p.descs.end());
        prep_dict_fsst.any_inline_nulls = prep_dict_fsst.any_inline_nulls || p.any_inline_nulls;
        prep_dict_fsst.total_predecode_bytes += p.total_predecode_bytes;
        break;
      }
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: {
        auto p = prepare_uncomp(run);
        prep_uncomp.descs.insert(prep_uncomp.descs.end(), p.descs.begin(), p.descs.end());
        break;
      }
      default:
        throw std::runtime_error(
          "gpu_decode_strings_column: viability invariant violated — "
          "data codec " +
          std::to_string(static_cast<int>(run.codec)) + " not implemented");
    }
    // Upper-bound from walker stats; 0 means unknown → take the sync path.
    for (auto const& seg : run.segments) {
      if (seg.max_string_length == 0u) {
        needs_exact_total = true;
        continue;
      }
      cum_chars_upper += size_t{seg.row_count} * seg.max_string_length;
    }
  }

  // Allocate output and intermediate buffers.
  rmm::device_uvector<uint32_t> d_lengths(size_t{total_rows} + 1, stream, mr);
  rmm::device_uvector<int32_t> d_offsets(size_t{total_rows} + 1, stream, mr);
  rmm::device_buffer d_comp_offsets(prep_fsst.total_fsst_row_count * sizeof(uint32_t), stream, mr);

  // Per-row kernels take chunked descriptors; predecode + mark_nulls stay
  // per-segment via prep_dict_fsst.descs.
  auto const target_ctas       = get_target_ctas();
  auto const uncomp_chunks     = expand_chunks(prep_uncomp.descs, target_ctas);
  auto const dict_chunks_short = expand_chunks(prep_dict.descs_short, target_ctas);
  auto const dict_chunks_long  = expand_chunks(prep_dict.descs_long, target_ctas);
  auto const dict_fsst_chunks  = expand_chunks(prep_dict_fsst.descs, target_ctas);

  auto upload = [&](void const* src, size_t bytes) {
    rmm::device_buffer buf(bytes, stream, mr);
    if (bytes > 0) {
      RMM_CUDA_TRY(cudaMemcpyAsync(buf.data(), src, bytes, cudaMemcpyHostToDevice, stream.value()));
    }
    return buf;
  };
  rmm::device_buffer d_uncomp_chunks_buf =
    upload(uncomp_chunks.data(), uncomp_chunks.size() * sizeof(string_chunk_desc));
  rmm::device_buffer d_dict_short_buf =
    upload(dict_chunks_short.data(), dict_chunks_short.size() * sizeof(string_chunk_desc));
  rmm::device_buffer d_dict_long_buf =
    upload(dict_chunks_long.data(), dict_chunks_long.size() * sizeof(string_chunk_desc));
  rmm::device_buffer d_dict_fsst_chunks_buf =
    upload(dict_fsst_chunks.data(), dict_fsst_chunks.size() * sizeof(dict_fsst_desc));
  rmm::device_buffer d_fsst_lengths_buf = upload(
    prep_fsst.length_descs.data(), prep_fsst.length_descs.size() * sizeof(string_chunk_desc));
  rmm::device_buffer d_fsst_chunks_buf = upload(
    prep_fsst.gather_chunks.data(), prep_fsst.gather_chunks.size() * sizeof(fsst_chunk_desc));
  rmm::device_buffer d_fsst_starts_buf =
    upload(prep_fsst.row_starts.data(), prep_fsst.row_starts.size() * sizeof(uint32_t));
  rmm::device_buffer d_fsst_decoders_buf =
    upload(prep_fsst.decoders.data(), prep_fsst.decoders.size() * sizeof(fsst_decoder_compact));
  rmm::device_buffer d_dict_fsst_descs_buf =
    upload(prep_dict_fsst.descs.data(), prep_dict_fsst.descs.size() * sizeof(dict_fsst_desc));
  rmm::device_buffer d_dict_fsst_decoders_buf = upload(
    prep_dict_fsst.decoders.data(), prep_dict_fsst.decoders.size() * sizeof(fsst_decoder_compact));
  rmm::device_buffer d_byte_offsets_buf = upload(
    prep_dict_fsst.byte_offsets.data(), prep_dict_fsst.byte_offsets.size() * sizeof(uint32_t));
  rmm::device_buffer d_decoded_offsets_buf =
    upload(prep_dict_fsst.decoded_offsets.data(),
           prep_dict_fsst.decoded_offsets.size() * sizeof(uint32_t));

  // Pageable host sources — sync before kernels consume to avoid free-mid-copy.
  RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  auto* d_comp_offsets_p     = static_cast<uint32_t*>(d_comp_offsets.data());
  auto* d_uncomp_chunks_p    = static_cast<string_chunk_desc*>(d_uncomp_chunks_buf.data());
  auto* d_dict_short_p       = static_cast<string_chunk_desc*>(d_dict_short_buf.data());
  auto* d_dict_long_p        = static_cast<string_chunk_desc*>(d_dict_long_buf.data());
  auto* d_fsst_lengths_p     = static_cast<string_chunk_desc*>(d_fsst_lengths_buf.data());
  auto* d_fsst_chunks_p      = static_cast<fsst_chunk_desc*>(d_fsst_chunks_buf.data());
  auto* d_fsst_starts_p      = static_cast<uint32_t*>(d_fsst_starts_buf.data());
  auto* d_fsst_decs_p        = static_cast<fsst_decoder_compact*>(d_fsst_decoders_buf.data());
  auto* d_dict_fsst_p        = static_cast<dict_fsst_desc*>(d_dict_fsst_descs_buf.data());
  auto* d_dict_fsst_chunks_p = static_cast<dict_fsst_desc*>(d_dict_fsst_chunks_buf.data());
  auto* d_dict_fsst_decs_p   = static_cast<fsst_decoder_compact*>(d_dict_fsst_decoders_buf.data());
  auto* d_byte_off_p         = static_cast<uint32_t*>(d_byte_offsets_buf.data());
  auto* d_decoded_off_p      = static_cast<uint32_t*>(d_decoded_offsets_buf.data());

  // Pass 1: lengths. Same kernel for short/long DICTIONARY — only gather forks.
  launch_uncomp_lengths(
    d_uncomp_chunks_p, d_lengths.data(), static_cast<uint32_t>(uncomp_chunks.size()), stream);
  launch_dict_lengths(
    d_dict_short_p, d_lengths.data(), static_cast<uint32_t>(dict_chunks_short.size()), stream);
  launch_dict_lengths(
    d_dict_long_p, d_lengths.data(), static_cast<uint32_t>(dict_chunks_long.size()), stream);
  if (!prep_fsst.length_descs.empty()) {
    // On-device symbol-table parse: one CTA per segment, coalesced symtab
    // load into shmem then serial host-style parse. Replaces the per-segment
    // sync D2H + duckdb_fsst_import that used to happen in prepare_fsst.
    auto const n_decoders = static_cast<uint32_t>(prep_fsst.length_descs.size());
    kernel_build_fsst_decoders<<<n_decoders, BLOCK_DIM, 0, stream.value()>>>(
      d_fsst_lengths_p, n_decoders, d_fsst_decs_p);

    // A+B per-segment (prefix-sum state lives in one CTA); C per-chunk.
    kernel_compute_compressed_offsets_fsst<<<static_cast<uint32_t>(prep_fsst.length_descs.size()),
                                             BLOCK_DIM,
                                             0,
                                             stream.value()>>>(
      d_comp_offsets_p,
      d_fsst_lengths_p,
      d_fsst_starts_p,
      static_cast<uint32_t>(prep_fsst.length_descs.size()));
    kernel_compute_decompressed_lengths_fsst<<<static_cast<uint32_t>(
                                                 prep_fsst.gather_chunks.size()),
                                               BLOCK_DIM,
                                               0,
                                               stream.value()>>>(
      d_fsst_chunks_p,
      d_lengths.data(),
      d_comp_offsets_p,
      d_fsst_decs_p,
      static_cast<uint32_t>(prep_fsst.gather_chunks.size()));
  }
  // Predecode buffer holds decoded dict bytes for mode-1 segments.
  rmm::device_buffer d_predecode_buf(
    prep_dict_fsst.total_predecode_bytes > 0 ? prep_dict_fsst.total_predecode_bytes : 1u,
    stream,
    mr);
  auto* d_predecode_p = static_cast<uint8_t*>(d_predecode_buf.data());

  if (!prep_dict_fsst.descs.empty()) {
    // Lengths chunk for SM-fill; predecode stays per-segment (one decode/dict).
    kernel_compute_lengths_dict_fsst<<<static_cast<uint32_t>(dict_fsst_chunks.size()),
                                       BLOCK_DIM,
                                       0,
                                       stream.value()>>>(
      d_dict_fsst_chunks_p,
      d_lengths.data(),
      d_decoded_off_p,
      static_cast<uint32_t>(dict_fsst_chunks.size()));
    if (prep_dict_fsst.total_predecode_bytes > 0) {
      kernel_predecode_dict_fsst<<<static_cast<uint32_t>(prep_dict_fsst.descs.size()),
                                   BLOCK_DIM,
                                   0,
                                   stream.value()>>>(
        d_dict_fsst_p,
        d_byte_off_p,
        d_decoded_off_p,
        d_dict_fsst_decs_p,
        d_predecode_p,
        static_cast<uint32_t>(prep_dict_fsst.descs.size()));
    }
  }

  // Prefix-sum lengths → byte offsets per row.
  size_t cub_bytes  = 0;
  auto const scan_n = static_cast<int>(total_rows) + 1;
  cub::DeviceScan::ExclusiveSum(nullptr,
                                cub_bytes,
                                d_lengths.data(),
                                reinterpret_cast<uint32_t*>(d_offsets.data()),
                                scan_n,
                                stream.value());
  rmm::device_buffer cub_temp_buf(cub_bytes, stream, mr);
  cub::DeviceScan::ExclusiveSum(cub_temp_buf.data(),
                                cub_bytes,
                                d_lengths.data(),
                                reinterpret_cast<uint32_t*>(d_offsets.data()),
                                scan_n,
                                stream.value());

  // cudf strings offsets are int32; reject up front if the upper bound exceeds it.
  constexpr auto INT32_MAX_SIZE = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (!needs_exact_total && cum_chars_upper > INT32_MAX_SIZE) {
    throw std::runtime_error("gpu_decode_strings_column: estimated total_chars (" +
                             std::to_string(cum_chars_upper) + ") exceeds int32 max");
  }
  size_t alloc_chars = 0;
  if (!needs_exact_total && cum_chars_upper <= HOST_UPPER_BOUND_LIMIT) {
    alloc_chars = cum_chars_upper;
  } else {
    RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    uint32_t total_chars_u = 0;
    RMM_CUDA_TRY(cudaMemcpy(
      &total_chars_u, d_offsets.data() + total_rows, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (total_chars_u > static_cast<uint32_t>(INT32_MAX_SIZE)) {
      throw std::runtime_error("gpu_decode_strings_column: total_chars (" +
                               std::to_string(total_chars_u) + ") exceeds int32 max");
    }
    alloc_chars = total_chars_u;
  }

  rmm::device_buffer d_chars(alloc_chars > 0 ? alloc_chars : 1u, stream, mr);
  auto* d_chars_p = static_cast<uint8_t*>(d_chars.data());

  // Pass 2: gather. See DICT_WARP_COOP_MIN_LEN for the partition rationale.
  launch_uncomp_gather(d_uncomp_chunks_p,
                       d_offsets.data(),
                       d_chars_p,
                       static_cast<uint32_t>(uncomp_chunks.size()),
                       stream);
  launch_dict_gather_short(d_dict_short_p,
                           d_offsets.data(),
                           d_chars_p,
                           static_cast<uint32_t>(dict_chunks_short.size()),
                           stream);
  launch_dict_gather_long(d_dict_long_p,
                          d_offsets.data(),
                          d_chars_p,
                          static_cast<uint32_t>(dict_chunks_long.size()),
                          stream);
  if (!prep_fsst.gather_chunks.empty()) {
    kernel_gather_fsst_chunked<<<static_cast<uint32_t>(prep_fsst.gather_chunks.size()),
                                 BLOCK_DIM,
                                 0,
                                 stream.value()>>>(
      d_fsst_chunks_p,
      d_offsets.data(),
      d_chars_p,
      d_comp_offsets_p,
      d_fsst_decs_p,
      static_cast<uint32_t>(prep_fsst.gather_chunks.size()));
  }
  if (!prep_dict_fsst.descs.empty()) {
    kernel_gather_dict_fsst<<<static_cast<uint32_t>(dict_fsst_chunks.size()),
                              BLOCK_DIM,
                              0,
                              stream.value()>>>(d_dict_fsst_chunks_p,
                                                d_offsets.data(),
                                                d_chars_p,
                                                d_byte_off_p,
                                                d_decoded_off_p,
                                                d_predecode_p,
                                                d_dict_fsst_decs_p,
                                                static_cast<uint32_t>(dict_fsst_chunks.size()));
  }

  // All-valid → overlay UNCOMPRESSED validity → fold in DICT_FSST inline NULLs.
  rmm::device_buffer null_mask{};
  cudf::size_type null_count = 0;
  bool need_mask             = col.has_nulls || prep_dict_fsst.any_inline_nulls;
  if (need_mask) {
    null_mask = cudf::create_null_mask(
      static_cast<cudf::size_type>(total_rows), cudf::mask_state::ALL_VALID, stream, mr);
    for (auto const& run : col.validity) {
      overlay_validity_run(run, static_cast<uint8_t*>(null_mask.data()), stream);
    }
    if (prep_dict_fsst.any_inline_nulls && !prep_dict_fsst.descs.empty()) {
      kernel_dict_fsst_mark_nulls<<<static_cast<uint32_t>(prep_dict_fsst.descs.size()),
                                    BLOCK_DIM,
                                    0,
                                    stream.value()>>>(
        d_dict_fsst_p,
        static_cast<uint8_t*>(null_mask.data()),
        static_cast<uint32_t>(prep_dict_fsst.descs.size()));
    }
    null_count = cudf::null_count(static_cast<cudf::bitmask_type const*>(null_mask.data()),
                                  0,
                                  static_cast<cudf::size_type>(total_rows),
                                  stream);
  }

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    static_cast<cudf::size_type>(total_rows + 1u),
                                                    d_offsets.release(),
                                                    rmm::device_buffer{0, stream, mr},
                                                    0);

  RMM_CUDA_TRY(cudaPeekAtLastError());
  return cudf::make_strings_column(static_cast<cudf::size_type>(total_rows),
                                   std::move(offsets_col),
                                   std::move(d_chars),
                                   null_count,
                                   std::move(null_mask));
}

}  // namespace sirius::cuda::scan
