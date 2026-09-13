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

//===----------------------------------------------------------------------===//
// DICT_FSST string codec. On-disk segment layout (DuckDB):
//
//   header | dictionary bytes | symbol table | packed lengths | packed indices
//
// Regions are 8-byte aligned. Bitpacked lengths and indices occupy whole
// groups of 32 values. Mode 0 has no symbol table. Offsets follow
// duckdb/src/storage/compression/dict_fsst/decompression.cpp.
//
// Modes (hdr.mode):
//   0 DICTIONARY  — dict bytes raw; gather is memcpy.
//   1 DICT_FSST   — dict bytes FSST-compressed; predecode kernel decompresses
//                   each dict entry once into a per-segment buffer; gather
//                   memcpys from there.
//   2 FSST_ONLY   — all rows unique; row i → dict entry i+1; no dict_indices
//                   region; gather inline-decompresses each row.
//
// Dictionary index 0 encodes NULL. mark_nulls updates the mask even when
// DuckDB omits a separate validity payload.
// The on-device FSST decode core (detail/fsst.cuh) is shared with the FSST codec.
//===----------------------------------------------------------------------===//

#include "cuda/scan/detail/byte_copy.cuh"
#include "cuda/scan/detail/fsst.cuh"
#include "cuda/scan/strings/dict_fsst.cuh"
#include "cuda/scan/unpack_value.cuh"

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>

#include <cub/cub.cuh>
#include <cuda/cmath>
#include <cuda/std/algorithm>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace sirius::cuda::scan {

namespace {

using detail::device_fsst_import;
using detail::FSST_SCRATCH_U32_PER_WARP;
using detail::FSST_WARPS_PER_CTA;
using detail::warp_decode_fsst;

//! On-disk DICT_FSST header.
struct dict_fsst_header_t {
  uint32_t dict_size;
  uint32_t dict_count;  ///< includes reserved idx 0 (NULL, length 0)
  uint8_t mode;         ///< 0=DICTIONARY, 1=DICT_FSST, 2=FSST_ONLY
  uint8_t string_lengths_width;
  uint8_t dictionary_indices_width;
  uint8_t _pad;
  uint32_t symbol_table_size;
};

//! @brief Per-segment scratch input for `kernel_build_dict_fsst_data`. Filled
//! host-side after a single batched header D2H so the kernel has all metadata
//! it needs without re-reading the header on device. `base_off` is the
//! cumulative `(dict_count + 1)` over prior valid segments — it indexes into
//! the global d_byte_offsets / d_decoded_offsets arrays.
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
  uint8_t valid;  ///< 0 for a zero-row segment
  uint8_t _pad;
};
static_assert(sizeof(dict_fsst_pre_desc) % 4 == 0);

struct dict_fsst_layout {
  uint32_t off_dict;
  uint32_t off_symtab;
  uint32_t off_slens;
  uint32_t off_didx;  ///< 0 in FSST_ONLY (no indices region)
};

constexpr char const* DICT_FSST_CODEC_NAME = "DICT_FSST";

//! A segment holds at most one row group of entries plus the NULL slot
//! (DuckDB Storage::MAX_ROW_GROUP_SIZE = 2^30).
constexpr uint64_t DICT_FSST_MAX_DICT_COUNT = (uint64_t{1} << 30) + 1u;

//! Kernels form bit positions and row indices in 32-bit signed arithmetic.
constexpr uint64_t KERNEL_INDEX_LIMIT = std::numeric_limits<int32_t>::max();

constexpr uint64_t align_up8_u64(uint64_t n) { return (n + 7u) & ~uint64_t{7}; }

//! Validate header bounds and row/bit index ranges before decoding. Zero widths are legal: no
//! indices region in FSST_ONLY, every index 0 for a NULL-slot-only dictionary, every entry empty
//! for zero-width lengths. Payloads are not checked here.
dict_fsst_layout validate_dict_fsst_segment(size_t seg_idx,
                                            gpu_string_segment_desc const& seg,
                                            dict_fsst_header_t const& hdr)
{
  auto fail = [&](std::string const& what) {
    throw_malformed_segment(DICT_FSST_CODEC_NAME, seg_idx, seg, what);
  };
  uint64_t const bytes_size = seg.bytes_size;
  if (hdr.mode > DICT_FSST_MODE_FSST_ONLY) {
    fail("mode " + std::to_string(hdr.mode) + " is not one of 0/1/2");
  }
  if (hdr.dict_count == 0) { fail("dict_count 0 (the NULL slot is always present)"); }
  if (hdr.dict_count > DICT_FSST_MAX_DICT_COUNT) {
    fail("dict_count " + std::to_string(hdr.dict_count) + " exceeds a row group");
  }
  if (hdr.string_lengths_width > MAX_BITPACKING_WIDTH) {
    fail("string_lengths_width " + std::to_string(hdr.string_lengths_width) + " > 32");
  }
  if (hdr.dictionary_indices_width > MAX_BITPACKING_WIDTH) {
    fail("dictionary_indices_width " + std::to_string(hdr.dictionary_indices_width) + " > 32");
  }
  bool const has_symtab = hdr.mode != DICT_FSST_MODE_DICTIONARY;
  if (has_symtab && hdr.symbol_table_size > FSST_SYMTAB_MAX_BYTES) {
    fail("symbol_table_size " + std::to_string(hdr.symbol_table_size) + " > " +
         std::to_string(FSST_SYMTAB_MAX_BYTES));
  }
  // Keep this predicate aligned with has_fsst in kernel_build_dict_fsst_data.
  bool const imports_symtab = has_symtab && hdr.dict_count > 1;
  if (imports_symtab && hdr.symbol_table_size < FSST_SYMTAB_HEADER_BYTES) {
    fail("symbol_table_size " + std::to_string(hdr.symbol_table_size) +
         " is shorter than the FSST symbol table header");
  }

  uint64_t const off_dict = align_up8_u64(sizeof(dict_fsst_header_t));
  uint64_t const dict_end = off_dict + hdr.dict_size;
  if (dict_end > bytes_size) { fail("dictionary bytes end past bytes_size"); }
  uint64_t const off_symtab  = align_up8_u64(dict_end);
  uint64_t const symtab_size = has_symtab ? hdr.symbol_table_size : 0u;
  if (has_symtab && off_symtab + symtab_size > bytes_size) {
    fail("symbol table ends past bytes_size");
  }
  uint64_t const off_slens = align_up8_u64(off_symtab + symtab_size);
  uint64_t const slens_end =
    off_slens + bitpacked_region_bytes(hdr.dict_count, hdr.string_lengths_width);
  if (slens_end > bytes_size) { fail("string_lengths region ends past bytes_size"); }

  uint64_t const rows_end = uint64_t{seg.seg_row_start} + seg.row_count;
  if (rows_end > KERNEL_INDEX_LIMIT ||
      uint64_t{hdr.dict_count} * hdr.string_lengths_width > KERNEL_INDEX_LIMIT ||
      rows_end * hdr.dictionary_indices_width > KERNEL_INDEX_LIMIT) {
    fail("row or bit index for rows " + std::to_string(rows_end) +
         " does not fit the kernels' 32-bit index arithmetic");
  }
  uint64_t off_didx = 0;
  if (hdr.mode == DICT_FSST_MODE_FSST_ONLY) {
    if (rows_end + 1u > hdr.dict_count) {
      fail("FSST_ONLY rows " + std::to_string(rows_end) + " exceed dict_count " +
           std::to_string(hdr.dict_count));
    }
  } else {
    off_didx = align_up8_u64(slens_end);
    if (off_didx + bitpacked_region_bytes(rows_end, hdr.dictionary_indices_width) > bytes_size) {
      fail("dictionary_indices region for rows " + std::to_string(rows_end) +
           " ends past bytes_size");
    }
  }
  return {static_cast<uint32_t>(off_dict),
          static_cast<uint32_t>(off_symtab),
          static_cast<uint32_t>(off_slens),
          static_cast<uint32_t>(off_didx)};
}

//! @brief One CTA per segment: parse the symbol table on device, unpack
//! string_lengths into byte_offsets and inclusive-scan it, then for FSST modes
//! walk each dict entry to fill decoded_offsets. Writes per-segment outputs
//! (total decoded bytes + inline-null flag) for the host to aggregate.
__global__ __launch_bounds__(STRINGS_BLOCK_DIM) void kernel_build_dict_fsst_data(
  dict_fsst_pre_desc const* __restrict__ pre,
  int num_segments,
  fsst_decoder_compact* __restrict__ d_decoders,
  uint32_t* __restrict__ d_byte_offsets,
  uint32_t* __restrict__ d_decoded_offsets,
  uint32_t* __restrict__ d_per_seg_decoded_total,
  uint8_t* __restrict__ d_per_seg_inline_null)
{
  using BlockScanT = cub::BlockScan<uint32_t, STRINGS_BLOCK_DIM>;

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
      for (int i = 0; i < FSST_NUM_SYMBOLS; ++i) {
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
    for (int i = threadIdx.x; i < symtab_size; i += blockDim.x) {
      sm_symtab[i] = d.d_bytes[d.off_symtab + i];
    }
    __syncthreads();
    if (threadIdx.x == 0) { device_fsst_import(sm_symtab, &d_decoders[seg_idx]); }
    __syncthreads();
    // Cache sm_len[] for the per-entry walks below.
    for (int i = threadIdx.x; i < FSST_NUM_SYMBOLS; i += blockDim.x) {
      sm_len[i] = d_decoders[seg_idx].len[i];
    }
  } else {
    for (int i = threadIdx.x; i < FSST_NUM_SYMBOLS; i += blockDim.x) {
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
  for (int k = threadIdx.x; k < d.dict_count; k += blockDim.x) {
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
    for (int k = threadIdx.x; k <= d.dict_count; k += blockDim.x) {
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
    for (int k = threadIdx.x + 1; k < d.dict_count; k += blockDim.x) {
      auto const comp_start = my_byte_off[k];
      auto const comp_len   = my_byte_off[k + 1] - comp_start;
      auto const* cp        = dict_bytes_base + comp_start;
      int decomp_len        = 0;
      int pos               = 0;
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
    for (int i = start; i < end; ++i) {
      thread_sum += my_dec_off[i];
      my_dec_off[i] = thread_sum;
    }
    uint32_t exclusive_sum = 0;
    BlockScanT(scan_temp).ExclusiveSum(thread_sum, exclusive_sum);
    if (exclusive_sum > 0) {
      for (int i = start; i < end; ++i) {
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
                                                 int num_segments)
{
  int seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc         = descs[seg_idx];
  uint32_t const* dec_off = d_decoded_offsets + desc.seg_dict_offset_base;
  uint32_t const* d_idx =
    reinterpret_cast<uint32_t const*>(desc.d_bytes + desc.dict_indices_offset);
  bool const fsst_only = (desc.mode == DICT_FSST_MODE_FSST_ONLY);

  for (int i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    int seg_i = desc.seg_row_start + i;
    int idx =
      fsst_only ? (seg_i + 1) : unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
    int len = 0;
    if (idx != 0 && idx < desc.dict_count) { len = dec_off[idx + 1] - dec_off[idx]; }
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
                                           int num_segments)
{
  int seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc = descs[seg_idx];
  if (desc.mode != DICT_FSST_MODE_DICT_FSST) return;
  if (desc.dict_count <= 1u) return;  // only entry 0 = NULL, nothing to decode

  __shared__ uint8_t sm_len[FSST_NUM_SYMBOLS];
  __shared__ unsigned long long sm_sym[FSST_NUM_SYMBOLS];

  fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
  for (int i = threadIdx.x; i < FSST_NUM_SYMBOLS; i += blockDim.x) {
    sm_len[i] = dec.len[i];
    sm_sym[i] = dec.symbol[i];
  }
  __syncthreads();

  uint8_t const* dict_data = desc.d_bytes + desc.dict_data_offset;
  uint32_t const* byte_off = d_byte_offsets + desc.seg_dict_offset_base;
  uint32_t const* dec_off  = d_decoded_offsets + desc.seg_dict_offset_base;
  uint8_t* out_base        = predecode_buf + desc.predecode_seg_offset;

  // Skip k=0 (reserved NULL slot, length 0). One thread per dict entry.
  for (int k = threadIdx.x + 1; k < desc.dict_count; k += blockDim.x) {
    int comp_start = byte_off[k];
    int comp_end   = byte_off[k + 1];
    int comp_len   = comp_end - comp_start;
    int out_pos    = dec_off[k];

    uint8_t const* comp_ptr = dict_data + comp_start;
    uint8_t* out_ptr        = out_base + out_pos;
    int pos                 = 0;
    int op                  = 0;
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

/// Per-row gather for the DICT_FSST codec (see codec banner above). Templated on
/// @p FsstOnly so the mode-2 inline-decode path (symbol table in shared memory +
/// warp_decode_fsst) compiles out of the common DICTIONARY / DICT_FSST
/// instantiation. Each instantiation early-returns on segments outside its mode
/// class, so the host launches both over the same descriptor array.
template <bool FsstOnly>
__global__ void kernel_gather_dict_fsst(dict_fsst_desc const* __restrict__ descs,
                                        int32_t const* __restrict__ d_offsets,
                                        uint8_t* __restrict__ d_chars,
                                        uint32_t const* __restrict__ d_byte_offsets,
                                        uint32_t const* __restrict__ d_decoded_offsets,
                                        uint8_t const* __restrict__ predecode_buf,
                                        fsst_decoder_compact const* __restrict__ d_decoders,
                                        int num_segments)
{
  int seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc = descs[seg_idx];
  // This instantiation handles only its own mode class; the complementary
  // instantiation (launched over the same array) covers the rest.
  if ((desc.mode == DICT_FSST_MODE_FSST_ONLY) != FsstOnly) return;

  uint8_t const* base           = desc.d_bytes;
  uint32_t const* dict_byte_off = d_byte_offsets + desc.seg_dict_offset_base;

  int const lane          = threadIdx.x & (cub::detail::warp_threads - 1u);
  int const warp_id       = threadIdx.x / cub::detail::warp_threads;
  int const warps_per_cta = blockDim.x / cub::detail::warp_threads;

  if constexpr (!FsstOnly) {
    // Modes 0/1: warp-cooperative memcpy. Mode 1 (DICT_FSST) copies the
    // predecoded bytes at decoded offsets; mode 0 (DICTIONARY) copies the raw
    // dict bytes at byte offsets.
    bool const mode_dict_fsst    = (desc.mode == DICT_FSST_MODE_DICT_FSST);
    uint32_t const* dict_dec_off = d_decoded_offsets + desc.seg_dict_offset_base;
    uint32_t const* memcpy_off   = mode_dict_fsst ? dict_dec_off : dict_byte_off;
    uint8_t const* memcpy_src =
      mode_dict_fsst ? (predecode_buf + desc.predecode_seg_offset) : (base + desc.dict_data_offset);
    uint32_t const* d_idx = reinterpret_cast<uint32_t const*>(base + desc.dict_indices_offset);

    for (int i = warp_id; i < desc.row_count; i += warps_per_cta) {
      int seg_i = desc.seg_row_start + i;
      int idx   = unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
      if (idx == 0) continue;  // NULL — pass-1 emitted length 0
      int op          = d_offsets[desc.global_row_start + i];
      int entry_start = memcpy_off[idx];
      int entry_len   = memcpy_off[idx + 1] - entry_start;
      detail::warp_copy_bytes(d_chars + op, memcpy_src + entry_start, entry_len, lane);
    }
  } else {
    // Mode 2 (FSST_ONLY): row i → dict entry i+1 (never NULL); inline-decompress
    // each row through the warp FSST helper, which needs the symbol table staged
    // in shared memory.
    __shared__ uint8_t sm_len[256];
    __shared__ uint32_t sm_sym_lo[256];
    __shared__ uint32_t sm_sym_hi[256];
    __shared__ uint32_t sm_scratch_u32[FSST_WARPS_PER_CTA][FSST_SCRATCH_U32_PER_WARP];

    fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
      sm_len[i]    = (i < FSST_NUM_SYMBOLS) ? dec.len[i] : uint8_t{0};
      uint64_t sym = (i < FSST_NUM_SYMBOLS) ? dec.symbol[i] : 0ull;
      sm_sym_lo[i] = static_cast<uint32_t>(sym);
      sm_sym_hi[i] = static_cast<uint32_t>(sym >> 32);
    }
    __syncthreads();

    uint8_t const* comp_base = base + desc.dict_data_offset;
    for (int i = warp_id; i < desc.row_count; i += warps_per_cta) {
      int seg_i      = desc.seg_row_start + i;
      int idx        = seg_i + 1;
      int byte_start = dict_byte_off[idx];
      int comp_len   = dict_byte_off[idx + 1] - byte_start;
      if (comp_len == 0) continue;
      int op = d_offsets[desc.global_row_start + i];
      warp_decode_fsst(comp_base + byte_start,
                       comp_len,
                       d_chars + op,
                       sm_len,
                       sm_sym_lo,
                       sm_sym_hi,
                       &sm_scratch_u32[warp_id][0],
                       lane);
    }
  }
}

/// Fold index-0 NULLs into the column mask.
__global__ void kernel_dict_fsst_mark_nulls(dict_fsst_desc const* __restrict__ descs,
                                            uint8_t* __restrict__ d_mask,
                                            int num_segments)
{
  int seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc = descs[seg_idx];
  uint32_t const* d_idx =
    reinterpret_cast<uint32_t const*>(desc.d_bytes + desc.dict_indices_offset);

  // FSST_ONLY can't encode NULL via idx==0 (row i → entry i+1 always non-zero).
  if (desc.mode == DICT_FSST_MODE_FSST_ONLY) return;

  auto* d_mask_words = reinterpret_cast<unsigned int*>(d_mask);
  for (int i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    int seg_i = desc.seg_row_start + i;
    int idx   = unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
    if (idx != 0) continue;
    int row = desc.global_row_start + i;
    atomicAnd(d_mask_words + (row >> 5), ~(1u << (row & 31u)));
  }
}

}  // namespace

//! Prepare DICT_FSST state after host validation. Synchronizes for headers and, when rows
//! are present, for device-built offsets, decoders and decoded totals needed by the host.
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

  // Phase 1: batched async D2H of all headers into pinned host memory, one sync.
  static thread_local pinned_host_pool headers_pool;
  auto const* headers =
    fetch_segment_headers<dict_fsst_header_t>(run, headers_pool, DICT_FSST_CODEC_NAME, stream);

  // Phase 2: validate, compute per-segment region offsets + cumulative
  // base_off into the global byte/decoded_offsets arrays.
  std::vector<dict_fsst_pre_desc> pre(num_segs);
  std::vector<uint32_t> indices_offsets(num_segs, 0u);
  // Sum of (dict_count + 1) over non-empty segments: the index base into the
  // flat byte/decoded offset arrays, which the kernels address with uint32.
  uint64_t total_dict_entries = 0;
  for (uint32_t i = 0; i < num_segs; ++i) {
    auto const& seg = run.segments[i];
    auto& p         = pre[i];
    p.valid         = 0;
    p.d_bytes       = seg.d_bytes;
    p.bytes_size    = seg.bytes_size;
    if (seg.row_count == 0) continue;
    auto const& hdr   = headers[i];
    auto const layout = validate_dict_fsst_segment(i, seg, hdr);
    if (total_dict_entries + hdr.dict_count + 1u > std::numeric_limits<uint32_t>::max()) {
      throw_malformed_segment(
        DICT_FSST_CODEC_NAME, i, seg, "cumulative dictionary entries exceed the 32-bit index");
    }

    p.off_dict             = layout.off_dict;
    p.off_symtab           = layout.off_symtab;
    p.off_slens            = layout.off_slens;
    p.dict_count           = hdr.dict_count;
    p.mode                 = hdr.mode;
    p.string_lengths_width = hdr.string_lengths_width;
    p.base_off             = static_cast<uint32_t>(total_dict_entries);
    p.valid                = 1;
    indices_offsets[i]     = layout.off_didx;
    total_dict_entries += hdr.dict_count + 1u;
  }

  if (total_dict_entries == 0) return out;  // every segment is empty

  // Phase 3: launch the on-device prep kernel.
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

  kernel_build_dict_fsst_data<<<num_segs, STRINGS_BLOCK_DIM, 0, stream.value()>>>(
    static_cast<dict_fsst_pre_desc const*>(d_pre_buf.data()),
    num_segs,
    static_cast<fsst_decoder_compact*>(d_decoders_buf.data()),
    static_cast<uint32_t*>(d_byte_off_buf.data()),
    static_cast<uint32_t*>(d_dec_off_buf.data()),
    static_cast<uint32_t*>(d_per_seg_total_buf.data()),
    static_cast<uint8_t*>(d_per_seg_inline_null_buf.data()));

  // Phase 4: pull results back into host vectors via a single batched D2H.
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
    auto const& p          = pre[i];
    auto const& hdr        = headers[i];
    uint32_t predecode_off = 0;
    if (p.mode == DICT_FSST_MODE_DICT_FSST) {
      predecode_off = predecode_cursor;
      predecode_cursor += per_seg_total[i];
    }
    out.descs.push_back({seg.d_bytes,
                         seg.bytes_size,
                         seg.row_count,
                         seg.row_offset,
                         seg.seg_row_start,
                         p.off_dict,
                         indices_offsets[i],
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

void launch_dict_fsst_lengths(dict_fsst_desc const* d_chunks,
                              uint32_t* d_lengths,
                              uint32_t const* d_decoded_offsets,
                              uint32_t n_chunks,
                              rmm::cuda_stream_view stream)
{
  if (n_chunks == 0) return;
  kernel_compute_lengths_dict_fsst<<<n_chunks, STRINGS_BLOCK_DIM, 0, stream.value()>>>(
    d_chunks, d_lengths, d_decoded_offsets, n_chunks);
}

void launch_dict_fsst_predecode(dict_fsst_desc const* d_descs,
                                uint32_t const* d_byte_offsets,
                                uint32_t const* d_decoded_offsets,
                                fsst_decoder_compact const* d_decoders,
                                uint8_t* d_predecode,
                                uint32_t n_segments,
                                uint32_t total_predecode_bytes,
                                rmm::cuda_stream_view stream)
{
  if (n_segments == 0 || total_predecode_bytes == 0) return;
  kernel_predecode_dict_fsst<<<n_segments, STRINGS_BLOCK_DIM, 0, stream.value()>>>(
    d_descs, d_byte_offsets, d_decoded_offsets, d_decoders, d_predecode, n_segments);
}

void launch_dict_fsst_gather(dict_fsst_desc const* d_chunks,
                             int32_t const* d_offsets,
                             uint8_t* d_chars,
                             uint32_t const* d_byte_offsets,
                             uint32_t const* d_decoded_offsets,
                             uint8_t const* d_predecode,
                             fsst_decoder_compact const* d_decoders,
                             uint32_t n_chunks,
                             rmm::cuda_stream_view stream)
{
  if (n_chunks == 0) return;
  // Launch both instantiations over the full descriptor array: FsstOnly=false
  // handles DICTIONARY / DICT_FSST, FsstOnly=true handles FSST_ONLY. Each
  // early-returns on segments outside its mode class.
  kernel_gather_dict_fsst<false>
    <<<n_chunks, STRINGS_BLOCK_DIM, 0, stream.value()>>>(d_chunks,
                                                         d_offsets,
                                                         d_chars,
                                                         d_byte_offsets,
                                                         d_decoded_offsets,
                                                         d_predecode,
                                                         d_decoders,
                                                         n_chunks);
  kernel_gather_dict_fsst<true>
    <<<n_chunks, STRINGS_BLOCK_DIM, 0, stream.value()>>>(d_chunks,
                                                         d_offsets,
                                                         d_chars,
                                                         d_byte_offsets,
                                                         d_decoded_offsets,
                                                         d_predecode,
                                                         d_decoders,
                                                         n_chunks);
}

void launch_dict_fsst_mark_nulls(dict_fsst_desc const* d_descs,
                                 uint8_t* d_null_mask,
                                 uint32_t n_segments,
                                 rmm::cuda_stream_view stream)
{
  if (n_segments == 0) return;
  kernel_dict_fsst_mark_nulls<<<n_segments, STRINGS_BLOCK_DIM, 0, stream.value()>>>(
    d_descs, d_null_mask, n_segments);
}

}  // namespace sirius::cuda::scan
