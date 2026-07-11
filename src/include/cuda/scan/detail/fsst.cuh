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
//! On-device FSST decode core, shared by the FSST codec and DICT_FSST
//! (mode 1 dict predecode, mode 2 inline decompress): the symbol-table import,
//! the warp length-walk, and the warp decode-and-emit. The decode scratch
//! tuning constants live here too, next to the code that sizes against them.

#pragma once

#include "cuda/scan/detail/byte_copy.cuh"
#include "cuda/scan/strings/common.cuh"

#include <cub/config.cuh>
#include <cub/util_ptx.cuh>
#include <cuda/std/algorithm>

#include <cstdint>
#include <cstring>

namespace sirius::cuda::scan::detail {

//===----------------------------------------------------------------------===//
// FSST decode scratch tuning
//===----------------------------------------------------------------------===//
constexpr uint32_t FSST_SCRATCH_U32_PER_WARP   = 256;
constexpr uint32_t FSST_SCRATCH_BYTES_PER_WARP = FSST_SCRATCH_U32_PER_WARP * sizeof(uint32_t);
constexpr uint32_t FSST_MAX_CHUNK_EMIT =
  cub::detail::warp_threads * 8;  ///< worst-case bytes emitted per 32B input chunk (8B max symbol)
constexpr uint32_t FSST_WARPS_PER_CTA = STRINGS_BLOCK_DIM / cub::detail::warp_threads;

//! @brief Import an FSST symbol table from its on-disk blob into @p out.
//!
//! Device port of DuckDB's duckdb_fsst_import. Blob layout: 8B version | 1B
//! zeroTerminated | 8B lenHisto | packed symbols by length group 1..8,1.
_CCCL_DEVICE _CCCL_FORCEINLINE bool device_fsst_import(uint8_t const* buf,
                                                       fsst_decoder_compact* out)
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

//! @brief Decompressed byte length of one row's `comp_len` FSST-compressed bytes,
//! computed cooperatively by a warp. @p sm_len is the symbol length table.
_CCCL_DEVICE _CCCL_FORCEINLINE uint32_t
warp_compute_decomp_len(uint8_t const* __restrict__ comp_ptr,
                        uint32_t comp_len,
                        uint8_t const* __restrict__ sm_len,
                        uint32_t lane)
{
  uint32_t total             = 0;
  int prev_chunk_last_is_esc = 0;
  for (uint32_t off = 0; off < comp_len; off += cub::detail::warp_threads) {
    auto const bytes_in_chunk =
      ::cuda::std::min(comp_len - off, uint32_t{cub::detail::warp_threads});
    auto const is_active  = lane < bytes_in_chunk;
    uint8_t const my_byte = is_active ? comp_ptr[off + lane] : 0;
    int const is_esc      = is_active && my_byte == FSST_ESC ? 1 : 0;

    auto const neighbor_is_esc = cub::ShuffleUp<cub::detail::warp_threads>(is_esc, 1, 0, FULL_MASK);
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

#pragma unroll
    for (uint32_t s = cub::detail::warp_threads / 2; s > 0; s /= 2) {
      // HIP's __shfl_xor_sync requires a 64-bit mask (static_assert on ROCm);
      // CUDA accepts both 32-bit and 64-bit. Cast to unsigned long long for
      // portability — FULL_MASK (unsigned int) widens cleanly on both backends.
      my_len += __shfl_xor_sync(static_cast<unsigned long long>(FULL_MASK), my_len, s);
    }
    total += my_len;

    // Broadcast whether the last byte in this 32B stripe was an escape for the next stripe.
    auto const last_active_lane = bytes_in_chunk - 1;
    prev_chunk_last_is_esc =
      cub::ShuffleIndex<cub::detail::warp_threads>(is_esc, last_active_lane, FULL_MASK);
  }
  return total;
}

//! @brief Decode `comp_len` FSST-compressed bytes with one warp, in 32-byte chunks.
//!
//! Decoded symbols accumulate in a per-warp scratch slab and flush to `dst` when
//! the next chunk's worst-case emit would overflow it, then once at end of row.
//! The symbol table is split lo/hi; `sm_sym_hi` is read only when len[code] > 4.
_CCCL_DEVICE _CCCL_FORCEINLINE void warp_decode_fsst(uint8_t const* __restrict__ comp_ptr,
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

  for (uint32_t off = 0; off < comp_len; off += cub::detail::warp_threads) {
    auto const bytes_in_chunk =
      ::cuda::std::min(comp_len - off, uint32_t{cub::detail::warp_threads});
    auto const active       = lane < bytes_in_chunk;
    uint8_t const my_byte   = active ? comp_ptr[off + lane] : 0;
    int const is_esc        = (active && my_byte == FSST_ESC) ? 1 : 0;
    auto const neighbor_esc = cub::ShuffleUp<cub::detail::warp_threads>(is_esc, 1, 0, FULL_MASK);
    auto const prev_was_esc = (lane == 0) ? prev_chunk_last_is_esc : neighbor_esc;

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
    for (uint32_t step = 1; step < cub::detail::warp_threads; step <<= 1) {
      auto const add = cub::ShuffleUp<cub::detail::warp_threads>(scan, step, 0, FULL_MASK);
      if (lane >= step) { scan += add; }
    }
    auto const exclusive = scan - my_len;
    auto const base      = scratch_used + exclusive;
    auto const chunk_total =
      cub::ShuffleIndex<cub::detail::warp_threads>(scan, cub::detail::warp_threads - 1, FULL_MASK);

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
    prev_chunk_last_is_esc =
      cub::ShuffleIndex<cub::detail::warp_threads>(is_esc, last_active_lane, FULL_MASK);

    // Lazy flush: only when the next chunk might overflow scratch.
    auto const more_chunks = (off + cub::detail::warp_threads) < comp_len;
    if (more_chunks && scratch_used + FSST_MAX_CHUNK_EMIT > FSST_SCRATCH_BYTES_PER_WARP) {
      __syncwarp();
      warp_copy_bytes(dst + cumulative_out_offset, warp_scratch_u8, scratch_used, lane);
      __syncwarp();
      cumulative_out_offset += scratch_used;
      scratch_used = 0;
    }
  }

  // Final flush at end of row — always required, scratch holds the tail.
  if (scratch_used > 0) {
    __syncwarp();
    warp_copy_bytes(dst + cumulative_out_offset, warp_scratch_u8, scratch_used, lane);
    __syncwarp();
  }
}

}  // namespace sirius::cuda::scan::detail
