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

//! @file
//! Global -> shared staging for packed bitstreams. Every codec that unpacks
//! fixed-bit-width values (BITPACKING, ALP/ALPRD) stages the packed bytes into
//! shared memory through this one primitive, then reads them back word-aligned
//! with `unpack_value` (detail/bit_unpack.cuh).

#pragma once

#include <cub/config.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <cuda/__memory/aligned_size.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cstring>

namespace sirius::cuda::scan::detail
{

//! @brief Stage @p n_live_words 32-bit words of a packed bitstream from global @p src_bytes into the
//! shared @p dst_words buffer, then zero @p guard_words trailing words so the bit-unpacker's
//! cross-word reads stay in bounds.
//!
//! Block-wide collective: every thread must call it. @p src_bytes may be arbitrarily aligned; the
//! bytes land in @p dst_words at offset 0 (no shift), so reading them back as words via
//! `unpack_value` is correct regardless of source alignment. The whole-word copy is exact for every
//! caller here: each codec's packed stream length is a FastPFor 32-value-group multiple, hence a
//! multiple of 4 bytes.
//!
//! The copy path is chosen by the joint alignment of `src`, `dst`, and the byte count:
//!   - >= 4-byte aligned: pass `cuda::aligned_size_t<N>` so `cg::memcpy_async` knows the run is
//!     N-byte aligned and uses the 4/8/16-byte `cp.async` engine (SM80+). A plain size would fall
//!     to the 1-byte path (it infers `alignof(uint8_t) == 1`).
//!   - sub-word (e.g. ALPRD's `+2` streams): a manual per-word copy reads 4 bytes per thread, which
//!     beats the 1-byte `cp.async` path — and `cp.async` has no 2-byte specialization to inform.
//!
//! @tparam BlockThreads  Block width (compile-time; also the grid stride of the manual path).
//! @param dst_words    Shared buffer of at least `n_live_words + guard_words` words (>= 4B aligned).
//! @param src_bytes    Global source holding at least `n_live_words * 4` readable bytes.
//! @param n_live_words Number of live 32-bit words to copy.
//! @param guard_words  Trailing words to zero past the live data (1 when the unpack target is
//!                     <= 32-bit, 2 when a 64-bit value can straddle into a third word).
template <int BlockThreads>
_CCCL_DEVICE _CCCL_FORCEINLINE void
stage_packed_to_shmem(uint32_t* dst_words, const uint8_t* src_bytes, uint32_t n_live_words, uint32_t guard_words)
{
  namespace cg          = cooperative_groups;
  const auto block      = cg::this_thread_block();
  auto* const dst_bytes = reinterpret_cast<uint8_t*>(dst_words);
  const auto n_bytes    = static_cast<::cuda::std::size_t>(n_live_words) * sizeof(uint32_t);
  const auto low        = reinterpret_cast<::cuda::std::uintptr_t>(src_bytes)
                 | reinterpret_cast<::cuda::std::uintptr_t>(dst_bytes) | n_bytes;

  if ((low & 0x3u) == 0)
  {
    // Aligned: inform memcpy_async of the widest cp.async transaction the data jointly supports.
    if ((low & 0xFu) == 0)
    {
      cg::memcpy_async(block, dst_bytes, src_bytes, ::cuda::aligned_size_t<16>{n_bytes});
    }
    else if ((low & 0x7u) == 0)
    {
      cg::memcpy_async(block, dst_bytes, src_bytes, ::cuda::aligned_size_t<8>{n_bytes});
    }
    else
    {
      cg::memcpy_async(block, dst_bytes, src_bytes, ::cuda::aligned_size_t<4>{n_bytes});
    }
    // Guard words lie past the copied range, so this write never races the async copy.
    for (uint32_t w = threadIdx.x; w < guard_words; w += BlockThreads)
    {
      dst_words[n_live_words + w] = 0;
    }
    cg::wait(block); // publishes both the copied words and the guard words
  }
  else
  {
    // Sub-word source: a 4-byte memcpy per word beats the 1-byte cp.async fallback.
    for (uint32_t w = threadIdx.x; w < n_live_words; w += BlockThreads)
    {
      uint32_t v;
      ::memcpy(&v, src_bytes + static_cast<::cuda::std::size_t>(w) * sizeof(uint32_t), sizeof(uint32_t));
      dst_words[w] = v;
    }
    for (uint32_t w = threadIdx.x; w < guard_words; w += BlockThreads)
    {
      dst_words[n_live_words + w] = 0;
    }
    __syncthreads(); // publishes the manually copied words and the guard words
  }
}

} // namespace sirius::cuda::scan::detail
