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

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cub/config.cuh>
#include <cuda/__memory/aligned_size.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>

namespace sirius::cuda::scan::detail {

//! @brief Stage @p n_live_words 32-bit words of a packed bitstream from @p src_bytes into the
//! shared @p dst_words buffer, then zero @p guard_words trailing words so the bit-unpacker's
//! cross-word reads stay in bounds.
//!
//! Block-wide collective (every thread must call it). @p src_bytes may be arbitrarily aligned; the
//! bytes land at @p dst_words offset 0, so reading them back as words via `unpack_value` is correct
//! regardless of source alignment.
//!
//! @tparam BlockThreads  Block width (also the grid stride of the manual copy path).
//! @param dst_words    Shared buffer of at least `n_live_words + guard_words` words (>= 4B
//! aligned).
//! @param src_bytes    Global source holding at least `n_live_words * 4` readable bytes.
//! @param n_live_words Number of live 32-bit words to copy.
//! @param guard_words  Trailing words to zero (1 for a <= 32-bit unpack target, 2 if a 64-bit value
//!                     can straddle into a third word).
template <int BlockThreads>
_CCCL_DEVICE _CCCL_FORCEINLINE void stage_packed_to_shmem(uint32_t* dst_words,
                                                          const uint8_t* src_bytes,
                                                          int n_live_words,
                                                          int guard_words)
{
  namespace cg          = cooperative_groups;
  auto const block      = cg::this_thread_block();
  auto* const dst_bytes = reinterpret_cast<uint8_t*>(dst_words);
  auto const n_bytes    = static_cast<::cuda::std::size_t>(n_live_words) * sizeof(uint32_t);
  auto const low        = reinterpret_cast<::cuda::std::uintptr_t>(src_bytes) |
                   reinterpret_cast<::cuda::std::uintptr_t>(dst_bytes) | n_bytes;

  if ((low & 0x3u) == 0) {
    // Aligned: inform memcpy_async of the widest cp.async transaction the data jointly supports.
    if ((low & 0xFu) == 0) {
      cg::memcpy_async(block, dst_bytes, src_bytes, ::cuda::aligned_size_t<16>{n_bytes});
    } else if ((low & 0x7u) == 0) {
      cg::memcpy_async(block, dst_bytes, src_bytes, ::cuda::aligned_size_t<8>{n_bytes});
    } else {
      cg::memcpy_async(block, dst_bytes, src_bytes, ::cuda::aligned_size_t<4>{n_bytes});
    }
    // Guard words lie past the copied range, so this write never races the async copy.
    for (int w = threadIdx.x; w < guard_words; w += BlockThreads) {
      dst_words[n_live_words + w] = 0;
    }
    cg::wait(block);  // publishes both the copied words and the guard words
  } else {
    // Sub-word alignment: copy one word per thread.
    for (int w = threadIdx.x; w < n_live_words; w += BlockThreads) {
      uint32_t v;
      memcpy(
        &v, src_bytes + static_cast<::cuda::std::size_t>(w) * sizeof(uint32_t), sizeof(uint32_t));
      dst_words[w] = v;
    }
    for (int w = threadIdx.x; w < guard_words; w += BlockThreads) {
      dst_words[n_live_words + w] = 0;
    }
    __syncthreads();  // publishes the manually copied words and the guard words
  }
}

}  // namespace sirius::cuda::scan::detail
