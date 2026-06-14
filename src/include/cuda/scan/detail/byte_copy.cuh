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
//! Warp-cooperative byte copy shared by the string gather kernels (DICTIONARY,
//! FSST, DICT_FSST): move a row's or dictionary entry's bytes from the source
//! (dictionary bytes or a decode scratch slab) into the output chars buffer.

#pragma once

#include <cub/config.cuh>
#include <cuda/scan/detail/warp.cuh>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>

namespace sirius::cuda::scan::detail {

//! @brief One warp copies @p n bytes from @p src to @p dst: lane-strided 4-byte words plus a
//! per-byte tail for the last < 4 bytes.
//!
//! Either pointer may be arbitrarily aligned. When @p src is 4-byte aligned the word reads are
//! direct `uint32_t` loads (the FSST scratch drain depends on this fast path off word-aligned
//! shared memory); otherwise they go through `memcpy`. Stores always go through `memcpy` because
//! the output chars buffer is byte-addressed.
//!
//! @param dst   Output (byte-addressed; any alignment).
//! @param src   Source bytes (any alignment).
//! @param n     Number of bytes to copy.
//! @param lane  Caller's lane id in [0, 32).
_CCCL_DEVICE _CCCL_FORCEINLINE void warp_copy_bytes(uint8_t* dst,
                                                    const uint8_t* src,
                                                    int n,
                                                    int lane)
{
  int constexpr WORD_BYTES = sizeof(uint32_t);
  int const n_full_words   = n / WORD_BYTES;

  if ((reinterpret_cast<::cuda::std::uintptr_t>(src) % WORD_BYTES) == 0) {
    // 4B aligned
    auto const* const src_words = reinterpret_cast<const uint32_t*>(src);
    for (int w = lane; w < n_full_words; w += WARP_THREADS) {
      auto const v = src_words[w];
      ::cuda::std::memcpy(dst + w * WORD_BYTES, &v, WORD_BYTES);
    }
  } else {
    // Unaligned
    for (int w = lane; w < n_full_words; w += WARP_THREADS) {
      ::cuda::std::memcpy(dst + w * WORD_BYTES, src + w * WORD_BYTES, WORD_BYTES);
    }
  }

  for (int t = n_full_words * WORD_BYTES + lane; t < n; t += WARP_THREADS) {
    dst[t] = src[t];
  }
}

}  // namespace sirius::cuda::scan::detail
