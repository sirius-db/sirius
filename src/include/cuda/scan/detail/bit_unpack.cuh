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
//! Bit-unpacking primitive shared by every fixed-bit-width decode path
//! (BITPACKING, ALP/ALPRD, DICTIONARY selection buffers, FSST length streams).

#pragma once

#include <cub/config.cuh>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

namespace sirius::cuda::scan::detail {

//! @brief Read one @p width -bit value at logical index @p idx from an LSB-first packed stream.
//!
//! Values are packed contiguously, least-significant-bit first, into a stream of 32-bit words:
//!
//!   word:   [........ word_idx ........][...... word_idx+1 ......]
//!   bits:    0      bit_off          31  32                     63
//!            |<--- value at idx --->|  (may straddle the 32-bit boundary)
//!
//! A value spans up to two 32-bit words for `sizeof(T) <= 4`, and up to three words for 64-bit `T`
//! when it straddles both a 32-bit and the 64-bit boundary (`bit_off > 0 && bit_off + width > 64`).
//!
//! @tparam T  Result type; one of `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`.
//! @param packed  Pointer to the packed words (shared or global). Must be 4-byte aligned and padded
//!                so the highest read (`packed[word_idx + 2]` for 64-bit `T`, else `packed[word_idx
//!                + 1]`) stays in bounds for the largest @p idx — one guard word past the live data
//!                suffices.
//! @param idx     Logical value index.
//! @param width   Bit width per value, in `[0, 64]`; `width == 0` returns 0.
//! @return The unpacked value, zero-extended into @p T.
template <typename T>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T unpack_value(const uint32_t* packed,
                                                            int idx,
                                                            int width)
{
  int constexpr WORD_BITS  = ::cuda::std::numeric_limits<uint32_t>::digits;
  int constexpr WORD_BYTES = sizeof(uint32_t);
  int constexpr DWORD_BITS = ::cuda::std::numeric_limits<uint64_t>::digits;
  if (width == 0) { return T(0); }

  auto const bit_pos  = idx * width;
  auto const word_idx = bit_pos / WORD_BITS;
  auto const bit_off  = bit_pos % WORD_BITS;

  // Accumulate in 64 bits; OR in the next word when the value crosses the 32-bit boundary.
  auto result = static_cast<uint64_t>(packed[word_idx]);
  if (bit_off + width > WORD_BITS) {
    result |= static_cast<uint64_t>(packed[word_idx + 1]) << WORD_BITS;
  }
  result >>= bit_off;

  // A 64-bit value with a non-zero bit offset can need a third word past the 64-bit boundary.
  if constexpr (sizeof(T) > WORD_BYTES) {
    if (bit_off > 0 && bit_off + width > DWORD_BITS) {
      result |= static_cast<uint64_t>(packed[word_idx + 2]) << (DWORD_BITS - bit_off);
    }
  }

  if (width < DWORD_BITS) {
    auto const mask = (1ULL << width) - 1;
    return static_cast<T>(result & mask);
  } else {
    return static_cast<T>(result);
  }
}

}  // namespace sirius::cuda::scan::detail
