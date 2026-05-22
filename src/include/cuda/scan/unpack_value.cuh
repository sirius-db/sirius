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

#pragma once

#include <cuda/std/numeric>

#include <cstdint>

namespace sirius::cuda::scan {

/// Read one width-bit value from `packed[]` at logical index `idx`. Values
/// are stored LSB-first within 32-bit words.
///
/// For T wider than 32 bits a value can span three 32-bit words when both
/// `bit_off > 0` and `bit_off + width > 64` (e.g. width=50, bit_off=20 reads
/// bits 20..69, which crosses two 32-bit boundaries). Callers must ensure
/// `packed[]` has one guard word past the live data so that third-word read
/// is in-bounds; the kernel writes the guard word itself.
template <typename T>
__device__ __forceinline__ T unpack_value(uint32_t const* packed, uint32_t idx, uint32_t width)
{
  auto constexpr WORD_BITS  = ::cuda::std::numeric_limits<uint32_t>::digits;
  auto constexpr WORD_BYTES = sizeof(uint32_t);
  if (width == 0) return T(0);

  auto const bit_pos  = static_cast<uint64_t>(idx) * width;
  auto const word_idx = static_cast<uint32_t>(bit_pos / WORD_BITS);
  auto const bit_off  = static_cast<uint32_t>(bit_pos % WORD_BITS);

  // We need to upcast in case we need bits from the next word.
  auto result = static_cast<uint64_t>(packed[word_idx]);
  if (bit_off + width > WORD_BITS) {
    result |= static_cast<uint64_t>(packed[word_idx + 1]) << WORD_BITS;
  }
  result >>= bit_off;

  // For 8B types, we may need bits from the second next word.
  if constexpr (sizeof(T) > WORD_BYTES) {
    if (bit_off > 0 && bit_off + width > 2 * WORD_BITS) {
      result |= static_cast<uint64_t>(packed[word_idx + 2]) << (64 - bit_off);
    }
  }

  auto const mask = (width >= 64) ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
  return static_cast<T>(result & mask);
}

}  // namespace sirius::cuda::scan
