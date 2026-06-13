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
//! Vectorized fill of a contiguous output range, shared by the codecs that
//! materialize one value per row (CONSTANT broadcast, BITPACKING CONSTANT and
//! CONSTANT_DELTA). int4 stores cover the 16B-aligned interior; a scalar
//! prologue/epilogue absorb the unaligned head and ragged tail — so the
//! vectorized path no longer requires `out` itself to be 16B aligned, only the
//! 256B column buffer it points into (which RMM already guarantees).

#pragma once

#include <cub/config.cuh>
#include <cub/thread/thread_store.cuh>
#include <cuda/std/cstdint>

namespace sirius::cuda::scan::detail {

//! @brief Fill `out[0, rows)` with `out[i] = gen(i)`, vectorizing the 16B-aligned interior with
//! int4 stores and covering the unaligned head + ragged tail with scalar stores. Uses the
//! cache-streaming store hint (`STORE_CS`) since the output is write-once decode data.
//!
//! Strided by (@p tid, @p stride): pass (threadIdx.x, blockDim.x) for a per-CTA fill, or the global
//! thread id + `gridDim.x * blockDim.x` for a grid-wide fill. No barrier — every output index is
//! written by exactly one thread.
//!
//!   out: [ head (scalar, to 16B) | int4 int4 int4 ... (aligned interior) | tail (scalar) ]
//!
//! @tparam T   Element type. Vectorized when `sizeof(T)` divides 16 and is <= 8; else plain scalar
//!             (e.g. 16-byte decimal128, where one int4 would hold a single value anyway).
//! @tparam Gen Callable `T gen(uint32_t i)` producing the value for output index i.
template <typename T, typename Gen>
_CCCL_DEVICE _CCCL_FORCEINLINE void vec_fill(
  T* out, uint32_t rows, uint32_t tid, uint32_t stride, Gen gen)
{
  if constexpr (sizeof(T) <= 8 && 16u % sizeof(T) == 0) {
    constexpr uint32_t TPV = 16u / sizeof(T);  // T values per int4
    const uint32_t misalign =
      static_cast<uint32_t>(reinterpret_cast<::cuda::std::uintptr_t>(out) & 15u);
    uint32_t head = (misalign == 0u) ? 0u : (16u - misalign) / static_cast<uint32_t>(sizeof(T));
    head          = (head < rows) ? head : rows;

    // Scalar prologue up to the first 16B boundary.
    for (uint32_t i = tid; i < head; i += stride) {
      cub::ThreadStore<cub::STORE_CS>(out + i, gen(i));
    }

    // int4 interior — `out + head` is now 16B aligned.
    const uint32_t vec_count = (rows - head) / TPV;
    auto* const out4         = reinterpret_cast<int4*>(out + head);
    for (uint32_t v = tid; v < vec_count; v += stride) {
      int4 packed;
      auto* const lanes   = reinterpret_cast<T*>(&packed);
      const uint32_t base = head + v * TPV;
#pragma unroll
      for (uint32_t k = 0; k < TPV; ++k) {
        lanes[k] = gen(base + k);
      }
      cub::ThreadStore<cub::STORE_CS>(out4 + v, packed);
    }

    // Scalar epilogue for the ragged tail.
    for (uint32_t i = head + vec_count * TPV + tid; i < rows; i += stride) {
      cub::ThreadStore<cub::STORE_CS>(out + i, gen(i));
    }
  } else {
    for (uint32_t i = tid; i < rows; i += stride) {
      out[i] = gen(i);
    }
  }
}

}  // namespace sirius::cuda::scan::detail
