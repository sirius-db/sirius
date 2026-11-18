// SPDX-License-Identifier: Apache-2.0
#pragma once

// Device helper shared by the kernels of the chunk decoders. Include from .cu files only.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace simpatico::chunk {

/// Little-endian element at `bytes`, which may sit at any address when !aligned (a leaf buffer
/// can start at any payload offset).
template <typename U>
__device__ __forceinline__ U load_element(const std::uint8_t* __restrict__ bytes, bool aligned)
{
  if (aligned) { return *reinterpret_cast<const U*>(bytes); }
  std::uint64_t v = 0;
  for (std::size_t k = 0; k < sizeof(U); ++k) {
    v |= static_cast<std::uint64_t>(bytes[k]) << (8 * k);
  }
  return static_cast<U>(v);
}

}  // namespace simpatico::chunk
