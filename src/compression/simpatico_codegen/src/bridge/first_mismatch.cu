// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

// Each row is `Words` words of `Word`.
template <typename Word, int Words>
__global__ void first_mismatch_kernel(Word const* a,
                                      Word const* b,
                                      unsigned long long rows,
                                      unsigned long long* first)
{
  auto const stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (auto row = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < rows;
       row += stride) {
    bool differs = false;
    for (int word = 0; word < Words; ++word)
      differs |= a[row * Words + word] != b[row * Words + word];
    if (differs) atomicMin(first, row);
  }
}

template <typename Word, int Words>
int launch(void const* a, void const* b, std::size_t rows, void* first, cudaStream_t stream)
{
  constexpr unsigned kThreads = 256;
  auto const blocks =
    static_cast<unsigned>(std::min<std::size_t>((rows + kThreads - 1) / kThreads, 4096));
  first_mismatch_kernel<Word, Words>
    <<<blocks, kThreads, 0, stream>>>(static_cast<Word const*>(a),
                                      static_cast<Word const*>(b),
                                      rows,
                                      static_cast<unsigned long long*>(first));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

extern "C" {

// Writes to the device's `*first` the first of `rows` rows of `row_bytes` (1, 2, 4, 8 or 16)
// where `a` and `b` differ, or `rows` where none does. Returns a cudaError_t as int.
int simpatico_first_mismatch(void const* a,
                             void const* b,
                             std::size_t rows,
                             std::size_t row_bytes,
                             void* first,
                             void* stream_v)
{
  auto const stream = static_cast<cudaStream_t>(stream_v);
  auto const none   = static_cast<unsigned long long>(rows);
  if (auto const e = cudaMemcpyAsync(first, &none, sizeof(none), cudaMemcpyHostToDevice, stream);
      e != cudaSuccess)
    return static_cast<int>(e);
  if (rows == 0) return 0;
  switch (row_bytes) {
    case 1: return launch<std::uint8_t, 1>(a, b, rows, first, stream);
    case 2: return launch<std::uint16_t, 1>(a, b, rows, first, stream);
    case 4: return launch<std::uint32_t, 1>(a, b, rows, first, stream);
    case 8: return launch<std::uint64_t, 1>(a, b, rows, first, stream);
    case 16: return launch<std::uint64_t, 2>(a, b, rows, first, stream);
    default: return static_cast<int>(cudaErrorInvalidValue);
  }
}

}  // extern "C"
