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

// Replaces the memset's all-ones with `rows` where no row differs, else copies the row's values.
template <typename Word, int Words>
__global__ void mismatch_values_kernel(Word const* a,
                                       Word const* b,
                                       unsigned long long rows,
                                       unsigned long long* record)
{
  auto const first = record[0];
  if (first >= rows) {
    record[0] = rows;
    return;
  }
  for (int word = 1; word < 5; ++word)
    record[word] = 0;
  auto* values = reinterpret_cast<Word*>(record + 1);
  for (int word = 0; word < Words; ++word) {
    values[word]                     = a[first * Words + word];
    values[16 / sizeof(Word) + word] = b[first * Words + word];
  }
}

template <typename Word, int Words>
int launch(void const* a, void const* b, std::size_t rows, void* record, cudaStream_t stream)
{
  constexpr unsigned kThreads = 256;
  auto const typed_a          = static_cast<Word const*>(a);
  auto const typed_b          = static_cast<Word const*>(b);
  auto const words            = static_cast<unsigned long long*>(record);
  if (rows > 0) {
    auto const blocks =
      static_cast<unsigned>(std::min<std::size_t>((rows + kThreads - 1) / kThreads, 4096));
    first_mismatch_kernel<Word, Words>
      <<<blocks, kThreads, 0, stream>>>(typed_a, typed_b, rows, words);
  }
  mismatch_values_kernel<Word, Words><<<1, 1, 0, stream>>>(typed_a, typed_b, rows, words);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

extern "C" {

// Writes the device's `record`, 5 words: the first of `rows` rows of `row_bytes` (1, 2, 4, 8 or 16)
// where `a` and `b` differ, or `rows` where none does; then, where one does, that row of `a` and
// that row of `b`, each in two words from the low end. Queues work on `stream` only, never waiting.
// Returns a cudaError_t as int.
int simpatico_first_mismatch(void const* a,
                             void const* b,
                             std::size_t rows,
                             std::size_t row_bytes,
                             void* record,
                             void* stream_v)
{
  auto const stream = static_cast<cudaStream_t>(stream_v);
  if (auto const e = cudaMemsetAsync(record, 0xff, 5 * sizeof(std::uint64_t), stream);
      e != cudaSuccess)
    return static_cast<int>(e);
  switch (row_bytes) {
    case 1: return launch<std::uint8_t, 1>(a, b, rows, record, stream);
    case 2: return launch<std::uint16_t, 1>(a, b, rows, record, stream);
    case 4: return launch<std::uint32_t, 1>(a, b, rows, record, stream);
    case 8: return launch<std::uint64_t, 1>(a, b, rows, record, stream);
    case 16: return launch<std::uint64_t, 2>(a, b, rows, record, stream);
    default: return static_cast<int>(cudaErrorInvalidValue);
  }
}

}  // extern "C"
