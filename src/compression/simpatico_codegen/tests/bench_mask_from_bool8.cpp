// SPDX-License-Identifier: Apache-2.0
//
// Micro-benchmark for mask_from_bool8, sized to be profiled with ncu.
// Usage: bench_mask_from_bool8 [rows=1073741824] [iters=20] [byte_offset=0]

#include "codegen/selection/selection.hpp"

#include <cuda/stream>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv)
{
  std::int64_t const rows = argc > 1 ? std::atoll(argv[1]) : (std::int64_t{1} << 30);
  int const iters         = argc > 2 ? std::atoi(argv[2]) : 20;
  std::size_t const off   = argc > 3 ? std::atoll(argv[3]) : 0;

  std::uint8_t* d_flags = nullptr;
  std::uint32_t* d_words = nullptr;
  std::int64_t const words =
    sirius::codegen::selection_mask::ChunksFor(rows) * 32;  // full padded strip
  cudaMalloc(&d_flags, rows + 64);
  cudaMalloc(&d_words, words * sizeof(std::uint32_t));
  std::vector<std::uint8_t> h(rows);
  for (std::int64_t i = 0; i < rows; ++i)
    h[i] = ((i * 2654435761u) >> 13) % 100 < 30 ? 1 : 0;
  cudaMemcpy(d_flags + off, h.data(), rows, cudaMemcpyHostToDevice);

  cudaStream_t s;
  cudaStreamCreate(&s);
  ::cuda::stream_ref ref{s};
  sirius::codegen::mask_from_bool8(d_flags + off, rows, d_words, ref);  // warmup
  cudaStreamSynchronize(s);

  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a, s);
  for (int i = 0; i < iters; ++i)
    sirius::codegen::mask_from_bool8(d_flags + off, rows, d_words, ref);
  cudaEventRecord(b, s);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  double const per = ms / iters;
  std::printf("rows=%lld off=%zu: %.4f ms/iter, %.1f GB/s read\n",
              static_cast<long long>(rows), off, per, rows / (per * 1e6));

  std::vector<std::uint32_t> hw(words);
  cudaMemcpy(hw.data(), d_words, words * 4, cudaMemcpyDeviceToHost);
  std::uint64_t sum = 0;
  for (auto w : hw) sum += __builtin_popcount(w);
  std::printf("popcount=%llu\n", static_cast<unsigned long long>(sum));
  return 0;
}
