// SPDX-License-Identifier: Apache-2.0
//
// Times the selection wave's phases (mask_from_bool8, run_selection_cnt, mask_to_row_indices,
// build_chunk_row_set, row_set_to_mask). Usage: bench_selection_wave [rows=268435456]
// [selectivity_pct=10] [iters=20]

#include "codegen/selection/chunk_row_set.hpp"
#include "codegen/selection/selection.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

using namespace sirius::codegen;

static float time_ms(cudaStream_t s, int iters, std::function<void()> const& fn)
{
  fn();
  cudaStreamSynchronize(s);
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a, s);
  for (int i = 0; i < iters; ++i)
    fn();
  cudaEventRecord(b, s);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  return ms / iters;
}

int main(int argc, char** argv)
{
  std::int64_t const rows = argc > 1 ? std::atoll(argv[1]) : (std::int64_t{1} << 28);
  int const sel           = argc > 2 ? std::atoi(argv[2]) : 10;
  int const iters         = argc > 3 ? std::atoi(argv[3]) : 20;

  cudaStream_t s;
  cudaStreamCreate(&s);
  ::cuda::stream_ref ref{s};
  auto mr = rmm::mr::get_current_device_resource_ref();

  std::vector<std::uint8_t> h(rows);
  for (std::int64_t i = 0; i < rows; ++i)
    h[i] = (((i * 2654435761u) >> 13) % 100) < static_cast<unsigned>(sel) ? 1 : 0;
  std::uint8_t* d_flags;
  cudaMalloc(&d_flags, rows);
  cudaMemcpy(d_flags, h.data(), rows, cudaMemcpyHostToDevice);

  std::int64_t const nc    = selection_mask::ChunksFor(rows);
  std::int64_t const words = selection_mask::WordsFor(rows);
  std::uint32_t *d_words, *d_offsets;
  cudaMalloc(&d_words, nc * 32 * 4);
  cudaMalloc(&d_offsets, (nc + 1) * 4);

  selection_mask mask;
  mask.num_rows       = rows;
  mask.words          = d_words;
  mask.chunk_offsets  = d_offsets;
  mask.survivor_count = -1;

  mask_from_bool8(d_flags, rows, d_words, ref);
  std::int64_t const surv = run_selection_cnt(mask, ref, mr);
  std::int32_t* d_idx;
  cudaMalloc(&d_idx, (surv + 1) * 4);

  std::printf("rows=%lld sel=%d%% survivors=%lld\n",
              static_cast<long long>(rows),
              sel,
              static_cast<long long>(surv));
  std::printf("mask_from_bool8     %8.4f ms\n",
              time_ms(s, iters, [&] { mask_from_bool8(d_flags, rows, d_words, ref); }));
  std::printf("run_selection_cnt   %8.4f ms\n",
              time_ms(s, iters, [&] { run_selection_cnt(mask, ref, mr); }));
  std::printf("mask_to_row_indices %8.4f ms\n",
              time_ms(s, iters, [&] { mask_to_row_indices(mask, d_idx, ref); }));

  chunk_row_set_owner owner;
  std::printf("build_chunk_row_set %8.4f ms\n",
              time_ms(s, iters, [&] { owner = build_chunk_row_set(d_idx, surv, rows, ref, mr); }));
  std::uint32_t *m2, *o2;
  cudaMalloc(&m2, nc * 32 * 4);
  cudaMalloc(&o2, (nc + 1) * 4);
  std::printf("row_set_to_mask     %8.4f ms\n",
              time_ms(s, iters, [&] { row_set_to_mask(owner.view(), m2, o2, ref, mr); }));
  std::vector<std::uint32_t> a(nc * 32), b(nc * 32);
  cudaMemcpy(a.data(), d_words, nc * 128, cudaMemcpyDeviceToHost);
  cudaMemcpy(b.data(), m2, nc * 128, cudaMemcpyDeviceToHost);
  std::printf("roundtrip mask %s\n", a == b ? "OK" : "MISMATCH");
  (void)words;
  return 0;
}
