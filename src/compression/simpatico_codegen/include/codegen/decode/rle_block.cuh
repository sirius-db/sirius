#pragma once

#include "codegen/stdint_shim.hpp"
// nvrtc provides device builtins implicitly; <cuda_runtime.h> is host-only.
#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif
#include "codegen/tree.hpp"  // kChunkSize, kTBSize

#include <cub/block/block_scan.cuh>

// Block-collective RLE decode primitives (plain CUDA + CUB, no JIT codegen), shared
// by the AOT and JIT decode paths.  The bitpack and delta block primitives live
// in their own decode headers (decode/bitpack.cuh, decode/delta.cuh).

namespace codegen {

// Hybrid RLE decompress: run-driven (one scan round per TBSize runs, each
// thread scatters its run) when there are many runs, else output-driven
// (scan all counts into a starts array, then binary-search per output
// position).  Entirely in shared memory via a CUB block scan.  read_count is a
// functor(int32_t idx) -> int32_t giving run idx's length.  Does NOT sync at end.
template <typename T, int ChunkSize = kChunkSize, int TBSize = kTBSize, typename ReadCount>
__device__ __forceinline__ void block_rle_decompress_dispatch(T const* __restrict__ values,
                                                              ReadCount read_count,
                                                              int32_t* __restrict__ starts_scratch,
                                                              int32_t num_runs,
                                                              int32_t n_out,
                                                              T* __restrict__ out)
{
  static_assert(ChunkSize % TBSize == 0, "ChunkSize must be a multiple of TBSize");
  using BlockScan = cub::BlockScan<int32_t, TBSize>;
  __shared__ typename BlockScan::TempStorage scan_storage;

  if (num_runs <= 0 || n_out <= 0) return;

  constexpr int kRunDrivenThreshold = TBSize / 4;

  if (num_runs >= kRunDrivenThreshold) {
    int32_t base_local   = 0;
    int const num_rounds = (num_runs + TBSize - 1) / TBSize;
    for (int round = 0; round < num_rounds; ++round) {
      int const idx = round * TBSize + threadIdx.x;

      int32_t const my_count = (idx < num_runs) ? read_count(idx) : 0;
      int32_t my_offset;
      int32_t round_total;
      BlockScan(scan_storage).ExclusiveSum(my_count, my_offset, round_total);

      if (idx < num_runs) {
        T const my_val = values[idx];
        int const base = base_local + my_offset;
        for (int k = 0; k < my_count; ++k) {
          out[base + k] = my_val;
        }
      }

      base_local += round_total;
      __syncthreads();
    }
    return;
  }

  constexpr int IPT = ChunkSize / TBSize;
  int32_t cnts[IPT];
#pragma unroll
  for (int j = 0; j < IPT; ++j) {
    int const idx = threadIdx.x * IPT + j;
    cnts[j]       = (idx < num_runs) ? read_count(idx) : 0;
  }
  int32_t starts_local[IPT];
  BlockScan(scan_storage).ExclusiveSum(cnts, starts_local);
  __syncthreads();

#pragma unroll
  for (int j = 0; j < IPT; ++j) {
    int const idx = threadIdx.x * IPT + j;
    if (idx < num_runs) { starts_scratch[idx] = starts_local[j]; }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < n_out; i += TBSize) {
    int lo = 0;
    int hi = num_runs - 1;
    while (lo < hi) {
      int const mid = (lo + hi + 1) / 2;
      if (starts_scratch[mid] <= i) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    out[i] = values[lo];
  }
}

namespace detail {
struct SmemCountsReader {
  int32_t const* counts;
  __device__ int32_t operator()(int32_t idx) const noexcept { return counts[idx]; }
};
template <typename T>
struct SmemValuesReader {
  T const* values;
  __device__ T operator()(int32_t idx) const noexcept { return values[idx]; }
};
}  // namespace detail

// Thin wrapper: run counts already materialised in shared/device memory.
template <typename T, int ChunkSize = kChunkSize, int TBSize = kTBSize>
__device__ inline void block_rle_decompress(T const* __restrict__ values,
                                            int32_t const* __restrict__ counts,
                                            int32_t* __restrict__ starts_scratch,
                                            int32_t num_runs,
                                            int32_t n_out,
                                            T* __restrict__ out)
{
  block_rle_decompress_dispatch<T, ChunkSize, TBSize>(
    values, detail::SmemCountsReader{counts}, starts_scratch, num_runs, n_out, out);
}

// Both values and counts supplied as functors — eliminates sh_values from smem.
// Saves sizeof(T) * kChunkSize bytes of dynamic shared memory vs block_rle_decompress.
// Use when the values child is a leaf (Bitpack or Raw): pass the inline read
// expression as a functor.  The ReadValues functor is called at most once per
// run (run-driven path) or once per output position (binary-search path).
template <typename T,
          int ChunkSize = kChunkSize,
          int TBSize    = kTBSize,
          typename ReadValues,
          typename ReadCount>
__device__ __forceinline__ void block_rle_decompress_fv(
  ReadValues read_values,  // (int32_t run_idx) -> T
  ReadCount read_count,    // (int32_t run_idx) -> int32_t
  int32_t* __restrict__ starts_scratch,
  int32_t num_runs,
  int32_t n_out,
  T* __restrict__ out)
{
  static_assert(ChunkSize % TBSize == 0, "ChunkSize must be a multiple of TBSize");
  using BlockScan = cub::BlockScan<int32_t, TBSize>;
  __shared__ typename BlockScan::TempStorage scan_storage;

  if (num_runs <= 0 || n_out <= 0) return;

  constexpr int kRunDrivenThreshold = TBSize / 4;

  if (num_runs >= kRunDrivenThreshold) {
    int32_t base_local   = 0;
    int const num_rounds = (num_runs + TBSize - 1) / TBSize;
    for (int round = 0; round < num_rounds; ++round) {
      int const idx = round * TBSize + threadIdx.x;

      int32_t const my_count = (idx < num_runs) ? read_count(idx) : 0;
      int32_t my_offset;
      int32_t round_total;
      BlockScan(scan_storage).ExclusiveSum(my_count, my_offset, round_total);

      if (idx < num_runs) {
        T const my_val = read_values(idx);
        int const base = base_local + my_offset;
        for (int k = 0; k < my_count; ++k) {
          out[base + k] = my_val;
        }
      }

      base_local += round_total;
      __syncthreads();
    }
    return;
  }

  constexpr int IPT = ChunkSize / TBSize;
  int32_t cnts[IPT];
#pragma unroll
  for (int j = 0; j < IPT; ++j) {
    int const idx = threadIdx.x * IPT + j;
    cnts[j]       = (idx < num_runs) ? read_count(idx) : 0;
  }
  int32_t starts_local[IPT];
  BlockScan(scan_storage).ExclusiveSum(cnts, starts_local);
  __syncthreads();

#pragma unroll
  for (int j = 0; j < IPT; ++j) {
    int const idx = threadIdx.x * IPT + j;
    if (idx >= num_runs) break;
    starts_scratch[idx] = starts_local[j];
  }
  __syncthreads();

  for (int32_t i = threadIdx.x; i < n_out; i += TBSize) {
    int lo = 0, hi = num_runs - 1;
    while (lo < hi) {
      int mid = (lo + hi + 1) / 2;
      if (starts_scratch[mid] <= i)
        lo = mid;
      else
        hi = mid - 1;
    }
    out[i] = read_values(lo);
  }
}

}  // namespace codegen
