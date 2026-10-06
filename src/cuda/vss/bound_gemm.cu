/*
 * Copyright 2025, Sirius Contributors.
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

#include "vss/bound_gemm.hpp"

#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

namespace sirius::vss {

namespace {

// Block tile 128 probe rows x 128 corpus rows, 8 warps in a 2 x 4 grid of 64 x 32 warp tiles.
// The reduction dimension is staged through shared memory 64 components at a time; the 16-byte
// row pad keeps the fragment loads off a single bank.
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kChunk = 64;
constexpr int kWarps = 8;

template <class T>
struct mma_traits;
template <>
struct mma_traits<int8_t> {
  using element     = signed char;
  using accumulator = int;
  using norm        = int32_t;
};
template <>
struct mma_traits<__half> {
  using element     = __half;
  using accumulator = float;
  using norm        = float;
};

/// Per-row slack added to the bound: 0 for exact int8 dots; for FP16 an upper bound on how far the
/// FP16 distance can sit below the FP32 one, a |q| X + b (|q| + X)^2 + c (|q| + X) with X the
/// largest corpus row norm.
struct slack_terms {
  float a{0}, b{0}, c{0}, x_max{0};
};

// Many slices in one launch: block b belongs to the slice s with prefix[s] <= b < prefix[s + 1],
// and is that slice's (b - prefix[s])-th block. A null table means a single-slice launch.
struct group_view {
  bound_slice const* slices{nullptr};
  int64_t const* prefix{nullptr};
  int n_slices{0};
};

__device__ inline int find_slice(group_view const& g, int64_t b)
{
  int lo = 0, hi = g.n_slices;
  while (hi - lo > 1) {
    int const mid = (lo + hi) / 2;
    if (g.prefix[mid] <= b) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Two blocks per SM: under separable compilation ptxas otherwise spends 177 registers on this
// kernel and runs one block per SM, 1.5x slower.
//
// Probe tiles run fastest across the grid: the probe side is small enough to stay in L2, so each
// corpus tile is read from memory about once instead of once per probe tile row.
//
// kRegisterEpilogue scores the accumulators where they are: a 16 x 16 accumulator's x[t] holds
// row g + 8 ((t >> 1) & 1), column 8 (t >> 2) + 2 q + (t & 1), g = lane / 4, q = lane % 4 -- the
// mma m16n8 layout WMMA uses on sm_80+, which accumulator_layout_is_m16n8 checks on the device.
// Otherwise each fragment goes through shared memory, whatever its layout.
template <class T, bool kRegisterEpilogue, bool kGrouped = false>
__global__ void __launch_bounds__(kWarps * 32, 2)
  bound_filter_kernel(T const* __restrict__ x,
                      typename mma_traits<T>::norm const* __restrict__ x_sq,
                      int64_t n,
                      int64_t id_base,
                      int64_t const* __restrict__ id_map,
                      T const* __restrict__ probe,
                      typename mma_traits<T>::norm const* __restrict__ probe_sq,
                      int64_t const* __restrict__ rows,
                      int64_t m,
                      int d,
                      float const* __restrict__ bound,
                      slack_terms slack,
                      int32_t* out_rows,
                      int64_t* out_ids,
                      float* out_d,
                      unsigned long long* count,
                      unsigned long long capacity,
                      group_view group)
{
#if __CUDA_ARCH__ >= 720
  using namespace nvcuda;
  using E              = typename mma_traits<T>::element;
  using A              = typename mma_traits<T>::accumulator;
  constexpr int kVec   = 16 / sizeof(T);  // components per 16-byte load
  constexpr int kPitch = kChunk + kVec;
  __shared__ __align__(32) T tile_a[kTileM * kPitch];
  __shared__ __align__(32) T tile_b[kTileN * kPitch];
  __shared__ __align__(32) A scores[kRegisterEpilogue ? 1 : kWarps][16 * 16];
  __shared__ int64_t tile_rows[kTileM];
  __shared__ float tile_qsq[kTileM];
  __shared__ float tile_limit[kTileM];
  // The tile's corpus norms, read coalesced up front rather than eight scattered loads a lane at
  // the end, where nothing hides them.
  __shared__ typename mma_traits<T>::norm tile_xsq[kTileN];

  int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int const wm = warp / 4, wn = warp % 4;
  int64_t q0 = static_cast<int64_t>(blockIdx.x) * kTileM;
  int64_t x0 = static_cast<int64_t>(blockIdx.y) * kTileN;
  // A separate instantiation, so a single-slice launch carries none of this in its registers.
  if constexpr (kGrouped) {
    auto const b       = static_cast<int64_t>(blockIdx.x);
    auto const si      = find_slice(group, b);
    auto const& sl     = group.slices[si];
    x                  = static_cast<T const*>(sl.x);
    x_sq               = static_cast<typename mma_traits<T>::norm const*>(sl.x_sq);
    n                  = sl.n;
    id_base            = sl.id_base;
    id_map             = sl.id_map;
    rows               = sl.rows;
    m                  = sl.m;
    auto const local   = b - group.prefix[si];
    auto const m_tiles = (m + kTileM - 1) / kTileM;
    q0                 = (local % m_tiles) * kTileM;
    x0                 = (local / m_tiles) * kTileN;
  }
  static_assert(kTileM == kTileN, "the setup loop fills both tiles' per-row values");
  for (int r = threadIdx.x; r < kTileM; r += blockDim.x) {
    tile_xsq[r]       = x0 + r < n ? x_sq[x0 + r] : 0;
    int64_t const row = q0 + r < m ? rows[q0 + r] : -1;
    tile_rows[r]      = row;
    if (row >= 0) {
      auto const qsq = static_cast<float>(probe_sq[row]);
      auto const qn  = sqrtf(qsq);
      tile_qsq[r]    = qsq;
      tile_limit[r]  = bound[row] + slack.a * qn * slack.x_max +
                      slack.b * (qn + slack.x_max) * (qn + slack.x_max) +
                      slack.c * (qn + slack.x_max);
    } else {
      tile_qsq[r]   = 0.f;
      tile_limit[r] = -INFINITY;
    }
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, A> acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      wmma::fill_fragment(acc[i][j], A(0));
    }
  }
  for (int k0 = 0; k0 < d; k0 += kChunk) {
    int const kc = min(kChunk, d - k0);
    for (int e = threadIdx.x; e < kTileM * (kChunk / kVec); e += blockDim.x) {
      int const r = e / (kChunk / kVec), c = (e % (kChunk / kVec)) * kVec;
      int4 a = make_int4(0, 0, 0, 0), b = make_int4(0, 0, 0, 0);
      if (tile_rows[r] >= 0 && c < kc) {
        a = *reinterpret_cast<int4 const*>(probe + tile_rows[r] * d + k0 + c);
      }
      if (x0 + r < n && c < kc) { b = *reinterpret_cast<int4 const*>(x + (x0 + r) * d + k0 + c); }
      *reinterpret_cast<int4*>(tile_a + r * kPitch + c) = a;
      *reinterpret_cast<int4*>(tile_b + r * kPitch + c) = b;
    }
    __syncthreads();
    for (int kk = 0; kk < kc; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, E, wmma::row_major> fa[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, E, wmma::col_major> fb[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        wmma::load_matrix_sync(
          fa[i], reinterpret_cast<E const*>(tile_a) + (wm * 64 + i * 16) * kPitch + kk, kPitch);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        wmma::load_matrix_sync(
          fb[j], reinterpret_cast<E const*>(tile_b) + (wn * 32 + j * 16) * kPitch + kk, kPitch);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  if constexpr (kRegisterEpilogue) {
    // Each lane scores its own elements of each fragment: its 8 columns' |x|^2 are loaded once,
    // and a fragment with no survivor in the warp costs one vote.
    int const g = lane / 4, q = lane % 4;
    typename mma_traits<T>::norm xs[8];
    unsigned col_ok = 0;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      int const col = wn * 32 + (c >> 2) * 16 + 8 * ((c & 3) >> 1) + 2 * q + (c & 1);
      col_ok |= x0 + col < n ? (1u << c) : 0u;
      xs[c] = tile_xsq[col];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int const r0         = wm * 64 + i * 16 + g;
      float const qs[2]    = {tile_qsq[r0], tile_qsq[r0 + 8]};
      float const limit[2] = {tile_limit[r0], tile_limit[r0 + 8]};
      // A pair's distance is only formed again for a survivor, where it is written out.
      auto distance = [&](int h, int c, typename mma_traits<T>::accumulator a) {
        if constexpr (sizeof(T) == 1) {
          return static_cast<float>(static_cast<int32_t>(qs[h]) + xs[c] - 2 * a);
        } else {
          return qs[h] + xs[c] - 2.f * a;
        }
      };
      // INT8: qs + xs - 2 acc is an integer below 2^24, so it is <= limit exactly when it is <=
      // floor(limit), i.e. when acc >= ceil((qs - floor(limit) + xs) / 2) -- one integer compare
      // per pair against a threshold per (row, column), +inf for a column past the slice.
      int32_t base[2] = {0, 0};
      if constexpr (sizeof(T) == 1) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          auto const l = limit[h] >= 1e9f    ? 1000000000
                         : limit[h] <= -1e9f ? -1000000000
                                             : static_cast<int32_t>(floorf(limit[h]));
          base[h]      = static_cast<int32_t>(qs[h]) - l;
        }
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        unsigned keep = 0;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          int const h = (t >> 1) & 1, c = j * 4 + ((t >> 2) << 1) + (t & 1);
          bool pass;
          if constexpr (sizeof(T) == 1) {
            int32_t const threshold =
              ((col_ok >> c) & 1u) ? (base[h] + xs[c] + 1) >> 1 : 0x7fffffff;
            pass = acc[i][j].x[t] >= threshold;
          } else {
            pass = ((col_ok >> c) & 1u) && distance(h, c, acc[i][j].x[t]) <= limit[h];
          }
          keep |= pass ? (1u << t) : 0u;
        }
        if (!__any_sync(0xffffffffu, keep != 0)) { continue; }
        // One atomic for the warp: an inclusive scan of the lanes' survivor counts places each.
        int const own = __popc(keep);
        int before    = own;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          int const v = __shfl_up_sync(0xffffffffu, before, o);
          if (lane >= o) { before += v; }
        }
        unsigned long long base = 0;
        if (lane == 31) { base = atomicAdd(count, static_cast<unsigned long long>(before)); }
        auto p = __shfl_sync(0xffffffffu, base, 31) + static_cast<unsigned long long>(before - own);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          if (((keep >> t) & 1u) == 0) { continue; }
          if (p < capacity) {
            int const h = (t >> 1) & 1, c = j * 4 + ((t >> 2) << 1) + (t & 1);
            int64_t const xj = x0 + wn * 32 + j * 16 + 8 * (t >> 2) + 2 * q + (t & 1);
            out_rows[p]      = static_cast<int32_t>(tile_rows[r0 + 8 * h]);
            out_ids[p]       = id_map != nullptr ? id_map[xj] : id_base + xj;
            out_d[p]         = distance(h, c, acc[i][j].x[t]);
          }
          ++p;
        }
      }
    }
  } else {
    // Epilogue, one 16 x 16 fragment at a time through the warp's own scratch tile: a lane scores
    // 8 pairs, and each warp-wide batch of survivors takes one atomic.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        wmma::store_matrix_sync(scores[warp], acc[i][j], 16, wmma::mem_row_major);
        __syncwarp();
        int const rb     = wm * 64 + i * 16;
        int64_t const xb = x0 + wn * 32 + j * 16;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          int const e = t * 32 + lane, r = rb + e / 16;
          int64_t const xj  = xb + e % 16;
          int64_t const row = tile_rows[r];
          bool keep         = false;
          float dist        = 0.f;
          if (row >= 0 && xj < n) {
            if constexpr (sizeof(T) == 1) {
              dist = static_cast<float>(static_cast<int32_t>(tile_qsq[r]) + x_sq[xj] -
                                        2 * scores[warp][e]);
            } else {
              dist = tile_qsq[r] + x_sq[xj] - 2.f * scores[warp][e];
            }
            keep = dist <= tile_limit[r];
          }
          unsigned const mask = __ballot_sync(0xffffffffu, keep);
          if (mask != 0) {
            int const leader        = __ffs(mask) - 1;
            unsigned long long base = 0;
            if (lane == leader) {
              base = atomicAdd(count, static_cast<unsigned long long>(__popc(mask)));
            }
            base = __shfl_sync(0xffffffffu, base, leader);
            if (keep) {
              auto const p = base + __popc(mask & ((1u << lane) - 1));
              if (p < capacity) {
                out_rows[p] = static_cast<int32_t>(row);
                out_ids[p]  = id_map != nullptr ? id_map[xj] : id_base + xj;
                out_d[p]    = dist;
              }
            }
          }
        }
        __syncwarp();
      }
    }
  }
#endif
}

// Writes to *mismatches how many of a 16 x 16 accumulator's elements are not where the register
// epilogue expects them.
template <class A>
__global__ void accumulator_layout_kernel(int* mismatches)
{
#if __CUDA_ARCH__ >= 720
  using namespace nvcuda;
  __shared__ A known[256];
  for (int i = threadIdx.x; i < 256; i += 32) {
    known[i] = A(i);
  }
  __syncwarp();
  wmma::fragment<wmma::accumulator, 16, 16, 16, A> f;
  wmma::load_matrix_sync(f, known, 16, wmma::mem_row_major);
  int const g = threadIdx.x / 4, q = threadIdx.x % 4;
  int bad = f.num_elements == 8 ? 0 : 1;
  for (int t = 0; t < f.num_elements && t < 8; ++t) {
    int const want = (g + 8 * ((t >> 1) & 1)) * 16 + 8 * (t >> 2) + 2 * q + (t & 1);
    bad += static_cast<int>(f.x[t]) != want ? 1 : 0;
  }
  if (bad != 0) { atomicAdd(mismatches, bad); }
#else
  if (threadIdx.x == 0) { *mismatches = 1; }
#endif
}

/// Whether the current device lays out accumulators as the register epilogue reads them; checked
/// once per process (SIRIUS_VSS_REGISTER_EPILOGUE=0 turns the register epilogue off).
template <class A>
bool accumulator_layout_is_m16n8()
{
  static bool const ok = [] {
    auto const* v = std::getenv("SIRIUS_VSS_REGISTER_EPILOGUE");
    if (v != nullptr && std::strcmp(v, "0") == 0) { return false; }
    int* mismatches = nullptr;
    CUDF_CUDA_TRY(cudaMalloc(&mismatches, sizeof(int)));
    CUDF_CUDA_TRY(cudaMemset(mismatches, 0, sizeof(int)));
    accumulator_layout_kernel<A><<<1, 32>>>(mismatches);
    int host = 1;
    CUDF_CUDA_TRY(cudaMemcpy(&host, mismatches, sizeof(int), cudaMemcpyDeviceToHost));
    CUDF_CUDA_TRY(cudaFree(mismatches));
    return host == 0;
  }();
  return ok;
}

// A few probe rows against a slice: with m this small a 128 x 128 tile is almost all padding and
// the search is bound by reading the slice, so each thread takes one corpus row, streams it with
// 16-byte loads and scores it against every probe row held in shared memory. Same bound, same
// append, same error model (exact int8 dots; FP16 products summed in FP32).
constexpr int kSmallThreads = 256;
constexpr int kSmallMaxDim  = 256;

template <class T, int M, bool kGrouped = false>
__global__ void __launch_bounds__(kSmallThreads)
  bound_filter_small_kernel(T const* __restrict__ x,
                            typename mma_traits<T>::norm const* __restrict__ x_sq,
                            int64_t n,
                            int64_t id_base,
                            int64_t const* __restrict__ id_map,
                            T const* __restrict__ probe,
                            typename mma_traits<T>::norm const* __restrict__ probe_sq,
                            int64_t const* __restrict__ rows,
                            int m,
                            int d,
                            float const* __restrict__ bound,
                            slack_terms slack,
                            int32_t* out_rows,
                            int64_t* out_ids,
                            float* out_d,
                            unsigned long long* count,
                            unsigned long long capacity,
                            group_view group)
{
  constexpr bool kInt8 = sizeof(T) == 1;
  // A grouped launch gives each block one kSmallThreads-row stretch of one slice.
  auto j_first = static_cast<int64_t>(blockIdx.x) * blockDim.x;
  auto j_step  = static_cast<int64_t>(gridDim.x) * blockDim.x;
  if constexpr (kGrouped) {
    auto const si  = find_slice(group, blockIdx.x);
    auto const& sl = group.slices[si];
    x              = static_cast<T const*>(sl.x);
    x_sq           = static_cast<typename mma_traits<T>::norm const*>(sl.x_sq);
    n              = sl.n;
    id_base        = sl.id_base;
    id_map         = sl.id_map;
    rows           = sl.rows;
    m              = static_cast<int>(sl.m);
    j_first        = (static_cast<int64_t>(blockIdx.x) - group.prefix[si]) * blockDim.x;
    j_step         = n;
  }
  // Probe components: int8 packed four to a word for __dp4a, FP16 widened to FP32.
  using Q                = std::conditional_t<kInt8, int, float>;
  constexpr int kPerWord = kInt8 ? 4 : 1;
  __shared__ __align__(16) Q s_probe[M][kSmallMaxDim / kPerWord];
  __shared__ int64_t s_rows[M];
  __shared__ float s_qsq[M];
  __shared__ float s_limit[M];

  int const words = d / kPerWord;
  for (int e = threadIdx.x; e < M * words; e += blockDim.x) {
    int const p = e / words, w = e % words;
    Q v = 0;
    if (p < m) {
      auto const row = rows[p];
      if constexpr (kInt8) {
        v = reinterpret_cast<int const*>(probe + row * d)[w];
      } else {
        v = __half2float(probe[row * d + w]);
      }
    }
    s_probe[p][w] = v;
  }
  for (int p = threadIdx.x; p < M; p += blockDim.x) {
    s_rows[p] = p < m ? rows[p] : -1;
    if (p < m) {
      auto const qsq = static_cast<float>(probe_sq[rows[p]]);
      auto const qn  = sqrtf(qsq);
      s_qsq[p]       = qsq;
      s_limit[p]     = bound[rows[p]] + slack.a * qn * slack.x_max +
                   slack.b * (qn + slack.x_max) * (qn + slack.x_max) + slack.c * (qn + slack.x_max);
    }
  }
  __syncthreads();

  int const lane = threadIdx.x % 32;
  for (int64_t j0 = j_first; j0 < n; j0 += j_step) {
    int64_t const j = j0 + threadIdx.x;
    bool const live = j < n;
    typename mma_traits<T>::accumulator acc[M];
#pragma unroll
    for (int p = 0; p < M; ++p) {
      acc[p] = 0;
    }
    if (live) {
      auto const* xr = reinterpret_cast<int4 const*>(x + j * d);
      // The probe side is read from shared memory 16 bytes at a time: one load per probe row
      // per 16 bytes of the corpus row, not one per component.
      for (int c = 0; c < d / (16 / static_cast<int>(sizeof(T))); ++c) {
        int4 const v = __ldg(xr + c);
        if constexpr (kInt8) {
#pragma unroll
          for (int p = 0; p < M; ++p) {
            int4 const q = *reinterpret_cast<int4 const*>(&s_probe[p][c * 4]);
            acc[p]       = __dp4a(v.x, q.x, acc[p]);
            acc[p]       = __dp4a(v.y, q.y, acc[p]);
            acc[p]       = __dp4a(v.z, q.z, acc[p]);
            acc[p]       = __dp4a(v.w, q.w, acc[p]);
          }
        } else {
          __half2 const* h = reinterpret_cast<__half2 const*>(&v);
          float f[8];
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            float2 const g = __half22float2(h[t]);
            f[2 * t]       = g.x;
            f[2 * t + 1]   = g.y;
          }
#pragma unroll
          for (int p = 0; p < M; ++p) {
            float4 const q0 = *reinterpret_cast<float4 const*>(&s_probe[p][c * 8]);
            float4 const q1 = *reinterpret_cast<float4 const*>(&s_probe[p][c * 8 + 4]);
            acc[p]          = fmaf(f[0], q0.x, acc[p]);
            acc[p]          = fmaf(f[1], q0.y, acc[p]);
            acc[p]          = fmaf(f[2], q0.z, acc[p]);
            acc[p]          = fmaf(f[3], q0.w, acc[p]);
            acc[p]          = fmaf(f[4], q1.x, acc[p]);
            acc[p]          = fmaf(f[5], q1.y, acc[p]);
            acc[p]          = fmaf(f[6], q1.z, acc[p]);
            acc[p]          = fmaf(f[7], q1.w, acc[p]);
          }
        }
      }
    }
    auto const xsq = live ? x_sq[j] : 0;
#pragma unroll
    for (int p = 0; p < M; ++p) {
      bool keep  = false;
      float dist = 0.f;
      if (live && p < m) {
        if constexpr (kInt8) {
          dist = static_cast<float>(static_cast<int32_t>(s_qsq[p]) + xsq - 2 * acc[p]);
        } else {
          dist = s_qsq[p] + xsq - 2.f * acc[p];
        }
        keep = dist <= s_limit[p];
      }
      unsigned const mask = __ballot_sync(0xffffffffu, keep);
      if (mask != 0) {
        int const leader        = __ffs(mask) - 1;
        unsigned long long base = 0;
        if (lane == leader) {
          base = atomicAdd(count, static_cast<unsigned long long>(__popc(mask)));
        }
        base = __shfl_sync(0xffffffffu, base, leader);
        if (keep) {
          auto const q = base + __popc(mask & ((1u << lane) - 1));
          if (q < capacity) {
            out_rows[q] = static_cast<int32_t>(s_rows[p]);
            out_ids[q]  = id_map != nullptr ? id_map[j] : id_base + j;
            out_d[q]    = dist;
          }
        }
      }
    }
  }
}

/// Probe-row counts up to this go to bound_filter_small_kernel (SIRIUS_VSS_SMALL_M overrides).
int small_m_limit()
{
  static int const limit = [] {
    auto const* v = std::getenv("SIRIUS_VSS_SMALL_M");
    return v == nullptr ? 16 : std::clamp(std::atoi(v), 0, 16);
  }();
  return limit;
}

template <class T>
bool launch_small(T const* x,
                  typename mma_traits<T>::norm const* x_sq,
                  int64_t n,
                  int64_t id_base,
                  int64_t const* id_map,
                  T const* probe,
                  typename mma_traits<T>::norm const* probe_sq,
                  int64_t const* rows,
                  int64_t m,
                  int64_t dim,
                  float const* bound,
                  slack_terms slack,
                  bound_candidates& out,
                  rmm::cuda_stream_view stream)
{
  if (m > small_m_limit() || dim > kSmallMaxDim) { return false; }
  auto const grid =
    static_cast<unsigned>(std::clamp<int64_t>((n + kSmallThreads - 1) / kSmallThreads, 1, 65535));
  auto go = [&](auto kernel) {
    kernel<<<grid, kSmallThreads, 0, stream.value()>>>(
      x,
      x_sq,
      n,
      id_base,
      id_map,
      probe,
      probe_sq,
      rows,
      static_cast<int>(m),
      static_cast<int>(dim),
      bound,
      slack,
      out.rows.data(),
      out.ids.data(),
      out.distances.data(),
      out.count.data(),
      static_cast<unsigned long long>(out.capacity()),
      group_view{});
  };
  if (m <= 1) {
    go(bound_filter_small_kernel<T, 1>);
  } else if (m <= 2) {
    go(bound_filter_small_kernel<T, 2>);
  } else if (m <= 4) {
    go(bound_filter_small_kernel<T, 4>);
  } else if (m <= 8) {
    go(bound_filter_small_kernel<T, 8>);
  } else {
    go(bound_filter_small_kernel<T, 16>);
  }
  CUDF_CUDA_TRY(cudaGetLastError());
  return true;
}

// One warp per pair: the exact FP32 squared distance between probe row rows[i] (or i / k when
// rows is null) and layout row ids[i], read from pinned host blocks through their device mapping.
__global__ void exact_distances_kernel(float const* __restrict__ probe,
                                       int32_t const* __restrict__ rows,
                                       int64_t k,
                                       int64_t const* __restrict__ ids,
                                       float* __restrict__ distances,
                                       int64_t n_pairs,
                                       float const* const* __restrict__ blocks,
                                       int64_t rows_per_block,
                                       int d,
                                       float const* __restrict__ certain_below,
                                       float certain_value)
{
  int const lane = threadIdx.x % 32;
  for (int64_t i = (blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x) / 32; i < n_pairs;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x / 32) {
    auto const id = ids[i];
    if (id < 0) { continue; }
    auto const row = rows != nullptr ? static_cast<int64_t>(rows[i]) : i / k;
    if (certain_below != nullptr) {
      // Lane 0 decides for the warp, so no lane can read the value it overwrites.
      int certain = lane == 0 && distances[i] <= certain_below[row];
      certain     = __shfl_sync(0xffffffffu, certain, 0);
      if (certain) {
        if (lane == 0) { distances[i] = certain_value; }
        continue;
      }
    }
    float const* q = probe + row * d;
    float const* x = blocks[id / rows_per_block] + (id % rows_per_block) * d;
    float s        = 0.f;
    for (int c = lane; c < d; c += 32) {
      float const t = q[c] - x[c];
      s             = fmaf(t, t, s);
    }
#pragma unroll
    for (int o = 16; o > 0; o /= 2) {
      s += __shfl_xor_sync(0xffffffffu, s, o);
    }
    if (lane == 0) { distances[i] = s; }
  }
}

__global__ void row_max_kernel(float const* acc_d, int64_t n_rows, int64_t k, float* bound)
{
  for (int64_t r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; r < n_rows;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float b = 0.f;
    for (int64_t j = 0; j < k; ++j) {
      b = fmaxf(b, acc_d[r * k + j]);
    }
    bound[r] = b;
  }
}

__global__ void fill_misses_kernel(float* d, int64_t* ids, int64_t n)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    d[i]   = __int_as_float(0x7f800000);
    ids[i] = -1;
  }
}

__global__ void map_ids_kernel(int64_t* ids, int64_t n, int64_t const* __restrict__ id_map)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (ids[i] >= 0) { ids[i] = id_map[ids[i]]; }
  }
}

__global__ void merge_keys_kernel(float const* acc_d,
                                  int64_t acc_n,
                                  int64_t k,
                                  int32_t const* cand_rows,
                                  float const* cand_d,
                                  int64_t n_cand,
                                  uint64_t* keys,
                                  int64_t* order,
                                  int32_t* per_row)
{
  auto const total = acc_n + n_cand;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint64_t row;
    float d;
    if (i < acc_n) {
      row = static_cast<uint64_t>(i / k);
      d   = acc_d[i];
    } else {
      row = static_cast<uint64_t>(cand_rows[i - acc_n]);
      d   = cand_d[i - acc_n];
      atomicAdd(per_row + row, 1);
    }
    // Distances are >= 0, so their bit patterns order as unsigned integers (+inf included).
    keys[i]  = (row << 32) | __float_as_uint(fmaxf(d, 0.f));
    order[i] = i;
  }
}

__global__ void merge_take_kernel(int64_t const* sorted_order,
                                  int32_t const* cand_before,
                                  int64_t n_rows,
                                  int64_t k,
                                  float const* old_d,
                                  int64_t const* old_n,
                                  float const* cand_d,
                                  int64_t const* cand_ids,
                                  float* acc_d,
                                  int64_t* acc_n,
                                  float* bound)
{
  auto const total = n_rows * k;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const r   = i / k;
    auto const src = sorted_order[i + cand_before[r]];
    float d;
    int64_t id;
    if (src < total) {
      d  = old_d[src];
      id = old_n[src];
    } else {
      d  = cand_d[src - total];
      id = cand_ids[src - total];
    }
    acc_d[i] = d;
    acc_n[i] = id;
    // A row still short of k is not unbounded: the bound it searched under still holds.
    if (i - r * k == k - 1) { bound[r] = fminf(bound[r], d); }
  }
}

__global__ void kth_bound_kernel(float const* acc_d, int64_t n_rows, int64_t k, float* bound)
{
  for (int64_t r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; r < n_rows;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    bound[r] = acc_d[r * k + k - 1];
  }
}

__global__ void sqrt_kernel(float* d, int64_t n)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    d[i] = sqrtf(d[i]);
  }
}

__global__ void fill_kernel(float* d, int64_t n, float value)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    d[i] = value;
  }
}

struct within_distance {
  float const* distances;
  float max_distance;
  __device__ bool operator()(int64_t i) const { return distances[i] <= max_distance; }
};

__global__ void take_kernel(int64_t const* picked,
                            int64_t n,
                            int32_t const* rows,
                            int64_t const* ids,
                            float const* distances,
                            int64_t const* id_map,
                            bool take_sqrt,
                            int32_t* out_rows,
                            int64_t* out_ids,
                            float* out_d)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const src = picked[i];
    out_rows[i]    = rows[src];
    out_ids[i]     = id_map != nullptr ? id_map[ids[src]] : ids[src];
    out_d[i]       = take_sqrt ? sqrtf(fmaxf(distances[src], 0.f)) : distances[src];
  }
}

int grid_for(int64_t n) { return static_cast<int>(std::clamp<int64_t>((n + 255) / 256, 1, 65535)); }

}  // namespace

bound_candidates::bound_candidates(int64_t capacity,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
  : rows(static_cast<std::size_t>(capacity), stream, mr),
    ids(static_cast<std::size_t>(capacity), stream, mr),
    distances(static_cast<std::size_t>(capacity), stream, mr),
    count(1, stream, mr)
{
  CUDF_CUDA_TRY(cudaMemsetAsync(count.data(), 0, sizeof(unsigned long long), stream.value()));
}

bool bound_filter_int8_supports(int64_t dim)
{
  // 16-byte loads of whole row segments, and exact int32 distances that also convert to float
  // exactly: 4 * 128^2 * dim < 2^24.
  return dim > 0 && dim % 16 == 0 && dim <= 256;
}

void bound_filter_int8(int8_t const* x,
                       int32_t const* x_sq,
                       int64_t n,
                       int64_t id_base,
                       int64_t const* id_map,
                       int8_t const* probe,
                       int32_t const* probe_sq,
                       int64_t const* rows,
                       int64_t m,
                       int64_t dim,
                       float const* bound,
                       bound_candidates& out,
                       rmm::cuda_stream_view stream)
{
  if (n == 0 || m == 0) { return; }
  CUDF_EXPECTS(bound_filter_int8_supports(dim), "bound_filter_int8: unsupported vector width");
  if (launch_small(x,
                   x_sq,
                   n,
                   id_base,
                   id_map,
                   probe,
                   probe_sq,
                   rows,
                   m,
                   dim,
                   bound,
                   slack_terms{},
                   out,
                   stream)) {
    return;
  }
  CUDF_EXPECTS((n + kTileN - 1) / kTileN <= 65535, "bound_filter_int8: too many corpus rows");
  dim3 const grid(static_cast<unsigned>((m + kTileM - 1) / kTileM),
                  static_cast<unsigned>((n + kTileN - 1) / kTileN));
  auto const kernel = accumulator_layout_is_m16n8<typename mma_traits<int8_t>::accumulator>()
                        ? bound_filter_kernel<int8_t, true>
                        : bound_filter_kernel<int8_t, false>;
  kernel<<<grid, kWarps * 32, 0, stream.value()>>>(x,
                                                   x_sq,
                                                   n,
                                                   id_base,
                                                   id_map,
                                                   probe,
                                                   probe_sq,
                                                   rows,
                                                   m,
                                                   static_cast<int>(dim),
                                                   bound,
                                                   slack_terms{},
                                                   out.rows.data(),
                                                   out.ids.data(),
                                                   out.distances.data(),
                                                   out.count.data(),
                                                   static_cast<unsigned long long>(out.capacity()),
                                                   group_view{});
  CUDF_CUDA_TRY(cudaGetLastError());
}

void bound_filter_group(std::vector<bound_slice> const& slices,
                        bool f16,
                        void const* probe,
                        void const* probe_sq,
                        int64_t dim,
                        float const* bound,
                        bound_candidates& out,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  // Classes 0..4 go to the few-rows kernel with M = 1, 2, 4, 8, 16 probe rows; class 5 to tiles.
  std::array<std::vector<bound_slice>, 6> classes;
  bool const small_ok = dim <= kSmallMaxDim;
  for (auto const& sl : slices) {
    if (sl.n == 0 || sl.m == 0) { continue; }
    std::size_t c = 5;
    if (small_ok && sl.m <= small_m_limit()) {
      c = sl.m <= 1 ? 0 : sl.m <= 2 ? 1 : sl.m <= 4 ? 2 : sl.m <= 8 ? 3 : 4;
    }
    classes[c].push_back(sl);
  }
  // The slice table and each slice's first block, copied beside each other; freed on the stream
  // once the launch that reads them is done.
  auto launch = [&](std::vector<bound_slice> const& v, bool tile, auto&& go) {
    if (v.empty()) { return; }
    std::vector<int64_t> prefix(v.size() + 1, 0);
    for (std::size_t i = 0; i < v.size(); ++i) {
      auto const blocks = tile ? ((v[i].m + kTileM - 1) / kTileM) * ((v[i].n + kTileN - 1) / kTileN)
                               : (v[i].n + kSmallThreads - 1) / kSmallThreads;
      prefix[i + 1]     = prefix[i] + blocks;
    }
    CUDF_EXPECTS(prefix.back() <= std::numeric_limits<int>::max(),
                 "bound_filter_group: too many blocks for one launch");
    auto const table_bytes = v.size() * sizeof(bound_slice);
    rmm::device_buffer table(table_bytes + prefix.size() * sizeof(int64_t), stream, mr);
    auto* device_prefix =
      reinterpret_cast<int64_t*>(static_cast<std::byte*>(table.data()) + table_bytes);
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(table.data(), v.data(), table_bytes, cudaMemcpyHostToDevice, stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(device_prefix,
                                  prefix.data(),
                                  prefix.size() * sizeof(int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    go(static_cast<unsigned>(prefix.back()),
       group_view{
         static_cast<bound_slice const*>(table.data()), device_prefix, static_cast<int>(v.size())});
    CUDF_CUDA_TRY(cudaGetLastError());
  };
  auto run = [&](auto element) {
    using T         = decltype(element);
    using N         = typename mma_traits<T>::norm;
    auto const* p   = static_cast<T const*>(probe);
    auto const* psq = static_cast<N const*>(probe_sq);
    auto small      = [&](std::size_t c, auto kernel) {
      launch(classes[c], false, [&](unsigned blocks, group_view g) {
        kernel<<<blocks, kSmallThreads, 0, stream.value()>>>(
          nullptr,
          nullptr,
          0,
          0,
          nullptr,
          p,
          psq,
          nullptr,
          0,
          static_cast<int>(dim),
          bound,
          slack_terms{},
          out.rows.data(),
          out.ids.data(),
          out.distances.data(),
          out.count.data(),
          static_cast<unsigned long long>(out.capacity()),
          g);
      });
    };
    small(0, bound_filter_small_kernel<T, 1, true>);
    small(1, bound_filter_small_kernel<T, 2, true>);
    small(2, bound_filter_small_kernel<T, 4, true>);
    small(3, bound_filter_small_kernel<T, 8, true>);
    small(4, bound_filter_small_kernel<T, 16, true>);
    auto const kernel = accumulator_layout_is_m16n8<typename mma_traits<T>::accumulator>()
                          ? bound_filter_kernel<T, true, true>
                          : bound_filter_kernel<T, false, true>;
    launch(classes[5], true, [&](unsigned blocks, group_view g) {
      kernel<<<blocks, kWarps * 32, 0, stream.value()>>>(
        nullptr,
        nullptr,
        0,
        0,
        nullptr,
        p,
        psq,
        nullptr,
        0,
        static_cast<int>(dim),
        bound,
        slack_terms{},
        out.rows.data(),
        out.ids.data(),
        out.distances.data(),
        out.count.data(),
        static_cast<unsigned long long>(out.capacity()),
        g);
    });
  };
  if (f16) {
    run(__half{});
  } else {
    run(int8_t{});
  }
}

bool bound_filter_f16_supports(int64_t dim) { return dim > 0 && dim % 16 == 0; }

void bound_filter_f16(std::uint16_t const* x,
                      float const* x_sq,
                      int64_t n,
                      int64_t id_base,
                      std::uint16_t const* probe,
                      float const* probe_sq,
                      int64_t const* rows,
                      int64_t m,
                      int64_t dim,
                      float const* bound,
                      float16_slack const& slack,
                      bound_candidates& out,
                      rmm::cuda_stream_view stream)
{
  if (n == 0 || m == 0) { return; }
  CUDF_EXPECTS(bound_filter_f16_supports(dim), "bound_filter_f16: unsupported vector width");
  if (launch_small(reinterpret_cast<__half const*>(x),
                   x_sq,
                   n,
                   id_base,
                   nullptr,
                   reinterpret_cast<__half const*>(probe),
                   probe_sq,
                   rows,
                   m,
                   dim,
                   bound,
                   slack_terms{slack.a, slack.b, slack.c, slack.x_max},
                   out,
                   stream)) {
    return;
  }
  CUDF_EXPECTS((n + kTileN - 1) / kTileN <= 65535, "bound_filter_f16: too many corpus rows");
  dim3 const grid(static_cast<unsigned>((m + kTileM - 1) / kTileM),
                  static_cast<unsigned>((n + kTileN - 1) / kTileN));
  auto const kernel = accumulator_layout_is_m16n8<typename mma_traits<__half>::accumulator>()
                        ? bound_filter_kernel<__half, true>
                        : bound_filter_kernel<__half, false>;
  kernel<<<grid, kWarps * 32, 0, stream.value()>>>(
    reinterpret_cast<__half const*>(x),
    x_sq,
    n,
    id_base,
    nullptr,
    reinterpret_cast<__half const*>(probe),
    probe_sq,
    rows,
    m,
    static_cast<int>(dim),
    bound,
    slack_terms{slack.a, slack.b, slack.c, slack.x_max},
    out.rows.data(),
    out.ids.data(),
    out.distances.data(),
    out.count.data(),
    static_cast<unsigned long long>(out.capacity()),
    group_view{});
  CUDF_CUDA_TRY(cudaGetLastError());
}

void exact_distances(float const* probe,
                     int32_t const* rows,
                     int64_t k,
                     int64_t const* ids,
                     float* distances,
                     int64_t n_pairs,
                     float const* const* blocks,
                     int64_t rows_per_block,
                     int64_t dim,
                     rmm::cuda_stream_view stream,
                     float const* certain_below,
                     float certain_value)
{
  if (n_pairs == 0) { return; }
  auto const warps_per_block = 8;
  auto const grid            = static_cast<int>(
    std::clamp<int64_t>((n_pairs + warps_per_block - 1) / warps_per_block, 1, 65535));
  exact_distances_kernel<<<grid, warps_per_block * 32, 0, stream.value()>>>(probe,
                                                                            rows,
                                                                            k,
                                                                            ids,
                                                                            distances,
                                                                            n_pairs,
                                                                            blocks,
                                                                            rows_per_block,
                                                                            static_cast<int>(dim),
                                                                            certain_below,
                                                                            certain_value);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void row_max_bound(
  float const* acc_distances, int64_t n_rows, int64_t k, float* bound, rmm::cuda_stream_view stream)
{
  if (n_rows == 0) { return; }
  row_max_kernel<<<grid_for(n_rows), 256, 0, stream.value()>>>(acc_distances, n_rows, k, bound);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void fill_misses(float* distances, int64_t* ids, int64_t n, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  fill_misses_kernel<<<grid_for(n), 256, 0, stream.value()>>>(distances, ids, n);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void map_ids(int64_t* ids, int64_t n, int64_t const* id_map, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  map_ids_kernel<<<grid_for(n), 256, 0, stream.value()>>>(ids, n, id_map);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void merge_bound_candidates(float* acc_distances,
                            int64_t* acc_neighbors,
                            int64_t n_rows,
                            int64_t k,
                            bound_candidates const& candidates,
                            int64_t n_candidates,
                            float* bound,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  if (n_candidates == 0) { return; }
  auto const acc_n = n_rows * k;
  auto const total = acc_n + n_candidates;
  rmm::device_uvector<uint64_t> keys(total, stream, mr), keys_sorted(total, stream, mr);
  rmm::device_uvector<int64_t> order(total, stream, mr), order_sorted(total, stream, mr);
  rmm::device_uvector<int32_t> per_row(n_rows + 1, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(per_row.data(), 0, per_row.size() * sizeof(int32_t), stream.value()));
  merge_keys_kernel<<<grid_for(total), 256, 0, stream.value()>>>(acc_distances,
                                                                 acc_n,
                                                                 k,
                                                                 candidates.rows.data(),
                                                                 candidates.distances.data(),
                                                                 n_candidates,
                                                                 keys.data(),
                                                                 order.data(),
                                                                 per_row.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // Radix sort is stable and the accumulator entries come first, so they win ties.
  int row_bits = 1;
  while ((int64_t{1} << row_bits) < n_rows) {
    ++row_bits;
  }
  std::size_t sort_bytes = 0, scan_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                sort_bytes,
                                                keys.data(),
                                                keys_sorted.data(),
                                                order.data(),
                                                order_sorted.data(),
                                                total,
                                                0,
                                                32 + row_bits,
                                                stream.value()));
  rmm::device_uvector<int32_t> cand_before(n_rows + 1, stream, mr);
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    nullptr, scan_bytes, per_row.data(), cand_before.data(), n_rows + 1, stream.value()));
  rmm::device_buffer temp(std::max(sort_bytes, scan_bytes), stream, mr);
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(temp.data(),
                                                sort_bytes,
                                                keys.data(),
                                                keys_sorted.data(),
                                                order.data(),
                                                order_sorted.data(),
                                                total,
                                                0,
                                                32 + row_bits,
                                                stream.value()));
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    temp.data(), scan_bytes, per_row.data(), cand_before.data(), n_rows + 1, stream.value()));

  rmm::device_uvector<float> old_d(acc_n, stream, mr);
  rmm::device_uvector<int64_t> old_n(acc_n, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    old_d.data(), acc_distances, acc_n * sizeof(float), cudaMemcpyDeviceToDevice, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(old_n.data(),
                                acc_neighbors,
                                acc_n * sizeof(int64_t),
                                cudaMemcpyDeviceToDevice,
                                stream.value()));
  merge_take_kernel<<<grid_for(acc_n), 256, 0, stream.value()>>>(order_sorted.data(),
                                                                 cand_before.data(),
                                                                 n_rows,
                                                                 k,
                                                                 old_d.data(),
                                                                 old_n.data(),
                                                                 candidates.distances.data(),
                                                                 candidates.ids.data(),
                                                                 acc_distances,
                                                                 acc_neighbors,
                                                                 bound);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void kth_distance_bound(
  float const* acc_distances, int64_t n_rows, int64_t k, float* bound, rmm::cuda_stream_view stream)
{
  if (n_rows == 0) { return; }
  kth_bound_kernel<<<grid_for(n_rows), 256, 0, stream.value()>>>(acc_distances, n_rows, k, bound);
  CUDF_CUDA_TRY(cudaGetLastError());
}

namespace {

__global__ void scale_kernel(float* d, int64_t n, float factor)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    d[i] *= factor;
  }
}

__global__ void normalize_rows_kernel(float const* x, int64_t n, int64_t dim, float* out)
{
  int const lane   = threadIdx.x % 32;
  auto const warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / 32);
  for (int64_t r = (blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x) / 32; r < n;
       r += warps) {
    float s = 0.f;
    for (int64_t c = lane; c < dim; c += 32) {
      float const v = x[r * dim + c];
      s             = fmaf(v, v, s);
    }
#pragma unroll
    for (int o = 16; o > 0; o /= 2) {
      s += __shfl_xor_sync(0xffffffffu, s, o);
    }
    float const inv = s > 0.f ? rsqrtf(s) : 0.f;
    for (int64_t c = lane; c < dim; c += 32) {
      out[r * dim + c] = x[r * dim + c] * inv;
    }
  }
}

}  // namespace

namespace {

constexpr int kSeedWarps = 8;

// k rounds of a warp argmin over a probe row's sample distances in shared memory; the last one
// taken is the k-th smallest. Consumes @p my.
__device__ float seed_kth(float* my, int cnt, int k, int lane)
{
  float kth = 0.f;
  for (int t = 0; t < k; ++t) {
    float best = __int_as_float(0x7f800000);
    int at     = -1;
    for (int i = lane; i < cnt; i += 32) {
      if (my[i] < best) {
        best = my[i];
        at   = i;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o /= 2) {
      float const ob = __shfl_xor_sync(0xffffffffu, best, o);
      int const oa   = __shfl_xor_sync(0xffffffffu, at, o);
      if (ob < best || (ob == best && oa > at)) {
        best = ob;
        at   = oa;
      }
    }
    kth = best;
    if (lane == 0 && at >= 0) { my[at] = __int_as_float(0x7f800000); }
    __syncwarp();
  }
  return kth;
}

// One warp per probe row, its sample's distances to shared memory, then seed_kth. W lanes share a
// sample row, each reading a contiguous span of it, so a warp's loads coalesce whatever the width.
template <int W>
__global__ void __launch_bounds__(kSeedWarps * 32)
  seed_bound_int8_kernel(int8_t const* __restrict__ x,
                         int32_t const* __restrict__ x_sq,
                         int8_t const* __restrict__ probe,
                         int32_t const* __restrict__ probe_sq,
                         int64_t const* __restrict__ first,
                         int32_t const* __restrict__ count,
                         int64_t const* __restrict__ order,
                         int64_t n,
                         int d,
                         int k,
                         float* __restrict__ bound)
{
  __shared__ float s_d[kSeedWarps][kSeedSample];
  int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int const sub = lane / W, sl = lane % W;
  float* my       = s_d[warp];
  int const words = d / 4;
  for (int64_t at = static_cast<int64_t>(blockIdx.x) * kSeedWarps + warp; at < n;
       at += static_cast<int64_t>(gridDim.x) * kSeedWarps) {
    auto const r  = order != nullptr ? order[at] : at;
    int const cnt = count[r];
    if (cnt < k) {
      if (lane == 0) { bound[r] = __int_as_float(0x7f800000); }
      continue;
    }
    auto const* q = reinterpret_cast<int const*>(probe + r * d);
    for (int i0 = 0; i0 < cnt; i0 += 32 / W) {
      int const i    = i0 + sub;
      auto const row = first[r] + (i < cnt ? i : 0);
      auto const* xr = reinterpret_cast<int const*>(x + row * d);
      int dot        = 0;
      for (int w = sl; w < words; w += W) {
        dot = __dp4a(xr[w], q[w], dot);
      }
#pragma unroll
      for (int o = W / 2; o > 0; o /= 2) {
        dot += __shfl_xor_sync(0xffffffffu, dot, o, W);
      }
      if (sl == 0 && i < cnt) { my[i] = static_cast<float>(probe_sq[r] + x_sq[row] - 2 * dot); }
    }
    __syncwarp();
    float const kth = seed_kth(my, cnt, k, lane);
    if (lane == 0) { bound[r] = kth; }
    __syncwarp();
  }
}

// The same over FLOAT16 rows, the dot accumulated in FP32.
template <int W>
__global__ void __launch_bounds__(kSeedWarps * 32)
  seed_bound_f16_kernel(uint16_t const* __restrict__ x,
                        float const* __restrict__ x_sq,
                        uint16_t const* __restrict__ probe,
                        float const* __restrict__ probe_sq,
                        int64_t const* __restrict__ first,
                        int32_t const* __restrict__ count,
                        int64_t const* __restrict__ order,
                        int64_t n,
                        int d,
                        int k,
                        float* __restrict__ bound)
{
  __shared__ float s_d[kSeedWarps][kSeedSample];
  int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int const sub = lane / W, sl = lane % W;
  float* my       = s_d[warp];
  int const pairs = d / 2;
  for (int64_t at = static_cast<int64_t>(blockIdx.x) * kSeedWarps + warp; at < n;
       at += static_cast<int64_t>(gridDim.x) * kSeedWarps) {
    auto const r  = order != nullptr ? order[at] : at;
    int const cnt = count[r];
    if (cnt < k) {
      if (lane == 0) { bound[r] = __int_as_float(0x7f800000); }
      continue;
    }
    auto const* q = reinterpret_cast<__half2 const*>(probe + r * d);
    for (int i0 = 0; i0 < cnt; i0 += 32 / W) {
      int const i    = i0 + sub;
      auto const row = first[r] + (i < cnt ? i : 0);
      auto const* xr = reinterpret_cast<__half2 const*>(x + row * d);
      float dot      = 0.f;
      for (int w = sl; w < pairs; w += W) {
        float2 const a = __half22float2(xr[w]);
        float2 const b = __half22float2(q[w]);
        dot            = fmaf(a.x, b.x, fmaf(a.y, b.y, dot));
      }
#pragma unroll
      for (int o = W / 2; o > 0; o /= 2) {
        dot += __shfl_xor_sync(0xffffffffu, dot, o, W);
      }
      if (sl == 0 && i < cnt) { my[i] = probe_sq[r] + x_sq[row] - 2.f * dot; }
    }
    __syncwarp();
    float const kth = seed_kth(my, cnt, k, lane);
    if (lane == 0) { bound[r] = kth; }
    __syncwarp();
  }
}

// Lanes per sample row: about eight 4-byte words each, between 4 and 32.
int seed_width(int64_t words)
{
  int w = 4;
  while (w < 32 && words >= 16 * w) {
    w *= 2;
  }
  return w;
}

template <typename Launch>
void launch_seed_width(int width, Launch&& launch)
{
  switch (width) {
    case 4: launch(std::integral_constant<int, 4>{}); break;
    case 8: launch(std::integral_constant<int, 8>{}); break;
    case 16: launch(std::integral_constant<int, 16>{}); break;
    default: launch(std::integral_constant<int, 32>{}); break;
  }
}

}  // namespace

void seed_bound_int8(int8_t const* x,
                     int32_t const* x_sq,
                     int8_t const* probe,
                     int32_t const* probe_sq,
                     int64_t const* first,
                     int32_t const* count,
                     int64_t const* order,
                     int64_t n,
                     int64_t dim,
                     int k,
                     float* bound,
                     rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  auto const grid =
    static_cast<int>(std::clamp<int64_t>((n + kSeedWarps - 1) / kSeedWarps, 1, 65535));
  launch_seed_width(seed_width(dim / 4), [&](auto w) {
    seed_bound_int8_kernel<decltype(w)::value><<<grid, kSeedWarps * 32, 0, stream.value()>>>(
      x, x_sq, probe, probe_sq, first, count, order, n, static_cast<int>(dim), k, bound);
  });
  CUDF_CUDA_TRY(cudaGetLastError());
}

void seed_bound_f16(uint16_t const* x,
                    float const* x_sq,
                    uint16_t const* probe,
                    float const* probe_sq,
                    int64_t const* first,
                    int32_t const* count,
                    int64_t const* order,
                    int64_t n,
                    int64_t dim,
                    int k,
                    float* bound,
                    rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  auto const grid =
    static_cast<int>(std::clamp<int64_t>((n + kSeedWarps - 1) / kSeedWarps, 1, 65535));
  launch_seed_width(seed_width(dim / 2), [&](auto w) {
    seed_bound_f16_kernel<decltype(w)::value><<<grid, kSeedWarps * 32, 0, stream.value()>>>(
      x, x_sq, probe, probe_sq, first, count, order, n, static_cast<int>(dim), k, bound);
  });
  CUDF_CUDA_TRY(cudaGetLastError());
}

void scale_in_place(float* d, int64_t n, float factor, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  scale_kernel<<<grid_for(n), 256, 0, stream.value()>>>(d, n, factor);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void normalize_rows(
  float const* x, int64_t n, int64_t dim, float* out, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  auto const grid = static_cast<int>(std::clamp<int64_t>((n + 7) / 8, 1, 65535));
  normalize_rows_kernel<<<grid, 256, 0, stream.value()>>>(x, n, dim, out);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void sqrt_in_place(float* d, int64_t n, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  sqrt_kernel<<<grid_for(n), 256, 0, stream.value()>>>(d, n);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void fill_bound(float* bound, int64_t n, float value, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  fill_kernel<<<grid_for(n), 256, 0, stream.value()>>>(bound, n, value);
  CUDF_CUDA_TRY(cudaGetLastError());
}

radius_pairs take_within(bound_candidates const& candidates,
                         int64_t n_candidates,
                         float max_distance,
                         int64_t const* id_map,
                         bool take_sqrt,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  radius_pairs out{rmm::device_uvector<int32_t>(0, stream, mr),
                   rmm::device_uvector<int64_t>(0, stream, mr),
                   rmm::device_uvector<float>(0, stream, mr)};
  if (n_candidates == 0) { return out; }
  rmm::device_uvector<int64_t> picked(n_candidates, stream, mr);
  rmm::device_uvector<int64_t> n_picked(1, stream, mr);
  thrust::counting_iterator<int64_t> const all(0);
  within_distance const keep{candidates.distances.data(), max_distance};
  std::size_t temp_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceSelect::If(
    nullptr, temp_bytes, all, picked.data(), n_picked.data(), n_candidates, keep, stream.value()));
  rmm::device_buffer temp(temp_bytes, stream, mr);
  CUDF_CUDA_TRY(cub::DeviceSelect::If(temp.data(),
                                      temp_bytes,
                                      all,
                                      picked.data(),
                                      n_picked.data(),
                                      n_candidates,
                                      keep,
                                      stream.value()));
  int64_t n = 0;
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(&n, n_picked.data(), sizeof(n), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  if (n == 0) { return out; }
  out.rows.resize(n, stream);
  out.ids.resize(n, stream);
  out.distances.resize(n, stream);
  take_kernel<<<grid_for(n), 256, 0, stream.value()>>>(picked.data(),
                                                       n,
                                                       candidates.rows.data(),
                                                       candidates.ids.data(),
                                                       candidates.distances.data(),
                                                       id_map,
                                                       take_sqrt,
                                                       out.rows.data(),
                                                       out.ids.data(),
                                                       out.distances.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  return out;
}

}  // namespace sirius::vss
