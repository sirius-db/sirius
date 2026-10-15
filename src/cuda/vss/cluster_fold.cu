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

#include "vss/cluster_fold.hpp"
#include "vss/cluster_lists.hpp"

#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sirius::vss {

namespace {

constexpr int kBlock = 256;

int grid_for(int64_t n)
{
  return static_cast<int>(std::max<int64_t>((n + kBlock - 1) / kBlock, 1));
}

__global__ void gather_rows_kernel(
  float const* src, int64_t dim, int64_t const* rows, int64_t m, float* out)
{
  auto const total = m * dim;
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const i = p / dim;
    out[p]       = src[rows[i] * dim + (p - i * dim)];
  }
}

// One thread per part row. The merged row is written back into the accumulator row it was
// read from, which is safe back to front: first count how many of the k survivors come from
// the accumulator, then fill positions k-1 .. 0 with the larger remaining head. Every write
// lands at or after the accumulator entry still to be read, so nothing is overwritten unread
// and no scratch row is needed.
__global__ void fold_topk_rows_kernel(float* acc_d,
                                      int64_t* acc_n,
                                      int64_t k,
                                      float const* part_d,
                                      int64_t const* part_n,
                                      int64_t part_width,
                                      int64_t k_eff,
                                      int64_t const* rows,
                                      int64_t m,
                                      int64_t id_base,
                                      int64_t const* id_map)
{
  auto const i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= m) { return; }
  float* ad      = acc_d + rows[i] * k;
  int64_t* an    = acc_n + rows[i] * k;
  auto const* pd = part_d + i * part_width;
  auto const* pn = part_n + i * part_width;

  // Forward pass, reads only: ties go to the accumulator.
  int64_t ia = 0;
  int64_t ib = 0;
  while (ia + ib < k) {
    if (ib < k_eff && (ia >= k || pd[ib] < ad[ia])) {
      ++ib;
    } else {
      ++ia;
    }
  }
  if (ib == 0) { return; }

  // Backward pass: of the two heads, the later one in the forward order is written last. On a
  // tie that is the part's, mirroring the forward rule.
  for (int64_t p = k - 1; p >= 0; --p) {
    if (ia == 0 || (ib > 0 && pd[ib - 1] >= ad[ia - 1])) {
      --ib;
      ad[p] = pd[ib];
      an[p] = id_map != nullptr ? id_map[pn[ib]] : pn[ib] + id_base;
    } else {
      --ia;
      ad[p] = ad[ia];
      an[p] = an[ia];
    }
  }
}

// One warp per probe row, one lane per slice row: each lane takes a whole dot product, reading its
// row from a shared tile (rows padded to an odd word stride, so the lanes' reads fall in distinct
// banks) and the probe row from shared memory (one address, broadcast). Lanes meet only when a
// row beats the probe's current k-th: lane 0 keeps the top k and broadcasts its threshold.
template <bool F16>
__global__ void nearest_cluster_topk_kernel(nearest_task const* tasks,
                                            std::uint32_t const* compact,
                                            int64_t row_words,
                                            int64_t tile_rows,
                                            void const* row_sq,
                                            int64_t const* routed_rows,
                                            std::uint32_t const* probes,
                                            void const* probe_sq,
                                            int64_t k,
                                            float* acc_d,
                                            int64_t* acc_n)
{
  constexpr int kMaxK          = 64;
  constexpr unsigned kFullMask = 0xffffffffu;
  extern __shared__ std::uint32_t smem[];
  auto const task     = tasks[blockIdx.x];
  int const warp      = static_cast<int>(threadIdx.x / 32);
  int const lane      = static_cast<int>(threadIdx.x % 32);
  int const n_warps   = static_cast<int>(blockDim.x / 32);
  auto const stride   = row_words + 1;
  auto* probe_rows    = smem;
  auto* tile          = smem + n_warps * row_words;
  bool const active   = warp < task.n_probes;
  int64_t const probe = active ? routed_rows[task.edge_begin + warp] : 0;
  auto* my_probe      = probe_rows + warp * row_words;
  if (active) {
    for (int64_t w = lane; w < row_words; w += 32) {
      my_probe[w] = probes[probe * row_words + w];
    }
  }
  float const qs = !active ? 0.f
                   : F16   ? static_cast<float const*>(probe_sq)[probe]
                           : static_cast<float>(static_cast<int32_t const*>(probe_sq)[probe]);

  float best_d[kMaxK];
  int64_t best_i[kMaxK];
  int filled      = 0;
  int worst_at    = 0;
  float thr       = INFINITY;  // admits everything until k are held, then the k-th distance
  auto const* src = compact + task.row_offset * row_words;
  for (int64_t r0 = 0; r0 < task.n_rows; r0 += tile_rows) {
    auto const rn = std::min<int64_t>(tile_rows, task.n_rows - r0);
    __syncthreads();
    for (int64_t i = threadIdx.x; i < rn * row_words; i += blockDim.x) {
      auto const r                         = i / row_words;
      tile[r * stride + i - r * row_words] = src[r0 * row_words + i];
    }
    __syncthreads();
    if (!active) { continue; }
    for (int64_t rb = 0; rb < rn; rb += 32) {
      auto const r  = rb + lane;
      bool const ok = r < rn;
      float s       = INFINITY;
      if (ok) {
        auto const* row = tile + r * stride;
        float dot;
        if constexpr (F16) {
          float acc = 0.f;
          for (int64_t w = 0; w < row_words; ++w) {
            std::uint32_t const a = row[w];
            std::uint32_t const b = my_probe[w];
            float2 const fa       = __half22float2(*reinterpret_cast<__half2 const*>(&a));
            float2 const fb       = __half22float2(*reinterpret_cast<__half2 const*>(&b));
            acc                   = fmaf(fa.x, fb.x, fmaf(fa.y, fb.y, acc));
          }
          dot = acc;
        } else {
          int acc = 0;
          for (int64_t w = 0; w < row_words; ++w) {
            acc = __dp4a(static_cast<int>(row[w]), static_cast<int>(my_probe[w]), acc);
          }
          dot = static_cast<float>(acc);
        }
        auto const id  = task.id_base + r0 + r;
        float const x2 = F16 ? static_cast<float const*>(row_sq)[id]
                             : static_cast<float>(static_cast<int32_t const*>(row_sq)[id]);
        s              = fmaxf(x2 - 2.f * dot + qs, 0.f);
      }
      unsigned mask = __ballot_sync(kFullMask, s < thr);
      while (mask != 0) {
        int const from = __ffs(mask) - 1;
        mask &= mask - 1;
        float const cs = __shfl_sync(kFullMask, s, from);
        if (lane == 0) {
          auto const cid = task.id_base + r0 + rb + from;
          if (filled < k) {
            best_d[filled] = cs;
            best_i[filled] = cid;
            ++filled;
            if (filled == k) {
              worst_at = 0;
              for (int i = 1; i < filled; ++i) {
                if (best_d[i] > best_d[worst_at]) { worst_at = i; }
              }
              thr = best_d[worst_at];
            }
          } else if (cs < thr) {
            best_d[worst_at] = cs;
            best_i[worst_at] = cid;
            worst_at         = 0;
            for (int i = 1; i < filled; ++i) {
              if (best_d[i] > best_d[worst_at]) { worst_at = i; }
            }
            thr = best_d[worst_at];
          }
        }
        thr = __shfl_sync(kFullMask, thr, 0);
      }
    }
  }
  if (!active || lane != 0) { return; }
  for (int i = 1; i < filled; ++i) {
    auto const d  = best_d[i];
    auto const id = best_i[i];
    int j         = i - 1;
    while (j >= 0 && best_d[j] > d) {
      best_d[j + 1] = best_d[j];
      best_i[j + 1] = best_i[j];
      --j;
    }
    best_d[j + 1] = d;
    best_i[j + 1] = id;
  }
  for (int i = 0; i < filled; ++i) {
    acc_d[probe * k + i] = best_d[i];
    acc_n[probe * k + i] = best_i[i];
  }
}

__global__ void remap_radius_edges_kernel(int64_t const* query_rows,
                                          int64_t const* rows,
                                          int32_t* left,
                                          int64_t* neighbors,
                                          int64_t n_edges,
                                          int64_t id_base,
                                          int64_t const* id_map)
{
  for (int64_t e = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; e < n_edges;
       e += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    left[e]      = static_cast<int32_t>(rows[query_rows[e]]);
    neighbors[e] = id_map != nullptr ? id_map[neighbors[e]] : neighbors[e] + id_base;
  }
}

__global__ void scatter_row_ids_kernel(int64_t const* dest,
                                       int64_t n,
                                       int64_t row_base,
                                       int64_t* row_ids)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    row_ids[dest[i]] = row_base + i;
  }
}

template <typename T>
__global__ void scatter_rows_kernel(
  T const* src, int64_t width, int64_t const* dest, int64_t n, T* out)
{
  auto const total = n * width;
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const i                           = p / width;
    out[dest[i] * width + (p - i * width)] = src[p];
  }
}

__global__ void count_non_uint8_kernel(float const* v, int64_t n, unsigned long long* out)
{
  unsigned long long bad = 0;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const x = v[i];
    bad += !(x >= 0.f && x <= 255.f && x == rintf(x));
  }
  if (bad != 0) { atomicAdd(out, bad); }
}

// Byte-valued lists are stored shifted to int8 (x - 128): L2 is invariant under the shift, and
// int8 is what the tensor-core GEMM multiplies exactly.
__global__ void narrow_to_shifted_int8_kernel(float const* in, int64_t n, int8_t* out)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = static_cast<int8_t>(static_cast<int>(in[i]) - 128);
  }
}

__global__ void widen_shifted_int8_kernel(int8_t const* in, int64_t n, float* out)
{
  // Four bytes in, four floats out per thread: the widening runs at device bandwidth.
  auto const n4   = n / 4;
  auto const* in4 = reinterpret_cast<char4 const*>(in);
  auto* out4      = reinterpret_cast<float4*>(out);
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n4;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const b = in4[i];
    out4[i]      = make_float4(b.x + 128.f, b.y + 128.f, b.z + 128.f, b.w + 128.f);
  }
  for (int64_t i = n4 * 4 + blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = in[i] + 128.f;
  }
}

// |x|^2 of shifted int8 rows, exact in int32 (at most dim * 128^2), one warp per row.
__global__ void int8_row_sq_norms_kernel(int8_t const* x, int64_t rows, int64_t d, int32_t* out)
{
  auto const warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / 32);
  auto const lane  = static_cast<int64_t>(threadIdx.x % 32);
  for (int64_t r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; r < rows; r += warps) {
    int32_t acc = 0;
    for (int64_t j = lane; j < d; j += 32) {
      int32_t const v = x[r * d + j];
      acc += v * v;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) { out[r] = acc; }
  }
}

__global__ void float_row_sq_norms_kernel(
  float const* x, int64_t rows, int64_t d, float* out, unsigned int* max_bits)
{
  auto const warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / 32);
  auto const lane  = static_cast<int64_t>(threadIdx.x % 32);
  for (int64_t r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; r < rows; r += warps) {
    float acc = 0.f;
    for (int64_t j = lane; j < d; j += 32) {
      float const v = x[r * d + j];
      acc           = fmaf(v, v, acc);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
      out[r] = acc;
      // Non-negative floats order as their bit patterns.
      atomicMax(max_bits, __float_as_uint(acc));
    }
  }
}

__global__ void gather_bytes_kernel(
  uint8_t const* src, int64_t row_bytes, int64_t const* rows, int64_t m, uint8_t* out)
{
  auto const total = m * row_bytes;
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const i = p / row_bytes;
    out[p]       = src[rows[i] * row_bytes + (p - i * row_bytes)];
  }
}

__global__ void gather_int32_kernel(int32_t const* src,
                                    int64_t const* rows,
                                    int64_t m,
                                    int32_t* out)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < m;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = src[rows[i]];
  }
}

__global__ void narrow_to_float16_kernel(float const* in, int64_t n, uint16_t* out)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = __half_as_ushort(__float2half_rn(in[i]));
  }
}

__global__ void widen_float16_kernel(uint16_t const* in, int64_t n, float* out)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = __half2float(__ushort_as_half(in[i]));
  }
}

// Order-preserving float <-> uint encoding, so atomicMin/atomicMax on the bits order the floats.
__device__ __forceinline__ unsigned int ordered_bits(float f)
{
  auto const b = __float_as_uint(f);
  return (b & 0x80000000u) != 0 ? ~b : (b | 0x80000000u);
}

// One thread per component, a block per span of rows: one atomic per (block, component).
__global__ void column_min_max_kernel(
  float const* x, int64_t rows, int64_t d, int64_t rows_per_block, unsigned* min_b, unsigned* max_b)
{
  auto const r0 = static_cast<int64_t>(blockIdx.x) * rows_per_block;
  auto const r1 = min(rows, r0 + rows_per_block);
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
    float lo = INFINITY, hi = -INFINITY;
    for (int64_t r = r0; r < r1; ++r) {
      float const v = x[r * d + j];
      lo            = fminf(lo, v);
      hi            = fmaxf(hi, v);
    }
    if (r1 > r0) {
      atomicMin(min_b + j, ordered_bits(lo));
      atomicMax(max_b + j, ordered_bits(hi));
    }
  }
}

// One warp per row.
__global__ void quantize_rows_int8_kernel(float const* x,
                                          int64_t rows,
                                          int64_t d,
                                          float const* offset,
                                          float scale,
                                          int8_t* out,
                                          float* row_error,
                                          unsigned* max_error_bits)
{
  auto const warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / 32);
  auto const lane  = static_cast<int64_t>(threadIdx.x % 32);
  float const inv  = 1.f / scale;
  for (int64_t r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; r < rows; r += warps) {
    float err = 0.f;
    for (int64_t j = lane; j < d; j += 32) {
      float const v  = x[r * d + j];
      float const q  = fminf(fmaxf(rintf((v - offset[j]) * inv), -127.f), 127.f);
      out[r * d + j] = static_cast<int8_t>(q);
      float const e  = v - fmaf(scale, q, offset[j]);
      err            = fmaf(e, e, err);
    }
    for (int o = 16; o > 0; o /= 2) {
      err += __shfl_xor_sync(0xffffffffu, err, o);
    }
    if (lane == 0) {
      // Rounded up by a relative 1e-6 so the float sum cannot understate the error it bounds.
      float const e = sqrtf(err) * (1.f + 1e-6f);
      if (row_error != nullptr) { row_error[r] = e; }
      if (max_error_bits != nullptr) { atomicMax(max_error_bits, __float_as_uint(e)); }
    }
  }
}

__global__ void int8_code_limit_kernel(float const* bound,
                                       float const* probe_error,
                                       float code_error,
                                       float scale,
                                       int64_t n,
                                       float* limit,
                                       bool lower)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float const b = bound[i];
    if (!(b < INFINITY)) {
      limit[i] = INFINITY;
      continue;
    }
    if (lower) {
      float const reach = (sqrtf(fmaxf(b, 0.f)) - probe_error[i] - code_error) / scale;
      limit[i]          = reach > 0.f ? reach * reach * (1.f - 1e-6f) : -INFINITY;
      continue;
    }
    float const reach = (sqrtf(fmaxf(b, 0.f)) + probe_error[i] + code_error) / scale;
    limit[i]          = reach * reach * (1.f + 1e-6f);
  }
}

// |half(x)|^2 per row, with the row's rounding error |x - half(x)|: to row_error[r] and/or into
// *max_error_bits, and the largest |half(x)|^2 into *max_norm_bits (all non-negative floats' bits).
__global__ void half_rows_norms_kernel(float const* x,
                                       uint16_t const* h,
                                       int64_t rows,
                                       int64_t d,
                                       float* sq,
                                       float* row_error,
                                       unsigned* max_error_bits,
                                       unsigned* max_norm_bits)
{
  auto const warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / 32);
  auto const lane  = static_cast<int64_t>(threadIdx.x % 32);
  for (int64_t r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; r < rows; r += warps) {
    float acc = 0.f, err = 0.f;
    for (int64_t j = lane; j < d; j += 32) {
      float const v = __half2float(__ushort_as_half(h[r * d + j]));
      float const e = x[r * d + j] - v;
      acc           = fmaf(v, v, acc);
      err           = fmaf(e, e, err);
    }
    for (int o = 16; o > 0; o /= 2) {
      acc += __shfl_xor_sync(0xffffffffu, acc, o);
      err += __shfl_xor_sync(0xffffffffu, err, o);
    }
    if (lane == 0) {
      sq[r]         = acc;
      float const e = sqrtf(err) * (1.f + 1e-6f);
      if (row_error != nullptr) { row_error[r] = e; }
      if (max_error_bits != nullptr) { atomicMax(max_error_bits, __float_as_uint(e)); }
      if (max_norm_bits != nullptr) { atomicMax(max_norm_bits, __float_as_uint(acc)); }
    }
  }
}

// FP16 rows are compared as |half(q)|^2 + |half(x)|^2 - 2 half(q).half(x) in FP32; that is the
// squared distance of the rounded pair up to the FP32 sums' error, at most
// 2 d 2^-24 (|half(q)| + max |half(x)|)^2 (twice the textbook bound: the tensor cores' order is
// unspecified). The rounded pair is within e_q + e_x of the true one by the triangle inequality.
__global__ void float16_bound_limit_kernel(float const* bound,
                                           float const* probe_sq,
                                           float const* probe_error,
                                           float row_error,
                                           float row_norm,
                                           int64_t d,
                                           int64_t n,
                                           float* limit,
                                           bool lower)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float const b = bound[i];
    if (!(b < INFINITY)) {
      limit[i] = INFINITY;
      continue;
    }
    float const norms = sqrtf(probe_sq[i]) + row_norm;
    float const sums  = 2.f * static_cast<float>(d) * 5.9604645e-8f * norms * norms;
    if (lower) {
      float const reach = sqrtf(fmaxf(b, 0.f)) - probe_error[i] - row_error;
      limit[i]          = reach > 0.f ? (reach * reach - sums) * (1.f - 1e-6f) : -INFINITY;
      continue;
    }
    float const reach = sqrtf(fmaxf(b, 0.f)) + probe_error[i] + row_error;
    limit[i]          = (reach * reach + sums) * (1.f + 1e-6f);
  }
}

__global__ void int8_seed_upper_bound_kernel(
  float* bound, float const* probe_error, float code_error, float scale, int64_t n)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float const b = bound[i];
    if (!(b < INFINITY)) { continue; }
    float const reach = scale * sqrtf(fmaxf(b, 0.f)) + probe_error[i] + code_error;
    bound[i]          = reach * reach * (1.f + 1e-6f);
  }
}

__global__ void float16_seed_upper_bound_kernel(float* bound,
                                                float const* probe_sq,
                                                float const* probe_error,
                                                float row_error,
                                                float row_norm,
                                                int64_t d,
                                                int64_t n)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float const b = bound[i];
    if (!(b < INFINITY)) { continue; }
    float const norms = sqrtf(probe_sq[i]) + row_norm;
    float const sums  = 2.f * static_cast<float>(d) * 5.9604645e-8f * norms * norms;
    float const reach = sqrtf(fmaxf(b + sums, 0.f)) + probe_error[i] + row_error;
    bound[i]          = reach * reach * (1.f + 1e-6f);
  }
}

__global__ void fill_list_offsets_kernel(int32_t* out, int64_t n, int64_t dim)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i <= n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = static_cast<int32_t>(i * dim);
  }
}

}  // namespace

void scatter_row_ids(
  int64_t const* dest, int64_t n, int64_t row_base, int64_t* row_ids, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  auto const grid = std::min(grid_for(n), 65535);
  scatter_row_ids_kernel<<<grid, kBlock, 0, stream.value()>>>(dest, n, row_base, row_ids);
  CUDF_CHECK_CUDA(stream.value());
}

void scatter_rows(std::byte const* src,
                  int64_t row_bytes,
                  int64_t const* dest,
                  int64_t n,
                  std::byte* out,
                  rmm::cuda_stream_view stream)
{
  if (n == 0 || row_bytes == 0) { return; }
  // Whole words when the rows are made of them, which every encoding but an odd-width UINT8 is.
  auto const words = row_bytes % 4 == 0 && reinterpret_cast<std::uintptr_t>(src) % 4 == 0 &&
                     reinterpret_cast<std::uintptr_t>(out) % 4 == 0;
  if (words) {
    auto const width = row_bytes / 4;
    scatter_rows_kernel<uint32_t>
      <<<std::min(grid_for(n * width), 65535), kBlock, 0, stream.value()>>>(
        reinterpret_cast<uint32_t const*>(src), width, dest, n, reinterpret_cast<uint32_t*>(out));
  } else {
    scatter_rows_kernel<uint8_t>
      <<<std::min(grid_for(n * row_bytes), 65535), kBlock, 0, stream.value()>>>(
        reinterpret_cast<uint8_t const*>(src), row_bytes, dest, n, reinterpret_cast<uint8_t*>(out));
  }
  CUDF_CHECK_CUDA(stream.value());
}

void count_non_uint8(float const* values,
                     int64_t n,
                     unsigned long long* out,
                     rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  count_non_uint8_kernel<<<std::min(grid_for(n), 4096), kBlock, 0, stream.value()>>>(
    values, n, out);
  CUDF_CHECK_CUDA(stream.value());
}

void narrow_to_shifted_int8(float const* in, int64_t n, int8_t* out, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  narrow_to_shifted_int8_kernel<<<std::min(grid_for(n), 65535), kBlock, 0, stream.value()>>>(
    in, n, out);
  CUDF_CHECK_CUDA(stream.value());
}

void widen_shifted_int8(int8_t const* in, int64_t n, float* out, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  // char4/float4 access needs 4- and 16-byte alignment; the callers' buffers start at
  // allocation boundaries and chunks start on whole rows of a dim that is a multiple of 4.
  CUDF_EXPECTS(
    reinterpret_cast<uintptr_t>(in) % 4 == 0 && reinterpret_cast<uintptr_t>(out) % 16 == 0,
    "widen_shifted_int8: misaligned buffers");
  widen_shifted_int8_kernel<<<std::min(grid_for(n / 4 + 1), 65535), kBlock, 0, stream.value()>>>(
    in, n, out);
  CUDF_CHECK_CUDA(stream.value());
}

void int8_row_sq_norms(
  int8_t const* x, int64_t rows, int64_t d, int32_t* out, rmm::cuda_stream_view stream)
{
  if (rows == 0) { return; }
  auto const grid = static_cast<int>(std::min<int64_t>((rows + 7) / 8, 65535));
  int8_row_sq_norms_kernel<<<grid, kBlock, 0, stream.value()>>>(x, rows, d, out);
  CUDF_CHECK_CUDA(stream.value());
}

void float_row_sq_norms(float const* x,
                        int64_t rows,
                        int64_t d,
                        float* out,
                        unsigned int* max_bits,
                        rmm::cuda_stream_view stream)
{
  if (rows == 0) { return; }
  auto const grid = static_cast<int>(std::min<int64_t>((rows + 7) / 8, 65535));
  float_row_sq_norms_kernel<<<grid, kBlock, 0, stream.value()>>>(x, rows, d, out, max_bits);
  CUDF_CHECK_CUDA(stream.value());
}

void gather_bytes(void const* src,
                  int64_t row_bytes,
                  int64_t const* rows,
                  int64_t m,
                  void* out,
                  rmm::cuda_stream_view stream)
{
  if (m == 0) { return; }
  gather_bytes_kernel<<<std::min(grid_for(m * row_bytes), 65535), kBlock, 0, stream.value()>>>(
    static_cast<uint8_t const*>(src), row_bytes, rows, m, static_cast<uint8_t*>(out));
  CUDF_CHECK_CUDA(stream.value());
}

void gather_int32(
  int32_t const* src, int64_t const* rows, int64_t m, int32_t* out, rmm::cuda_stream_view stream)
{
  if (m == 0) { return; }
  gather_int32_kernel<<<std::min(grid_for(m), 65535), kBlock, 0, stream.value()>>>(
    src, rows, m, out);
  CUDF_CHECK_CUDA(stream.value());
}

void narrow_to_float16(float const* in, int64_t n, uint16_t* out, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  narrow_to_float16_kernel<<<std::min(grid_for(n), 65535), kBlock, 0, stream.value()>>>(in, n, out);
  CUDF_CHECK_CUDA(stream.value());
}

void widen_float16(uint16_t const* in, int64_t n, float* out, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  widen_float16_kernel<<<std::min(grid_for(n), 65535), kBlock, 0, stream.value()>>>(in, n, out);
  CUDF_CHECK_CUDA(stream.value());
}

void column_min_max(float const* x,
                    int64_t rows,
                    int64_t d,
                    unsigned int* min_bits,
                    unsigned int* max_bits,
                    rmm::cuda_stream_view stream)
{
  if (rows == 0) { return; }
  constexpr int64_t kRowsPerBlock = 4096;
  auto const blocks               = static_cast<int>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
  column_min_max_kernel<<<blocks, 128, 0, stream.value()>>>(
    x, rows, d, kRowsPerBlock, min_bits, max_bits);
  CUDF_CHECK_CUDA(stream.value());
}

float decode_ordered_float(unsigned int bits)
{
  unsigned int const b = (bits & 0x80000000u) != 0 ? (bits & 0x7fffffffu) : ~bits;
  float f;
  std::memcpy(&f, &b, sizeof(f));
  return f;
}

void quantize_rows_int8(float const* x,
                        int64_t rows,
                        int64_t d,
                        float const* offset,
                        float scale,
                        int8_t* out,
                        float* row_error,
                        unsigned int* max_error_bits,
                        rmm::cuda_stream_view stream)
{
  if (rows == 0) { return; }
  auto const grid = static_cast<int>(std::min<int64_t>((rows + 7) / 8, 65535));
  quantize_rows_int8_kernel<<<grid, 256, 0, stream.value()>>>(
    x, rows, d, offset, scale, out, row_error, max_error_bits);
  CUDF_CHECK_CUDA(stream.value());
}

void int8_code_limit(float const* bound,
                     float const* probe_error,
                     float code_error,
                     float scale,
                     int64_t n,
                     float* limit,
                     rmm::cuda_stream_view stream,
                     bool lower)
{
  if (n == 0) { return; }
  auto const grid = std::min(grid_for(n), 65535);
  int8_code_limit_kernel<<<grid, kBlock, 0, stream.value()>>>(
    bound, probe_error, code_error, scale, n, limit, lower);
  CUDF_CHECK_CUDA(stream.value());
}

void int8_seed_upper_bound(float* bound,
                           float const* probe_error,
                           float code_error,
                           float scale,
                           int64_t n,
                           rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  int8_seed_upper_bound_kernel<<<std::min(grid_for(n), 65535), kBlock, 0, stream.value()>>>(
    bound, probe_error, code_error, scale, n);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void float16_seed_upper_bound(float* bound,
                              float const* probe_sq,
                              float const* probe_error,
                              float row_error,
                              float row_norm,
                              int64_t d,
                              int64_t n,
                              rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  float16_seed_upper_bound_kernel<<<std::min(grid_for(n), 65535), kBlock, 0, stream.value()>>>(
    bound, probe_sq, probe_error, row_error, row_norm, d, n);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void half_rows_norms(float const* x,
                     uint16_t const* h,
                     int64_t rows,
                     int64_t d,
                     float* sq,
                     float* row_error,
                     unsigned int* max_error_bits,
                     unsigned int* max_norm_bits,
                     rmm::cuda_stream_view stream)
{
  if (rows == 0) { return; }
  auto const grid = static_cast<int>(std::min<int64_t>((rows + 7) / 8, 65535));
  half_rows_norms_kernel<<<grid, 256, 0, stream.value()>>>(
    x, h, rows, d, sq, row_error, max_error_bits, max_norm_bits);
  CUDF_CHECK_CUDA(stream.value());
}

void float16_bound_limit(float const* bound,
                         float const* probe_sq,
                         float const* probe_error,
                         float row_error,
                         float row_norm,
                         int64_t d,
                         int64_t n,
                         float* limit,
                         rmm::cuda_stream_view stream,
                         bool lower)
{
  if (n == 0) { return; }
  auto const grid = std::min(grid_for(n), 65535);
  float16_bound_limit_kernel<<<grid, kBlock, 0, stream.value()>>>(
    bound, probe_sq, probe_error, row_error, row_norm, d, n, limit, lower);
  CUDF_CHECK_CUDA(stream.value());
}

void fill_list_offsets(int32_t* out, int64_t n, int64_t dim, rmm::cuda_stream_view stream)
{
  auto const grid = std::min(grid_for(n + 1), 65535);
  fill_list_offsets_kernel<<<grid, kBlock, 0, stream.value()>>>(out, n, dim);
  CUDF_CHECK_CUDA(stream.value());
}

void gather_rows(float const* src,
                 int64_t dim,
                 int64_t const* rows,
                 int64_t m,
                 float* out,
                 rmm::cuda_stream_view stream)
{
  if (m == 0) { return; }
  auto const grid = std::min(grid_for(m * dim), 65535);
  gather_rows_kernel<<<grid, kBlock, 0, stream.value()>>>(src, dim, rows, m, out);
  CUDF_CHECK_CUDA(stream.value());
}

void fold_topk_rows(float* acc_distances,
                    int64_t* acc_neighbors,
                    int64_t k,
                    float const* part_distances,
                    int64_t const* part_neighbors,
                    int64_t part_width,
                    int64_t k_eff,
                    int64_t const* rows,
                    int64_t m,
                    int64_t id_base,
                    rmm::cuda_stream_view stream,
                    int64_t const* id_map)
{
  if (m == 0 || k_eff == 0) { return; }
  CUDF_EXPECTS(k_eff <= part_width && k_eff <= k, "fold_topk_rows: k_eff exceeds its row");
  fold_topk_rows_kernel<<<grid_for(m), kBlock, 0, stream.value()>>>(acc_distances,
                                                                    acc_neighbors,
                                                                    k,
                                                                    part_distances,
                                                                    part_neighbors,
                                                                    part_width,
                                                                    k_eff,
                                                                    rows,
                                                                    m,
                                                                    id_base,
                                                                    id_map);
  CUDF_CHECK_CUDA(stream.value());
}

void nearest_cluster_topk(nearest_task const* tasks,
                          int64_t n_tasks,
                          bool f16,
                          void const* compact,
                          int64_t row_bytes,
                          void const* row_sq,
                          int64_t const* routed_rows,
                          void const* probes,
                          void const* probe_sq,
                          int64_t k,
                          float* acc_distances,
                          int64_t* acc_neighbors,
                          rmm::cuda_stream_view stream)
{
  if (n_tasks == 0) { return; }
  CUDF_EXPECTS(row_bytes % 4 == 0 && row_bytes / 4 <= 32 * 16,
               "nearest_cluster_topk: rows must be whole 32-bit words, at most 512 of them");
  CUDF_EXPECTS(k >= 1 && k <= 64, "nearest_cluster_topk: k must be in [1, 64]");
  auto const row_words = row_bytes / 4;
  auto const threads   = 32 * kNearestTaskProbes;
  // The block's probe rows, then a tile of slice rows at the padded stride: whole warps of rows,
  // as many as fit beside the probe rows in about 48 KiB (at least one warp's worth).
  auto const probe_bytes = static_cast<std::size_t>(kNearestTaskProbes * row_words) * 4;
  auto const row_slot    = static_cast<std::size_t>(row_words + 1) * 4;
  auto const fit         = (std::size_t{48} << 10) > probe_bytes
                             ? ((std::size_t{48} << 10) - probe_bytes) / row_slot
                             : std::size_t{0};
  auto const tile_rows =
    static_cast<int64_t>(std::max<std::size_t>(32, std::min<std::size_t>(256, fit / 32 * 32)));
  auto const smem        = probe_bytes + static_cast<std::size_t>(tile_rows) * row_slot;
  auto const* rows       = static_cast<std::uint32_t const*>(compact);
  auto const* probe_rows = static_cast<std::uint32_t const*>(probes);
  if (smem > (std::size_t{48} << 10)) {
    CUDF_CUDA_TRY(cudaFuncSetAttribute(
      f16 ? nearest_cluster_topk_kernel<true> : nearest_cluster_topk_kernel<false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem)));
  }
  if (f16) {
    nearest_cluster_topk_kernel<true>
      <<<static_cast<unsigned>(n_tasks), threads, smem, stream.value()>>>(tasks,
                                                                          rows,
                                                                          row_words,
                                                                          tile_rows,
                                                                          row_sq,
                                                                          routed_rows,
                                                                          probe_rows,
                                                                          probe_sq,
                                                                          k,
                                                                          acc_distances,
                                                                          acc_neighbors);
  } else {
    nearest_cluster_topk_kernel<false>
      <<<static_cast<unsigned>(n_tasks), threads, smem, stream.value()>>>(tasks,
                                                                          rows,
                                                                          row_words,
                                                                          tile_rows,
                                                                          row_sq,
                                                                          routed_rows,
                                                                          probe_rows,
                                                                          probe_sq,
                                                                          k,
                                                                          acc_distances,
                                                                          acc_neighbors);
  }
  CUDF_CHECK_CUDA(stream.value());
}

void remap_radius_edges(int64_t const* query_rows,
                        int64_t const* rows,
                        int32_t* left,
                        int64_t* neighbors,
                        int64_t n_edges,
                        int64_t id_base,
                        rmm::cuda_stream_view stream,
                        int64_t const* id_map)
{
  if (n_edges == 0) { return; }
  auto const grid = std::min(grid_for(n_edges), 65535);
  remap_radius_edges_kernel<<<grid, kBlock, 0, stream.value()>>>(
    query_rows, rows, left, neighbors, n_edges, id_base, id_map);
  CUDF_CHECK_CUDA(stream.value());
}

namespace {

constexpr unsigned int kNoMatchOrder = 0xFFFFFFFFu;  // sorts after every distance a pair can have
constexpr int64_t kNoMatchId         = INT64_MAX;

__device__ __forceinline__ float from_ordered_bits(unsigned int u)
{
  return __uint_as_float((u & 0x80000000u) != 0 ? (u & 0x7FFFFFFFu) : ~u);
}

__device__ __forceinline__ bool best_match_counts(int32_t row,
                                                  int64_t id,
                                                  float distance,
                                                  float max_distance,
                                                  float const* limit,
                                                  earlier_keys const& keys)
{
  if (!(distance <= max_distance)) { return false; }
  if (limit != nullptr && !(distance <= limit[row])) { return false; }
  if (keys.probe == nullptr) { return true; }
  auto const corpus_key = keys.corpus != nullptr ? keys.corpus[id] : id;
  return keys.differ ? corpus_key != keys.probe[row] : corpus_key < keys.probe[row];
}

__global__ void best_match_reset_kernel(unsigned int* order, int64_t* ids, int64_t n)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    order[i] = kNoMatchOrder;
    ids[i]   = kNoMatchId;
  }
}

__global__ void best_match_min_kernel(int32_t const* rows,
                                      int64_t const* ids,
                                      float const* distances,
                                      int64_t n,
                                      float max_distance,
                                      float const* limit,
                                      earlier_keys keys,
                                      unsigned int* order)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const r = rows[i];
    if (best_match_counts(r, ids[i], distances[i], max_distance, limit, keys)) {
      atomicMin(order + r, ordered_bits(distances[i]));
    }
  }
}

// Among a row's pairs at its smallest distance, the smallest id (or buffer index) wins, so the
// answer does not depend on the order the threads run in.
__global__ void best_match_pick_kernel(int32_t const* rows,
                                       int64_t const* ids,
                                       float const* distances,
                                       int64_t n,
                                       float max_distance,
                                       float const* limit,
                                       earlier_keys keys,
                                       unsigned int const* order,
                                       long long* best,
                                       bool by_index)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const r = rows[i];
    if (best_match_counts(r, ids[i], distances[i], max_distance, limit, keys) &&
        ordered_bits(distances[i]) == order[r]) {
      atomicMin(best + r, static_cast<long long>(by_index ? i : ids[i]));
    }
  }
}

// A row whose best distance just dropped forgets the id it held at the old one.
__global__ void best_match_settle_kernel(
  unsigned int const* order, unsigned int const* before, int64_t* ids, float* bound, int64_t n)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (order[i] < before[i]) {
      ids[i] = kNoMatchId;
      if (bound != nullptr) { bound[i] = fminf(bound[i], from_ordered_bits(order[i])); }
    }
  }
}

struct is_winner {
  int32_t const* rows;
  int64_t const* winner;
  __device__ bool operator()(int64_t i) const { return winner[rows[i]] == i; }
};

struct is_contender {
  int32_t const* rows;
  int64_t const* ids;
  float const* distances;
  float const* limit;
  earlier_keys keys;
  int64_t const* winner;
  __device__ bool operator()(int64_t i) const
  {
    auto const r = rows[i];
    return winner[r] != i && best_match_counts(r, ids[i], distances[i], INFINITY, limit, keys);
  }
};

struct has_match {
  int64_t const* ids;
  __device__ bool operator()(int64_t r) const { return ids[r] != kNoMatchId; }
};

struct is_kept {
  int32_t const* rows;
  int64_t const* ids;
  float const* distances;
  float max_distance;
  earlier_keys keys;
  __device__ bool operator()(int64_t i) const
  {
    return best_match_counts(rows[i], ids[i], distances[i], max_distance, nullptr, keys);
  }
};

__global__ void take_pairs_kernel(int64_t const* picked,
                                  int64_t n,
                                  int32_t const* rows,
                                  int64_t const* ids,
                                  float const* distances,
                                  int64_t const* id_map,
                                  bool take_sqrt,
                                  int32_t* out_rows,
                                  int64_t* out_ids,
                                  float* out_distances)
{
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < n;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const i     = picked[p];
    out_rows[p]      = rows[i];
    out_ids[p]       = id_map != nullptr ? id_map[ids[i]] : ids[i];
    out_distances[p] = take_sqrt ? sqrtf(fmaxf(distances[i], 0.f)) : distances[i];
  }
}

__global__ void best_match_emit_kernel(int64_t const* picked,
                                       int64_t n,
                                       unsigned int const* order,
                                       int64_t const* ids,
                                       int64_t const* id_map,
                                       bool take_sqrt,
                                       int32_t* out_rows,
                                       int64_t* out_ids,
                                       float* out_distances)
{
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < n;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const r     = picked[p];
    auto const d     = from_ordered_bits(order[r]);
    out_rows[p]      = static_cast<int32_t>(r);
    out_ids[p]       = id_map != nullptr ? id_map[ids[r]] : ids[r];
    out_distances[p] = take_sqrt ? sqrtf(fmaxf(d, 0.f)) : d;
  }
}

// The indices in [0, n) that @p keep selects, in order, and how many (read back, so this blocks).
template <typename Keep>
int64_t select_indices(int64_t n,
                       Keep keep,
                       rmm::device_uvector<int64_t>& picked,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<int64_t> n_picked(1, stream, mr);
  thrust::counting_iterator<int64_t> const all(0);
  std::size_t temp_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceSelect::If(
    nullptr, temp_bytes, all, picked.data(), n_picked.data(), n, keep, stream.value()));
  rmm::device_buffer temp(temp_bytes, stream, mr);
  CUDF_CUDA_TRY(cub::DeviceSelect::If(
    temp.data(), temp_bytes, all, picked.data(), n_picked.data(), n, keep, stream.value()));
  int64_t count = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &count, n_picked.data(), sizeof(count), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  return count;
}

}  // namespace

void best_match_reset(uint32_t* best_order,
                      int64_t* best_ids,
                      int64_t n_rows,
                      rmm::cuda_stream_view stream)
{
  if (n_rows == 0) { return; }
  best_match_reset_kernel<<<grid_for(n_rows), kBlock, 0, stream.value()>>>(
    best_order, best_ids, n_rows);
  CUDF_CHECK_CUDA(stream.value());
}

void best_match_fold(int32_t const* rows,
                     int64_t const* ids,
                     float const* distances,
                     int64_t n,
                     float max_distance,
                     earlier_keys keys,
                     uint32_t* best_order,
                     int64_t* best_ids,
                     uint32_t* scratch,
                     int64_t n_rows,
                     float* bound,
                     rmm::cuda_stream_view stream)
{
  if (n == 0 || n_rows == 0) { return; }
  auto const grid = std::min(grid_for(n), 65535);
  CUDF_CUDA_TRY(cudaMemcpyAsync(scratch,
                                best_order,
                                static_cast<std::size_t>(n_rows) * sizeof(uint32_t),
                                cudaMemcpyDeviceToDevice,
                                stream.value()));
  best_match_min_kernel<<<grid, kBlock, 0, stream.value()>>>(
    rows, ids, distances, n, max_distance, nullptr, keys, best_order);
  best_match_settle_kernel<<<grid_for(n_rows), kBlock, 0, stream.value()>>>(
    best_order, scratch, best_ids, bound, n_rows);
  best_match_pick_kernel<<<grid, kBlock, 0, stream.value()>>>(
    rows,
    ids,
    distances,
    n,
    max_distance,
    nullptr,
    keys,
    best_order,
    reinterpret_cast<long long*>(best_ids),
    false);
  CUDF_CHECK_CUDA(stream.value());
}

void best_match_winners(int32_t const* rows,
                        int64_t const* ids,
                        float const* distances,
                        int64_t n,
                        earlier_keys keys,
                        float const* limit,
                        uint32_t* scratch,
                        int64_t* winner,
                        int64_t n_rows,
                        rmm::cuda_stream_view stream)
{
  if (n_rows == 0) { return; }
  best_match_reset_kernel<<<grid_for(n_rows), kBlock, 0, stream.value()>>>(scratch, winner, n_rows);
  if (n > 0) {
    auto const grid = std::min(grid_for(n), 65535);
    best_match_min_kernel<<<grid, kBlock, 0, stream.value()>>>(
      rows, ids, distances, n, INFINITY, limit, keys, scratch);
    best_match_pick_kernel<<<grid, kBlock, 0, stream.value()>>>(
      rows,
      ids,
      distances,
      n,
      INFINITY,
      limit,
      keys,
      scratch,
      reinterpret_cast<long long*>(winner),
      true);
  }
  CUDF_CHECK_CUDA(stream.value());
}

kept_pairs best_match_take(int32_t const* rows,
                           int64_t const* ids,
                           float const* distances,
                           int64_t n,
                           earlier_keys keys,
                           float const* limit,
                           int64_t const* winner,
                           bool winners,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  kept_pairs out{rmm::device_uvector<int32_t>(0, stream, mr),
                 rmm::device_uvector<int64_t>(0, stream, mr),
                 rmm::device_uvector<float>(0, stream, mr)};
  if (n == 0) { return out; }
  rmm::device_uvector<int64_t> picked(n, stream, mr);
  auto const count =
    winners ? select_indices(n, is_winner{rows, winner}, picked, stream, mr)
            : select_indices(
                n, is_contender{rows, ids, distances, limit, keys, winner}, picked, stream, mr);
  if (count == 0) { return out; }
  out.rows.resize(count, stream);
  out.ids.resize(count, stream);
  out.distances.resize(count, stream);
  take_pairs_kernel<<<grid_for(count), kBlock, 0, stream.value()>>>(picked.data(),
                                                                    count,
                                                                    rows,
                                                                    ids,
                                                                    distances,
                                                                    nullptr,
                                                                    false,
                                                                    out.rows.data(),
                                                                    out.ids.data(),
                                                                    out.distances.data());
  CUDF_CHECK_CUDA(stream.value());
  // `picked` is freed on return; the gather that reads it is ordered before that on this stream.
  return out;
}

kept_pairs take_pairs(int32_t const* rows,
                      int64_t const* ids,
                      float const* distances,
                      int64_t n,
                      earlier_keys keys,
                      float max_distance,
                      int64_t const* id_map,
                      bool take_sqrt,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  kept_pairs out{rmm::device_uvector<int32_t>(0, stream, mr),
                 rmm::device_uvector<int64_t>(0, stream, mr),
                 rmm::device_uvector<float>(0, stream, mr)};
  if (n == 0) { return out; }
  rmm::device_uvector<int64_t> picked(n, stream, mr);
  auto const count =
    select_indices(n, is_kept{rows, ids, distances, max_distance, keys}, picked, stream, mr);
  if (count == 0) { return out; }
  out.rows.resize(count, stream);
  out.ids.resize(count, stream);
  out.distances.resize(count, stream);
  take_pairs_kernel<<<grid_for(count), kBlock, 0, stream.value()>>>(picked.data(),
                                                                    count,
                                                                    rows,
                                                                    ids,
                                                                    distances,
                                                                    id_map,
                                                                    take_sqrt,
                                                                    out.rows.data(),
                                                                    out.ids.data(),
                                                                    out.distances.data());
  CUDF_CHECK_CUDA(stream.value());
  return out;
}

kept_pairs best_match_emit(uint32_t const* best_order,
                           int64_t const* best_ids,
                           int64_t n_rows,
                           int64_t const* id_map,
                           bool take_sqrt,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  kept_pairs out{rmm::device_uvector<int32_t>(0, stream, mr),
                 rmm::device_uvector<int64_t>(0, stream, mr),
                 rmm::device_uvector<float>(0, stream, mr)};
  if (n_rows == 0) { return out; }
  rmm::device_uvector<int64_t> picked(n_rows, stream, mr);
  auto const count = select_indices(n_rows, has_match{best_ids}, picked, stream, mr);
  if (count == 0) { return out; }
  out.rows.resize(count, stream);
  out.ids.resize(count, stream);
  out.distances.resize(count, stream);
  best_match_emit_kernel<<<grid_for(count), kBlock, 0, stream.value()>>>(picked.data(),
                                                                         count,
                                                                         best_order,
                                                                         best_ids,
                                                                         id_map,
                                                                         take_sqrt,
                                                                         out.rows.data(),
                                                                         out.ids.data(),
                                                                         out.distances.data());
  CUDF_CHECK_CUDA(stream.value());
  return out;
}

}  // namespace sirius::vss
