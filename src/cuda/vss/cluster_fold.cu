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

#include <cuda_fp16.h>

#include <algorithm>
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
// *max_error_bits, and the largest |half(x)| into *max_norm_bits (all non-negative floats' bits).
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
      if (max_norm_bits != nullptr) { atomicMax(max_norm_bits, __float_as_uint(sqrtf(acc))); }
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

}  // namespace sirius::vss
