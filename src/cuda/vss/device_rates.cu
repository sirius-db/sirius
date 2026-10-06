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

#include "vss/device_rates.hpp"

#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace sirius::vss {

namespace {

constexpr int kThreads = 256;
constexpr int kChains  = 8;

__global__ void fma_kernel(float* out, int iters)
{
  float c[kChains];
  float const a = 1.0f + threadIdx.x * 1e-7f;
  float const b = 0.999999f;
  for (int j = 0; j < kChains; ++j) {
    c[j] = j;
  }
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < kChains; ++j) {
      c[j] = fmaf(c[j], b, a);
    }
  }
  float sum = 0;
  for (int j = 0; j < kChains; ++j) {
    sum += c[j];
  }
  if (sum == -1.0f) { out[blockIdx.x] = sum; }  // never true; keeps the chains live
}

template <class In, class Acc>
__global__ void mma_kernel(Acc* out, int iters)
{
#if __CUDA_ARCH__ >= 720
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, In, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, In, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, Acc> c[4];
  wmma::fill_fragment(a, In(1));
  wmma::fill_fragment(b, In(1));
  for (auto& f : c) {
    wmma::fill_fragment(f, Acc(0));
  }
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (auto& f : c) {
      wmma::mma_sync(f, a, b, f);
    }
  }
  for (int j = 1; j < 4; ++j) {
    for (int e = 0; e < c[0].num_elements; ++e) {
      c[0].x[e] += c[j].x[e];
    }
  }
  if (c[0].x[0] == Acc(-1)) {  // never true; keeps the products live
    wmma::store_matrix_sync(out, c[0], 16, wmma::mem_row_major);
  }
#endif
}

/// Seconds for @p launch on @p stream, best of three after two warm-up launches.
template <class F>
bool time_best(cudaStream_t stream, F&& launch, double& seconds)
{
  cudaEvent_t start = nullptr, stop = nullptr;
  if (cudaEventCreate(&start) != cudaSuccess) { return false; }
  if (cudaEventCreate(&stop) != cudaSuccess) {
    cudaEventDestroy(start);
    return false;
  }
  bool ok = true;
  seconds = 0;
  for (int warm = 0; warm < 2; ++warm) {
    launch();
  }
  for (int run = 0; run < 3 && ok; ++run) {
    cudaEventRecord(start, stream);
    launch();
    cudaEventRecord(stop, stream);
    float ms = 0;
    ok       = cudaEventSynchronize(stop) == cudaSuccess &&
         cudaEventElapsedTime(&ms, start, stop) == cudaSuccess && ms > 0;
    double const s = ms * 1e-3;
    seconds        = run == 0 ? s : std::min(seconds, s);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ok && cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool measure_device_rates(device_rates& out)
{
  int device = 0, sms = 0, major = 0, minor = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    return false;
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { return false; }

  constexpr std::size_t kBytes = std::size_t{64} << 20;
  void* host                   = nullptr;
  void* dev_a                  = nullptr;
  void* dev_b                  = nullptr;
  bool ok                      = cudaMallocHost(&host, kBytes) == cudaSuccess &&
            cudaMalloc(&dev_a, kBytes) == cudaSuccess && cudaMalloc(&dev_b, kBytes) == cudaSuccess;

  int const blocks = sms * 4;
  double s         = 0;
  if (ok) {
    constexpr int kIters = 1 << 14;
    ok                   = time_best(
      stream,
      [&] { fma_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<float*>(dev_a), kIters); },
      s);
    out.fp32 = 2.0 * kChains * kIters * static_cast<double>(blocks) * kThreads / s;
  }
  bool const tensor        = major > 7 || (major == 7 && minor >= 2);
  constexpr double kMmaOps = 2.0 * 16 * 16 * 16 * 4;  // per warp per iteration
  double const warps       = static_cast<double>(blocks) * (kThreads / 32);
  if (ok && tensor) {
    constexpr int kIters = 1 << 10;
    ok                   = time_best(
      stream,
      [&] {
        mma_kernel<half, float>
          <<<blocks, kThreads, 0, stream>>>(static_cast<float*>(dev_a), kIters);
      },
      s);
    out.f16 = kMmaOps * kIters * warps / s;
  }
  if (ok && tensor) {
    constexpr int kIters = 1 << 11;
    ok                   = time_best(
      stream,
      [&] {
        mma_kernel<signed char, int>
          <<<blocks, kThreads, 0, stream>>>(static_cast<int*>(dev_a), kIters);
      },
      s);
    out.int8 = kMmaOps * kIters * warps / s;
  }
  if (ok && !tensor) {
    out.f16  = out.fp32;
    out.int8 = out.fp32;
  }
  if (ok) {
    constexpr int kCopies = 2;
    ok                    = time_best(
      stream,
      [&] {
        for (int i = 0; i < kCopies; ++i) {
          cudaMemcpyAsync(dev_a, host, kBytes, cudaMemcpyHostToDevice, stream);
        }
      },
      s);
    out.pcie = kCopies * static_cast<double>(kBytes) / s;
  }
  if (ok) {
    constexpr int kCopies = 10;
    ok                    = time_best(
      stream,
      [&] {
        for (int i = 0; i < kCopies; ++i) {
          cudaMemcpyAsync(dev_b, dev_a, kBytes, cudaMemcpyDeviceToDevice, stream);
        }
      },
      s);
    out.hbm = 2.0 * kCopies * static_cast<double>(kBytes) / s;
  }

  cudaStreamSynchronize(stream);
  if (dev_b != nullptr) { cudaFree(dev_b); }
  if (dev_a != nullptr) { cudaFree(dev_a); }
  if (host != nullptr) { cudaFreeHost(host); }
  cudaStreamDestroy(stream);
  cudaGetLastError();  // a failed allocation above is reported by the return value, not left sticky
  return ok;
}

}  // namespace sirius::vss
