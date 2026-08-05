// hwsim WS5: GPU compute throttling primitives shared by throttle_compute and victim_bench.
//
// Two throttling mechanisms:
//   1. SM-stealing: a persistent spin kernel launched as exactly one full wave
//      (num_sms * blocks_per_sm blocks). Every block records its %smid, then blocks
//      whose %smid >= n_stolen exit immediately, freeing their SM for the victim.
//      Blocks on stolen SMs spin on FMAs (register-only, no DRAM traffic) until a
//      host-mapped stop flag is set.
//   2. Duty-cycling: a full-GPU burst kernel that spins for busy_ns (measured with
//      %globaltimer), driven by a host loop that alternates launch/sleep.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                              \
  do {                                                                                \
    cudaError_t err_ = (call);                                                        \
    if (err_ != cudaSuccess) {                                                        \
      fprintf(stderr, "CUDA error at %s:%d: %s -> %s\n", __FILE__, __LINE__, #call,   \
              cudaGetErrorString(err_));                                              \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

#define CU_CHECK(call)                                                                \
  do {                                                                                \
    CUresult res_ = (call);                                                           \
    if (res_ != CUDA_SUCCESS) {                                                       \
      const char* msg_ = nullptr;                                                     \
      cuGetErrorString(res_, &msg_);                                                  \
      fprintf(stderr, "CU driver error at %s:%d: %s -> %s\n", __FILE__, __LINE__,     \
              #call, msg_ ? msg_ : "?");                                              \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

__device__ __forceinline__ unsigned hwsim_smid() {
  unsigned id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

__device__ __forceinline__ unsigned long long hwsim_globaltimer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Register-only FMA burn loop. No memory traffic.
__device__ __forceinline__ float hwsim_fma_burn(float seed, int iters) {
  float a = seed;
  const float b = 1.0000001f;
  const float c = 1e-7f;
#pragma unroll 8
  for (int i = 0; i < iters; ++i) {
    a = fmaf(a, b, c);
  }
  return a;
}

// ---------------------------------------------------------------------------
// SM-stealing persistent kernel.
// Must be launched as exactly ONE full wave: grid = num_sms * blocks_per_sm,
// block = block_size chosen so blocks_per_sm blocks saturate an SM's thread slots.
// coverage[smid] counts blocks placed on each SM (for the full wave, so we can
// verify one-wave placement); blocks on smid >= n_stolen exit immediately.
__global__ void smsteal_kernel(const volatile int* __restrict__ stop,
                               int n_stolen,
                               unsigned int* __restrict__ coverage,
                               float* __restrict__ sink) {
  const unsigned my_sm = hwsim_smid();
  if (threadIdx.x == 0 && coverage != nullptr && my_sm < 1024) {
    atomicAdd(&coverage[my_sm], 1u);
  }
  if ((int)my_sm >= n_stolen) return;  // leave this SM to the victim

  __shared__ int s_stop;
  float acc = 1.0f + (float)threadIdx.x * 1e-6f;
  do {
    if (threadIdx.x == 0) s_stop = *stop;
    __syncthreads();
    acc = hwsim_fma_burn(acc, 1 << 13);  // ~tens of microseconds between flag checks
    __syncthreads();
  } while (!s_stop);
  if (acc == 123.456f) *sink = acc;  // never true; defeats dead-code elimination
}

// ---------------------------------------------------------------------------
// Duty-cycle burst kernel: every thread spins on FMAs until busy_ns of wall time
// (per %globaltimer) has elapsed since the block started.
__global__ void burst_kernel(unsigned long long busy_ns, float* __restrict__ sink) {
  const unsigned long long start = hwsim_globaltimer_ns();
  float acc = 1.0f + (float)threadIdx.x * 1e-6f;
  do {
    acc = hwsim_fma_burn(acc, 1 << 12);
  } while (hwsim_globaltimer_ns() - start < busy_ns);
  if (acc == 123.456f) *sink = acc;
}

// ---------------------------------------------------------------------------
struct ThrottleShape {
  int num_sms;
  int blocks_per_sm;
  int block_size;
};

inline ThrottleShape throttle_shape(const void* kernel) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  ThrottleShape s;
  s.num_sms = prop.multiProcessorCount;
  s.block_size = prop.maxThreadsPerBlock;
  s.blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&s.blocks_per_sm, kernel,
                                                           s.block_size, 0));
  if (s.blocks_per_sm < 1) s.blocks_per_sm = 1;
  return s;
}

// ---------------------------------------------------------------------------
// RAII-ish handle for a running SM-steal throttler (usable in-process on its own
// stream, or as the body of the standalone throttler process).
struct SmStealHandle {
  cudaStream_t stream = nullptr;
  int* stop_host = nullptr;       // host-mapped stop flag
  int* stop_dev = nullptr;        // device view of the flag
  unsigned int* coverage = nullptr;  // device coverage histogram (1024 entries)
  float* sink = nullptr;
  ThrottleShape shape{};
  int n_stolen = 0;
  bool running = false;

  // fraction in [0,1] -> steals round(fraction * num_sms) SMs.
  void start(double fraction, bool report_coverage) {
    shape = throttle_shape((const void*)smsteal_kernel);
    n_stolen = (int)(fraction * shape.num_sms + 0.5);
    if (n_stolen < 0) n_stolen = 0;
    if (n_stolen > shape.num_sms) n_stolen = shape.num_sms;

    CUDA_CHECK(cudaHostAlloc(&stop_host, sizeof(int), cudaHostAllocMapped));
    *stop_host = 0;
    CUDA_CHECK(cudaHostGetDevicePointer(&stop_dev, stop_host, 0));
    CUDA_CHECK(cudaMalloc(&coverage, 1024 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(coverage, 0, 1024 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (n_stolen > 0) {
      const int grid = shape.num_sms * shape.blocks_per_sm;  // exactly one wave
      smsteal_kernel<<<grid, shape.block_size, 0, stream>>>(stop_dev, n_stolen,
                                                            coverage, sink);
      CUDA_CHECK(cudaGetLastError());
      running = true;
    }

    if (report_coverage && n_stolen > 0) {
      // Give the wave a moment to place all blocks, then read the histogram on a
      // separate stream (the throttle stream never goes idle while spinning).
      struct timespec ts { 0, 50 * 1000 * 1000 };  // 50 ms
      nanosleep(&ts, nullptr);
      unsigned int host_cov[1024];
      cudaStream_t s2;
      CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
      CUDA_CHECK(cudaMemcpyAsync(host_cov, coverage, sizeof(host_cov),
                                 cudaMemcpyDeviceToHost, s2));
      CUDA_CHECK(cudaStreamSynchronize(s2));
      CUDA_CHECK(cudaStreamDestroy(s2));

      int sms_seen = 0, min_b = 1 << 30, max_b = 0;
      for (int i = 0; i < 1024; ++i) {
        if (host_cov[i] > 0) {
          ++sms_seen;
          if ((int)host_cov[i] < min_b) min_b = (int)host_cov[i];
          if ((int)host_cov[i] > max_b) max_b = (int)host_cov[i];
        }
      }
      printf("smsteal: stealing %d/%d SMs (fraction %.3f), %d blocks/SM x %d threads\n",
             n_stolen, shape.num_sms, (double)n_stolen / shape.num_sms,
             shape.blocks_per_sm, shape.block_size);
      printf("smsteal: wave placement: %d distinct SMs covered, blocks/SM min=%d max=%d "
             "(expect %d SMs, %d/%d)\n",
             sms_seen, min_b, max_b, shape.num_sms, shape.blocks_per_sm,
             shape.blocks_per_sm);
      fflush(stdout);
    }
  }

  void stop() {
    if (stop_host != nullptr) *stop_host = 1;
    if (running) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      running = false;
    }
    if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    if (coverage) { cudaFree(coverage); coverage = nullptr; }
    if (sink) { cudaFree(sink); sink = nullptr; }
    if (stop_host) { cudaFreeHost(stop_host); stop_host = nullptr; }
  }
};
