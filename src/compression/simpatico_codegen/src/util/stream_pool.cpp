// SPDX-License-Identifier: Apache-2.0
#include "codegen/util/stream_pool.hpp"

#include <rmm/error.hpp>

#include <atomic>
#include <map>
#include <stdexcept>
#include <string>

namespace simpatico {

namespace {

std::atomic<size_t> g_injected_create_failures{0};

/// Consume one injected failure, if any are pending.
bool take_injected_failure() noexcept
{
  auto pending = g_injected_create_failures.load(std::memory_order_relaxed);
  while (pending > 0) {
    if (g_injected_create_failures.compare_exchange_weak(
          pending, pending - 1, std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

}  // namespace

void inject_stream_create_failures_for_testing(size_t count) noexcept
{
  g_injected_create_failures.store(count, std::memory_order_relaxed);
}

stream_pool& thread_device_stream_pool(size_t n)
{
  // One pool per (thread, device). A map rather than a single pool so a thread
  // that works on several devices gets streams belonging to each.
  //
  // Per thread, not shared per device: run_column_workers ends every call with a
  // sync_all() over the pool's streams, which would wait on other threads' work
  // if the streams were shared, and buffers allocated on these streams free on
  // them later, so the handles must outlive every such buffer -- the thread's
  // lifetime gives that without any cross-thread bookkeeping.
  thread_local std::map<int, stream_pool> pools;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    throw std::runtime_error("stream_pool: cannot query the current device");
  }
  auto& pool = pools[device];
  if (pool.streams.empty()) {
    if (cudaError_t const err = pool.try_init(n); err != cudaSuccess) {
      // Clear the (non-sticky) error so it is not mis-attributed to whatever CUDA
      // call this thread makes next.
      (void)cudaGetLastError();
      throw rmm::out_of_memory("stream_pool: failed to create " + std::to_string(n) +
                               " streams on device " + std::to_string(device) + ": " +
                               cudaGetErrorName(err) + " (" + cudaGetErrorString(err) + ")");
    }
  }
  return pool;
}

cudaError_t stream_pool::try_init(size_t n)
{
  if (!streams.empty()) return cudaSuccess;  // Already initialized
  if (take_injected_failure()) return cudaErrorMemoryAllocation;
  streams.resize(n);
  for (size_t i = 0; i < n; ++i) {
    cudaError_t err = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      for (size_t j = 0; j < i; ++j) {
        cudaStreamDestroy(streams[j]);
      }
      streams.clear();
      return err;
    }
  }
  return cudaSuccess;
}

void stream_pool::shutdown()
{
  for (auto& stream : streams) {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
  streams.clear();
}

cudaError_t stream_pool::sync_all()
{
  cudaError_t first = cudaSuccess;
  for (auto& stream : streams) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (first == cudaSuccess && err != cudaSuccess) { first = err; }
  }
  return first;
}

}  // namespace simpatico
