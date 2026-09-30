// SPDX-License-Identifier: Apache-2.0
#include "codegen/util/stream_pool.hpp"

#include <map>
#include <stdexcept>
#include <string>

namespace simpatico {

stream_pool& thread_device_stream_pool(size_t n)
{
  // One pool per (thread, device). A map rather than a single pool so a thread
  // that works on several devices gets streams belonging to each.
  thread_local std::map<int, stream_pool> pools;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    throw std::runtime_error("stream_pool: cannot query the current device");
  }
  auto& pool = pools[device];
  if (pool.streams.empty()) {
    if (!pool.init(n)) {
      throw std::runtime_error("stream_pool: failed to create " + std::to_string(n) +
                               " streams on device " + std::to_string(device));
    }
  }
  return pool;
}

bool stream_pool::init(size_t n)
{
  if (!streams.empty()) return true;  // Already initialized
  streams.resize(n);
  for (size_t i = 0; i < n; ++i) {
    cudaError_t err = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      for (size_t j = 0; j < i; ++j) {
        cudaStreamDestroy(streams[j]);
      }
      streams.clear();
      return false;
    }
  }
  return true;
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
