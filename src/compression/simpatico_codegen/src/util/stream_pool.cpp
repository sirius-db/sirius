// SPDX-License-Identifier: Apache-2.0
#include "codegen/util/stream_pool.hpp"

#include <algorithm>
#include <cstdio>
#ifdef SIMPATICO_USE_CUCASCADE_STREAM_POOL
#include <cucascade/memory/stream_pool.hpp>
#endif

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace simpatico {

struct stream_pool::lease_state {
#ifdef SIMPATICO_USE_CUCASCADE_STREAM_POOL
  std::vector<cucascade::memory::borrowed_stream> streams;
#endif
};

#ifdef SIMPATICO_USE_CUCASCADE_STREAM_POOL
namespace {
cucascade::memory::exclusive_stream_pool& device_pool(int device)
{
  // Returned columns retain their allocation stream for asynchronous frees.
  // Keep the owners alive even after a lease or its submitting thread ends.
  struct registry {
    std::mutex mutex;
    std::map<int, std::unique_ptr<cucascade::memory::exclusive_stream_pool>> pools;
  };
  static auto* owners = new registry;
  std::lock_guard lock(owners->mutex);
  auto& pool = owners->pools[device];
  if (!pool) {
    pool = std::make_unique<cucascade::memory::exclusive_stream_pool>(rmm::cuda_device_id{device});
  }
  return *pool;
}
}  // namespace
#endif

stream_pool::stream_pool() = default;
stream_pool::~stream_pool() { shutdown(); }
stream_pool::stream_pool(stream_pool&& other) noexcept
  : streams(std::exchange(other.streams, {})), leases_(std::move(other.leases_))
{
}
stream_pool& stream_pool::operator=(stream_pool&& other) noexcept
{
  if (this != &other) {
    shutdown();
    streams = std::exchange(other.streams, {});
    leases_ = std::move(other.leases_);
  }
  return *this;
}

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
#ifdef SIMPATICO_USE_CUCASCADE_STREAM_POOL
  if (n == 0) return true;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return false;
  auto& owner = device_pool(device);
  auto leases = std::make_unique<lease_state>();
  leases->streams.reserve(n);
  streams.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    // Blocking while holding part of a multi-stream lease could deadlock
    // concurrent callers. Grow just as the pipeline executor does.
    leases->streams.push_back(
      owner.acquire_stream(cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW));
    streams.push_back(leases->streams.back().get().get());
  }
  leases_ = std::move(leases);
#else
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
#endif
  return true;
}

void stream_pool::shutdown()
{
  (void)sync_all();
#ifdef SIMPATICO_USE_CUCASCADE_STREAM_POOL
  leases_.reset();
#else
  for (auto& stream : streams) {
    cudaStreamDestroy(stream);
  }
#endif
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

std::vector<::cuda::stream_ref> stream_pool::refs() const
{
  return {streams.begin(), streams.end()};
}

cudaError_t synchronize_distinct(std::span<const ::cuda::stream_ref> streams) noexcept
{
  cudaError_t first = cudaSuccess;
  for (auto it = streams.begin(); it != streams.end(); ++it) {
    auto const handle = it->get();
    if (std::any_of(
          streams.begin(), it, [handle](auto earlier) { return earlier.get() == handle; }))
      continue;
    auto status = cudaStreamQuery(handle);
    if (status == cudaSuccess) continue;
    if (status != cudaErrorNotReady && first == cudaSuccess) first = status;
    status = cudaStreamSynchronize(handle);
    if (status != cudaSuccess && first == cudaSuccess) first = status;
  }
  return first;
}

void synchronize_distinct_or_log(std::span<const ::cuda::stream_ref> streams,
                                 char const* context) noexcept
{
  if (auto const status = synchronize_distinct(streams); status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", context, cudaGetErrorString(status));
  }
}

}  // namespace simpatico
