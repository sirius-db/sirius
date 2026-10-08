/*
 * Copyright 2026, Sirius Contributors.
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

#include "compression_device_pool.hpp"

#include "compression_alloc_stats.hpp"
#include "log/logging.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/memory_resource>

#include <atomic>
#include <mutex>
#include <optional>

namespace sirius::compression {

namespace {

/// Bytes currently allocated from the arena. One relaxed atomic per allocation,
/// always on: the spill path reads it to skip compressing when the arena is
/// nearly full (see compression_device_pool_used_bytes), rather than letting the
/// encode fail into it.
std::atomic<std::size_t> g_arena_used{0};

/// Forwards to the arena and keeps g_arena_used. Same shape as the optional
/// counting adaptor in compression_alloc_stats.cpp, which may wrap this one.
class arena_usage_resource {
 public:
  explicit arena_usage_resource(rmm::device_async_resource_ref upstream) : _upstream(upstream) {}

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    void* ptr = _upstream.allocate(stream, bytes, alignment);
    g_arena_used.fetch_add(bytes, std::memory_order_relaxed);
    return ptr;
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    _upstream.deallocate(stream, ptr, bytes, alignment);
    g_arena_used.fetch_sub(bytes, std::memory_order_relaxed);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    void* ptr = _upstream.allocate_sync(bytes, alignment);
    g_arena_used.fetch_add(bytes, std::memory_order_relaxed);
    return ptr;
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    _upstream.deallocate_sync(ptr, bytes, alignment);
    g_arena_used.fetch_sub(bytes, std::memory_order_relaxed);
  }

  bool operator==(arena_usage_resource const& other) const noexcept { return this == &other; }
  bool operator!=(arena_usage_resource const& other) const noexcept { return this != &other; }

  friend void get_property(arena_usage_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  rmm::device_async_resource_ref _upstream;
};

static_assert(cuda::mr::resource_with<arena_usage_resource, cuda::mr::device_accessible>,
              "arena_usage_resource does not satisfy the cuda::mr::resource concept");

std::mutex g_pool_mutex;
// Never destroyed: the arena outlives the converters that allocate from it, and
// tearing a device arena down during static destruction races with CUDA's own
// teardown. Leaking it at exit is the conventional trade for that.
rmm::mr::pool_memory_resource* g_pool = nullptr;
arena_usage_resource* g_pool_usage    = nullptr;
std::atomic<std::size_t> g_pool_bytes{0};

}  // namespace

bool init_compression_device_pool(std::size_t bytes, int device_id)
{
  std::lock_guard<std::mutex> lock(g_pool_mutex);

  if (bytes == 0) { return false; }
  if (g_pool != nullptr) {
    if (g_pool_bytes.load(std::memory_order_relaxed) != bytes) {
      SIRIUS_LOG_WARN(
        "[compression] device arena already initialized at {} bytes; ignoring request for {}",
        g_pool_bytes.load(std::memory_order_relaxed),
        bytes);
    }
    return true;
  }

  try {
    // initial == maximum: take the whole arena up front and never grow. Growing
    // later would defeat the point — compression's demand is decided before the
    // query starts competing for the device, not renegotiated under pressure.
    // On the device the config carved the arena out of, so the capacity it
    // subtracted and the memory taken here are the same device's.
    std::optional<rmm::cuda_set_device_raii> set_device;
    if (device_id >= 0) { set_device.emplace(rmm::cuda_device_id{device_id}); }
    g_pool = new rmm::mr::pool_memory_resource(rmm::mr::cuda_memory_resource{}, bytes, bytes);
  } catch (const std::exception& e) {
    delete g_pool;
    g_pool = nullptr;
    SIRIUS_LOG_WARN(
      "[compression] could not reserve a {} byte device arena ({}); spill compression will "
      "allocate from the query pool",
      bytes,
      e.what());
    return false;
  }

  g_pool_usage = new arena_usage_resource(*g_pool);
  g_pool_bytes.store(bytes, std::memory_order_relaxed);
  SIRIUS_LOG_INFO(
    "[compression] reserved {} MiB device arena for spill compression on device {} (carved out "
    "of the GPU memory space's capacity)",
    bytes / (1024 * 1024),
    device_id);
  return true;
}

rmm::device_async_resource_ref compression_device_mr()
{
  // Wrapped in the counting adaptor only when SIRIUS_COMPRESSION_ALLOC_STATS is
  // set; otherwise this returns the underlying resource unchanged.
  if (g_pool_usage != nullptr) { return alloc_stats_wrap(*g_pool_usage); }
  return alloc_stats_wrap(rmm::mr::get_current_device_resource_ref());
}

bool compression_device_pool_enabled() noexcept { return g_pool_usage != nullptr; }

std::size_t compression_device_pool_used_bytes() noexcept
{
  return g_arena_used.load(std::memory_order_relaxed);
}

std::size_t compression_device_pool_bytes() noexcept
{
  return g_pool_bytes.load(std::memory_order_relaxed);
}

}  // namespace sirius::compression
