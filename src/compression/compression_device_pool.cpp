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

/// Bytes currently allocated from the arena, and their high-water mark. A couple
/// of relaxed atomics per allocation, always on: the spill path reads the usage
/// to skip compressing when the arena is nearly full (see
/// compression_device_pool_used_bytes), and the per-query pool log reports the
/// peak (compression_device_pool_peak_bytes).
usage_high_water_mark g_arena_usage;

/// Forwards to the arena and keeps g_arena_usage. Same shape as the optional
/// counting adaptor in compression_alloc_stats.cpp, which may wrap this one.
class arena_usage_resource {
 public:
  explicit arena_usage_resource(rmm::device_async_resource_ref upstream) : _upstream(upstream) {}

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    void* ptr = _upstream.allocate(stream, bytes, alignment);
    g_arena_usage.on_allocate(bytes);
    return ptr;
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    _upstream.deallocate(stream, ptr, bytes, alignment);
    g_arena_usage.on_deallocate(bytes);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    void* ptr = _upstream.allocate_sync(bytes, alignment);
    g_arena_usage.on_allocate(bytes);
    return ptr;
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    _upstream.deallocate_sync(ptr, bytes, alignment);
    g_arena_usage.on_deallocate(bytes);
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

std::size_t compression_device_pool_used_bytes() noexcept { return g_arena_usage.used(); }

std::size_t compression_device_pool_peak_bytes() noexcept { return g_arena_usage.peak(); }

void compression_device_pool_reset_peak() noexcept { (void)g_arena_usage.reset_peak(); }

std::size_t compression_device_pool_bytes() noexcept
{
  return g_pool_bytes.load(std::memory_order_relaxed);
}

namespace {

std::atomic<std::uint64_t> g_enc_granted{0};
std::atomic<std::uint64_t> g_enc_declined{0};
usage_high_water_mark g_enc_reserved;
std::atomic<std::size_t> g_enc_largest_reserved{0};
std::atomic<std::size_t> g_enc_largest_used{0};

void raise_to(std::atomic<std::size_t>& mark, std::size_t value) noexcept
{
  auto cur = mark.load(std::memory_order_relaxed);
  while (value > cur && !mark.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {}
}

}  // namespace

void note_encode_reservation_granted(std::size_t reserved_bytes) noexcept
{
  g_enc_granted.fetch_add(1, std::memory_order_relaxed);
  g_enc_reserved.on_allocate(reserved_bytes);
  raise_to(g_enc_largest_reserved, reserved_bytes);
}

void note_encode_reservation_released(std::size_t reserved_bytes, std::size_t peak_used) noexcept
{
  g_enc_reserved.on_deallocate(reserved_bytes);
  raise_to(g_enc_largest_used, peak_used);
}

void note_encode_reservation_declined() noexcept
{
  g_enc_declined.fetch_add(1, std::memory_order_relaxed);
}

encode_reservation_stats read_encode_reservation_stats() noexcept
{
  encode_reservation_stats s;
  s.granted                   = g_enc_granted.load(std::memory_order_relaxed);
  s.declined                  = g_enc_declined.load(std::memory_order_relaxed);
  s.outstanding_reserved      = g_enc_reserved.used();
  s.peak_outstanding_reserved = g_enc_reserved.peak();
  s.largest_reserved          = g_enc_largest_reserved.load(std::memory_order_relaxed);
  s.largest_used              = g_enc_largest_used.load(std::memory_order_relaxed);
  return s;
}

void reset_encode_reservation_window() noexcept
{
  g_enc_granted.store(0, std::memory_order_relaxed);
  g_enc_declined.store(0, std::memory_order_relaxed);
  (void)g_enc_reserved.reset_peak();
  g_enc_largest_reserved.store(0, std::memory_order_relaxed);
  g_enc_largest_used.store(0, std::memory_order_relaxed);
}

}  // namespace sirius::compression
