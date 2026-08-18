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

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <atomic>
#include <mutex>

namespace sirius::compression {

namespace {

std::mutex g_pool_mutex;
// Never destroyed: the arena outlives the converters that allocate from it, and
// tearing a device arena down during static destruction races with CUDA's own
// teardown. Leaking it at exit is the conventional trade for that.
rmm::mr::pool_memory_resource* g_pool = nullptr;
std::atomic<std::size_t> g_pool_bytes{0};

}  // namespace

bool init_compression_device_pool(std::size_t bytes)
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

  g_pool_bytes.store(bytes, std::memory_order_relaxed);
  SIRIUS_LOG_INFO("[compression] reserved {} MiB device arena for spill compression",
                  bytes / (1024 * 1024));
  return true;
}

rmm::device_async_resource_ref compression_device_mr()
{
  // Wrapped in the counting adaptor only when SIRIUS_COMPRESSION_ALLOC_STATS is
  // set; otherwise this returns the underlying resource unchanged.
  if (g_pool != nullptr) { return alloc_stats_wrap(*g_pool); }
  return alloc_stats_wrap(rmm::mr::get_current_device_resource_ref());
}

bool compression_device_pool_enabled() noexcept { return g_pool != nullptr; }

std::size_t compression_device_pool_bytes() noexcept
{
  return g_pool_bytes.load(std::memory_order_relaxed);
}

}  // namespace sirius::compression
