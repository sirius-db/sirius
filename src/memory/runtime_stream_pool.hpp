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

#pragma once
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/stream_pool.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
namespace sirius::memory {
// The memory manager owns these pools until all query and pinned buffers are retired.
// Borrowing is exclusive during an operation; returning a lease preserves the stream
// for later buffer deallocation. These streams do not alias memory_space's raw pool.
class runtime_stream_pool {
  inline static std::mutex mutex;
  inline static std::map<const cucascade::memory::memory_space*,
                         std::weak_ptr<cucascade::memory::exclusive_stream_pool>>
    pools;

 public:
  static std::shared_ptr<cucascade::memory::exclusive_stream_pool> install(
    const cucascade::memory::memory_space& space)
  {
    auto pool = std::make_shared<cucascade::memory::exclusive_stream_pool>(
      rmm::cuda_device_id{space.get_device_id()});
    std::lock_guard lock(mutex);
    pools[&space] = pool;
    return pool;
  }
  static void remove(const cucascade::memory::memory_space& space)
  {
    std::lock_guard lock(mutex);
    pools.erase(&space);
  }
  static cucascade::memory::borrowed_stream acquire(const cucascade::memory::memory_space& space)
  {
    std::shared_ptr<cucascade::memory::exclusive_stream_pool> pool;
    {
      std::lock_guard lock(mutex);
      auto it = pools.find(&space);
      if (it != pools.end()) pool = it->second.lock();
    }
    if (!pool) throw std::logic_error("GPU memory space has no Sirius runtime stream pool");
    // Work pools bound concurrent borrowers. GROW avoids waiting while holding a batch
    // lock or another device's lease; parked workers retain no borrowed stream.
    return pool->acquire_stream(
      cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
  }
};
}  // namespace sirius::memory
