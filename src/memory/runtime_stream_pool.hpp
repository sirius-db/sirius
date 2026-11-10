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
#include <utility>
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
  // cuCascade's borrowed_stream returns its stream through a raw pool pointer.
  // Keep that pool alive until the return completes, including during move assignment.
  // This does not extend the lifetime of memory spaces, allocators, or GPU buffers:
  // those still require the engine's normal shutdown ordering.
  class lease {
    friend class runtime_stream_pool;
    std::shared_ptr<cucascade::memory::exclusive_stream_pool> pool_;
    cucascade::memory::borrowed_stream stream_;  // Destroy before pool_.

    explicit lease(std::shared_ptr<cucascade::memory::exclusive_stream_pool> pool)
      : pool_(std::move(pool)),
        stream_(pool_->acquire_stream(
          cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW))
    {
    }

   public:
    lease(const lease&)            = delete;
    lease& operator=(const lease&) = delete;
    lease(lease&&) noexcept        = default;
    lease& operator=(lease&& other) noexcept
    {
      if (this != &other) {
        stream_ = std::move(other.stream_);
        pool_   = std::move(other.pool_);
      }
      return *this;
    }
    ~lease() = default;

    void reset() noexcept
    {
      stream_.reset();
      pool_.reset();
    }
    ::cuda::stream_ref get() const noexcept { return stream_.get(); }
    const rmm::cuda_stream* operator->() const noexcept { return stream_.operator->(); }
    operator ::cuda::stream_ref() const { return get(); }
  };

  // Registration controls discovery of a pool; leases independently keep it alive.
  // Owning this as a manager member also unregisters during constructor unwinding.
  class registration {
    friend class runtime_stream_pool;
    const cucascade::memory::memory_space* space_;
    std::shared_ptr<cucascade::memory::exclusive_stream_pool> pool_;

    registration(const cucascade::memory::memory_space* space,
                 std::shared_ptr<cucascade::memory::exclusive_stream_pool> pool)
      : space_(space), pool_(std::move(pool))
    {
    }

   public:
    registration(const registration&)            = delete;
    registration& operator=(const registration&) = delete;
    registration(registration&& other) noexcept
      : space_(std::exchange(other.space_, nullptr)), pool_(std::move(other.pool_))
    {
    }
    registration& operator=(registration&& other) noexcept
    {
      if (this != &other) {
        reset();
        space_ = std::exchange(other.space_, nullptr);
        pool_  = std::move(other.pool_);
      }
      return *this;
    }
    ~registration() { reset(); }

    void reset() noexcept
    {
      if (!pool_) return;
      {
        std::lock_guard lock(mutex);
        auto it = pools.find(space_);
        // An old registration must not erase a replacement at the same address.
        if (it != pools.end() && it->second.lock() == pool_) pools.erase(it);
      }
      // Release CUDA resources outside the registry mutex.
      pool_.reset();
      space_ = nullptr;
    }
  };

  [[nodiscard]] static registration install(const cucascade::memory::memory_space& space)
  {
    auto pool = std::make_shared<cucascade::memory::exclusive_stream_pool>(
      rmm::cuda_device_id{space.get_device_id()});
    registration result{&space, std::move(pool)};
    {
      std::lock_guard lock(mutex);
      pools[&space] = result.pool_;
    }
    return result;
  }
  static lease acquire(const cucascade::memory::memory_space& space)
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
    return lease{std::move(pool)};
  }
};
}  // namespace sirius::memory
