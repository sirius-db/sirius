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

#pragma once

#include "io/templated_ioctx.hpp"
#include "io/uring/uring_reactor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>

namespace sirius::io {

// ---------------------------------------------------------------------------
// ring_pool
// ---------------------------------------------------------------------------

/**
 * @brief Pool of @c io_uring instances for host (buffered) reads.
 */
class ring_pool {
 public:
  struct guard {
    guard(ring_pool* p, size_t i) noexcept : _pool(p), _idx(i) {}

    io_uring& ring() const { return _pool->_rings[_idx]; }

    ~guard()
    {
      if (_pool) _pool->release(_idx);
    }

    guard(guard const&)            = delete;
    guard& operator=(guard const&) = delete;
    guard(guard&& o) noexcept : _pool(std::exchange(o._pool, nullptr)), _idx(o._idx) {}

   private:
    ring_pool* _pool{nullptr};
    size_t _idx{0};
  };

  ring_pool()                            = default;
  ring_pool(ring_pool const&)            = delete;
  ring_pool& operator=(ring_pool const&) = delete;

  void init(size_t n_rings, unsigned ring_entries);
  ~ring_pool();
  guard acquire();

 private:
  void release(size_t idx);

  std::unique_ptr<io_uring[]> _rings;
  std::unique_ptr<bool[]> _in_use;
  size_t _n{0};
  std::mutex _mtx;
  std::condition_variable _cv;
};

// ---------------------------------------------------------------------------
// uring_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief io_uring-backed ioctx.  Thin specialisation of
 *        @c templated_ioctx<uring_reactor> adding a side-channel
 *        @c ring_pool for callers that need direct access to a host ring.
 */
class uring_ioctx : public templated_ioctx<uring_reactor> {
 public:
  explicit uring_ioctx(unsigned host_ring_depth = 16,
                       unsigned ring_entries    = 64,
                       size_t n_reactors        = 4,
                       size_t bounce_slot_size  = 1UL * 1024 * 1024);
};

}  // namespace sirius::io
