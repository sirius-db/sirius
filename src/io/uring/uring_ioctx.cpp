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

#include "io/uring/uring_ioctx.hpp"

#include <algorithm>
#include <cstring>
#include <ranges>
#include <stdexcept>

namespace sirius::io {

// ---------------------------------------------------------------------------
// ring_pool
// ---------------------------------------------------------------------------

void ring_pool::init(size_t n_rings, unsigned ring_entries)
{
  _rings  = std::make_unique<io_uring[]>(n_rings);
  _in_use = std::make_unique<bool[]>(n_rings);
  _n      = n_rings;

  for (auto i : std::views::iota(size_t{0}, _n)) {
    int ret = io_uring_queue_init(ring_entries, &_rings[i], 0);
    if (ret < 0)
      throw std::runtime_error("ring_pool: io_uring_queue_init failed: " +
                               std::string(strerror(-ret)));
  }
}

ring_pool::~ring_pool()
{
  std::for_each_n(_rings.get(), _n, [](io_uring& r) { io_uring_queue_exit(&r); });
}

ring_pool::guard ring_pool::acquire()
{
  std::unique_lock lk{_mtx};
  size_t found = _n;
  _cv.wait(lk, [&] {
    auto* first = _in_use.get();
    auto* it    = std::find(first, first + _n, false);
    if (it == first + _n) return false;
    *it   = true;
    found = static_cast<size_t>(it - first);
    return true;
  });
  return guard{this, found};
}

void ring_pool::release(size_t idx)
{
  {
    std::lock_guard lk{_mtx};
    _in_use[idx] = false;
  }
  _cv.notify_one();
}

// ---------------------------------------------------------------------------
// uring_ioctx
// ---------------------------------------------------------------------------

uring_ioctx::uring_ioctx(unsigned host_ring_depth,
                         unsigned ring_entries,
                         size_t n_reactors,
                         size_t bounce_slot_size)
  : templated_ioctx<uring_reactor>(n_reactors, [ring_entries, bounce_slot_size] {
      return std::make_unique<uring_reactor>(ring_entries, bounce_slot_size);
    })
{
}

}  // namespace sirius::io
