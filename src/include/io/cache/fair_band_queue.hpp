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

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <utility>

namespace sirius::io::cache {

// ---------------------------------------------------------------------------
// fair_band_queue — blocking MPMC queue with round-robin banding
// ---------------------------------------------------------------------------
//
// A drop-in replacement for the prefetching cache's strict-FIFO request
// queues (concurrency issue F4): one query flooding thousands of requests
// used to fully block another query's *first* prefetch, because the cache's
// single preparation/prefetch/evictor threads drained one global FIFO.
//
// Items are kept FIFO *within* a band (a band is one query's epoch), and the
// pop side rotates round-robin *across* the non-empty bands: each pop serves
// the next band strictly after the last one served, wrapping.  With a single
// band the order is bit-identical to a plain FIFO, so single-query behavior
// is unchanged.  With N non-empty bands, any band's oldest item is served
// within N pops — a flood in one band cannot starve the others.
//
// This is deliberately a simple mutex + per-band deque structure (the F1
// fix's rotation shape, minus the priority index): the cache enqueues one
// request per fadvise batch, not per chunk, so the queues are never hot
// enough to need a lock-free design.
template <class T>
class fair_band_queue {
 public:
  using band_t = std::uint32_t;

  /// Band for items with no query affiliation (wake-up sentinels, requests
  /// from an unregistered caller).  Participates in the rotation like any
  /// other band, so sentinels are never starved.
  static constexpr band_t no_band = 0;

  void enqueue(band_t band, T item)
  {
    {
      std::lock_guard lk(_mtx);
      _bands[band].push_back(std::move(item));
      ++_size;
    }
    _cv.notify_one();
  }

  /// Block until an item is available, then pop it (round-robin across
  /// bands, FIFO within a band).
  void wait_dequeue(T& out)
  {
    std::unique_lock lk(_mtx);
    _cv.wait(lk, [&] { return _size > 0; });
    out = pop_next_locked();
  }

  /// Non-blocking pop.  Returns false when the queue is empty.
  bool try_dequeue(T& out)
  {
    std::lock_guard lk(_mtx);
    if (_size == 0) { return false; }
    out = pop_next_locked();
    return true;
  }

  [[nodiscard]] std::size_t size_approx() const
  {
    std::lock_guard lk(_mtx);
    return _size;
  }

 private:
  T pop_next_locked()
  {
    // Serve the first band strictly after the last one served, wrapping to
    // the lowest.  _bands only holds non-empty bands, so this always lands
    // on a poppable deque.
    auto it = _bands.upper_bound(_last_band);
    if (it == _bands.end()) { it = _bands.begin(); }
    _last_band = it->first;
    T item     = std::move(it->second.front());
    it->second.pop_front();
    if (it->second.empty()) { _bands.erase(it); }
    --_size;
    return item;
  }

  mutable std::mutex _mtx;
  std::condition_variable _cv;
  std::map<band_t, std::deque<T>> _bands;  // non-empty bands only
  band_t _last_band{no_band};
  std::size_t _size{0};
};

}  // namespace sirius::io::cache
