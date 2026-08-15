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

#include "query_id.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

namespace sirius::io::cache {

// ---------------------------------------------------------------------------
// query_epoch_tracker — the cache's set of LIVE query epochs
// ---------------------------------------------------------------------------
//
// Replaces the prefetching cache's single global newest-wins ticker
// (concurrency issue F3): scoring chunks against the *newest* epoch meant a
// second query's prepare_for_query demoted every chunk the first query had
// prefetched-but-not-yet-read to eviction tier 0.
//
// Each query gets a monotonically increasing epoch at begin_query() (epochs
// start at 1; 0 means "before any query").  The epoch stays *live* until
// end_query() retires it at window cleanup.  Chunk scoring uses
// min_live_epoch() as the staleness bar: a chunk is stale only when its
// stamped epoch is older than EVERY live query's epoch — i.e. no live query
// can still be counting on it.  With a single query at a time this collapses
// to the old ticker semantics exactly (min live == newest).
//
// When no query is live, min_live_epoch() falls back to newest_epoch(): the
// last query's chunks keep their demand-based tier between queries, which is
// what the old single-ticker behavior provided (the ticker only advanced on
// the *next* prepare).  A missed end_query() therefore never corrupts
// anything — it only over-protects that query's chunks, and the evictor's
// cutoff fallback still reclaims them under pressure.
//
// Reads of newest/min-live are lock-free atomics (they sit on the insert and
// eviction paths); the map of live queries is tiny (bounded by the admission
// gate's max_concurrent_queries) and only touched at query begin/end.
class query_epoch_tracker {
 public:
  /// Register @p id as live and return its freshly minted epoch.
  /// Registering an id twice (a stale window whose cleanup never ran)
  /// simply re-stamps it with the new epoch.
  std::uint32_t begin_query(sirius::query_id_t id)
  {
    std::lock_guard lk(_mtx);
    auto const epoch = _newest.load(std::memory_order_relaxed) + 1;
    _newest.store(epoch, std::memory_order_relaxed);
    _live[id] = epoch;
    recompute_min_live_locked();
    return epoch;
  }

  /// Retire @p id's epoch.  Idempotent; unknown ids are ignored.
  void end_query(sirius::query_id_t id) noexcept
  {
    std::lock_guard lk(_mtx);
    _live.erase(id);
    recompute_min_live_locked();
  }

  /// The epoch @p id's requests should be stamped with.  Falls back to the
  /// newest epoch when @p id is not (or no longer) live, matching the old
  /// ticker-stamp behavior for unregistered callers.
  [[nodiscard]] std::uint32_t epoch_of(sirius::query_id_t id) const
  {
    std::lock_guard lk(_mtx);
    auto const it = _live.find(id);
    return it != _live.end() ? it->second : _newest.load(std::memory_order_relaxed);
  }

  /// The most recently minted epoch (0 before the first begin_query).
  [[nodiscard]] std::uint32_t newest_epoch() const noexcept
  {
    return _newest.load(std::memory_order_acquire);
  }

  /// The staleness bar for chunk scoring: the oldest LIVE epoch, or the
  /// newest epoch when no query is live (see the class comment).
  [[nodiscard]] std::uint32_t min_live_epoch() const noexcept
  {
    return _min_live.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t live_count() const
  {
    std::lock_guard lk(_mtx);
    return _live.size();
  }

 private:
  void recompute_min_live_locked() noexcept
  {
    auto bar = _newest.load(std::memory_order_relaxed);
    for (auto const& [id, epoch] : _live) {
      bar = std::min(bar, epoch);
    }
    _min_live.store(bar, std::memory_order_release);
  }

  mutable std::mutex _mtx;
  std::atomic<std::uint32_t> _newest{0};
  std::atomic<std::uint32_t> _min_live{0};
  std::map<sirius::query_id_t, std::uint32_t> _live;
};

}  // namespace sirius::io::cache
