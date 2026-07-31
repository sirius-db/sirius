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

#include "query_id.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>

namespace sirius::exec {

/// \brief Where a query is in its lifecycle, from the point of view of "may work be enqueued?".
enum class query_lifecycle_state : std::uint8_t {
  /// Normal execution: every enqueue point accepts work for this query.
  open,
  /// Teardown has begun and the drains are running. Enqueues are refused so that work cannot be
  /// added back behind a drain that already passed.
  quiescing,
};

/**
 * @brief The single authority on whether work may still be enqueued for a query.
 *
 * Sirius has four thread pools and seven places that enqueue work, several of which fire from
 * *completion callbacks* — a finishing task schedules its downstream consumers, and a downgraded
 * task pushes itself back onto the scheduler queue. During teardown those late enqueues must be
 * refused, or a drain that already ran leaves work behind pointing at a plan that is about to be
 * destroyed.
 *
 * Historically that was achieved by interrupting the *shared* queues (`multi_index_priority_queue
 * ::interrupt()`), which refuses pushes for every query at once — see the comment block in
 * `task_scheduler::drain_after_error`. This registry does the same thing per query, so one
 * query's teardown no longer silently eats another query's work.
 *
 * ### Semantics
 *
 * - `open_query()` at the start of an execution window, `quiesce()` at the start of its cleanup,
 *   `close()` once the drains are done.
 * - An **unknown** query id is treated as accepting work. This is deliberate: the failure mode of
 *   a missed `open_query()` would otherwise be a query that silently never schedules anything,
 *   i.e. a hang, which is exactly the class of bug this registry exists to remove. Components
 *   constructed without a registry (most unit tests) behave as they did before.
 * - `accepts_work()` is therefore "not known to be tearing down" rather than "known to be live".
 *
 * Thread-safe. Every method takes the mutex for a map lookup only; nothing is called with it held.
 */
class query_lifecycle_registry {
 public:
  query_lifecycle_registry()  = default;
  ~query_lifecycle_registry() = default;

  query_lifecycle_registry(const query_lifecycle_registry&)            = delete;
  query_lifecycle_registry& operator=(const query_lifecycle_registry&) = delete;
  query_lifecycle_registry(query_lifecycle_registry&&)                 = delete;
  query_lifecycle_registry& operator=(query_lifecycle_registry&&)      = delete;

  /**
   * @brief Register @p query_id as accepting work.
   *
   * Called once per execution window, before anything can be scheduled. Idempotent: re-opening an
   * already-open query is a no-op, and re-opening a quiescing one returns it to `open` (which no
   * production path does — window ids are monotonic).
   */
  void open_query(sirius::query_id_t query_id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _states[query_id] = query_lifecycle_state::open;
  }

  /**
   * @brief Refuse further work for @p query_id.
   *
   * Called at the top of the query's cleanup, before any drain runs, so that a drain cannot be
   * outrun by a completion callback enqueuing behind it. Idempotent, and a no-op for a query that
   * was never opened.
   */
  void quiesce(sirius::query_id_t query_id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _states.find(query_id);
    if (it != _states.end()) { it->second = query_lifecycle_state::quiescing; }
  }

  /**
   * @brief Forget @p query_id entirely; its window is over.
   *
   * Called after the drains complete. The entry is erased rather than kept as a tombstone, so the
   * map stays bounded by the number of in-flight queries rather than growing for the life of the
   * process. Work arriving after this point is by definition work that no drain will ever see;
   * preventing *that* is the job of the per-query pool drains and shared repository ownership,
   * not of this registry.
   */
  void close(sirius::query_id_t query_id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _states.erase(query_id);
  }

  /**
   * @brief Whether work may still be enqueued for @p query_id.
   *
   * @return false only when @p query_id is registered and quiescing. Unknown ids return true; see
   *         the class docs for why that direction is the safe one.
   */
  [[nodiscard]] bool accepts_work(sirius::query_id_t query_id) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _states.find(query_id);
    return it == _states.end() || it->second == query_lifecycle_state::open;
  }

  /// \brief The recorded state of @p query_id, or nullopt if it is not registered.
  [[nodiscard]] std::optional<query_lifecycle_state> state(sirius::query_id_t query_id) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _states.find(query_id);
    return it == _states.end() ? std::nullopt : std::optional{it->second};
  }

  /// \brief Number of registered (open or quiescing) queries.
  [[nodiscard]] std::size_t size() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _states.size();
  }

  /// \brief Drop every entry. Teardown only.
  void clear()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _states.clear();
  }

 private:
  mutable std::mutex _mutex;
  /// std::map rather than unordered_map: entries are bounded by in-flight query count (a handful),
  /// so ordered lookup is fine and iteration order stays deterministic for debugging.
  std::map<sirius::query_id_t, query_lifecycle_state> _states;
};

}  // namespace sirius::exec
