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

#include "log/logging.hpp"
#include "query_id.hpp"

#include <cucascade/data/data_repository_manager.hpp>

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::data {

/**
 * @brief Owns one `cucascade::shared_data_repository_manager` per in-flight query.
 *
 * Replaces the single SiriusContext-wide manager. Operator ids restart at 0 for every query
 * (see `pipeline::assign_operator_ids`), so the `{operator_id, port_id}` repository keys are
 * only unique *within* a query — each query therefore needs its own manager. Ending a query
 * drops only that query's repositories instead of wiping every in-flight query's data.
 *
 * Managers are handed out as `shared_ptr` rather than references: a downgrade worker may be
 * sweeping a manager while its query ends, and shared ownership means the manager object
 * survives until the last borrower releases it instead of dangling.
 *
 * Thread-safe. `get_all()` returns a snapshot built under the lock so callers can iterate
 * without holding it — memory-pressure sweeps are long and blocking, and holding this mutex
 * across one would serialize query begin/end behind spilling.
 */
class data_repository_manager_registry {
 public:
  using manager_type = cucascade::shared_data_repository_manager;
  using manager_ptr  = std::shared_ptr<manager_type>;

  data_repository_manager_registry()  = default;
  ~data_repository_manager_registry() = default;

  data_repository_manager_registry(const data_repository_manager_registry&)            = delete;
  data_repository_manager_registry& operator=(const data_repository_manager_registry&) = delete;
  data_repository_manager_registry(data_repository_manager_registry&&)                 = delete;
  data_repository_manager_registry& operator=(data_repository_manager_registry&&)      = delete;

  /**
   * @brief RAII borrow token for a memory-pressure sweep over the registry's repositories.
   *
   * A downgrade sweep walks `get_all()` and then holds raw `data_repository*` obtained from
   * each manager across blocking work (thread-pool reserve, host/device copies). Shared
   * ownership protects the *manager* object, not the repositories inside it: `erase()` calls
   * `clear_all_repositories()`, which destroys those repositories while a sweep may still be
   * dereferencing them.
   *
   * The token turns that precondition into an enforced fence: `erase()`/`clear()` wait for
   * every active sweep to finish (and block new sweeps from starting while they wait, so a
   * steady stream of sweeps cannot starve a query's teardown). This became load-bearing when
   * per-query cleanup stopped quiescing the downgrade executors globally — a peer query's or
   * the monitor's sweep can now legitimately be mid-flight while a query tears down.
   *
   * Interim by design: shared-ownership repositories (planned) subsume it.
   */
  class sweep_guard {
   public:
    sweep_guard() = default;
    explicit sweep_guard(data_repository_manager_registry* registry) : _registry(registry) {}
    sweep_guard(sweep_guard&& other) noexcept : _registry(std::exchange(other._registry, nullptr))
    {
    }
    sweep_guard(const sweep_guard&)            = delete;
    sweep_guard& operator=(const sweep_guard&) = delete;
    sweep_guard& operator=(sweep_guard&& other) noexcept
    {
      if (this != &other) {
        if (_registry != nullptr) { _registry->end_sweep(); }
        _registry = std::exchange(other._registry, nullptr);
      }
      return *this;
    }
    ~sweep_guard()
    {
      if (_registry != nullptr) { _registry->end_sweep(); }
    }

   private:
    data_repository_manager_registry* _registry{nullptr};
  };

  /**
   * @brief Register a sweep. Blocks while a teardown (`erase()`/`clear()`) is waiting or
   *        running, then returns a token that must outlive every raw repository pointer the
   *        sweep borrows (including pointers captured by worker lambdas — hold the token until
   *        the workers are joined).
   *
   * Never called with the registry's map mutex held, and acquires no lock while held, so it
   * cannot deadlock against `get_all()`/`get()`/`create_for_query()`.
   */
  [[nodiscard]] sweep_guard begin_sweep()
  {
    std::unique_lock<std::mutex> lock(_sweep_mutex);
    _sweep_cv.wait(lock, [this] { return _pending_teardowns == 0; });
    ++_active_sweeps;
    return sweep_guard(this);
  }

  /**
   * @brief Create the manager for @p query_id.
   *
   * Called once per execution window, before any repository is wired. Windows that never wire
   * repositories (e.g. `pin_table`) simply end up with an empty manager, which keeps the
   * "inside a window implies a manager exists" invariant cheap to rely on.
   *
   * The manager gets a leak handler attributing destructor-side reports to this query: under
   * shared ownership, batches that die un-consumed are logged when their repository finally
   * dies (usually inside erase(); later if a sweep still borrows it), not at erase time.
   *
   * @throws std::runtime_error if @p query_id is already registered — window ids are
   *         monotonic, so a duplicate means a lifecycle bug rather than a recoverable state.
   */
  manager_ptr create_for_query(sirius::query_id_t query_id)
  {
    auto manager = std::make_shared<manager_type>();
    manager->set_leak_handler(
      [query_id](std::size_t operator_id, const std::string& port_id, std::size_t count) {
        // Runs on whatever thread drops the last repository reference (the erasing
        // cleanup thread, or a downgrade worker releasing its sweep snapshot); the
        // manager already swallows a throwing handler, this is belt to that suspender.
        try {
          SIRIUS_LOG_WARN(
            "data_repository_manager_registry: query {} operator {} port '{}' repository died "
            "with {} un-consumed data batch(es) (memory leak).",
            sirius::value_of(query_id),
            operator_id,
            port_id,
            count);
        } catch (...) {  // best-effort observability
        }
      });
    {
      std::lock_guard<std::mutex> lock(_mutex);
      auto [it, inserted] = _managers.emplace(query_id, std::move(manager));
      if (!inserted) {
        throw std::runtime_error(
          "data_repository_manager_registry: a manager is already registered for query " +
          std::to_string(sirius::value_of(query_id)));
      }
      return it->second;
    }
  }

  /// \brief The manager for @p query_id, or nullptr if none is registered.
  [[nodiscard]] manager_ptr get(sirius::query_id_t query_id) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _managers.find(query_id);
    return it == _managers.end() ? nullptr : it->second;
  }

  /**
   * @brief Snapshot of every live manager, in ascending query-id order.
   *
   * Used by the downgrade executors, which must see across all in-flight queries because
   * memory pressure is a global condition. Ordering is deterministic so spill-candidate
   * selection stays reproducible across runs. Note that the executors walk this snapshot in
   * REVERSE (newest query first) to keep the oldest query's data resident; ascending is just
   * the canonical order to return, not the sweep order.
   */
  [[nodiscard]] std::vector<manager_ptr> get_all() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<manager_ptr> result;
    result.reserve(_managers.size());
    for (const auto& [id, manager] : _managers) {
      result.push_back(manager);
    }
    return result;
  }

  /**
   * @brief Drop @p query_id's manager. Erasing an unknown id is a no-op.
   *
   * Only the MAP ENTRY is dropped — the manager (and each repository inside it) dies when
   * its last holder releases it. In the common case that is right here, synchronously; if a
   * memory-pressure sweep still holds the manager or one of its repositories (both are
   * handed out as shared_ptr), the sweep keeps them alive until it naturally finishes, and
   * teardown never has to wait for it. Batches that die un-consumed are reported by the
   * repository destructors through the leak handler installed in create_for_query(),
   * attributed to this query — wherever and whenever they actually die.
   */
  void erase(sirius::query_id_t query_id)
  {
    teardown_scope fence(*this);
    manager_ptr manager;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      auto it = _managers.find(query_id);
      if (it == _managers.end()) { return; }
      manager = std::move(it->second);
      _managers.erase(it);
    }
    // Released outside the lock: if this is the last reference, manager destruction releases
    // data batches, which can run arbitrary deallocation work that must not happen with the
    // registry mutex held.
    manager.reset();
  }

  /// \brief Drop every manager. For SiriusContext teardown, after all workers are stopped
  ///        (the sweep fence is then uncontended and acquired immediately).
  void clear()
  {
    teardown_scope fence(*this);
    std::map<query_id_t, manager_ptr> drained;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      drained.swap(_managers);
    }
    // Destroyed outside the lock: manager destruction releases data batches, which can run
    // arbitrary deallocation work that must not happen with the registry mutex held.
  }

  /// \brief Number of queries currently holding a manager.
  [[nodiscard]] size_t size() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _managers.size();
  }

 private:
  /// Sweep-side release for sweep_guard.
  void end_sweep()
  {
    {
      std::lock_guard<std::mutex> lock(_sweep_mutex);
      --_active_sweeps;
    }
    _sweep_cv.notify_all();
  }

  /// Teardown-side bracket: registers intent first (so no NEW sweep can start and starve the
  /// wait), then waits until every active sweep has released its token. Multiple teardowns may
  /// proceed concurrently — the map mutex covers the map, and each erases a distinct manager.
  class teardown_scope {
   public:
    explicit teardown_scope(data_repository_manager_registry& registry) : _registry(registry)
    {
      std::unique_lock<std::mutex> lock(_registry._sweep_mutex);
      ++_registry._pending_teardowns;
      _registry._sweep_cv.wait(lock, [this] { return _registry._active_sweeps == 0; });
    }
    teardown_scope(const teardown_scope&)            = delete;
    teardown_scope& operator=(const teardown_scope&) = delete;
    ~teardown_scope()
    {
      {
        std::lock_guard<std::mutex> lock(_registry._sweep_mutex);
        --_registry._pending_teardowns;
      }
      _registry._sweep_cv.notify_all();
    }

   private:
    data_repository_manager_registry& _registry;
  };

  mutable std::mutex _mutex;
  /// std::map (not unordered_map) so get_all() iteration is ascending by query id.
  std::map<sirius::query_id_t, manager_ptr> _managers;

  /// Sweep fence state (see begin_sweep()/teardown_scope). Separate from _mutex on purpose:
  /// sweeps hold their token across long blocking work, and _mutex must stay cheap.
  std::mutex _sweep_mutex;
  std::condition_variable _sweep_cv;
  int _active_sweeps{0};
  int _pending_teardowns{0};
};

}  // namespace sirius::data
