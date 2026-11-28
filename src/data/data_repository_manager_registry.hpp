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

#include <cucascade/data/data_repository_manager.hpp>

#include <cstdint>
#include <functional>
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
 * survives until the last borrower releases it instead of dangling. The manager alone does
 * not protect its repositories: spill callers must also hold the victim's lifecycle work lease.
 *
 * Thread-safe. `get_all()` returns a snapshot built under the lock so callers can iterate
 * without holding it — memory-pressure sweeps are long and blocking, and holding this mutex
 * across one would serialize query begin/end behind spilling.
 */
class data_repository_manager_registry {
 public:
  using manager_type           = cucascade::shared_data_repository_manager;
  using manager_ptr            = std::shared_ptr<manager_type>;
  using leaked_repository_info = manager_type::leaked_repository_info;
  using repository_type        = cucascade::shared_data_repository;
  /// How far in the future a repository's batches are needed; larger means later. Spill sweeps
  /// visit a query's repositories in descending rank. Must be safe to call from any thread for
  /// as long as the query's manager is registered.
  using spill_rank_fn     = std::function<std::int64_t(repository_type const*)>;
  using spill_rank_fn_ptr = std::shared_ptr<spill_rank_fn const>;

  data_repository_manager_registry()  = default;
  ~data_repository_manager_registry() = default;

  data_repository_manager_registry(const data_repository_manager_registry&)            = delete;
  data_repository_manager_registry& operator=(const data_repository_manager_registry&) = delete;
  data_repository_manager_registry(data_repository_manager_registry&&)                 = delete;
  data_repository_manager_registry& operator=(data_repository_manager_registry&&)      = delete;

  /**
   * @brief Create the manager for @p query_id.
   *
   * Called once per execution window, before any repository is wired. Windows that never wire
   * repositories (e.g. `pin_table`) simply end up with an empty manager, which keeps the
   * "inside a window implies a manager exists" invariant cheap to rely on.
   *
   * @throws std::runtime_error if @p query_id is already registered — window ids are
   *         monotonic, so a duplicate means a lifecycle bug rather than a recoverable state.
   */
  manager_ptr create_for_query(sirius::query_id_t query_id)
  {
    auto manager = std::make_shared<manager_type>();
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

  /// Install @p query_id's spill rank. Ignored if the query has no manager.
  void set_spill_rank(sirius::query_id_t query_id, spill_rank_fn_ptr rank)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_managers.contains(query_id)) { _spill_ranks[query_id] = std::move(rank); }
  }

  /// @p query_id's spill rank, or nullptr if none was installed.
  [[nodiscard]] spill_rank_fn_ptr get_spill_rank(sirius::query_id_t query_id) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _spill_ranks.find(query_id);
    return it == _spill_ranks.end() ? nullptr : it->second;
  }

  /// Identity snapshot only. A spill sweep acquires one victim's lifecycle lease before
  /// dereferencing its manager, so a snapshot cannot prolong every query's retirement.
  [[nodiscard]] std::vector<std::pair<query_id_t, manager_ptr>> candidates() const
  {
    std::lock_guard lock(_mutex);
    return {_managers.begin(), _managers.end()};
  }

  /**
   * @brief Drop @p query_id's manager and report any repositories that still held batches.
   *
   * Unregisters first, then clears the extracted manager so the report carries the same
   * `{operator_id, port_id, count}` detail the single-manager path produced.
   *
   * Precondition: every borrower of this query's repositories has been quiesced (the query
   * cleanup path closes its gate and waits for its work leases). Clearing while a worker holds a
   * raw `data_repository*` obtained from this manager would dangle — shared ownership protects
   * the manager object, not the repositories inside it.
   *
   * @return Per-repository info for each repository that still had un-consumed batches; empty
   *         when @p query_id is not registered (erasing an unknown id is a no-op).
   */
  std::vector<leaked_repository_info> erase(sirius::query_id_t query_id)
  {
    manager_ptr manager;
    spill_rank_fn_ptr rank;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      if (auto r = _spill_ranks.find(query_id); r != _spill_ranks.end()) {
        rank = std::move(r->second);
        _spill_ranks.erase(r);
      }
      auto it = _managers.find(query_id);
      if (it == _managers.end()) { return {}; }
      manager = std::move(it->second);
      _managers.erase(it);
    }
    return manager ? manager->clear_all_repositories() : std::vector<leaked_repository_info>{};
  }

  /// \brief Drop every manager. For SiriusContext teardown, after all workers are stopped.
  void clear()
  {
    std::map<query_id_t, manager_ptr> drained;
    std::map<query_id_t, spill_rank_fn_ptr> drained_ranks;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      drained.swap(_managers);
      drained_ranks.swap(_spill_ranks);
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
  mutable std::mutex _mutex;
  /// std::map (not unordered_map) so get_all() iteration is ascending by query id.
  std::map<sirius::query_id_t, manager_ptr> _managers;
  std::map<sirius::query_id_t, spill_rank_fn_ptr> _spill_ranks;
};

}  // namespace sirius::data
