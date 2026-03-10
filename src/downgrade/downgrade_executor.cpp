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

#include "downgrade/downgrade_executor.hpp"

#include "duckdb/common/types/uuid.hpp"
#include "log/logging.hpp"

#include <rmm/cuda_stream.hpp>

#include <algorithm>
#include <optional>
#include <thread>

namespace sirius {
namespace parallel {

void downgrade_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  on_start();
  // Tell the queue how many workers to wake on close()
  auto* dq = dynamic_cast<downgrade_task_queue*>(_task_queue.get());
  if (dq) { dq->set_num_workers(_config.num_threads); }
  _threads.reserve(_config.num_threads);
  for (int i = 0; i < _config.num_threads; ++i) {
    _threads.emplace_back(&downgrade_executor::worker_loop, this, i);
  }
  _monitor_thread = std::thread(&downgrade_executor::monitor_loop, this);
}

void downgrade_executor::drain()
{
  // Stop then restart — ensures all in-flight tasks complete and the queue is empty
  stop();
  start();
}

void downgrade_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  on_stop();
  if (_monitor_thread.joinable()) { _monitor_thread.join(); }
  for (auto& thread : _threads) {
    if (thread.joinable()) { thread.join(); }
  }
  _threads.clear();
}

void downgrade_executor::worker_loop(int worker_id)
{
  // cudaMemcpyBatchAsync requires a real (non-default) CUDA stream
  rmm::cuda_stream stream;

  while (true) {
    if (!_running.load()) { break; }
    auto task = _task_queue->pull();
    if (task == nullptr) { break; }
    try {
      task->execute(stream);
    } catch (const std::exception& e) {
      // Downgrade failures are non-fatal — log and continue to the next task.
      // Do NOT call on_task_error() here: it calls stop() from within the worker
      // thread, which tries to join itself and causes a deadlock.
      SIRIUS_LOG_ERROR("[downgrade] worker {} task failed: {}", worker_id, e.what());
    }
  }
}

void downgrade_executor::monitor_loop()
{
  using namespace std::chrono_literals;

  while (_running.load()) {
    if (_memory_space && _memory_space->should_downgrade_memory()) {
      size_t amount = _memory_space->get_amount_to_downgrade();
      if (amount > 0) {
        // Collect all repositories from the manager
        std::vector<downgrade_repository_info> repos;
        _data_repo_mgr.for_each_repository(
          [&repos](cucascade::shared_data_repository* repo) { repos.push_back({repo}); });
        if (!repos.empty()) { run_downgrade_pass(std::move(repos), amount); }
      }
    }
    // Brief sleep to avoid busy-spinning; the monitor re-checks after each interval
    std::this_thread::sleep_for(10ms);
  }
}

// --- Static helpers ---

size_t downgrade_executor::get_repo_data_size_on_tier(cucascade::shared_data_repository* repo,
                                                      cucascade::memory::Tier tier)
{
  size_t total = 0;
  for (size_t p = 0; p < repo->num_partitions(); ++p) {
    auto batch_ids = repo->get_batch_ids(p);
    for (auto id : batch_ids) {
      auto batch = repo->get_data_batch_by_id(id, std::nullopt, p);
      if (!batch || !batch->get_data()) continue;
      auto* ms = batch->get_memory_space();
      if (!ms) continue;
      if (ms->get_tier() == tier) { total += batch->get_data()->get_size_in_bytes(); }
    }
  }
  return total;
}

bool downgrade_executor::is_partition_active(cucascade::shared_data_repository* repo,
                                             size_t partition_idx)
{
  auto batch_ids = repo->get_batch_ids(partition_idx);
  for (auto id : batch_ids) {
    auto batch = repo->get_data_batch_by_id(id, std::nullopt, partition_idx);
    if (!batch) continue;
    auto state = batch->get_state();
    if (state == cucascade::batch_state::task_created ||
        state == cucascade::batch_state::processing) {
      return true;
    }
  }
  return false;
}

std::vector<std::shared_ptr<cucascade::data_batch>>
downgrade_executor::collect_candidates_from_partition(
  cucascade::shared_data_repository* repo,
  size_t partition_idx,
  cucascade::memory::memory_space_id source_space,
  size_t max_bytes,
  size_t& collected_bytes)
{
  std::vector<std::shared_ptr<cucascade::data_batch>> candidates;
  auto batch_ids = repo->get_batch_ids(partition_idx);
  for (auto id : batch_ids) {
    if (max_bytes > 0 && collected_bytes >= max_bytes) break;
    auto batch = repo->get_data_batch_by_id(id, std::nullopt, partition_idx);
    if (!batch || !batch->get_data()) continue;
    if (batch->get_state() != cucascade::batch_state::idle) continue;
    auto* ms = batch->get_memory_space();
    if (!ms || ms->get_id() != source_space) continue;

    collected_bytes += batch->get_data()->get_size_in_bytes();
    candidates.push_back(std::move(batch));
  }
  return candidates;
}

// --- Main selection + scheduling logic ---

size_t downgrade_executor::run_downgrade_pass(std::vector<downgrade_repository_info> repositories,
                                              size_t amount_to_downgrade)
{
  auto source_tier = _space_id.tier;

  struct scored_repo {
    cucascade::shared_data_repository* repo;
    size_t tier_data_size;
    bool is_partitioned;
  };

  std::vector<scored_repo> scored_repos;
  for (auto& info : repositories) {
    if (!info.repo) continue;
    size_t tier_size = get_repo_data_size_on_tier(info.repo, source_tier);
    if (tier_size == 0) continue;
    scored_repos.push_back({info.repo, tier_size, info.repo->num_partitions() > 1});
  }

  std::sort(
    scored_repos.begin(), scored_repos.end(), [](const scored_repo& a, const scored_repo& b) {
      if (a.is_partitioned != b.is_partitioned) return a.is_partitioned > b.is_partitioned;
      return a.tier_data_size > b.tier_data_size;
    });

  std::vector<std::shared_ptr<cucascade::data_batch>> all_candidates;
  size_t collected_bytes = 0;

  // Pass 1: Non-active partitions (last to first)
  for (auto& sr : scored_repos) {
    if (collected_bytes >= amount_to_downgrade) break;
    for (size_t i = sr.repo->num_partitions(); i > 0; --i) {
      if (collected_bytes >= amount_to_downgrade) break;
      size_t pidx = i - 1;
      if (is_partition_active(sr.repo, pidx)) continue;
      auto candidates = collect_candidates_from_partition(
        sr.repo, pidx, _space_id, amount_to_downgrade, collected_bytes);
      for (auto& c : candidates) {
        all_candidates.push_back(std::move(c));
      }
    }
  }

  // Pass 2: Active partitions (last to first)
  if (collected_bytes < amount_to_downgrade) {
    for (auto& sr : scored_repos) {
      if (collected_bytes >= amount_to_downgrade) break;
      for (size_t i = sr.repo->num_partitions(); i > 0; --i) {
        if (collected_bytes >= amount_to_downgrade) break;
        size_t pidx = i - 1;
        if (!is_partition_active(sr.repo, pidx)) continue;
        auto candidates = collect_candidates_from_partition(
          sr.repo, pidx, _space_id, amount_to_downgrade, collected_bytes);
        for (auto& c : candidates) {
          all_candidates.push_back(std::move(c));
        }
      }
    }
  }

  if (all_candidates.empty()) return 0;

  auto global_state = std::make_shared<downgrade_task_global_state>(
    _reservation_manager, _data_repo_mgr, _message_queue);

  size_t task_count = 0;
  for (auto& batch : all_candidates) {
    auto data_size = batch->get_data() ? batch->get_data()->get_size_in_bytes() : 0;
    SIRIUS_LOG_TRACE("[downgrade] scheduling batch {} ({} B) on tier {}",
                     batch->get_batch_id(),
                     data_size,
                     static_cast<int>(source_tier));
    auto local_state = std::make_unique<downgrade_task_local_state>(
      duckdb::UUIDv7().GenerateRandomUUID().lower, 0, std::move(batch));
    auto task = std::make_unique<downgrade_task>(std::move(local_state), global_state);
    schedule_downgrade_task(std::move(task));
    ++task_count;
  }

  return task_count;
}

}  // namespace parallel
}  // namespace sirius
