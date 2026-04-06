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

#include "log/logging.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <optional>
#include <thread>

namespace sirius {
namespace parallel {

downgrade_executor::downgrade_executor(
  exec::thread_pool_config config,
  cucascade::shared_data_repository_manager& data_repo_mgr,
  cucascade::memory::memory_space_id space_id,
  cucascade::memory::memory_space* memory_space,
  sirius::memory::sirius_memory_reservation_manager& reservation_manager)
  : _config(std::move(config)),
    _data_repo_mgr(data_repo_mgr),
    _space_id(space_id),
    _memory_space(memory_space),
    _reservation_manager(reservation_manager)
{
}

downgrade_executor::~downgrade_executor() { stop(); }

void downgrade_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }

  if (_memory_space) { cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking); }

  _request_queue.reactivate();

  absl::AnyInvocable<void() noexcept> per_thread_init = nullptr;
  if (_memory_space) {
    auto device_id = _memory_space->get_device_id();
    per_thread_init = [device_id]() noexcept { cudaSetDevice(device_id); };
  }

  _pool = std::make_unique<exec::bounded_thread_pool>(
    _config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list,
    std::move(per_thread_init));

  _processing_thread = std::thread(&downgrade_executor::processing_loop, this);

  if (_memory_space) {
    _monitor_thread = std::thread(&downgrade_executor::monitor_loop, this);
  }
}

void downgrade_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }

  _pool->interrupt();
  _request_queue.interrupt();

  if (_monitor_thread.joinable()) { _monitor_thread.join(); }
  if (_processing_thread.joinable()) { _processing_thread.join(); }

  _pool->wait_all();
  _pool->stop();
  _pool.reset();

  if (_stream) {
    cudaStreamDestroy(_stream);
    _stream = nullptr;
  }
}

void downgrade_executor::drain()
{
  _pool->interrupt();
  _request_queue.interrupt();

  if (_processing_thread.joinable()) { _processing_thread.join(); }

  _pool->wait_all();
  _request_queue.drain();
  _pool->resume();
  _request_queue.reactivate();

  _processing_thread = std::thread(&downgrade_executor::processing_loop, this);
}

void downgrade_executor::processing_loop()
{
  while (_running.load()) {
    auto request = _request_queue.pop();
    if (!request) break;  // interrupted

    // 1. Collect candidates (single-threaded)
    std::vector<downgrade_repository_info> repos;
    _data_repo_mgr.for_each_repository(
      [&repos](cucascade::shared_data_repository* repo) { repos.push_back({repo}); });

    auto candidates = collect_all_candidates(repos, request->target_bytes);

    // 2. Dispatch all to pool
    for (auto& batch : candidates) {
      auto slot = _pool->reserve();
      if (!slot) break;  // interrupted
      _pool->dispatch(
        std::move(slot),
        [batch = std::move(batch), &res_mgr = _reservation_manager,
         stream = rmm::cuda_stream_view{_stream}]() {
          downgrade_task task{batch, res_mgr};
          try {
            task.execute(stream);
          } catch (const std::exception& e) {
            SIRIUS_LOG_ERROR("[downgrade] batch downgrade failed: {}", e.what());
          }
        });
    }

    // 3. Wait for all dispatched batch downgrades to complete
    _pool->wait_all();
  }
}

void downgrade_executor::monitor_loop()
{
  using namespace std::chrono_literals;

  while (_running.load()) {
    if (_memory_space && _memory_space->should_downgrade_memory()) {
      size_t amount = _memory_space->get_amount_to_downgrade();
      if (amount > 0) {
        auto req         = std::make_unique<downgrade_request>();
        req->target_bytes = amount;
        _request_queue.push(std::move(req));
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

// --- Candidate collection helper ---

std::vector<std::shared_ptr<cucascade::data_batch>> downgrade_executor::collect_all_candidates(
  const std::vector<downgrade_repository_info>& repositories, size_t target_bytes)
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
    if (collected_bytes >= target_bytes) break;
    for (size_t i = sr.repo->num_partitions(); i > 0; --i) {
      if (collected_bytes >= target_bytes) break;
      size_t pidx = i - 1;
      if (is_partition_active(sr.repo, pidx)) continue;
      auto candidates = collect_candidates_from_partition(
        sr.repo, pidx, _space_id, target_bytes, collected_bytes);
      for (auto& c : candidates) {
        all_candidates.push_back(std::move(c));
      }
    }
  }

  // Pass 2: Active partitions (last to first)
  if (collected_bytes < target_bytes) {
    for (auto& sr : scored_repos) {
      if (collected_bytes >= target_bytes) break;
      for (size_t i = sr.repo->num_partitions(); i > 0; --i) {
        if (collected_bytes >= target_bytes) break;
        size_t pidx = i - 1;
        if (!is_partition_active(sr.repo, pidx)) continue;
        auto candidates = collect_candidates_from_partition(
          sr.repo, pidx, _space_id, target_bytes, collected_bytes);
        for (auto& c : candidates) {
          all_candidates.push_back(std::move(c));
        }
      }
    }
  }

  return all_candidates;
}

// --- Public downgrade pass methods ---

size_t downgrade_executor::run_downgrade_pass_all_repos(size_t amount_to_downgrade)
{
  std::vector<downgrade_repository_info> repos;
  _data_repo_mgr.for_each_repository(
    [&repos](cucascade::shared_data_repository* repo) { repos.push_back({repo}); });
  if (repos.empty()) return 0;
  return run_downgrade_pass(std::move(repos), amount_to_downgrade);
}

size_t downgrade_executor::run_downgrade_pass(std::vector<downgrade_repository_info> repositories,
                                              size_t amount_to_downgrade)
{
  auto all_candidates = collect_all_candidates(repositories, amount_to_downgrade);

  if (all_candidates.empty()) return 0;

  size_t task_count = 0;
  for (auto& batch : all_candidates) {
    auto data_size = batch->get_data() ? batch->get_data()->get_size_in_bytes() : 0;
    SIRIUS_LOG_TRACE("[downgrade] scheduling batch {} ({} B) on tier {}",
                     batch->get_batch_id(),
                     data_size,
                     static_cast<int>(_space_id.tier));
    auto slot = _pool->reserve();
    if (!slot) break;
    _pool->dispatch(
      std::move(slot),
      [batch = std::move(batch), &res_mgr = _reservation_manager,
       stream = rmm::cuda_stream_view{_stream}]() {
        downgrade_task task{batch, res_mgr};
        try {
          task.execute(stream);
        } catch (const std::exception& e) {
          SIRIUS_LOG_ERROR("[downgrade] batch downgrade failed: {}", e.what());
        }
      });
    ++task_count;
  }

  return task_count;
}

}  // namespace parallel
}  // namespace sirius
