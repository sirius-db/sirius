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

#include <algorithm>
#include <chrono>
#include <optional>
#include <thread>

namespace sirius {
namespace parallel {

downgrade_executor::downgrade_executor(
  exec::downgrade_executor_config config,
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

  {
    auto device_id = _memory_space ? _memory_space->get_device_id() : 0;
    _stream_pool   = std::make_unique<cucascade::memory::exclusive_stream_pool>(
      rmm::cuda_device_id{device_id}, _config.thread_pool.num_threads);
  }

  _request_queue.reactivate();

  absl::AnyInvocable<void() noexcept> per_thread_init = nullptr;
  if (_memory_space) {
    auto device_id  = _memory_space->get_device_id();
    per_thread_init = [device_id]() noexcept { cudaSetDevice(device_id); };
  }

  _pool = std::make_unique<exec::bounded_thread_pool>(_config.thread_pool.num_threads,
                                                      _config.thread_pool.thread_name_prefix,
                                                      _config.thread_pool.cpu_affinity_list,
                                                      std::move(per_thread_init));

  _processing_thread = std::thread(&downgrade_executor::processing_loop, this);

  if (_memory_space && _config.monitor_period_ms > 0) {
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
  cancel_pending_requests();
  _pool->stop();
  _pool.reset();
  _stream_pool.reset();
}

void downgrade_executor::drain()
{
  _pool->interrupt();
  _request_queue.interrupt();

  if (_processing_thread.joinable()) { _processing_thread.join(); }

  _pool->wait_all();
  cancel_pending_requests();
  _pool->resume();
  _request_queue.reactivate();

  _processing_thread = std::thread(&downgrade_executor::processing_loop, this);
}

void downgrade_executor::processing_loop()
{
  while (_running.load()) {
    auto request = _request_queue.pop();
    if (!request) break;  // interrupted

    auto& req    = request;  // unique_ptr<downgrade_request>
    auto t_start = std::chrono::steady_clock::now();

    // 1. Collect repos
    std::vector<downgrade_repository_info> repos;
    _data_repo_mgr.for_each_repository(
      [&repos](cucascade::shared_data_repository* repo) { repos.push_back({repo}); });

    // 2. Collect candidates
    auto candidates = collect_all_candidates(repos, req->target_bytes);

    // 3. Incremental dispatch with predicate checking
    for (auto& weak_batch : candidates) {
      if (req->satisfied.load()) break;

      auto batch = weak_batch.lock();
      if (!batch) continue;  // batch was freed elsewhere — skip

      auto slot = _pool->reserve();
      if (!slot) break;  // interrupted

      auto batch_size = batch->get_data() ? batch->get_data()->get_size_in_bytes() : 0;

      auto exc_stream = _stream_pool->acquire_stream(
        cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);

      auto preferred_numa_node = _config.preferred_numa_node;

      _pool->dispatch(std::move(slot),
                      [batch      = std::move(batch),
                       req_ptr    = req.get(),
                       &res_mgr   = _reservation_manager,
                       exc_stream = std::move(exc_stream),
                       batch_size,
                       preferred_numa_node]() mutable {
                        // Thread the NUMA preference (from downgrade_executor_config,
                        // re-authored v1.0 dd86dd0 intent) into the POD task so its
                        // execute() path can select a NUMA-local HOST memory_space via
                        // cucascade::memory::any_memory_space_in_tier_with_preference.
                        downgrade_task task{batch, res_mgr, preferred_numa_node};
                        try {
                          if (task.execute(exc_stream)) {
                            req_ptr->bytes_freed.fetch_add(batch_size, std::memory_order_relaxed);
                            req_ptr->batches_downgraded.fetch_add(1, std::memory_order_relaxed);
                            if (req_ptr->predicate && req_ptr->predicate()) {
                              req_ptr->satisfied.store(true);
                            }
                          }
                        } catch (const std::exception& e) {
                          SIRIUS_LOG_ERROR("[downgrade] batch downgrade failed: {}", e.what());
                        }
                      });
    }

    // 4. Let in-flight batches finish
    _pool->wait_all();

    auto total_bytes   = req->bytes_freed.load(std::memory_order_relaxed);
    auto total_batches = req->batches_downgraded.load(std::memory_order_relaxed);
    auto duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    double throughput_mbs =
      (duration_ms > 0.0) ? (total_bytes / (1024.0 * 1024.0)) / (duration_ms / 1000.0) : 0.0;
    std::string is_monitor_request = req->is_monitor_request ? "from monitor " : "";
    SIRIUS_LOG_DEBUG(
      "[downgrade] request {}done: {} target_bytes, {} batches, {} bytes in {:.2f} ms ({:.1f} "
      "MB/s)",
      is_monitor_request,
      req->target_bytes,
      total_batches,
      total_bytes,
      duration_ms,
      throughput_mbs);

    // 5. Fulfill the promise with total bytes freed
    req->result.set_value(total_bytes);
  }
}

void downgrade_executor::monitor_loop()
{
  using namespace std::chrono_literals;

  while (_running.load()) {
    if (_memory_space && _memory_space->should_downgrade_memory()) {
      size_t amount = _memory_space->get_amount_to_downgrade();
      if (amount > 0) {
        auto req                = std::make_unique<downgrade_request>();
        req->target_bytes       = amount;
        req->is_monitor_request = true;
        req->predicate          = [&freed = req->bytes_freed, amount]() {
          return freed.load(std::memory_order_relaxed) >= amount;
        };
        // Fire-and-forget: monitor does not wait for the result
        _request_queue.push(std::move(req));
      }
    }
    // Brief sleep to avoid busy-spinning; the monitor re-checks after each interval
    std::this_thread::sleep_for(std::chrono::milliseconds(_config.monitor_period_ms));
  }
}

void downgrade_executor::cancel_pending_requests()
{
  while (auto req = _request_queue.try_pop()) {
    try {
      req->result.set_exception(
        std::make_exception_ptr(std::runtime_error("downgrade executor shutting down")));
    } catch (...) {
      // Promise may already be fulfilled — safe to ignore
    }
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

std::vector<std::weak_ptr<cucascade::data_batch>>
downgrade_executor::collect_candidates_from_partition(
  cucascade::shared_data_repository* repo,
  size_t partition_idx,
  cucascade::memory::memory_space_id source_space,
  size_t max_bytes,
  size_t& collected_bytes)
{
  std::vector<std::weak_ptr<cucascade::data_batch>> candidates;
  auto batch_ids = repo->get_batch_ids(partition_idx);
  for (auto id : batch_ids) {
    if (max_bytes > 0 && collected_bytes >= max_bytes) break;
    auto batch = repo->get_data_batch_by_id(id, std::nullopt, partition_idx);
    if (!batch || !batch->get_data()) continue;
    if (batch->get_state() != cucascade::batch_state::idle) continue;
    auto* ms = batch->get_memory_space();
    if (!ms || ms->get_id() != source_space) continue;

    collected_bytes += batch->get_data()->get_size_in_bytes();
    candidates.emplace_back(batch);
  }
  return candidates;
}

// --- Candidate collection helper ---

std::vector<std::weak_ptr<cucascade::data_batch>> downgrade_executor::collect_all_candidates(
  const std::vector<downgrade_repository_info>& repositories, size_t amount_to_downgrade)
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

  std::vector<std::weak_ptr<cucascade::data_batch>> all_candidates;
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

  return all_candidates;
}

// --- Public request API ---

std::future<size_t> downgrade_executor::request_free_memory(size_t bytes)
{
  auto req          = std::make_unique<downgrade_request>();
  req->target_bytes = bytes;
  req->predicate    = [&freed = req->bytes_freed, bytes]() {
    return freed.load(std::memory_order_relaxed) >= bytes;
  };
  auto future = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN(
      "[downgrade] request_free_memory: queue inactive, dropping request for {} bytes", bytes);
  }
  return future;
}

size_t downgrade_executor::request_free_memory_and_wait(size_t bytes)
{
  return request_free_memory(bytes).get();
}

std::future<size_t> downgrade_executor::request_downgrade(size_t target_bytes,
                                                          std::function<bool()> predicate)
{
  auto req          = std::make_unique<downgrade_request>();
  req->target_bytes = target_bytes;
  req->predicate    = std::move(predicate);
  auto future       = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request for {} bytes",
                    target_bytes);
    req->result.set_value(0);
    return future;
  }
  return future;
}

}  // namespace parallel
}  // namespace sirius
