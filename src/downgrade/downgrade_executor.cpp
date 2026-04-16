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

#include "data/convertible_data.hpp"
#include "data/convertible_data_batch.hpp"
#include "data/convertible_gpu_pipeline_task.hpp"
#include "log/logging.hpp"

#include <chrono>
#include <thread>

namespace sirius {
namespace parallel {

downgrade_executor::downgrade_executor(
  exec::downgrade_executor_config config,
  cucascade::shared_data_repository_manager& data_repo_mgr,
  cucascade::memory::memory_space_id space_id,
  cucascade::memory::memory_space* memory_space,
  sirius::memory::sirius_memory_reservation_manager& reservation_manager,
  sirius::exec::inspectable_mpsc<sirius::parallel::itask>* gpu_task_queue,
  sirius::exec::inspectable_mpsc<sirius::parallel::itask>* pipeline_task_queue)
  : _config(std::move(config)),
    _data_repo_mgr(data_repo_mgr),
    _space_id(space_id),
    _memory_space(memory_space),
    _reservation_manager(reservation_manager),
    _gpu_task_queue(gpu_task_queue),
    _pipeline_task_queue(pipeline_task_queue)
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

    auto& req    = request;
    auto t_start = std::chrono::steady_clock::now();

    // Per-tier tracking for LOG-01
    struct tier_stats {
      std::atomic<size_t> batches{0};
      std::atomic<size_t> bytes{0};
    };
    tier_stats repo_stats, gpu_queue_stats, pipeline_queue_stats;

    // Resolve the source memory space for filtering candidates
    auto* source_space = _reservation_manager.get_memory_space(_space_id.tier, _space_id.device_id);

    // Build target spaces list: for GPU->HOST downgrade, target is HOST tier
    std::vector<cucascade::memory::memory_space*> target_spaces;
    auto host_spaces =
      _reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    for (auto* hs : host_spaces) {
      target_spaces.push_back(const_cast<cucascade::memory::memory_space*>(hs));
    }

    // Helper lambda: dispatch a single convertible_data to the thread pool.
    // Returns false if the pool is interrupted.
    auto dispatch_candidate =
      [&](std::unique_ptr<convertible_data> candidate, tier_stats& stats) -> bool {
      if (req->satisfied.load()) return true;  // already done

      auto candidate_bytes = candidate->bytes_in_space(source_space);

      auto slot = _pool->reserve();
      if (!slot) return false;  // interrupted

      // Re-check after reserve() returns -- the previous candidate's worker may
      // have set satisfied while we were blocked waiting for a thread slot.
      if (req->satisfied.load()) return true;

      auto exc_stream = _stream_pool->acquire_stream(
        cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);

      _pool->dispatch(std::move(slot),
                      [cand       = std::move(candidate),
                       req_ptr    = req.get(),
                       &res_mgr   = _reservation_manager,
                       &targets   = target_spaces,
                       exc_stream = std::move(exc_stream),
                       candidate_bytes,
                       &stats]() mutable {
                        try {
                          if (cand->convert(targets, exc_stream, res_mgr)) {
                            req_ptr->bytes_freed.fetch_add(candidate_bytes,
                                                           std::memory_order_relaxed);
                            req_ptr->batches_downgraded.fetch_add(1, std::memory_order_relaxed);
                            stats.batches.fetch_add(1, std::memory_order_relaxed);
                            stats.bytes.fetch_add(candidate_bytes, std::memory_order_relaxed);
                            // D-08: Check predicate after each convert
                            if (req_ptr->predicate && req_ptr->predicate()) {
                              req_ptr->satisfied.store(true);
                            }
                          }
                        } catch (const std::exception& e) {
                          SIRIUS_LOG_ERROR("[downgrade] convert failed: {}", e.what());
                        }
                      });
      return true;
    };

    // === TIER 1: Data repositories (D-01, LOOP-01) ===
    // Create a convertible_data_batch_provider per repo and collect candidates.
    // We use get_all_convertible() to snapshot eligible batches once per repo,
    // avoiding re-scanning the same batch before its state changes from idle.
    _data_repo_mgr.for_each_repository(
      [&](cucascade::shared_data_repository* repo) {
        if (req->satisfied.load()) return;

        convertible_data_batch_provider provider(repo);
        auto candidates =
          provider.get_all_convertible(source_space, /*front_to_back=*/false);
        for (auto& candidate : candidates) {
          if (req->satisfied.load()) break;
          if (!dispatch_candidate(std::move(candidate), repo_stats)) return;
        }
      });

    // === TIER 2: gpu_pipeline_executor task queue (D-06, LOOP-02) ===
    if (!req->satisfied.load() && _gpu_task_queue) {
      convertible_gpu_pipeline_task_provider gpu_provider(*_gpu_task_queue);
      while (!req->satisfied.load()) {
        auto candidate =
          gpu_provider.get_next_convertible(source_space, /*front_to_back=*/false);
        if (!candidate) break;
        if (!dispatch_candidate(std::move(candidate), gpu_queue_stats)) break;
      }
    }

    // === TIER 3: pipeline_executor task queue (D-06, LOOP-03) ===
    if (!req->satisfied.load() && _pipeline_task_queue) {
      convertible_gpu_pipeline_task_provider pipeline_provider(*_pipeline_task_queue);
      while (!req->satisfied.load()) {
        auto candidate =
          pipeline_provider.get_next_convertible(source_space, /*front_to_back=*/false);
        if (!candidate) break;
        if (!dispatch_candidate(std::move(candidate), pipeline_queue_stats)) break;
      }
    }

    // Wait for all in-flight work to finish (D-08: predicate also checked in workers)
    _pool->wait_all();

    // === Logging (D-09, LOG-01) ===
    auto total_bytes   = req->bytes_freed.load(std::memory_order_relaxed);
    auto total_batches = req->batches_downgraded.load(std::memory_order_relaxed);
    auto duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start)
        .count();
    double throughput_mbs =
      (duration_ms > 0.0) ? (total_bytes / (1024.0 * 1024.0)) / (duration_ms / 1000.0) : 0.0;
    std::string source_label = req->is_monitor_request ? "monitor " : "";

    SIRIUS_LOG_DEBUG(
      "[downgrade] request {}done: {} batches, {} bytes in {:.2f} ms ({:.1f} MB/s) | "
      "repos: {}/{} batches/bytes, gpu_queue: {}/{}, pipeline_queue: {}/{}",
      source_label,
      total_batches,
      total_bytes,
      duration_ms,
      throughput_mbs,
      repo_stats.batches.load(std::memory_order_relaxed),
      repo_stats.bytes.load(std::memory_order_relaxed),
      gpu_queue_stats.batches.load(std::memory_order_relaxed),
      gpu_queue_stats.bytes.load(std::memory_order_relaxed),
      pipeline_queue_stats.batches.load(std::memory_order_relaxed),
      pipeline_queue_stats.bytes.load(std::memory_order_relaxed));

    // Fulfill the promise
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

// --- Public request API ---

std::future<size_t> downgrade_executor::request_free_memory(size_t bytes)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = [&freed = req->bytes_freed, bytes]() {
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

std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
    req->result.set_value(0);
    return future;
  }
  return future;
}

}  // namespace parallel
}  // namespace sirius
