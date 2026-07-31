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

#include "pipeline/downgraded_task_prefetcher.hpp"

#include "data/convertible_data_batch.hpp"
#include "data/convertible_gpu_pipeline_task.hpp"
#include "log/logging.hpp"

#include <cuda_runtime_api.h>

#include <cucascade/data/data_batch.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>

namespace sirius::pipeline {

downgraded_task_prefetcher::downgraded_task_prefetcher(
  config cfg,
  exec::multi_index_priority_queue<sirius::parallel::itask>& task_queue,
  sirius::memory::sirius_memory_reservation_manager& res_mgr,
  cucascade::memory::memory_space* gpu_space)
  : _config(cfg), _task_queue(task_queue), _res_mgr(res_mgr), _gpu_space(gpu_space)
{
  if (_gpu_space == nullptr) {
    _running.store(false);
    return;
  }
  _targets.push_back(_gpu_space);
  _stream_pool = std::make_unique<cucascade::memory::exclusive_stream_pool>(
    rmm::cuda_device_id{_gpu_space->get_device_id()}, _config.num_threads);
  _workers.reserve(_config.num_threads);
  for (std::size_t i = 0; i < _config.num_threads; ++i) {
    _workers.emplace_back([this] { worker_loop(); });
  }
  SIRIUS_LOG_INFO("[task_prefetch] started: {} threads, min_free_fraction={:.2f}",
                  _config.num_threads,
                  _config.min_free_fraction);
}

downgraded_task_prefetcher::~downgraded_task_prefetcher() { stop(); }

void downgraded_task_prefetcher::stop()
{
  bool expected = true;
  if (_running.compare_exchange_strong(expected, false)) {
    SIRIUS_LOG_INFO("[task_prefetch] stopping: upgraded {} batches / {} bytes ({} host, {} disk)",
                    _batches_prefetched.load(),
                    _bytes_prefetched.load(),
                    _host_batches.load(),
                    _disk_batches.load());
  }
  for (auto& worker : _workers) {
    if (worker.joinable()) { worker.join(); }
  }
  _workers.clear();
}

void downgraded_task_prefetcher::worker_loop()
{
  // Bind the worker to the target GPU so conversions allocate on the right
  // device context (the prefetcher is single-GPU scoped, but the process
  // default device is not guaranteed to match).
  cudaError_t err = cudaSetDevice(_gpu_space->get_device_id());
  if (err != cudaSuccess) {
    SIRIUS_LOG_ERROR("[task_prefetch] worker init: cudaSetDevice({}) failed: {}",
                     _gpu_space->get_device_id(),
                     cudaGetErrorString(err));
    return;
  }
  auto stream = _stream_pool->acquire_stream();
  while (_running.load(std::memory_order_relaxed)) {
    std::size_t converted = 0;
    try {
      converted = sweep(stream.get());
    } catch (const std::exception& e) {
      SIRIUS_LOG_WARN("[task_prefetch] sweep error (backing off): {}", e.what());
    }

    if (!_running.load(std::memory_order_relaxed)) { break; }

    if (converted == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(_config.poll_interval_ms));
    }
  }
}

std::size_t downgraded_task_prefetcher::sweep(rmm::cuda_stream_view stream)
{
  std::size_t converted     = 0;
  const auto max_memory     = _gpu_space->get_max_memory();
  const auto min_free_bytes = static_cast<std::size_t>(_config.min_free_fraction * max_memory);
  // get_available_memory() can exceed get_max_memory() when the usage limit is
  // larger than the reservation limit; cap it so the floor fraction is always
  // relative to the same (reservable) budget the conversions draw from.
  const auto available = [&] { return std::min(_gpu_space->get_available_memory(), max_memory); };
  const auto now_ms    = [] {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
  };
  // Pressure check with hysteresis: any pressure observation (downgrade
  // trigger or floor breach) idles the prefetcher until pressure_quiet_ms of
  // continuous calm. The instantaneous headroom right after an eviction looks
  // fine, but upgrading then would hand batches straight back to the eviction
  // path (observed as a downgrade<->prefetch bounce under small GPU budgets).
  const auto under_pressure = [&] {
    if (_gpu_space->should_downgrade_memory() || available() < min_free_bytes) {
      _last_pressure_ms.store(now_ms(), std::memory_order_relaxed);
      return true;
    }
    const auto last = _last_pressure_ms.load(std::memory_order_relaxed);
    return last != 0 && now_ms() - last < static_cast<std::int64_t>(_config.pressure_quiet_ms);
  };

  if (!_config.prefetch_during_pressure && under_pressure()) { return 0; }
  if (_config.prefetch_during_pressure && available() < min_free_bytes) { return 0; }
  if (_task_queue.empty()) { return 0; }

  // Snapshot the input batches of the next max_lookahead_tasks pending tasks
  // in dispatch order (the scheduler's matcher pops with
  // pop_if(front_to_back=true), so the front of the queue is the next task to
  // run). The lookahead bound keeps the prefetched-but-unconsumed footprint
  // proportional to imminent work. shared_ptr copies only — the visitor runs
  // under the queue mutex and must stay lightweight.
  std::vector<std::shared_ptr<cucascade::data_batch>> candidates;
  std::size_t tasks_seen = 0;
  _task_queue.for_each_mutable(
    [&candidates, &tasks_seen, this](sirius::parallel::itask& task) {
      auto* operator_data = convertible_gpu_pipeline_task::get_pipelineable_data(task);
      if (operator_data == nullptr) { return true; }
      for (const auto& batch : operator_data->get_data_batches()) {
        if (batch) { candidates.push_back(batch); }
      }
      return ++tasks_seen < _config.max_lookahead_tasks;
    },
    /*front_to_back=*/true);

  for (const auto& batch : candidates) {
    if (!_running.load(std::memory_order_relaxed)) { return converted; }

    // Cheap pre-check under a shared lock; skip GPU-resident or busy batches
    // (a busy batch is being handled by its task or by the downgrade executor).
    std::size_t batch_bytes = 0;
    auto source_tier        = cucascade::memory::Tier::GPU;
    {
      auto ro = batch->try_to_read_only();
      if (!ro || ro->get_data() == nullptr ||
          ro->get_current_tier() == cucascade::memory::Tier::GPU) {
        continue;
      }
      batch_bytes = ro->get_data()->get_size_in_bytes();
      source_tier = ro->get_current_tier();
    }

    // Headroom gate: never let prefetched bytes push the space below the free
    // floor (or into the downgrade trigger — that would hand the batch right
    // back to the eviction path). Later candidates belong to tasks at least as
    // far from dispatch, so stop the whole sweep and retry after tasks free
    // memory.
    if (!_config.prefetch_during_pressure && under_pressure()) { return converted; }
    if (available() < batch_bytes + min_free_bytes) { return converted; }

    // convertible_data_batch handles try_to_mutable (skip on contention), the
    // GPU reservation, the conversion, and idle-state restore on all paths.
    sirius::convertible_data_batch upgrader(batch);
    auto result = upgrader.convert(_targets, stream, _res_mgr, /*blocking=*/false);
    if (!result) { continue; }

    ++converted;
    _batches_prefetched.fetch_add(1, std::memory_order_relaxed);
    _bytes_prefetched.fetch_add(batch_bytes, std::memory_order_relaxed);
    if (source_tier == cucascade::memory::Tier::DISK) {
      _disk_batches.fetch_add(1, std::memory_order_relaxed);
    } else {
      _host_batches.fetch_add(1, std::memory_order_relaxed);
    }
    SIRIUS_LOG_DEBUG("[task_prefetch] upgraded batch {} ({} bytes, from {}) ahead of dispatch",
                     batch->get_batch_id(),
                     batch_bytes,
                     source_tier == cucascade::memory::Tier::DISK ? "DISK" : "HOST");
  }
  return converted;
}

}  // namespace sirius::pipeline
