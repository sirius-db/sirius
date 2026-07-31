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

#include "scan_manager/memory_prefetcher.hpp"

#include "data/sirius_converter_registry.hpp"
#include "log/logging.hpp"

#include <rmm/error.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>

#include <chrono>
#include <exception>

namespace sirius::scan_manager {

memory_prefetcher::memory_prefetcher(memory_prefetcher_config cfg,
                                     std::vector<std::shared_ptr<split_connector>> connectors,
                                     cucascade::memory::memory_space* gpu_space)
  : _config(cfg), _connectors(std::move(connectors)), _gpu_space(gpu_space)
{
  if (_gpu_space == nullptr || _connectors.empty()) {
    _running.store(false);
    return;
  }
  _stream_pool = std::make_unique<cucascade::memory::exclusive_stream_pool>(
    rmm::cuda_device_id{_gpu_space->get_device_id()}, _config.num_threads);
  _workers.reserve(_config.num_threads);
  for (std::size_t i = 0; i < _config.num_threads; ++i) {
    _workers.emplace_back([this] { worker_loop(); });
  }
  SIRIUS_LOG_INFO(
    "[memory_prefetcher] started: {} threads, {} connectors, min_free_fraction={:.2f}",
    _config.num_threads,
    _connectors.size(),
    _config.min_free_fraction);
}

memory_prefetcher::~memory_prefetcher() { stop(); }

void memory_prefetcher::stop()
{
  bool expected = true;
  if (_running.compare_exchange_strong(expected, false)) {
    SIRIUS_LOG_INFO("[memory_prefetcher] stopping: prefetched {} batches / {} bytes",
                    _batches_prefetched.load(),
                    _bytes_prefetched.load());
  }
  for (auto& worker : _workers) {
    if (worker.joinable()) { worker.join(); }
  }
  _workers.clear();
}

void memory_prefetcher::worker_loop()
{
  auto stream = _stream_pool->acquire_stream();
  while (_running.load(std::memory_order_relaxed)) {
    std::size_t converted = 0;
    try {
      converted = sweep(stream.get());
    } catch (const std::exception& e) {
      SIRIUS_LOG_WARN("[memory_prefetcher] sweep error (backing off): {}", e.what());
    }

    if (!_running.load(std::memory_order_relaxed)) { break; }

    // Exit when every connector is closed and drained — no more splits will
    // ever arrive this query. (A new query builds a new prefetcher.)
    bool all_closed = true;
    for (const auto& connector : _connectors) {
      if (!connector->is_closed()) {
        all_closed = false;
        break;
      }
    }
    if (all_closed) { break; }

    if (converted == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(_config.poll_interval_ms));
    }
  }
}

std::size_t memory_prefetcher::sweep(rmm::cuda_stream_view stream)
{
  std::size_t converted = 0;
  const auto min_free_bytes =
    static_cast<std::size_t>(_config.min_free_fraction * _gpu_space->get_max_memory());
  auto& registry = sirius::converter_registry::get();

  // Walk connectors in scan (execution) order so the head-of-line pipeline's
  // data lands on the GPU first.
  for (const auto& connector : _connectors) {
    if (!_running.load(std::memory_order_relaxed)) { return converted; }

    // Actively-draining connector: its scan tasks convert their own batches
    // on 1 stream per pipeline thread. Grabbing exclusive locks here would
    // serialize those conversions behind the (fewer) prefetch threads and
    // slow scan-bound queries down.
    if (connector->is_draining(_config.drain_quiet_ms)) { continue; }

    for (const auto& batch : connector->peek_resident_batches()) {
      // Re-check between conversions so we back off within one batch of the
      // scan starting to drain this connector.
      if (connector->is_draining(_config.drain_quiet_ms)) { break; }
      if (!_running.load(std::memory_order_relaxed)) { return converted; }

      // Cheap pre-check under a shared lock; skip GPU-resident batches.
      std::size_t batch_bytes = 0;
      {
        auto ro = batch->to_read_only();
        if (ro.get_data() == nullptr || ro.get_current_tier() == cucascade::memory::Tier::GPU) {
          continue;
        }
        batch_bytes = ro.get_data()->get_size_in_bytes();
      }

      // Headroom gate: never let prefetched (unreclaimable) bytes push the
      // space below the free floor. This also keeps the reservation below out
      // of the executors' way: it is only attempted while the space has at
      // least min_free_fraction headroom to spare.
      if (_gpu_space->get_available_memory() < batch_bytes + min_free_bytes) {
        // No room for this batch now; later batches are at least as far from
        // being consumed, so stop the whole sweep and retry after tasks free
        // memory.
        return converted;
      }

      // Exclusive lock, skip on contention (a task is consuming this batch —
      // its own prepare_for_processing does the upload).
      auto mut = batch->try_to_mutable();
      if (!mut) { continue; }
      if (mut->get_data() == nullptr || mut->get_current_tier() == cucascade::memory::Tier::GPU) {
        continue;
      }

      auto reservation = _gpu_space->make_reservation_or_null(batch_bytes);
      if (!reservation) { return converted; }

      try {
        // Convert against the reservation so the conversion's device
        // allocations draw from the arena reserved above.
        mut->convert_to<cucascade::gpu_table_representation>(registry, *reservation, stream);
      } catch (const rmm::out_of_memory&) {
        SIRIUS_LOG_DEBUG("[memory_prefetcher] OOM converting batch {}; backing off",
                         mut->get_batch_id());
        return converted;
      }

      ++converted;
      _batches_prefetched.fetch_add(1, std::memory_order_relaxed);
      _bytes_prefetched.fetch_add(batch_bytes, std::memory_order_relaxed);
    }
  }
  return converted;
}

}  // namespace sirius::scan_manager
