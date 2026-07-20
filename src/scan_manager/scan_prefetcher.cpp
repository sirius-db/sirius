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

#include "scan_manager/scan_prefetcher.hpp"

#include "data/sirius_converter_registry.hpp"
#include "log/logging.hpp"

#include <rmm/error.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>

#include <chrono>
#include <cstdint>
#include <exception>

namespace sirius::scan_manager {

scan_prefetcher::scan_prefetcher(config cfg,
                                 std::vector<std::shared_ptr<split_connector>> connectors,
                                 cucascade::memory::memory_space* gpu_space)
  : _config(cfg),
    _connectors(std::move(connectors)),
    _gpu_space(gpu_space),
    _last_pop_counts(_connectors.size()),
    _last_pop_time_ms(_connectors.size())
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
  SIRIUS_LOG_INFO("[scan_prefetcher] started: {} threads, {} connectors, min_free_fraction={:.2f}",
                  _config.num_threads,
                  _connectors.size(),
                  _config.min_free_fraction);
}

scan_prefetcher::~scan_prefetcher() { stop(); }

void scan_prefetcher::stop()
{
  bool expected = true;
  if (_running.compare_exchange_strong(expected, false)) {
    SIRIUS_LOG_INFO("[scan_prefetcher] stopping: prefetched {} batches / {} bytes",
                    _batches_prefetched.load(),
                    _bytes_prefetched.load());
  }
  for (auto& worker : _workers) {
    if (worker.joinable()) { worker.join(); }
  }
  _workers.clear();
}

void scan_prefetcher::worker_loop()
{
  auto stream = _stream_pool->acquire_stream();
  while (_running.load(std::memory_order_relaxed)) {
    std::size_t converted = 0;
    try {
      converted = sweep(stream.get());
    } catch (const std::exception& e) {
      SIRIUS_LOG_WARN("[scan_prefetcher] sweep error (backing off): {}", e.what());
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

std::size_t scan_prefetcher::sweep(rmm::cuda_stream_view stream)
{
  std::size_t converted = 0;
  const auto min_free_bytes =
    static_cast<std::size_t>(_config.min_free_fraction * _gpu_space->get_max_memory());
  auto& registry = sirius::converter_registry::get();

  // Walk connectors in scan (execution) order so the head-of-line pipeline's
  // data lands on the GPU first.
  for (std::size_t ci = 0; ci < _connectors.size(); ++ci) {
    const auto& connector = _connectors[ci];
    if (!_running.load(std::memory_order_relaxed)) { return converted; }

    // Actively-draining connector: its scan tasks convert their own batches
    // on 1 stream per pipeline thread. Grabbing exclusive locks here would
    // serialize those conversions behind the (fewer) prefetch threads and
    // slow scan-bound queries down. A pure between-sweeps delta is not enough
    // — sweeps run every ~2ms while an active scan pops only every ~10-40ms —
    // so a connector stays "draining" until drain_quiet_ms pass with no pops.
    auto is_draining = [&]() {
      const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
      const auto pops = connector->pop_count();
      if (pops != _last_pop_counts[ci].exchange(pops, std::memory_order_relaxed)) {
        _last_pop_time_ms[ci].store(now_ms, std::memory_order_relaxed);
        return true;
      }
      const auto last = _last_pop_time_ms[ci].load(std::memory_order_relaxed);
      return last != 0 && now_ms - last < static_cast<std::int64_t>(_config.drain_quiet_ms);
    };
    if (is_draining()) { continue; }

    for (const auto& batch : connector->peek_resident_batches()) {
      // Re-check between conversions so we back off within one batch of the
      // scan starting to drain this connector.
      if (is_draining()) { break; }
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
      // space below the free floor.
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
        mut->convert_to<cucascade::gpu_table_representation>(registry, _gpu_space, stream);
      } catch (const rmm::out_of_memory&) {
        SIRIUS_LOG_DEBUG("[scan_prefetcher] OOM converting batch {}; backing off",
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
