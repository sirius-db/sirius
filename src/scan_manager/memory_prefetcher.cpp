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

#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "log/logging.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
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
  _drain_claims = std::make_unique<std::atomic<bool>[]>(_connectors.size());
  for (std::size_t i = 0; i < _connectors.size(); ++i) {
    _drain_claims[i].store(false, std::memory_order_relaxed);
  }
  // Acquired before the workers start, so round-robin gives each a distinct stream.
  _worker_streams.reserve(_config.num_threads);
  for (std::size_t i = 0; i < _config.num_threads; ++i) {
    _worker_streams.push_back(_gpu_space->acquire_stream());
  }
  _workers.reserve(_config.num_threads);
  for (std::size_t i = 0; i < _config.num_threads; ++i) {
    _workers.emplace_back([this, i] { worker_loop(i); });
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
    SIRIUS_LOG_INFO(
      "[memory_prefetcher] stopping: prefetched {} batches / {} bytes "
      "(stops: headroom={} reservation={}, skips: lock={} draining={}, errors={})",
      _batches_prefetched.load(),
      _bytes_prefetched.load(),
      _stops_headroom.load(),
      _stops_reservation.load(),
      _skips_lock.load(),
      _skips_draining.load(),
      _errors_conversion.load());
  }
  for (auto& worker : _workers) {
    if (worker.joinable()) { worker.join(); }
  }
  _workers.clear();
}

void memory_prefetcher::worker_loop(std::size_t worker_index)
{
  // Bind this worker to the space's device: a fresh thread's current device is
  // 0, and the compression converters allocate from the CURRENT device's
  // resource rather than the target space's.
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{_gpu_space->get_device_id()}};
  const rmm::cuda_stream_view stream = _worker_streams[worker_index];
  while (_running.load(std::memory_order_relaxed)) {
    std::size_t converted = 0;
    try {
      converted = sweep(stream);
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
  for (std::size_t ci = 0; ci < _connectors.size(); ++ci) {
    const auto& connector = _connectors[ci];
    if (!_running.load(std::memory_order_relaxed)) { return converted; }

    // Actively-draining connector: its scan tasks convert their own popped
    // batches on 1 stream per pipeline thread, so racing them near the pop
    // point serializes those conversions behind the (fewer) prefetch threads.
    // Collisions only happen at the head of the queue, though — so instead of
    // skipping the connector entirely, convert TAIL-FIRST and leave a head
    // margin for the batches the scan will pop imminently. At most ONE worker
    // may do this per connector.
    const bool draining = connector->is_draining(_config.drain_quiet_ms);
    bool claimed        = false;
    if (draining) {
      bool expected = false;
      if (!_drain_claims[ci].compare_exchange_strong(expected, true)) {
        _skips_draining.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      claimed = true;
    }
    // Release the drain claim on every exit path from this connector.
    struct claim_release {
      std::atomic<bool>* flag;
      ~claim_release()
      {
        if (flag) { flag->store(false, std::memory_order_relaxed); }
      }
    } release{claimed ? &_drain_claims[ci] : nullptr};

    auto batches = connector->peek_resident_batches();
    if (draining) {
      constexpr std::size_t head_margin = 4;
      if (batches.size() <= head_margin) {
        _skips_draining.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      batches.erase(batches.begin(), batches.begin() + head_margin);
      std::reverse(batches.begin(), batches.end());
    }

    for (const auto& batch : batches) {
      // For a quiet connector, back off to tail-first within one batch of the
      // scan starting to drain it (the next sweep re-plans with the margin).
      if (!draining && connector->is_draining(_config.drain_quiet_ms)) { break; }
      if (!_running.load(std::memory_order_relaxed)) { return converted; }

      // Cheap pre-check under a shared lock; skip GPU-resident batches. Must
      // be NON-blocking: a sibling worker converting this batch holds its
      // exclusive lock for the whole conversion, and a blocking to_read_only
      // here would convoy the workers batch-by-batch.
      std::size_t batch_bytes = 0;
      std::size_t peak_bytes  = 0;
      {
        auto ro = batch->try_to_read_only();
        if (!ro || ro->get_data() == nullptr ||
            ro->get_current_tier() == cucascade::memory::Tier::GPU) {
          continue;
        }
        batch_bytes = ro->get_data()->get_size_in_bytes();
        // A compressed batch stages its encoded payload on device while the
        // decoded output is written, so gate and reserve the conversion PEAK,
        // not the resting size.
        peak_bytes = sirius::peak_materialization_bytes(ro->get_data());
      }

      // Exclusive lock, skip on contention (a task is consuming this batch —
      // its own prepare_for_processing does the upload).
      auto mut = batch->try_to_mutable();
      if (!mut) {
        _skips_lock.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      if (mut->get_data() == nullptr || mut->get_current_tier() == cucascade::memory::Tier::GPU) {
        continue;
      }

      auto reservation = _gpu_space->make_reservation_or_null(peak_bytes);
      if (!reservation) {
        _stops_reservation.fetch_add(1, std::memory_order_relaxed);
        return converted;
      }

      if (_gpu_space->get_available_memory() < min_free_bytes) {
        reservation.reset();
        _stops_headroom.fetch_add(1, std::memory_order_relaxed);
        return converted;
      }

      auto* allocator = reservation->get_memory_resource_of<cucascade::memory::Tier::GPU>();
      if (allocator == nullptr ||
          !allocator->attach_reservation_to_tracker(stream, std::move(reservation))) {
        _errors_conversion.fetch_add(1, std::memory_order_relaxed);
        SIRIUS_LOG_WARN(
          "[memory_prefetcher] could not attach reservation to allocation tracker for "
          "batch {} (skipping)",
          mut->get_batch_id());
        continue;
      }
      absl::Cleanup reservation_detacher = [allocator, stream] {
        allocator->reset_stream_reservation(stream);
      };

      try {
        mut->convert_to<cucascade::gpu_table_representation>(registry, _gpu_space, stream);
      } catch (const rmm::out_of_memory&) {
        SIRIUS_LOG_DEBUG("[memory_prefetcher] OOM converting batch {}; backing off",
                         mut->get_batch_id());
        return converted;
      } catch (const std::exception& e) {
        // Conversion is more than data movement (compressed batches decode
        // here), so non-OOM failures are possible. Skip the batch: the scan
        // task converts it itself on the authoritative path and surfaces the
        // error to the query if it persists.
        _errors_conversion.fetch_add(1, std::memory_order_relaxed);
        SIRIUS_LOG_WARN("[memory_prefetcher] conversion of batch {} failed (skipping): {}",
                        mut->get_batch_id(),
                        e.what());
        continue;
      }

      ++converted;
      _batches_prefetched.fetch_add(1, std::memory_order_relaxed);
      _bytes_prefetched.fetch_add(batch_bytes, std::memory_order_relaxed);
    }
  }
  return converted;
}

}  // namespace sirius::scan_manager
