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

#include "cuda/device_health.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "log/logging.hpp"
#include "memory/runtime_stream_pool.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>

namespace sirius::scan_manager {

namespace {

std::string device_list(const std::vector<cucascade::memory::memory_space*>& spaces)
{
  std::string out;
  for (auto const* space : spaces) {
    if (!out.empty()) { out += ','; }
    out += std::to_string(space->get_device_id());
  }
  return out;
}

}  // namespace

memory_prefetcher::memory_prefetcher(memory_prefetcher_config cfg,
                                     std::vector<std::shared_ptr<split_connector>> connectors,
                                     cucascade::memory::memory_space* gpu_space,
                                     exec::query_lifecycle_registry& lifecycle)
  : memory_prefetcher(
      cfg,
      std::move(connectors),
      gpu_space ? std::vector{gpu_space} : std::vector<cucascade::memory::memory_space*>{},
      lifecycle)
{
}

memory_prefetcher::memory_prefetcher(memory_prefetcher_config cfg,
                                     std::vector<std::shared_ptr<split_connector>> connectors,
                                     std::vector<cucascade::memory::memory_space*> gpu_spaces,
                                     exec::query_lifecycle_registry& lifecycle,
                                     std::shared_ptr<host_chunk_placement> placement)
  : _lifecycle(lifecycle),
    _config(cfg),
    _connectors(std::move(connectors)),
    _gpu_spaces(std::move(gpu_spaces)),
    _placement(std::move(placement))
{
  if (std::find(_gpu_spaces.begin(), _gpu_spaces.end(), nullptr) != _gpu_spaces.end()) {
    throw std::invalid_argument("Memory prefetch requires non-null GPU spaces");
  }
  if (_gpu_spaces.empty() || _connectors.empty()) {
    _running.store(false);
    return;
  }
  if (!_placement || _placement->admitted_gpu_ids().empty()) {
    std::vector<int> ids;
    ids.reserve(_gpu_spaces.size());
    for (auto const* space : _gpu_spaces) {
      ids.push_back(space->get_device_id());
    }
    _placement = std::make_shared<host_chunk_placement>(std::move(ids), nullptr);
  }
  _drain_claims = std::make_unique<std::atomic<std::size_t>[]>(_connectors.size());
  for (std::size_t i = 0; i < _connectors.size(); ++i) {
    _drain_claims[i].store(0, std::memory_order_relaxed);
  }
  const std::size_t n_spaces = _gpu_spaces.size();
  // num_threads is per admitted GPU: the scans whose uploads this overlaps run pipeline threads
  // per GPU, and a fixed pool shared by every GPU falls behind them as GPUs are added.
  const std::size_t n_workers = _config.num_threads * n_spaces;
  _min_free_bytes.reserve(n_spaces);
  for (auto const* space : _gpu_spaces) {
    _min_free_bytes.push_back(
      static_cast<std::size_t>(_config.min_free_fraction * space->get_max_memory()));
  }
  _batches_per_space = std::make_unique<std::atomic<std::size_t>[]>(n_spaces);
  for (std::size_t i = 0; i < n_spaces; ++i) {
    _batches_per_space[i].store(0, std::memory_order_relaxed);
  }
  try {
    _workers.reserve(n_workers);
    for (std::size_t i = 0; i < n_workers; ++i) {
      _workers.emplace_back([this] { worker_loop(); });
    }
  } catch (...) {
    stop();
    throw;
  }
  SIRIUS_LOG_INFO(
    "[memory_prefetcher] started: {} threads, {} connectors, GPUs [{}], min_free_fraction={:.2f}",
    n_workers,
    _connectors.size(),
    device_list(_gpu_spaces),
    _config.min_free_fraction);
}

memory_prefetcher::~memory_prefetcher() { stop(); }

void memory_prefetcher::stop()
{
  bool expected = true;
  if (_running.compare_exchange_strong(expected, false)) {
    std::string per_gpu;
    for (std::size_t i = 0; i < _gpu_spaces.size(); ++i) {
      if (!per_gpu.empty()) { per_gpu += ' '; }
      per_gpu += std::to_string(_gpu_spaces[i]->get_device_id()) + ':' +
                 std::to_string(_batches_per_space[i].load());
    }
    SIRIUS_LOG_INFO(
      "[memory_prefetcher] stopping: prefetched {} batches / {} bytes, per GPU [{}] "
      "(stops: headroom={} reservation={}, skips: lock={} draining={}, errors={})",
      _batches_prefetched.load(),
      _bytes_prefetched.load(),
      per_gpu,
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

void memory_prefetcher::handle_error(std::exception_ptr error) noexcept
{
  if (_lifecycle.report_failure(error)) _running.store(false, std::memory_order_relaxed);
}

void memory_prefetcher::worker_loop()
{
  // A fresh thread's current device is 0: start on the first admitted GPU, and let each
  // conversion make its own target current (the compression converters allocate from the
  // CURRENT device's resource rather than the target space's).
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{_gpu_spaces.front()->get_device_id()}};
  // Another query or worker can poison the shared runtime while this worker is idle.
  while (_running.load(std::memory_order_relaxed) && !_lifecycle.runtime_failed()) {
    std::size_t converted = 0;
    try {
      converted = sweep();
    } catch (const std::exception& e) {
      handle_error(std::current_exception());
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

std::optional<std::size_t> memory_prefetcher::space_index(int device) const
{
  for (std::size_t i = 0; i < _gpu_spaces.size(); ++i) {
    if (_gpu_spaces[i]->get_device_id() == device) { return i; }
  }
  return std::nullopt;
}

std::size_t memory_prefetcher::sweep()
{
  std::size_t converted      = 0;
  const std::size_t n_spaces = _gpu_spaces.size();
  auto& registry             = sirius::converter_registry::get();
  // Spaces whose admission gate refused during this sweep. A batch with no unsaturated
  // candidate is skipped; the sweep ends once every space has refused.
  std::vector<bool> saturated(n_spaces, false);
  std::size_t n_saturated = 0;
  // One runtime stream lease per space this sweep converts into, taken on first use and
  // returned when the sweep ends.
  std::vector<std::optional<sirius::memory::runtime_stream_pool::lease>> leases(n_spaces);

  // Walk connectors in scan (execution) order so the head-of-line pipeline's
  // data lands on the GPU first.
  for (std::size_t ci = 0; ci < _connectors.size(); ++ci) {
    const auto& connector = _connectors[ci];
    if (!_running.load(std::memory_order_relaxed) || _lifecycle.runtime_failed()) {
      return converted;
    }

    // Actively-draining connector: its scan tasks convert their own popped
    // batches on 1 stream per pipeline thread, so racing them near the pop
    // point serializes those conversions behind the (fewer) prefetch threads.
    // Collisions only happen at the head of the queue, though — so instead of
    // skipping the connector entirely, convert TAIL-FIRST and leave a head
    // margin for the batches the scan will pop imminently. At most one worker
    // per admitted GPU may do this per connector, and the margin is per GPU
    // too: the scan pops from every GPU at once.
    const bool draining = connector->is_draining(_config.drain_quiet_ms);
    bool claimed        = false;
    if (draining) {
      auto current = _drain_claims[ci].load(std::memory_order_relaxed);
      while (current < n_spaces && !_drain_claims[ci].compare_exchange_weak(
                                     current, current + 1, std::memory_order_relaxed)) {}
      if (current >= n_spaces) {
        _skips_draining.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      claimed = true;
    }
    // Release the drain claim on every exit path from this connector.
    struct claim_release {
      std::atomic<std::size_t>* count;
      ~claim_release()
      {
        if (count) { count->fetch_sub(1, std::memory_order_relaxed); }
      }
    } release{claimed ? &_drain_claims[ci] : nullptr};

    auto splits = connector->peek_resident_batches();
    if (draining) {
      const std::size_t head_margin = 4 * n_spaces;
      if (splits.size() <= head_margin) {
        _skips_draining.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      splits.erase(splits.begin(), splits.begin() + head_margin);
      std::reverse(splits.begin(), splits.end());
    }

    for (const auto& split : splits) {
      const auto& batch = split.batch;
      // For a quiet connector, back off to tail-first within one batch of the
      // scan starting to drain it (the next sweep re-plans with the margin).
      if (!draining && connector->is_draining(_config.drain_quiet_ms)) { break; }
      if (!_running.load(std::memory_order_relaxed) || _lifecycle.runtime_failed()) {
        return converted;
      }

      // Cheap pre-check under a shared lock; skip GPU-resident batches. Must
      // be NON-blocking: a sibling worker converting this batch holds its
      // exclusive lock for the whole conversion, and a blocking to_read_only
      // here would convoy the workers batch-by-batch.
      std::size_t batch_bytes                       = 0;
      std::size_t peak_bytes                        = 0;
      const cucascade::memory::memory_space* source = nullptr;
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
        source     = ro->get_memory_space();
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

      // A stamped split's task is created on the stamped device: convert there or not at all.
      // A stamp outside the admitted set means the task creator will re-home the task, so
      // leave that upload to it. Unstamped splits take the placement's choice, with
      // fall-through across its candidates (decided under the lock, so a batch another worker
      // holds does not advance the cursor).
      std::vector<std::size_t> candidates;
      if (split.preferred_device.has_value()) {
        if (auto idx = space_index(*split.preferred_device); idx && !saturated[*idx]) {
          candidates.push_back(*idx);
        }
      } else {
        for (int const device : _placement->candidates(source)) {
          if (auto idx = space_index(device); idx && !saturated[*idx]) {
            candidates.push_back(*idx);
          }
        }
      }
      if (candidates.empty()) { continue; }

      // Admit on the first candidate with headroom, reserving BEFORE gating (see the class
      // comment). A refusal saturates that space for the rest of this sweep.
      std::optional<std::size_t> target;
      std::unique_ptr<cucascade::memory::reservation> reservation;
      for (auto const idx : candidates) {
        auto* space = _gpu_spaces[idx];
        reservation = space->make_reservation_or_null(peak_bytes);
        if (!reservation) {
          _stops_reservation.fetch_add(1, std::memory_order_relaxed);
        } else if (space->get_available_memory() < _min_free_bytes[idx]) {
          reservation.reset();
          _stops_headroom.fetch_add(1, std::memory_order_relaxed);
        } else {
          target = idx;
          break;
        }
        saturated[idx] = true;
        ++n_saturated;
      }
      if (!target) {
        if (n_saturated == n_spaces) { return converted; }
        continue;
      }

      auto* space = _gpu_spaces[*target];
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{space->get_device_id()}};
      if (!leases[*target]) {
        leases[*target].emplace(sirius::memory::runtime_stream_pool::acquire(*space));
      }
      const ::cuda::stream_ref stream = leases[*target]->get();

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
        mut->convert_to<cucascade::gpu_table_representation>(registry, space, stream);
      } catch (const rmm::out_of_memory&) {
        // Tighter than its accounting shows; leave this space alone for the rest of the sweep.
        SIRIUS_LOG_DEBUG("[memory_prefetcher] OOM converting batch {} on GPU {}; backing off",
                         mut->get_batch_id(),
                         space->get_device_id());
        saturated[*target] = true;
        if (++n_saturated == n_spaces) { return converted; }
        continue;
      } catch (const std::exception& e) {
        handle_error(std::current_exception());
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
      _batches_per_space[*target].fetch_add(1, std::memory_order_relaxed);
      SIRIUS_LOG_DEBUG("[memory_prefetcher] batch {} ({} bytes) host space {} -> GPU {}",
                       mut->get_batch_id(),
                       batch_bytes,
                       source != nullptr ? source->get_device_id() : -1,
                       space->get_device_id());
    }
  }
  return converted;
}

}  // namespace sirius::scan_manager
