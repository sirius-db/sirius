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

#include "scan_manager/split_connector.hpp"

#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/stream_pool.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace sirius::scan_manager {

/**
 * @brief Background host->GPU upgrader for queued pinned-cache scan splits.
 *
 * Host-pinned scans put all their splits (per-query wrapper data_batches over
 * host slices of the pinned chunks) onto their split_connectors at query start,
 * but the H2D copy for each batch happens only when the scan's pipeline task
 * executes (scan_operator_input::prepare_for_processing). Because scan tasks
 * are created strictly on topology demand, the copy engines sit idle during
 * compute-bound phases and then the next scan's H2D burst runs with idle SMs.
 *
 * The prefetcher overlaps the two: worker threads walk the connectors in scan
 * (execution) order and convert pending resident batches to GPU tier early, on
 * their own streams, gated on GPU memory headroom. When the scan task later
 * runs, prepare_for_processing sees the batch already GPU-resident and skips
 * the upload (and the task creator's cached-scan locality derivation reads the
 * batch's post-conversion space, so device dispatch stays consistent).
 *
 * Races with a consumer are arbitrated by the data_batch state machine: the
 * conversion holds the exclusive (mutable) lock via try_to_mutable (skip on
 * contention), and prepare_for_processing re-checks the tier under its own
 * lock.
 *
 * Memory safety: converted batches live in connector queues, which the
 * downgrade executor does NOT scan (it walks data repositories), so
 * prefetched-but-unconsumed bytes cannot be reclaimed under pressure. The
 * headroom gate must therefore stay conservative: prefetch only while
 * (available - batch_size) >= min_free_fraction * max_memory.
 */
class scan_prefetcher {
 public:
  struct config {
    /// Number of prefetch worker threads (each with its own stream).
    std::size_t num_threads{2};
    /// Keep at least this fraction of the GPU space free after each prefetch.
    double min_free_fraction{0.4};
    /// Worker sweep interval while waiting for headroom / new splits.
    std::size_t poll_interval_ms{2};
    /// A connector is considered actively draining (and skipped) until this
    /// long has passed since its last observed pop. Must exceed the scan's
    /// inter-pop interval (~10-40ms per 5GB batch) or sweeps race the scan.
    std::size_t drain_quiet_ms{100};
  };

  scan_prefetcher(config cfg,
                  std::vector<std::shared_ptr<split_connector>> connectors,
                  cucascade::memory::memory_space* gpu_space);

  ~scan_prefetcher();

  scan_prefetcher(const scan_prefetcher&)            = delete;
  scan_prefetcher& operator=(const scan_prefetcher&) = delete;
  scan_prefetcher(scan_prefetcher&&)                 = delete;
  scan_prefetcher& operator=(scan_prefetcher&&)      = delete;

  /// Request stop and join all workers. Idempotent.
  void stop();

  [[nodiscard]] std::size_t batches_prefetched() const
  {
    return _batches_prefetched.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::size_t bytes_prefetched() const
  {
    return _bytes_prefetched.load(std::memory_order_relaxed);
  }

 private:
  void worker_loop();

  /// Attempt one sweep over all connectors; returns the number of batches converted.
  std::size_t sweep(rmm::cuda_stream_view stream);

  config _config;
  std::vector<std::shared_ptr<split_connector>> _connectors;
  cucascade::memory::memory_space* _gpu_space;
  /// Last observed pop_count per connector (parallel to _connectors). A delta
  /// between sweeps means the scan is actively draining that connector — skip
  /// it so the prefetcher never serializes conversions the scan's own tasks
  /// would run concurrently.
  std::vector<std::atomic<std::uint64_t>> _last_pop_counts;
  /// steady_clock ms timestamp of the last observed pop per connector; a
  /// connector stays "draining" until drain_quiet_ms elapse with no pops.
  std::vector<std::atomic<std::int64_t>> _last_pop_time_ms;

  /// Dedicated streams so prefetch copies never share a stream with pipeline
  /// task work (the memory_space's shared round-robin pool would interleave
  /// our 5GB copies with task kernels on the same stream).
  std::unique_ptr<cucascade::memory::exclusive_stream_pool> _stream_pool;

  std::atomic<bool> _running{true};
  std::atomic<std::size_t> _batches_prefetched{0};
  std::atomic<std::size_t> _bytes_prefetched{0};
  std::vector<std::thread> _workers;
};

}  // namespace sirius::scan_manager
