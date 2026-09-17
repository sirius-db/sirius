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

#include "scan_manager/config.hpp"
#include "scan_manager/split_connector.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

namespace sirius::scan_manager {

/**
 * @brief Background host->GPU memory prefetcher for queued pinned-cache scan splits.
 *
 * Host-pinned scans put all their splits (per-query wrapper data_batches over
 * host slices of the pinned chunks) onto their split_connectors at query start,
 * but the H2D copy for each batch happens only when the scan's pipeline task
 * executes (scan_operator_input::prepare_for_processing). Because scan tasks
 * are created strictly on topology demand, the copy engines sit idle during
 * compute-bound phases and then the next scan's H2D burst runs with idle SMs.
 *
 * The prefetcher overlaps the two: worker threads walk the connectors in scan
 * execution order (the scan manager passes them ordered by @c _scan_op_order,
 * the same order the task creator drains them) and convert pending resident
 * batches to GPU tier early, gated on GPU memory headroom. When the scan task
 * later runs, prepare_for_processing sees the batch already GPU-resident and
 * skips the upload (and the task creator's cached-scan locality derivation
 * reads the batch's post-conversion space, so device dispatch stays
 * consistent).
 *
 * Why worker threads (rather than one async stream of copies): a conversion is
 * not a bare cudaMemcpyAsync — convert_to allocates the device table,
 * reconstructs the cudf table from the host layout, and synchronizes its
 * stream before the batch's exclusive lock can be released, so each in-flight
 * conversion needs a thread to drive it. Concurrency across batches therefore
 * scales with num_threads. Each worker's private CUDA stream carries no copy
 * traffic (the converter allocates and copies on a pool stream it acquires
 * internally); it exists only as a stable per-worker key for attaching the
 * admission reservation to the allocation tracker.
 *
 * Races with a consumer are arbitrated by the data_batch state machine: the
 * conversion holds the exclusive (mutable) lock via try_to_mutable (skip on
 * contention), and prepare_for_processing re-checks the tier under its own
 * lock.
 *
 * Memory safety: converted batches live in connector queues, which the
 * downgrade executor does NOT scan (it walks data repositories), so
 * prefetched-but-unconsumed bytes cannot be reclaimed under pressure.
 * Admission must therefore reserve BEFORE gating: the worker reserves the
 * conversion peak (charging availability immediately) and only then checks
 * the min_free_fraction floor, so the check sees every concurrent worker's
 * in-flight admission and the floor holds regardless of num_threads. The
 * reservation is then attached to the worker thread's allocation tracker so
 * the conversion's allocations draw it down instead of being counted a
 * second time on top of it.
 */
class memory_prefetcher {
 public:
  memory_prefetcher(memory_prefetcher_config cfg,
                    std::vector<std::shared_ptr<split_connector>> connectors,
                    cucascade::memory::memory_space* gpu_space);

  ~memory_prefetcher();

  memory_prefetcher(const memory_prefetcher&)            = delete;
  memory_prefetcher& operator=(const memory_prefetcher&) = delete;
  memory_prefetcher(memory_prefetcher&&)                 = delete;
  memory_prefetcher& operator=(memory_prefetcher&&)      = delete;

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
  void worker_loop(std::size_t worker_index);

  /// Attempt one sweep over all connectors; returns the number of batches converted.
  std::size_t sweep(rmm::cuda_stream_view stream);

  memory_prefetcher_config _config;
  std::vector<std::shared_ptr<split_connector>> _connectors;
  /// Per-connector work claim (parallel to _connectors): only one worker at a
  /// time may convert on a DRAINING connector, so prefetch parallelism never
  /// stacks on top of the active scan's own conversion threads (regresses
  /// short scan-bound queries). Quiet connectors allow full parallelism.
  std::unique_ptr<std::atomic<bool>[]> _drain_claims;
  cucascade::memory::memory_space* _gpu_space;

  /// One stream per worker, borrowed (NOT owned): converted batches are
  /// dealloc-bound to it and outlive the worker that made them.
  std::vector<rmm::cuda_stream_view> _worker_streams;

  std::atomic<bool> _running{true};
  std::atomic<std::size_t> _batches_prefetched{0};
  std::atomic<std::size_t> _bytes_prefetched{0};
  /// Diagnostic gate counters (logged at stop): why sweeps stopped early.
  /// _stops_reservation: the peak reservation was refused; _stops_headroom:
  /// the reservation was charged but the post-charge floor check failed.
  std::atomic<std::size_t> _stops_headroom{0};
  std::atomic<std::size_t> _stops_reservation{0};
  std::atomic<std::size_t> _skips_lock{0};
  std::atomic<std::size_t> _skips_draining{0};
  std::atomic<std::size_t> _errors_conversion{0};
  std::vector<std::thread> _workers;
};

}  // namespace sirius::scan_manager
