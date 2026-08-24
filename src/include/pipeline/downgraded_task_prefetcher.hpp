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

#include "exec/multi_index_priority_queue.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "parallel/task.hpp"

#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/stream_pool.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace sirius::pipeline {

/**
 * @brief Background HOST/DISK->GPU upgrader for the inputs of queued pipeline tasks.
 *
 * Under memory pressure the downgrade executor spills the input batches of
 * queued gpu_pipeline_tasks (and of repository batches that later become task
 * inputs) to HOST or DISK. When such a task is finally dispatched, the upgrade
 * back to GPU happens synchronously inside prepare_for_processing
 * (lock_or_prepare_batch), serializing the H2D copy (or disk read) with the
 * task's compute — the same copy/compute alternation the scan prefetcher
 * eliminates for pinned-cache scan splits.
 *
 * This prefetcher overlaps the two: worker threads walk the pending tasks in
 * dispatch order (the scheduler's matcher pops with pop_if(front_to_back=true),
 * so the queue front is next to run) and convert their non-GPU-resident input
 * batches back to GPU tier early, on dedicated streams, gated on GPU memory
 * headroom. When the task later runs, prepare_for_processing finds the batch
 * already GPU-resident and skips the upload.
 *
 * Races with a consumer (or with the downgrade executor) are arbitrated by the
 * data_batch state machine: conversion goes through convertible_data_batch with
 * blocking=false (try_to_mutable, skip on contention), and
 * prepare_for_processing re-checks the tier under its own exclusive lock.
 *
 * Downgrade tug-of-war safety: the downgrade executor only fires under memory
 * pressure (should_downgrade_memory), while this prefetcher only upgrades while
 * at least min_free_fraction of the GPU space would remain free afterwards —
 * the two regimes are disjoint by construction, so the prefetcher can never
 * feed an eviction storm. Additionally the Tier-2 eviction search walks the
 * queue back-to-front (farthest from dispatch), the opposite end from the one
 * prefetched here.
 *
 * Unlike the scan prefetcher there is no drain guard: each dispatched task
 * upgrades only its own few input batches (not a long shared queue of
 * conversions), so lock contention with a preparing task at worst delays that
 * task by the remainder of one in-flight conversion it would have had to
 * perform itself anyway.
 */
class downgraded_task_prefetcher {
 public:
  struct config {
    /// Number of prefetch worker threads (each with its own stream).
    std::size_t num_threads{2};
    /// Keep at least this fraction of the GPU space free after each upgrade.
    double min_free_fraction{0.4};
    /// Worker sweep interval while waiting for headroom / new tasks.
    std::size_t poll_interval_ms{2};
    /// Only prefetch inputs of the first this-many tasks in dispatch order.
    /// Bounds the prefetched-but-unconsumed GPU footprint to what is about to
    /// be consumed anyway — an unbounded walk under a small GPU budget fills
    /// the space with queued-task inputs and starves the running pipeline.
    std::size_t max_lookahead_tasks{4};
    /// After observing memory pressure (downgrade trigger or floor breach),
    /// stay idle until this long passes with no further pressure. Prevents the
    /// downgrade<->prefetch bounce: right after an eviction frees memory the
    /// instantaneous headroom looks fine, but the system is still pressured
    /// and upgrading would hand batches straight back to the eviction path.
    std::size_t pressure_quiet_ms{250};
    /// EXPERIMENT: keep prefetching while the downgrade executor is actively
    /// evicting (skips the should_downgrade/quiet-period gate; the per-batch
    /// headroom floor still applies as the OOM guard). Safe only when eviction
    /// victims and prefetch targets are order-disjoint (front-of-queue targets
    /// vs back/last-consumed victims).
    bool prefetch_during_pressure{false};
  };

  downgraded_task_prefetcher(config cfg,
                             exec::multi_index_priority_queue<sirius::parallel::itask>& task_queue,
                             sirius::memory::sirius_memory_reservation_manager& res_mgr,
                             cucascade::memory::memory_space* gpu_space);

  ~downgraded_task_prefetcher();

  downgraded_task_prefetcher(const downgraded_task_prefetcher&)            = delete;
  downgraded_task_prefetcher& operator=(const downgraded_task_prefetcher&) = delete;
  downgraded_task_prefetcher(downgraded_task_prefetcher&&)                 = delete;
  downgraded_task_prefetcher& operator=(downgraded_task_prefetcher&&)      = delete;

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

  /// Attempt one sweep over the queued tasks; returns the number of batches converted.
  std::size_t sweep(rmm::cuda_stream_view stream);

  config _config;
  exec::multi_index_priority_queue<sirius::parallel::itask>& _task_queue;
  sirius::memory::sirius_memory_reservation_manager& _res_mgr;
  cucascade::memory::memory_space* _gpu_space;
  /// Single-element target list passed to convertible_data_batch::convert.
  std::vector<const cucascade::memory::memory_space*> _targets;

  /// Dedicated streams so prefetch copies never share a stream with pipeline
  /// task work.
  std::unique_ptr<cucascade::memory::exclusive_stream_pool> _stream_pool;

  /// steady_clock ms timestamp of the last observed memory pressure; sweeps
  /// stay idle until pressure_quiet_ms elapse with no pressure.
  std::atomic<std::int64_t> _last_pressure_ms{0};

  std::atomic<bool> _running{true};
  std::atomic<std::size_t> _batches_prefetched{0};
  std::atomic<std::size_t> _bytes_prefetched{0};
  /// Per-source-tier counts, for post-run analysis of what was upgraded.
  std::atomic<std::size_t> _host_batches{0};
  std::atomic<std::size_t> _disk_batches{0};
  std::vector<std::thread> _workers;
};

}  // namespace sirius::pipeline
