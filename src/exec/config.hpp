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

#pragma once

#include <cucascade/exec/config.hpp>

#include <chrono>
#include <cstddef>
#include <optional>

namespace sirius::exec {

using cucascade::exec::default_gpu_pipeline_num_threads;
using cucascade::exec::thread_pool_config;

/// Unresolved downgrade fallback; hardware resolution derives the configured default.
inline constexpr int default_downgrade_num_threads = 1;

/// Configuration for the downgrade executor.
/// Embeds the thread pool config plus downgrade-specific settings.
struct downgrade_executor_config {
  exec::thread_pool_config thread_pool{.num_threads        = default_downgrade_num_threads,
                                       .thread_name_prefix = "downgrade"};

  /// Period for the memory pressure monitor loop.
  /// Set to 0 to disable the monitor loop entirely.
  std::chrono::milliseconds monitor_period{std::chrono::milliseconds{10}};

  /// Copy submission granularity for GPU->HOST spill conversions. When non-zero, Sirius
  /// replaces the builtin cucascade converter (which submits an entire batch's D2H copies as
  /// one monolithic batched call) with a chunked one that flushes every ~copy_chunk_bytes while
  /// the column tree is still being walked. 0 keeps the builtin converter.
  std::size_t copy_chunk_bytes{1ull << 30};

  /// Preferred HOST memory_space device_id (NUMA node) for the downgrade target.
  /// When set, the GPU->HOST downgrade orders the matching HOST memory_space(s) first in its
  /// target list, so batches downgrade to the NUMA-local host memory_space when capacity is
  /// available, with cross-NUMA fallback. When unset (nullopt), HOST spaces are tried in the
  /// reservation manager's order (single-GPU default behavior).
  /// Populated by SiriusContext from config_.get_hw_topology().gpus[device_id].numa_node.
  std::optional<int> preferred_numa_node;

  /// Complete a reserve-path request (request_downgrade(): a task is waiting for its
  /// reservation) as soon as its predicate holds, also while another request is being
  /// processed, and process the ones still unsatisfied before the next queued request. false
  /// (the default) keeps strict FIFO order. Off by default: on sorts larger than the GPU it
  /// removes most of the time in which no task computes without shortening the run, which is
  /// bound by spill I/O, and tasks that start while the GPU is still full can make MERGE_SORT
  /// derive much smaller windows from the bytes free at arming.
  bool prioritize_reserve_requests{false};

  /// A request stops dispatching candidates after this many conversions in a row failed for a
  /// reason other than memory (an unwritable spill directory, an I/O error), with none
  /// succeeding. 0 disables the limit.
  std::size_t max_consecutive_convert_failures{8};

  /// After a monitor request that freed nothing because its conversions failed, the monitor
  /// waits this long before issuing the next one, doubling up to the maximum. A request that
  /// frees anything resets it.
  std::chrono::milliseconds failure_backoff_initial{std::chrono::milliseconds{100}};
  std::chrono::milliseconds failure_backoff_max{std::chrono::milliseconds{5000}};
};

/// How the GPU pipeline executor behaves when a task's memory reservation cannot be granted at
/// once. Process-wide; see gpu_pipeline_executor::set_reservation_policy().
struct reservation_wait_policy {
  /// Wait for the reservation on the task's worker thread, so the manager loop keeps
  /// dispatching other ready tasks. false (the default) waits on the manager loop: one waiting
  /// task stalls the executor. Off by default: it gives no wall-time gain on spilling sorts (see
  /// downgrade_executor_config::prioritize_reserve_requests), and a task waiting on a worker
  /// is no longer in the scheduler's queue, where the downgrade executor could evict its
  /// inputs.
  bool wait_on_worker{false};
  /// A waiting task may be overtaken by at most this many later tasks, and for at most
  /// max_bypass_time; after that the manager dispatches nothing new until it is served.
  std::size_t max_bypass_tasks{16};
  std::chrono::milliseconds max_bypass_time{std::chrono::milliseconds{1000}};
  /// How long a waiting task lets running tasks release memory before it asks the downgrade
  /// executor to evict. It asks at once when no other reservation is active.
  std::chrono::milliseconds patience{std::chrono::milliseconds{20}};
  /// Fail a task at once when it runs out of memory while its reservation already covers the
  /// whole memory space and its input cannot shrink: a retry cannot succeed.
  bool fail_fast_on_full_space_oom{true};
};

}  // namespace sirius::exec
