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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sirius::exec {

/// Default pool sizes. The scan-manager pool takes the cores these leave over
/// (see scan_manager::default_scan_manager_num_threads).
inline constexpr int default_gpu_pipeline_num_threads = 4;
inline constexpr int default_downgrade_num_threads    = 1;

struct thread_pool_config {
  int num_threads{0};
  std::string thread_name_prefix{"thread"};
  std::vector<int> cpu_affinity_list;
};

/// Retry policy for GPU pipeline tasks rescheduled after an OOM or a transient
/// launch failure (`sirius.executor.pipeline.retry`).
///
/// A rescheduled task waits for a memory-release event (a task completing or a
/// downgrade freeing bytes) instead of sleeping a fixed interval, and is failed
/// only when the whole engine has made no progress for `no_progress_timeout`, it
/// kept failing while nothing else was running (`max_idle_retries`), or it has
/// used up `max_retries` attempts that had their full reservation. The
/// count ignores attempts that ran on a partial reservation: those failed
/// because memory was short, which waiting is meant to fix.
struct gpu_task_retry_config {
  /// Attempts with a full reservation a task may fail before the query fails.
  /// The sanity cap behind the wall-clock limit; 0 disables it.
  std::uint32_t max_retries{10000};
  /// Consecutive retries a task may fail while the engine is idle -- no progress
  /// since its previous retry, no other GPU task executing, no downgrade in
  /// flight -- before the query fails: nothing is left that could free memory,
  /// so waiting longer cannot help. 0 disables the limit.
  std::uint32_t max_idle_retries{10};
  /// Fail a retrying task when nothing anywhere completed a GPU task or freed
  /// memory through a downgrade for this long. 0 disables the limit.
  std::chrono::milliseconds no_progress_timeout{std::chrono::minutes{10}};
  /// Minimum wait before a retry, and the first retry's longest wait.
  std::chrono::milliseconds initial_backoff{20};
  /// Longest wait for a memory-release event before retrying anyway; the wait
  /// doubles from `initial_backoff` per consecutive retry without progress.
  std::chrono::milliseconds max_backoff{2000};
};

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
  /// When set, the GPU->HOST downgrade dispatch uses
  /// cucascade::memory::any_memory_space_in_tier_with_preference{Tier::HOST, *preferred_numa_node}
  /// so batches downgrade to the NUMA-local host memory_space when capacity is available,
  /// with cross-NUMA fallback. When unset (nullopt), the downgrade dispatch uses the
  /// unpreferred any_memory_space_in_tier{Tier::HOST} strategy (single-GPU default behavior).
  /// Populated by SiriusContext from config_.get_hw_topology().gpus[device_id].numa_node.
  std::optional<int> preferred_numa_node;

  /// How long a task waits for a downgrade request it issued before giving up and
  /// failing the query.
  ///
  /// The wait was previously unbounded (`request_downgrade(...).get()`), so a
  /// request that is never satisfied parks the task forever: the query holds all
  /// GPU memory at 0% utilisation with no error, which is far harder to diagnose
  /// or survive than a failed query. This is a liveness backstop, NOT a latency
  /// target — a legitimate request can be slow (a single 9.9 GB compressed
  /// downgrade was measured at 18.4 s), so the default leaves room above any
  /// duration observed in practice. Set to 0 to restore the unbounded wait.
  std::chrono::milliseconds downgrade_wait_timeout{std::chrono::minutes{1}};
};

}  // namespace sirius::exec
