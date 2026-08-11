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

/// Configuration for the downgrade executor.
/// Embeds the thread pool config plus downgrade-specific settings.
struct downgrade_executor_config {
  exec::thread_pool_config thread_pool{.num_threads        = default_downgrade_num_threads,
                                       .thread_name_prefix = "downgrade"};

  /// Period for the memory pressure monitor loop.
  /// Set to 0 to disable the monitor loop entirely.
  std::chrono::milliseconds monitor_period{std::chrono::milliseconds{10}};

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
