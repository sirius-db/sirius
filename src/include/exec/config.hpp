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
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sirius::exec {

/// Default pool sizes. The scan-manager pool takes the cores these leave over
/// (see scan_manager::default_scan_manager_num_threads).
inline constexpr int default_gpu_pipeline_num_threads = 4;
inline constexpr int default_downgrade_num_threads    = 1;

/// Default cap on OOM/contention reschedules of one GPU pipeline task before
/// the query is failed with a classified retry-cap error. History: bumped from
/// 10 to 100 as part of follow-up #17 — SF100 Q11 with cache=table_gpu +
/// num_gpus=2 exhausted the old 10-retry budget against cross-GPU BUILD_PROBE
/// batch-lock contention (each convert-release cycle is O(100ms) at SF100
/// scale). 100 retries x 50 ms backoff (~5 s) clears the contention window
/// while still bailing out on truly wedged queries. Overridable via the
/// operator_params YAML section / `SET gpu_reservation_max_retries` (per-query
/// snapshot semantics, register E1).
inline constexpr uint32_t default_gpu_reservation_max_retries = 100;

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
};

}  // namespace sirius::exec
