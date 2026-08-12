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

  /// Copy submission granularity for GPU->HOST spill conversions. When non-zero, Sirius
  /// replaces the builtin cucascade converter (which submits an entire batch's D2H copies as
  /// one monolithic batched call) with a chunked one that flushes every ~copy_chunk_bytes while
  /// the column tree is still being walked. 0 keeps the builtin converter.
  std::size_t copy_chunk_bytes{1ull << 30};

  /// When true (default), a monitor-issued downgrade request stops as soon as live pressure
  /// drops back below the *trigger* threshold, so the spilled set is proportional to the actual
  /// overflow. When false, it runs down to the stop threshold, flushing the whole trigger->stop
  /// band.
  bool overflow_proportional_spill{true};

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
