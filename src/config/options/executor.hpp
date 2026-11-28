// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/block.hpp"
#include "config/option.hpp"
#include "config/options/scan_manager.hpp"
#include "config/size.hpp"
#include "config/validators.hpp"
#include "exec/config.hpp"

#include <chrono>
#include <vector>

namespace sirius::config {

namespace detail {
inline const exec::downgrade_executor_config& downgrade_defaults()
{
  static const exec::downgrade_executor_config value{};
  return value;
}
}  // namespace detail

/// `sirius.executor.task_creator`: the threads that turn ready inputs into GPU tasks.
struct task_creator_options : block_base<task_creator_options> {
  /// Unset: the resolver's fixed default.
  config_option<int> num_threads = option<int>("num_threads")
                                     .validate(positive<int>{})
                                     .mark_production()
                                     .describe("task creator worker threads")
                                     .build();
  config_option<std::vector<int>> cpu_affinity =
    option<std::vector<int>>("cpu_affinity")
      .fallback(std::vector<int>{})
      .describe("CPU cores task creator threads are pinned to")
      .build();

  auto fields()
  {
    return field_list(
      num_threads,
      cpu_affinity,
      removed("strategy",
              "removed; task creation policy is internal and currently demand-driven; remove "
              "this key"),
      removed("priority_order",
              "removed; scheduling priority is internal and currently source-first; remove this "
              "key"));
  }
};

/// `sirius.executor.pipeline`: the GPU pipeline threads.
struct pipeline_options : block_base<pipeline_options> {
  /// Unset: derived from the machine and the GPU count when the configuration is resolved.
  config_option<int> num_threads = option<int>("num_threads")
                                     .validate(positive<int>{})
                                     .mark_production()
                                     .describe("GPU pipeline worker threads")
                                     .build();

  auto fields() { return field_list(num_threads); }
};

/// `sirius.executor.downgrade`: the threads that spill batches to lower memory tiers.
struct downgrade_options : block_base<downgrade_options> {
  /// Unset: derived from the machine and the GPU count when the configuration is resolved.
  config_option<int> num_threads = option<int>("num_threads")
                                     .validate(positive<int>{})
                                     .mark_production()
                                     .describe("downgrade (spill) executor worker threads")
                                     .build();
  config_option<std::vector<int>> cpu_affinity =
    option<std::vector<int>>("cpu_affinity")
      .fallback(std::vector<int>{})
      .describe("CPU cores downgrade executor threads are pinned to")
      .build();
  /// Period of the memory pressure monitor; 0 turns the monitor off.
  config_option<std::chrono::milliseconds> monitor_period =
    option<std::chrono::milliseconds>("monitor_period")
      .validate(non_negative<std::chrono::milliseconds>{})
      .fallback(detail::downgrade_defaults().monitor_period)
      .describe("period of the memory pressure monitor")
      .build();
  /// Copy granularity of GPU-to-host spills; 0 submits a batch's copies at once.
  config_option<ubytes> copy_chunk_bytes =
    option<ubytes>("copy_chunk_bytes")
      .fallback(ubytes(detail::downgrade_defaults().copy_chunk_bytes))
      .describe("copy granularity of GPU-to-host spills")
      .build();

  auto fields() { return field_list(num_threads, cpu_affinity, monitor_period, copy_chunk_bytes); }
};

/// `sirius.executor`: the engine's thread pools and the scan manager.
struct executor_options : block_base<executor_options> {
  task_creator_options task_creator;
  scan_manager_options scan_manager;
  pipeline_options pipeline;
  downgrade_options downgrade;

  auto fields()
  {
    return field_list(sub("task_creator", task_creator),
                      sub("scan_manager", scan_manager),
                      sub("pipeline", pipeline),
                      sub("downgrade", downgrade));
  }
};

}  // namespace sirius::config
