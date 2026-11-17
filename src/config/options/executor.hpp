// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/block.hpp"
#include "config/option.hpp"
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
  /// Unset: derived from the machine when the configuration is resolved.
  config_option<int> num_threads = option<int>("num_threads").validate(positive<int>{}).build();
  config_option<std::vector<int>> cpu_affinity =
    option<std::vector<int>>("cpu_affinity").fallback(std::vector<int>{}).build();

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
  config_option<int> num_threads = option<int>("num_threads").validate(positive<int>{}).build();

  auto fields() { return field_list(num_threads); }
};

/// `sirius.executor.downgrade`: the threads that spill batches to lower memory tiers.
struct downgrade_options : block_base<downgrade_options> {
  /// Unset: derived from the machine and the GPU count when the configuration is resolved.
  config_option<int> num_threads = option<int>("num_threads").validate(positive<int>{}).build();
  config_option<std::vector<int>> cpu_affinity =
    option<std::vector<int>>("cpu_affinity").fallback(std::vector<int>{}).build();
  /// Period of the memory pressure monitor; 0 turns the monitor off.
  config_option<std::chrono::milliseconds> monitor_period =
    option<std::chrono::milliseconds>("monitor_period")
      .validate(non_negative<std::chrono::milliseconds>{})
      .fallback(detail::downgrade_defaults().monitor_period)
      .build();
  /// Copy granularity of GPU-to-host spills; 0 submits a batch's copies at once.
  config_option<bytes> copy_chunk_bytes =
    option<bytes>("copy_chunk_bytes")
      .fallback(to_bytes(detail::downgrade_defaults().copy_chunk_bytes))
      .build();

  auto fields() { return field_list(num_threads, cpu_affinity, monitor_period, copy_chunk_bytes); }
};

/// `sirius.executor`: the engine's thread pools.
struct executor_options : block_base<executor_options> {
  task_creator_options task_creator;
  pipeline_options pipeline;
  downgrade_options downgrade;

  auto fields()
  {
    return field_list(
      sub("task_creator", task_creator), sub("pipeline", pipeline), sub("downgrade", downgrade));
  }
};

}  // namespace sirius::config
