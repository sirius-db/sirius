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

#include "exec/config.hpp"

#include <string>
#include <string_view>
#include <unordered_map>

namespace sirius::creator {

/**
 * @brief How a scheduling request should be serviced.
 *
 * - active:    create tasks in response to input from an active operator
 * - lookahead: proactively create tasks for operators that might be needed soon.
 */
enum class request_type { active, lookahead };

/// ADL-discoverable string conversion so yaml_reader can parse request_type values.
inline bool string_to_enum(std::string_view sv, request_type& out)
{
  static const std::unordered_map<std::string_view, request_type> map = {
    {"active", request_type::active},
    {"lookahead", request_type::lookahead},
  };
  auto it = map.find(sv);
  if (it == map.end()) { return false; }
  out = it->second;
  return true;
}

inline bool enum_to_string(request_type type, std::string& s)
{
  switch (type) {
    case request_type::active: s = "active"; return true;
    case request_type::lookahead: s = "lookahead"; return true;
    default: return false;
  }
}

/**
 * @brief Within-branch pipeline scheduling priority direction.
 *
 * task_creator assigns each pipeline in a linear branch a scheduling priority; this selects which
 * end of the branch is dispatched first:
 * - source: the head (closest to the scan / upstream) runs first — plan order (default).
 * - sink:   reverses it, so pipelines farther from the scan (downstream) run first.
 */
enum class priority_order { sink, source };

/// ADL-discoverable string conversion so yaml_reader can parse priority_order values.
inline bool string_to_enum(std::string_view sv, priority_order& out)
{
  static const std::unordered_map<std::string_view, priority_order> map = {
    {"sink", priority_order::sink},
    {"source", priority_order::source},
  };
  auto it = map.find(sv);
  if (it == map.end()) { return false; }
  out = it->second;
  return true;
}

inline bool enum_to_string(priority_order order, std::string& s)
{
  switch (order) {
    case priority_order::sink: s = "sink"; return true;
    case priority_order::source: s = "source"; return true;
    default: return false;
  }
}

/// Default task-creator pool size; counted in the scan-manager sizing budget
/// (see scan_manager::default_scan_manager_num_threads).
inline constexpr int default_task_creator_num_threads = 1;

/// Configuration for the task creator.
/// Embeds the thread pool config plus internal scheduling policy.
struct task_creator_config {
  exec::thread_pool_config thread_pool{.num_threads        = default_task_creator_num_threads,
                                       .thread_name_prefix = "task_creator"};

  /// Internal policy for servicing scheduling requests. The current default is active
  /// (demand-driven only); lookahead remains available to engine-controlled policy.
  request_type strategy{request_type::active};

  /// Internal within-branch scheduling priority, consumed by compute_pipeline_priorities.
  /// The current engine policy keeps plan order (head/scan first); sink remains available to a
  /// future engine-controlled policy.
  priority_order priority{priority_order::source};
};

}  // namespace sirius::creator
