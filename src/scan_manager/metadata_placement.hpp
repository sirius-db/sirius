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

#include "log/logging.hpp"
#include "memory/topology_index.hpp"
#include "scan_manager/config.hpp"

#include <algorithm>
#include <atomic>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::scan_manager {
struct metadata_placement {
  // Parallel arrays indexed by pool domain. nodes contains OS NUMA node IDs
  // (-1 for unknown fallback locality); cpus contains the permitted worker masks.
  std::vector<int> nodes;
  std::vector<std::vector<int>> cpus;
};

// A manager-wide cursor spreads independent attempts across their eligible domains.
// Rotating only the starting point preserves each coalescer's stable locality and GPU subset.
inline void rotate_metadata_domains(std::vector<int>& domains, std::atomic<size_t>& next)
{
  if (domains.size() < 2) return;
  auto start = next.fetch_add(1, std::memory_order_relaxed) % domains.size();
  std::rotate(domains.begin(), domains.begin() + start, domains.end());
}

inline metadata_placement make_metadata_placement(memory::topology_index const& topology,
                                                  std::vector<int> const& allowed,
                                                  std::vector<int> const& configured,
                                                  size_t workers)
{
  metadata_placement result;
  std::set<int> seen;
  for (int device : topology.gpu_ids()) {
    int node = topology.numa_node_of(device);
    if (node < 0 || seen.contains(node)) continue;
    auto const& gpus = topology.get_topology().gpus;
    auto gpu         = std::find_if(gpus.begin(), gpus.end(), [device](auto const& g) {
      return static_cast<int>(g.id) == device;
    });
    if (gpu == gpus.end()) continue;
    std::vector<int> cpus;
    for (int cpu : gpu->cpu_cores) {
      if (std::find(allowed.begin(), allowed.end(), cpu) == allowed.end()) continue;
      if (!configured.empty() &&
          std::find(configured.begin(), configured.end(), cpu) == configured.end())
        continue;
      cpus.push_back(cpu);
    }
    if (cpus.empty()) continue;
    seen.insert(node);
    result.nodes.push_back(node);
    result.cpus.push_back(std::move(cpus));
  }
  if (result.cpus.empty()) {
    std::vector<int> fallback;
    for (int cpu : allowed)
      if (configured.empty() ||
          std::find(configured.begin(), configured.end(), cpu) != configured.end())
        fallback.push_back(cpu);
    if (fallback.empty()) throw std::invalid_argument("scan metadata workers have no allowed CPUs");
    SIRIUS_LOG_WARN("scan metadata placement: no eligible GPU-local domain; using allowed CPUs");
    result.nodes.push_back(-1);
    result.cpus.push_back(std::move(fallback));
  }
  if (result.cpus.size() > workers)
    throw std::invalid_argument("scan_manager.num_threads must cover each eligible NUMA domain");
  return result;
}
// Discover domains before resolving the automatic pool size. Explicit counts
// retain the same rejection policy as make_metadata_placement's worker check.
inline metadata_placement resolve_metadata_placement(memory::topology_index const& topology,
                                                     std::vector<int> const& allowed,
                                                     scan_manager_config& config)
{
  auto placement = make_metadata_placement(
    topology, allowed, config.thread_pool.cpu_affinity_list, std::numeric_limits<size_t>::max());
  config.thread_pool.num_threads = resolve_scan_manager_num_threads(config, placement.cpus.size());
  return placement;
}
}  // namespace sirius::scan_manager
