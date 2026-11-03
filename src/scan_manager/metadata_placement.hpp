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

#include "exec/thread_util.hpp"
#include "log/logging.hpp"
#include "memory/topology_index.hpp"
#include "scan_manager/config.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace sirius::scan_manager {
struct metadata_placement {
  // Parallel arrays indexed by pool domain. nodes contains OS NUMA node IDs
  // (-1 for unknown fallback locality); cpus contains the permitted worker masks.
  std::vector<int> nodes;
  std::vector<std::vector<int>> cpus;
};

inline std::vector<int> read_cpu_list(std::string const& path)
{
  std::ifstream file(path);
  std::string list;
  if (!(file >> list)) return {};
  std::vector<int> result;
  std::istringstream parts(list);
  std::string part;
  while (std::getline(parts, part, ',')) {
    auto dash = part.find('-');
    int first = std::stoi(part.substr(0, dash));
    int last  = dash == std::string::npos ? first : std::stoi(part.substr(dash + 1));
    if (first < 0 || last < first || last >= CPU_SETSIZE)
      throw std::runtime_error("unsupported CPU list in " + path);
    for (int cpu = first; cpu <= last; ++cpu)
      result.push_back(cpu);
  }
  return result;
}

// Read the deployment's cgroup allowance, not the DuckDB caller's potentially
// single-CPU affinity. Explicit scan-manager cpu_affinity_list further restricts it.
// On systems without readable cgroup v2 cpusets, preserve the inherited allowance.
inline std::vector<int> metadata_allowed_cpus()
{
  std::ifstream cgroups("/proc/self/cgroup");
  std::string line;
  while (std::getline(cgroups, line)) {
    if (!line.starts_with("0::/")) continue;
    auto path = line.substr(3);
    if (path.find("..") != std::string::npos) break;
    auto cpus = read_cpu_list("/sys/fs/cgroup" + path + "/cpuset.cpus.effective");
    if (!cpus.empty()) return cpus;
    // A cgroup namespace may expose this process's cgroup at the mount root.
    cpus = read_cpu_list("/sys/fs/cgroup/cpuset.cpus.effective");
    if (!cpus.empty()) return cpus;
  }
  SIRIUS_LOG_WARN(
    "scan metadata placement: cgroup CPU allowance unavailable; using caller affinity");
  return exec::thread_util::get_current_thread_affinity();
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
