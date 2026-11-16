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
#include <atomic>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace sirius::scan_manager {
struct metadata_placement {
  // Parallel arrays indexed by pool domain. nodes contains OS NUMA node IDs
  // (-1 for unknown fallback locality); cpus contains the permitted worker masks.
  std::vector<int> nodes;
  std::vector<std::vector<int>> cpus;
};

inline std::optional<std::vector<int>> read_cpu_list(std::string const& path)
{
  errno = 0;
  std::ifstream file(path);
  if (!file) {
    if (errno == ENOENT) return std::nullopt;
    throw std::system_error(errno, std::generic_category(), "reading " + path);
  }
  std::string list;
  if (!(file >> list)) return std::vector<int>{};
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

// mountinfo escapes whitespace and backslashes as three octal digits.
inline std::string unescape_mount_path(std::string const& value)
{
  std::string result;
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '\\' && i + 3 < value.size() &&
        value.find_first_not_of("01234567", i + 1) >= i + 4) {
      result += static_cast<char>(std::stoi(value.substr(i + 1, 3), nullptr, 8));
      i += 3;
    } else {
      result += value[i];
    }
  }
  return result;
}

// Resolve membership against the visible cgroup2 mount root, including bind mounts
// and namespace roots. Missing leaf controller files inherit the nearest ancestor's
// effective cpuset. An unreadable/malformed file must not authorize a broader ancestor.
inline std::vector<int> metadata_cgroup_cpus(
  std::istream& membership,
  std::istream& mounts,
  std::function<std::optional<std::vector<int>>(std::string const&)> const& read = read_cpu_list)
{
  namespace fs = std::filesystem;
  auto valid   = [](fs::path const& path) {
    return path.is_absolute() && std::none_of(path.begin(), path.end(), [](auto const& part) {
             return part == ".." || part == ".";
           });
  };
  fs::path group;
  std::string line;
  while (std::getline(membership, line)) {
    if (line.starts_with("0::/")) {
      group = line.substr(3);
      break;
    }
  }
  if (!valid(group)) return {};
  fs::path mount, relative;
  size_t specificity = 0;
  while (std::getline(mounts, line)) {
    auto separator = line.find(" - ");
    if (separator == std::string::npos) continue;
    std::istringstream suffix(line.substr(separator + 3));
    std::string type;
    if (!(suffix >> type) || type != "cgroup2") continue;
    std::istringstream fields(line.substr(0, separator));
    std::string id, parent, device, root_text, mount_text;
    if (!(fields >> id >> parent >> device >> root_text >> mount_text)) continue;
    fs::path root = unescape_mount_path(root_text), point = unescape_mount_path(mount_text);
    if (!valid(root) || !valid(point)) continue;
    auto within = group.lexically_relative(root);
    if (within.empty() || *within.begin() == "..") continue;
    if (root.native().size() >= specificity) {
      specificity = root.native().size();
      mount       = std::move(point);
      relative    = std::move(within);
    }
  }
  if (mount.empty()) return {};  // No proven mapping: use inherited thread affinity.
  auto current = (mount / relative).lexically_normal();
  for (;;) {
    if (auto cpus = read((current / "cpuset.cpus.effective").string())) return *cpus;
    if (current == mount) return {};
    current = current.parent_path();
  }
}

// Read the deployment allowance, not an incidental single-CPU DuckDB caller mask.
// If it cannot be established, keep the inherited allowance rather than guessing root.
inline std::vector<int> metadata_allowed_cpus()
{
  try {
    std::ifstream cgroups("/proc/self/cgroup"), mounts("/proc/self/mountinfo");
    auto cpus = metadata_cgroup_cpus(cgroups, mounts);
    if (!cpus.empty()) return cpus;
  } catch (std::exception const& error) {
    SIRIUS_LOG_WARN("scan metadata placement: {}", error.what());
  }
  SIRIUS_LOG_WARN(
    "scan metadata placement: cgroup CPU allowance unavailable; using caller affinity");
  return exec::thread_util::get_current_thread_affinity();
}

// A manager-wide cursor spreads independent attempts across their eligible domains.
// Rotating only the starting point preserves each source's stable locality and GPU subset.
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
