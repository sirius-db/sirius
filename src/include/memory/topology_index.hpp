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

#include <cucascade/memory/topology_discovery.hpp>

#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::memory {

/// @brief Fast bidirectional lookup over a discovered hardware topology,
///        scoped to a specific set of GPU device ids.
///
/// cucascade's @c system_topology_info stores GPUs as a flat vector, each
/// carrying its NUMA node — answering "which NUMA node owns this GPU?" or
/// "which GPUs sit on this NUMA node?" means scanning that vector every time.
/// This index builds both maps once at construction so callers (NUMA-aware
/// bounce-buffer placement, per-node reactor pools, ...) can resolve either
/// direction in O(1).
///
/// The index is scoped to the @p device_ids it is built with — typically the
/// GPUs Sirius actually reserved memory on, not necessarily every GPU the
/// topology discovered.  @c gpu_ids(), @c gpus_of() and @c numa_node_of() only
/// ever report those device ids; topology GPUs outside the set are ignored.
///
/// The index owns a copy of the topology, so it stays valid independently of
/// the @c topology_discovery that produced it.  NUMA node ids are taken
/// verbatim from the topology, including the sentinel @c -1 for "unknown" (also
/// used when a requested device id is absent from the topology).
class topology_index {
 public:
  /// @brief Build the index for @p device_ids against a discovered topology.
  /// @param topology    the system topology to resolve NUMA nodes from (copied
  ///                    and retained).
  /// @param device_ids  the GPU device ids to scope the index to, in the order
  ///                    callers should iterate them (e.g. the reserved GPU
  ///                    memory spaces).  Retained verbatim as @c gpu_ids().
  topology_index(cucascade::memory::system_topology_info topology, std::vector<int> device_ids)
    : _topology(std::move(topology)), _gpu_ids(std::move(device_ids))
  {
    // Resolve each device id's NUMA node from the topology once.
    std::unordered_map<int, int> topology_numa;
    for (auto const& gpu : _topology.gpus) {
      topology_numa[static_cast<int>(gpu.id)] = gpu.numa_node;
    }
    for (int const gpu_id : _gpu_ids) {
      auto it              = topology_numa.find(gpu_id);
      int const numa_node  = it == topology_numa.end() ? -1 : it->second;
      _gpu_to_numa[gpu_id] = numa_node;
      _numa_to_gpus[numa_node].push_back(gpu_id);
    }
  }

  /// @brief The topology this index was built from.
  [[nodiscard]] const cucascade::memory::system_topology_info& get_topology() const noexcept
  {
    return _topology;
  }

  /// @brief NUMA node hosting @p gpu.
  /// @param gpu  CUDA device id.
  /// @return the GPU's NUMA node, or @c -1 if the GPU is not in this index's
  ///         device set (the same sentinel the topology uses for an unknown
  ///         node).
  [[nodiscard]] int numa_node_of(int gpu) const
  {
    auto it = _gpu_to_numa.find(gpu);
    return it == _gpu_to_numa.end() ? -1 : it->second;
  }

  /// @brief GPUs attached to @p numa.
  /// @param numa  NUMA node id.
  /// @return a view of this index's device ids on that node (in scope order),
  ///         or an empty span if none map to it.  The span is valid for the
  ///         lifetime of this index.
  [[nodiscard]] std::span<const int> gpus_of(int numa) const
  {
    auto it = _numa_to_gpus.find(numa);
    return it == _numa_to_gpus.end() ? std::span<const int>{} : std::span<const int>{it->second};
  }

  [[nodiscard]] std::span<const int> gpu_ids() const noexcept { return _gpu_ids; }

 private:
  cucascade::memory::system_topology_info _topology;
  std::unordered_map<int, int> _gpu_to_numa;                ///< GPU device id -> NUMA node.
  std::unordered_map<int, std::vector<int>> _numa_to_gpus;  ///< NUMA node -> GPU device ids.
  std::vector<int> _gpu_ids;  ///< Scoped GPU device ids, in caller order (for span stability).
};

}  // namespace sirius::memory
