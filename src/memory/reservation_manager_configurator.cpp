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

#include "memory/reservation_manager_configurator.hpp"

#include "memory/common.hpp"
#include "memory/topology_discovery.hpp"
#include "mr/device/device_memory_resource.hpp"

#include <rmm/cuda_device.hpp>

#include <numa.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <set>
#include <variant>

namespace cucascade {
namespace memory {

using builder_reference   = reservation_manager_configurator::builder_reference;
using memory_space_config = reservation_manager_configurator::memory_space_config;

builder_reference& reservation_manager_configurator::set_number_of_gpus(std::size_t n_gpus)
{
  assert(n_gpus > 0 && "Number of GPUs must be positive");
  n_gpus_or_gpu_ids_ = n_gpus;
  return *this;
}

builder_reference& reservation_manager_configurator::set_gpu_ids(std::vector<int> gpu_ids)
{
  assert(!gpu_ids.empty() && "GPU IDs list cannot be empty");
  n_gpus_or_gpu_ids_ = std::move(gpu_ids);
  return *this;
}

builder_reference& reservation_manager_configurator::set_device_tier_id_to_gpu_id_map(
  const std::unordered_map<int, int>& device_to_gpu_map)
{
  assert(!device_to_gpu_map.empty() && "GPU IDs list cannot be empty");
  n_gpus_or_gpu_ids_ = device_to_gpu_map;
  return *this;
}

builder_reference& reservation_manager_configurator::set_gpu_usage_limit(std::size_t bytes)
{
  gpu_usage_limit_or_ratio_ = bytes;
  return *this;
}

builder_reference& reservation_manager_configurator::set_usage_limit_ratio_per_gpu(double fraction)
{
  assert(fraction > 0.0 && fraction <= 1.0 && "Usage limit ratio must be in (0.0, 1.0]");
  gpu_usage_limit_or_ratio_ = fraction;
  return *this;
}

builder_reference& reservation_manager_configurator::set_reservation_limit_ratio_per_gpu(
  double fraction)
{
  assert(fraction > 0.0 && fraction <= 1.0 && "Reservation limit ratio must be in (0.0, 1.0]");
  gpu_reservation_limit_ratio_ = fraction;
  return *this;
}

builder_reference& reservation_manager_configurator::set_capacity_per_numa_node(std::size_t bytes)
{
  assert(bytes > 0 && "Capacity per NUMA node must be positive");
  capacity_per_numa_node_ = bytes;
  return *this;
}

builder_reference& reservation_manager_configurator::set_reservation_limit_ratio_per_numa_node(
  double fraction)
{
  assert(fraction > 0.0 && fraction <= 1.0 && "Reservation limit ratio must be in (0.0, 1.0]");
  cpu_reservation_limit_ratio_ = fraction;
  return *this;
}

builder_reference& reservation_manager_configurator::set_gpu_memory_resource_factory(
  DeviceMemoryResourceFactoryFn mr_fn)
{
  assert(mr_fn && "GPU memory resource factory cannot be nullptr");
  gpu_mr_fn_ = std::move(mr_fn);
  return *this;
}

builder_reference& reservation_manager_configurator::set_cpu_memory_resource_factory(
  DeviceMemoryResourceFactoryFn mr_fn)
{
  assert(mr_fn && "CPU memory resource factory cannot be nullptr");
  cpu_mr_fn_ = std::move(mr_fn);
  return *this;
}

builder_reference& reservation_manager_configurator::set_numa_ids(const std::vector<int>& numa_ids)
{
  assert(!numa_ids.empty() && "NUMA IDs list cannot be empty");
  auto_binding_or_numa_ids_ = std::move(numa_ids);
  return *this;
}

builder_reference& reservation_manager_configurator::bind_cpu_tier_to_gpus()
{
  auto_binding_or_numa_ids_ = std::monostate{};
  return *this;
}

builder_reference& reservation_manager_configurator::set_host_id_to_numa_maps(
  const std::unordered_map<int, int>& host_to_numa_maps)
{
  auto_binding_or_numa_ids_ = host_to_numa_maps;
  return *this;
}

builder_reference& reservation_manager_configurator::ignore_topology()
{
  ignore_topology_ = true;
  return *this;
}

std::vector<memory_space_config> reservation_manager_configurator::build(
  const system_topology_info& topology) const
{
  return build(ignore_topology_ ? nullptr : &topology);
}

std::vector<memory_space_config> reservation_manager_configurator::build_with_topology() const
{
  topology_discovery discovery;
  [[maybe_unused]] bool status = discovery.discover();
  assert(status);
  auto& topology = discovery.get_topology();
  return build(&topology);
}

std::vector<memory_space_config> reservation_manager_configurator::build() const
{
  return build(nullptr);
}

std::vector<memory_space_config> reservation_manager_configurator::build(
  const system_topology_info* topology) const
{
  auto [gpu_ids, tier_ids]  = extract_gpu_ids(topology);
  auto gpu_memory_threshold = extract_gpu_memory_thresholds(gpu_ids, topology);

  std::vector<memory_space_config> configs;
  for (std::size_t index = 0; index < gpu_ids.size(); ++index) {
    int tier_id = tier_ids[index];
    configs.emplace_back(Tier::GPU,
                         tier_id,
                         gpu_memory_threshold[index].first,
                         gpu_memory_threshold[index].second,
                         gpu_mr_fn_);
  };

  std::vector<int> host_numa_ids = extract_host_ids(gpu_ids, topology);
  for (int numa_id : host_numa_ids) {
    configs.emplace_back(
      Tier::HOST,
      numa_id,
      static_cast<std::size_t>(capacity_per_numa_node_ * cpu_reservation_limit_ratio_),
      capacity_per_numa_node_,
      cpu_mr_fn_);
  }

  return configs;
}

std::pair<std::vector<int>, std::vector<int>> reservation_manager_configurator::extract_gpu_ids(
  const system_topology_info* topology) const
{
  std::vector<int> gpu_ids;
  if (std::holds_alternative<std::size_t>(n_gpus_or_gpu_ids_)) {
    std::size_t n_gpus = std::get<std::size_t>(n_gpus_or_gpu_ids_);
    assert((n_gpus <= topology->num_gpus) && "Requested number of GPUs exceeds available GPUs");
    gpu_ids.resize(n_gpus);
    for (int gpu_id = 0; gpu_id != n_gpus; ++gpu_id) {
      gpu_ids.push_back(gpu_id);
    }
    return {gpu_ids, gpu_ids};
  } else if (std::holds_alternative<std::unordered_map<int, int>>(n_gpus_or_gpu_ids_)) {
    const auto& device_to_gpu_map = std::get<std::unordered_map<int, int>>(n_gpus_or_gpu_ids_);
    std::vector<int> tier_ids;
    for (const auto& [device_tier_id, gpu_id] : device_to_gpu_map) {
      assert(gpu_id >= 0 && (gpu_id < static_cast<int>(topology->num_gpus)) &&
             "GPU ID out of range");

      tier_ids.push_back(device_tier_id);
      gpu_ids.push_back(gpu_id);
      gpu_mr_fn_ = [device_to_gpu_map, current_mr_fn = gpu_mr_fn_](
                     int tier_id,
                     size_t limit,
                     size_t capacity) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        auto it = device_to_gpu_map.find(tier_id);
        if (it == device_to_gpu_map.end()) {
          throw std::runtime_error("GPU ID not found in device to GPU map");
        }
        int gpu_id = it->second;
        return current_mr_fn(gpu_id, limit, capacity);
      };
    };
    return {gpu_ids, tier_ids};
  } else {
    gpu_ids = std::get<std::vector<int>>(n_gpus_or_gpu_ids_);
    for (int gpu_id : gpu_ids) {
      assert(gpu_id >= 0 && (gpu_id < static_cast<int>(topology->num_gpus)) &&
             "GPU ID out of range");
    }
    return {gpu_ids, gpu_ids};
  }
}

std::vector<std::pair<size_t, size_t>>
reservation_manager_configurator::extract_gpu_memory_thresholds(
  const std::vector<int>& gpu_ids, const system_topology_info* topology) const
{
  std::vector<std::pair<size_t, size_t>> ts;
  for (int gpu_id : gpu_ids) {
    rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{gpu_id});
    auto const [free, total] = rmm::available_device_memory();
    std::size_t capacity     = free;
    if (std::holds_alternative<std::size_t>(gpu_usage_limit_or_ratio_)) {
      capacity = std::get<std::size_t>(gpu_usage_limit_or_ratio_);
      assert(capacity <= total && "GPU usage limit cannot exceed total device memory");
    } else {
      capacity = std::get<double>(gpu_usage_limit_or_ratio_) * total;
    }
    ts.emplace_back(capacity * gpu_reservation_limit_ratio_, capacity);
  };
  return ts;
}

std::vector<int> reservation_manager_configurator::extract_host_ids(
  const std::vector<int>& gpu_ids, const system_topology_info* topology) const
{
  std::vector<int> host_numa_ids;
  if (std::holds_alternative<std::monostate>(auto_binding_or_numa_ids_)) {
    assert(topology != nullptr && "Topology must be provided when auto-binding to NUMA nodes");
    std::set<int> gpu_numa_ids;
    for (int gpu_id : gpu_ids) {
      const auto& gpu_info = topology->gpus[gpu_id];
      gpu_numa_ids.insert(gpu_info.numa_node);
    }
    host_numa_ids.insert(host_numa_ids.end(), gpu_numa_ids.begin(), gpu_numa_ids.end());
  } else if (std::holds_alternative<std::vector<int>>(auto_binding_or_numa_ids_)) {
    host_numa_ids = std::get<std::vector<int>>(auto_binding_or_numa_ids_);
  } else {
    const auto& host_to_numa_maps =
      std::get<std::unordered_map<int, int>>(auto_binding_or_numa_ids_);
    for (const auto& [host_id, numa_id] : host_to_numa_maps) {
      host_numa_ids.push_back(host_id);
    }
    auto current_mr_fn = cpu_mr_fn_;
    cpu_mr_fn_ = [host_to_numa_maps, current_mr_fn](int host_id, size_t limit, size_t capacity)
      -> std::unique_ptr<rmm::mr::device_memory_resource> {
      auto it = host_to_numa_maps.find(host_id);
      if (it != host_to_numa_maps.end()) {
        int numa_id = it->second;
        return current_mr_fn(numa_id, limit, capacity);
      } else {
        throw std::runtime_error("NUMA ID not found in host to NUMA maps");
      }
    };
  }
  return host_numa_ids;
}

}  // namespace memory
}  // namespace cucascade
