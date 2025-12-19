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

#include "memory/common.hpp"
#include "memory/memory_reservation_manager.hpp"
#include "memory/memory_space.hpp"
#include "memory/topology_discovery.hpp"

#include <span>
#include <unordered_map>
#include <variant>
#include <vector>

namespace cucascade {
namespace memory {

//===----------------------------------------------------------------------===//
// memory_reservation_manager
//===----------------------------------------------------------------------===//

/**
 * @class reservation_manager_config_builder
 * @brief Builder class for configuring memory reservation manager settings.
 *
 * This class provides a fluent interface to configure GPU and CPU memory reservation
 * parameters, including device selection, memory limits, reservation ratios, and
 * memory resource factories. It supports both explicit device IDs and automatic
 * configuration based on system topology.
 *
 * Usage example:
 * @code
 * reservation_manager_config_builder builder;
 * builder.set_number_of_gpus(2)
 *        .set_gpu_usage_limit(2UL << 30)
 *        .set_reservation_limit_ratio_per_gpu(0.8)
 *        .set_numa_ids({0, 1})
 *        .set_capacity_per_numa_node(8UL << 30)
 *        .set_gpu_memory_resource_factory(custom_gpu_factory)
 *        .set_cpu_memory_resource_factory(custom_cpu_factory);
 * auto configs = builder.build(system_topology);
 * @endcode
 *
 * @note Either GPU IDs or number of GPUs must be set. Similarly, either explicit memory
 * limits or ratios can be specified for GPUs and NUMA nodes.
 */
class reservation_manager_configurator {
 public:
  using builder_reference   = reservation_manager_configurator;
  using memory_space_config = memory_reservation_manager::memory_space_config;

  /// either set gpu ids or number of gpus

  /// @brief set number of gpus
  /// @param n_gpus Number of GPUs to configure.
  builder_reference& set_number_of_gpus(std::size_t n_gpus);

  /// @brief set gpu ids
  /// @param gpu_ids Vector of GPU IDs to configure.
  builder_reference& set_gpu_ids(std::vector<int> gpu_ids);

  /// @brief set map of device tier id to gpu id
  /// @param device_to_gpu_map Vector of pairs mapping host IDs to NUMA node
  /// @note this is meant to be used for testing purpose only
  builder_reference& set_device_tier_id_to_gpu_id_map(
    const std::unordered_map<int, int>& device_to_gpu_map);

  // either set space capacity or set a ratio of gpu total capacity
  /// @brief set gpu usage limit in bytes (i.e. capacity)
  /// @param bytes Memory usage limit in bytes.
  builder_reference& set_gpu_usage_limit(std::size_t bytes);

  /// @brief set gpu usage limit as a fraction of total GPU capacity
  /// @param fraction Fraction of total GPU capacity to use.
  builder_reference& set_usage_limit_ratio_per_gpu(double fraction);

  // either set host ids or create as many host tiers as numa nodes of gpus create by this builder
  /// @brief set numa ids for cpu tiers
  /// @param numa_ids Vector of NUMA node IDs to configure.
  builder_reference& set_numa_ids(const std::vector<int>& numa_ids);

  // either set host ids or create as many host tiers as numa nodes of gpus create by this builder
  /// @brief set host id to numa maps
  /// @param host_to_numa_maps Vector of pairs mapping host IDs to NUMA node IDs.
  /// @note this is meant to be used for testing purpose only
  builder_reference& set_host_id_to_numa_maps(
    const std::unordered_map<int, int>& host_to_numa_maps);

  /// @brief automatically bind cpu tiers to gpus based on topology
  builder_reference& bind_cpu_tier_to_gpus();

  /// set capacity per host tier
  /// @param bytes Memory capacity per NUMA node in bytes.
  builder_reference& set_capacity_per_numa_node(std::size_t bytes);

  /// \brief set reservation limit ratio per GPU
  /// @param fraction Fraction of GPU memory capacity to reserve.
  builder_reference& set_reservation_limit_ratio_per_gpu(double fraction);

  /// \brief set ratio of space capacity used for reservation in cpus
  /// @param fraction Fraction of NUMA node memory capacity to reserve.
  builder_reference& set_reservation_limit_ratio_per_numa_node(double fraction);

  /// \brief set the function that takes in the device id and create gpu memory resource
  /// @param mr_fn Function to create GPU memory resource.
  builder_reference& set_gpu_memory_resource_factory(DeviceMemoryResourceFactoryFn mr_fn);

  /// \brief set the function that takes in the numa node id and create cpu memory resource
  /// @param mr_fn Function to create CPU memory resource.
  builder_reference& set_cpu_memory_resource_factory(DeviceMemoryResourceFactoryFn mr_fn);

  /// \brief don't do topology checking
  builder_reference& ignore_topology();

  /// \brief build the memory space configurations based on the provided system topology
  /// @param topology System topology information.
  /// @return Vector of memory space configurations.
  std::vector<memory_space_config> build(const system_topology_info& topology) const;

  /// \brief build the memory space configurations after discovering the system topology
  /// @return Vector of memory space configurations.
  std::vector<memory_space_config> build_with_topology() const;

  /// \brief build the memory space configurations without topology information
  /// @return Vector of memory space configurations.
  std::vector<memory_space_config> build() const;

 private:
  std::vector<memory_space_config> build(const system_topology_info* topology) const;

  std::pair<std::vector<int>, std::vector<int>> extract_gpu_ids(
    const system_topology_info* topology) const;
  std::vector<std::pair<size_t, size_t>> extract_gpu_memory_thresholds(
    const std::vector<int>& gpus_ids, const system_topology_info* topology) const;
  std::vector<int> extract_host_ids(const std::vector<int>& gpu_ids,
                                    const system_topology_info* topology) const;

  bool ignore_topology_{false};
  std::variant<std::size_t, std::vector<int>, std::unordered_map<int, int>> n_gpus_or_gpu_ids_{1UL};
  std::variant<std::size_t, double> gpu_usage_limit_or_ratio_{
    static_cast<std::size_t>(1UL << 30)};          // uses 1GB of gpu memory
  double gpu_reservation_limit_ratio_{0.75};       // limit to 75% of GPU usagel limit
  std::size_t capacity_per_numa_node_{8UL << 30};  // 8GB per NUMA node by default
  std::variant<std::monostate, std::vector<int>, std::unordered_map<int, int>>
    auto_binding_or_numa_ids_{};
  double cpu_reservation_limit_ratio_{0.75};  // 75% limit per NUMA node by default
  mutable DeviceMemoryResourceFactoryFn gpu_mr_fn_ = make_default_gpu_memory_resource;
  mutable DeviceMemoryResourceFactoryFn cpu_mr_fn_ = make_default_host_memory_resource;
};

}  // namespace memory
}  // namespace cucascade
