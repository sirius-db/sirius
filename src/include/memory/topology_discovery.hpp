/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace cucascade::memory {

/**
 * @brief GPU information.
 */
struct gpu_topology_info {
  unsigned int id{0};                        ///< GPU device ID.
  std::string name;                          ///< GPU device name.
  std::string pci_bus_id;                    ///< PCI bus ID.
  std::string uuid;                          ///< GPU UUID.
  int numa_node{-1};                         ///< NUMA node ID (-1 if unknown).
  std::string cpu_affinity_list;             ///< CPU affinity list.
  std::vector<int> cpu_cores;                ///< List of CPU core IDs.
  std::vector<int> memory_binding;           ///< NUMA nodes for memory binding.
  std::vector<std::string> network_devices;  ///< Network devices (NICs) optimal for this GPU.
};

/**
 * @brief Network device information.
 */
struct network_device_info {
  std::string name;        ///< Device name (e.g., "mlx5_0").
  int numa_node;           ///< NUMA node ID (-1 if unknown).
  std::string pci_bus_id;  ///< PCI bus ID.
};

enum class StorageDriveType {
  NVME,      // NVMe SSD
  SATA_SSD,  // SATA Solid State Drive
  SATA_HDD,  // SATA Hard Disk Drive
  UNKNOWN
};

struct storage_device_info {
  StorageDriveType type = StorageDriveType::UNKNOWN;  ///< Type of storage drive.
  std::string name;                                   ///< Device name (e.g., "nvme0n1").
  int numa_node{-1};                                  ///< NUMA node ID (-1 if unknown).
  std::string pci_bus_id;                             ///< PCI bus ID.
};

/**
 * @brief System topology information.
 */
struct system_topology_info {
  std::string hostname;                              ///< System hostname.
  unsigned int num_gpus;                             ///< Total number of GPUs.
  int num_numa_nodes;                                ///< Total number of NUMA nodes.
  int num_network_devices;                           ///< Total number of network devices.
  std::vector<gpu_topology_info> gpus;               ///< GPU topology information.
  std::vector<network_device_info> network_devices;  ///< Network device information.
  std::vector<storage_device_info> storage_devices;  ///< Storage device information.
};

/**
 * @brief PCIe topology path types.
 */
enum class PciePathType {
  PIX  = 0,  ///< Connection traversing at most a single PCIe bridge (best).
  PXB  = 1,  ///< Connection traversing multiple PCIe bridges.
  PHB  = 2,  ///< Connection traversing PCIe Host Bridge.
  NODE = 3,  ///< Connection traversing PCIe and interconnect within NUMA node.
  SYS  = 4   ///< Connection traversing NUMA interconnect (worst).
};

/**
 * @brief Discover system topology including GPUs, NUMA nodes, and network devices.
 *
 * This class provides methods to discover system topology information using NVML
 * and /sys filesystem queries. It dynamically identifies GPU-to-NUMA-to-NIC mappings
 * based on PCIe topology.
 *
 * Example usage:
 * @code
 * cucascade::memory:topology_discovery discovery;
 * if (discovery.discover()) {
 *     auto topology = discovery.get_topology();
 * }
 * @endcode
 */
class topology_discovery {
 public:
  /**
   * @brief Discover system topology.
   *
   * This method performs the actual discovery of GPUs, NUMA nodes, CPU affinity,
   * and network devices. It must be called before `get_topology()`.
   *
   * @return true if discovery was successful, false otherwise.
   */
  [[nodiscard]] bool discover();

  /**
   * @brief Get the discovered topology information.
   *
   * @return system_topology_info structure containing all topology data.
   * @note `discover()` must be called first.
   */
  [[nodiscard]] system_topology_info const& get_topology() const { return topology_.value(); }

  /**
   * @brief Check if topology has been discovered.
   *
   * @return true if `discover()` has been called successfully.
   */
  [[nodiscard]] bool is_discovered() const { return topology_.has_value(); }

 private:
  std::optional<system_topology_info> topology_;  ///< Discovered topology information.
};

}  // namespace cucascade::memory
