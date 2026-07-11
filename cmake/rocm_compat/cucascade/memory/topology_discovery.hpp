/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade topology_discovery — ROCm stub.

#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace cucascade::memory {

enum class StorageDriveType { NVME, SATA_SSD, HDD, UNKNOWN, SIZE };
enum class NetworkDeviceVerification { EXISTS_ACTIVE_IP, EXISTS, NONE, SIZE };
enum class PciePathType { DIRECT, MULTIHOP, UNKNOWN, SIZE };

struct gpu_topology_info {
  int32_t device_id{0};
  std::size_t total_memory{0};
  std::vector<int32_t> peer_devices;
};

struct network_device_info {
  std::string name;
  std::string ip_address;
  bool is_active{false};
};

struct storage_device_info {
  std::string mount_point;
  StorageDriveType drive_type{StorageDriveType::UNKNOWN};
  std::size_t capacity{0};
};

struct system_topology_info {
  std::vector<gpu_topology_info> gpus;
  std::vector<network_device_info> network_devices;
  std::vector<storage_device_info> storage_devices;
};

class topology_discovery {
 public:
  bool discover(NetworkDeviceVerification /*verify*/ = NetworkDeviceVerification::EXISTS_ACTIVE_IP) {
    return false; // stub: no topology discovery
  }
  bool is_discovered() const { return false; }
  system_topology_info const& get_topology() const { return topology_; }
 private:
  system_topology_info topology_;
};

}  // namespace cucascade::memory
