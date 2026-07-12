/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade reservation_manager_configurator — ROCm real builder.
//! Stores all configuration values and emits memory_space_config vector.

#pragma once
#include "cucascade/memory/common.hpp"
#include "cucascade/memory/topology_discovery.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace cucascade::memory {

class reservation_manager_configurator {
 public:
  reservation_manager_configurator& set_number_of_gpus(std::size_t n) { num_gpus_ = n; return *this; }
  reservation_manager_configurator& set_gpu_ids(std::vector<int32_t> const& ids) { gpu_ids_ = ids; return *this; }
  reservation_manager_configurator& set_gpu_usage_limit(std::size_t limit) { gpu_usage_limit_ = limit; return *this; }
  reservation_manager_configurator& set_usage_limit_ratio_per_gpu(double ratio) { gpu_usage_ratio_ = ratio; return *this; }
  reservation_manager_configurator& set_reservation_limit_per_gpu(std::size_t limit) { gpu_reservation_limit_ = limit; return *this; }
  reservation_manager_configurator& set_reservation_fraction_per_gpu(double frac) { gpu_reservation_fraction_ = frac; return *this; }
  reservation_manager_configurator& set_downgrade_fractions_per_gpu(double trigger, double stop) {
    gpu_downgrade_trigger_ = trigger; gpu_downgrade_stop_ = stop; return *this;
  }
  reservation_manager_configurator& set_reservation_limit_per_host(std::size_t limit) { host_reservation_limit_ = limit; return *this; }
  reservation_manager_configurator& set_reservation_fraction_per_host(double frac) { host_reservation_fraction_ = frac; return *this; }
  reservation_manager_configurator& set_downgrade_fractions_per_host(double trigger, double stop) {
    host_downgrade_trigger_ = trigger; host_downgrade_stop_ = stop; return *this;
  }
  reservation_manager_configurator& set_total_host_capacity(std::size_t cap) { total_host_capacity_ = cap; return *this; }
  reservation_manager_configurator& set_per_host_capacity(std::size_t cap) { per_host_capacity_ = cap; return *this; }
  reservation_manager_configurator& use_host_per_gpu() { host_per_gpu_ = true; return *this; }
  reservation_manager_configurator& use_host_per_numa() { host_per_numa_ = true; return *this; }
  reservation_manager_configurator& set_host_portability(bool port) { host_portable_ = port; return *this; }
  reservation_manager_configurator& set_host_pool_features(std::size_t block_size, std::size_t pool_size, std::size_t initial_pools) {
    host_block_size_ = block_size; host_pool_size_ = pool_size; host_initial_pools_ = initial_pools; return *this;
  }
  reservation_manager_configurator& set_disk_mounting_point(int id, std::size_t capacity, std::string const& root) {
    disk_id_ = id; disk_capacity_ = capacity; disk_root_ = root; return *this;
  }
  reservation_manager_configurator& track_reservation_per_stream(bool track) { track_per_stream_ = track; return *this; }
  reservation_manager_configurator& set_gpu_memory_resource_factory(DeviceMemoryResourceFactoryFn fn) { gpu_factory_ = fn; return *this; }
  reservation_manager_configurator& set_host_memory_resource_factory(DeviceMemoryResourceFactoryFn fn) { host_factory_ = fn; return *this; }

  std::vector<memory_space_config> build(system_topology_info const& topo = {}) {
    std::vector<memory_space_config> configs;
    std::size_t ngpus = num_gpus_ > 0 ? num_gpus_ : topo.num_gpus;
    for (std::size_t i = 0; i < ngpus; ++i) {
      gpu_memory_space_config gpu_cfg;
      gpu_cfg.device_id = static_cast<int32_t>(i);
      gpu_cfg.usage_limit = gpu_usage_limit_;
      gpu_cfg.reservation_limit = gpu_reservation_limit_;
      configs.push_back(gpu_cfg);
    }
    if (total_host_capacity_ > 0 || per_host_capacity_ > 0) {
      host_memory_space_config host_cfg;
      host_cfg.device_id = 0;
      host_cfg.capacity = total_host_capacity_ > 0 ? total_host_capacity_ : per_host_capacity_;
      host_cfg.portable = host_portable_;
      configs.push_back(host_cfg);
    }
    if (!disk_root_.empty()) {
      disk_memory_space_config disk_cfg;
      disk_cfg.mounting_point = disk_root_;
      disk_cfg.capacity = disk_capacity_;
      configs.push_back(disk_cfg);
    }
    return configs;
  }

  std::vector<memory_space_config> build() { return build({}); }

 private:
  std::size_t num_gpus_{0};
  std::vector<int32_t> gpu_ids_;
  std::size_t gpu_usage_limit_{0};
  double gpu_usage_ratio_{0};
  std::size_t gpu_reservation_limit_{0};
  double gpu_reservation_fraction_{0};
  double gpu_downgrade_trigger_{0}, gpu_downgrade_stop_{0};
  std::size_t host_reservation_limit_{0};
  double host_reservation_fraction_{0};
  double host_downgrade_trigger_{0}, host_downgrade_stop_{0};
  std::size_t total_host_capacity_{0};
  std::size_t per_host_capacity_{0};
  bool host_per_gpu_{false};
  bool host_per_numa_{false};
  bool host_portable_{false};
  std::size_t host_block_size_{default_block_size};
  std::size_t host_pool_size_{default_pool_size};
  std::size_t host_initial_pools_{default_initial_number_pools};
  int disk_id_{0};
  std::size_t disk_capacity_{0};
  std::string disk_root_;
  bool track_per_stream_{false};
  DeviceMemoryResourceFactoryFn gpu_factory_;
  DeviceMemoryResourceFactoryFn host_factory_;
};

}  // namespace cucascade::memory
