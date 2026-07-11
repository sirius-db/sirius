/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade reservation_manager_configurator — ROCm stub (builder pattern).
//! Method arities match what src/sirius_config.cpp actually calls.

#pragma once
#include "cucascade/memory/common.hpp"
#include "cucascade/memory/topology_discovery.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace cucascade::memory {

class reservation_manager_configurator {
 public:
  reservation_manager_configurator& set_number_of_gpus(std::size_t) { return *this; }
  reservation_manager_configurator& set_gpu_ids(std::vector<int32_t> const&) { return *this; }
  reservation_manager_configurator& set_gpu_usage_limit(std::size_t) { return *this; }
  reservation_manager_configurator& set_usage_limit_ratio_per_gpu(double) { return *this; }
  reservation_manager_configurator& set_reservation_limit_per_gpu(std::size_t) { return *this; }
  reservation_manager_configurator& set_reservation_fraction_per_gpu(double) { return *this; }
  // Sirius calls with 2 doubles (trigger_fraction, stop_fraction)
  reservation_manager_configurator& set_downgrade_fractions_per_gpu(double, double) { return *this; }
  reservation_manager_configurator& set_reservation_limit_per_host(std::size_t) { return *this; }
  reservation_manager_configurator& set_reservation_fraction_per_host(double) { return *this; }
  // Sirius calls with 2 doubles (trigger_fraction, stop_fraction)
  reservation_manager_configurator& set_downgrade_fractions_per_host(double, double) { return *this; }
  reservation_manager_configurator& set_total_host_capacity(std::size_t) { return *this; }
  reservation_manager_configurator& set_per_host_capacity(std::size_t) { return *this; }
  reservation_manager_configurator& use_host_per_gpu() { return *this; }
  reservation_manager_configurator& use_host_per_numa() { return *this; }
  reservation_manager_configurator& set_host_portability(bool) { return *this; }
  // Sirius calls with 3 size_t (block_size, pool_size, initial_number_pools)
  reservation_manager_configurator& set_host_pool_features(std::size_t, std::size_t, std::size_t) { return *this; }
  // Sirius calls with (int id, size_t capacity, string root)
  reservation_manager_configurator& set_disk_mounting_point(int, std::size_t, std::string const&) { return *this; }
  reservation_manager_configurator& track_reservation_per_stream(bool) { return *this; }
  reservation_manager_configurator& set_gpu_memory_resource_factory(DeviceMemoryResourceFactoryFn) { return *this; }
  reservation_manager_configurator& set_host_memory_resource_factory(DeviceMemoryResourceFactoryFn) { return *this; }

  std::vector<memory_space_config> build(system_topology_info const& = {}) { return {}; }
  std::vector<memory_space_config> build() { return {}; }
};

}  // namespace cucascade::memory
