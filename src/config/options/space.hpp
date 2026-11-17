// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/block.hpp"
#include "config/option.hpp"
#include "config/size.hpp"
#include "config/validators.hpp"

#include <cucascade/memory/config.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::config {

namespace detail {
inline const cucascade::memory::gpu_memory_space_config& gpu_space_defaults()
{
  static const cucascade::memory::gpu_memory_space_config value{};
  return value;
}
inline const cucascade::memory::host_memory_space_config& host_space_defaults()
{
  static const cucascade::memory::host_memory_space_config value{};
  return value;
}
inline const cucascade::memory::disk_memory_space_config& disk_space_defaults()
{
  static const cucascade::memory::disk_memory_space_config value{};
  return value;
}
}  // namespace detail

/// One `sirius.space.gpu[]` entry: a GPU memory space, described directly.
struct gpu_space_options : block_base<gpu_space_options> {
  config_option<int> device_id =
    option<int>("device_id").fallback(detail::gpu_space_defaults().device_id).build();
  /// Sirius turns per-stream reservation tracking off unless asked.
  config_option<bool> per_stream_reservation =
    option<bool>("per_stream_reservation").fallback(false).build();
  config_option<double> reservation_limit_fraction =
    option<double>("reservation_limit_fraction")
      .validate(fraction{})
      .fallback(detail::gpu_space_defaults().reservation_limit_fraction)
      .build();
  config_option<double> downgrade_trigger_fraction =
    option<double>("downgrade_trigger_fraction")
      .validate(fraction{})
      .fallback(detail::gpu_space_defaults().downgrade_trigger_fraction)
      .build();
  config_option<double> downgrade_stop_fraction =
    option<double>("downgrade_stop_fraction")
      .validate(fraction{})
      .fallback(detail::gpu_space_defaults().downgrade_stop_fraction)
      .build();
  /// 0 = no capacity configured (cuCascade's default).
  config_option<ubytes> memory_capacity =
    option<ubytes>("memory_capacity")
      .fallback(ubytes(detail::gpu_space_defaults().memory_capacity))
      .build();

  auto fields()
  {
    return field_list(device_id,
                      per_stream_reservation,
                      reservation_limit_fraction,
                      downgrade_trigger_fraction,
                      downgrade_stop_fraction,
                      memory_capacity);
  }

  void validate(std::string_view path) const
  {
    check_downgrade_fractions(
      path, downgrade_trigger_fraction.get(), downgrade_stop_fraction.get());
  }
};

/// One `sirius.space.host[]` entry: a pinned host memory space, described directly.
struct host_space_options : block_base<host_space_options> {
  config_option<int> numa_id =
    option<int>("numa_id").fallback(detail::host_space_defaults().numa_id).build();
  config_option<double> reservation_limit_fraction =
    option<double>("reservation_limit_fraction")
      .validate(fraction{})
      .fallback(detail::host_space_defaults().reservation_limit_fraction)
      .build();
  config_option<double> downgrade_trigger_fraction =
    option<double>("downgrade_trigger_fraction")
      .validate(fraction{})
      .fallback(detail::host_space_defaults().downgrade_trigger_fraction)
      .build();
  config_option<double> downgrade_stop_fraction =
    option<double>("downgrade_stop_fraction")
      .validate(fraction{})
      .fallback(detail::host_space_defaults().downgrade_stop_fraction)
      .build();
  config_option<ubytes> memory_capacity =
    option<ubytes>("memory_capacity")
      .fallback(ubytes(detail::host_space_defaults().memory_capacity))
      .build();
  config_option<ubytes> block_size =
    option<ubytes>("block_size").fallback(ubytes(detail::host_space_defaults().block_size)).build();
  config_option<std::size_t> pool_size =
    option<std::size_t>("pool_size").fallback(detail::host_space_defaults().pool_size).build();
  config_option<std::size_t> initial_number_pools =
    option<std::size_t>("initial_number_pools")
      .fallback(detail::host_space_defaults().initial_number_pools)
      .build();

  auto fields()
  {
    return field_list(numa_id,
                      reservation_limit_fraction,
                      downgrade_trigger_fraction,
                      downgrade_stop_fraction,
                      memory_capacity,
                      block_size,
                      pool_size,
                      initial_number_pools);
  }

  void validate(std::string_view path) const
  {
    check_downgrade_fractions(
      path, downgrade_trigger_fraction.get(), downgrade_stop_fraction.get());
  }
};

/// One `sirius.space.disk[]` entry: a disk memory space, described directly.
struct disk_space_options : block_base<disk_space_options> {
  config_option<int> disk_id =
    option<int>("disk_id").fallback(detail::disk_space_defaults().disk_id).build();
  config_option<std::string> mount_path =
    option<std::string>("mount_path").fallback(detail::disk_space_defaults().mount_paths).build();
  config_option<ubytes> memory_capacity =
    option<ubytes>("memory_capacity")
      .fallback(ubytes(detail::disk_space_defaults().memory_capacity))
      .build();

  auto fields() { return field_list(disk_id, mount_path, memory_capacity); }
};

/// `sirius.space`: memory spaces listed one by one (the low-level alternative to sirius.memory).
/// Any non-empty list replaces the per-tier description and the topology's GPU selection.
struct space_options : block_base<space_options> {
  std::vector<gpu_space_options> gpu;
  std::vector<host_space_options> host;
  std::vector<disk_space_options> disk;

  auto fields() { return field_list(list("gpu", gpu), list("host", host), list("disk", disk)); }

  /// Whether any list has an entry.
  [[nodiscard]] bool configured() const noexcept
  {
    return !gpu.empty() || !host.empty() || !disk.empty();
  }

  /// Whether every listed GPU space names a nonzero capacity (and at least one is listed).
  [[nodiscard]] bool gpu_capacities_configured() const
  {
    if (gpu.empty()) { return false; }
    for (auto const& space : gpu) {
      if (!(space.memory_capacity.get() > ubytes::zero())) { return false; }
    }
    return true;
  }
};

}  // namespace sirius::config
