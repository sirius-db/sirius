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

namespace sirius::config {

/// `sirius.memory.gpu`: every selected GPU's memory space.
struct gpu_memory_options : block_base<gpu_memory_options> {
  /// The space's capacity: absolute bytes or a fraction of the GPU's memory.
  config_option<bytes, double> usage_limit = option<bytes>("usage_limit_bytes")
                                               .or_type<double>("usage_limit_fraction")
                                               .validate(fraction{})
                                               .fallback(0.95)
                                               .build();

  /// How much of the capacity reservations may take: bytes or a fraction of it.
  config_option<bytes, double> reservation_limit = option<bytes>("reservation_limit_bytes")
                                                     .or_type<double>("reservation_limit_fraction")
                                                     .validate(positive_fraction{})
                                                     .fallback(1.0)
                                                     .build();

  config_option<double> downgrade_trigger_fraction =
    option<double>("downgrade_trigger_fraction").validate(fraction{}).fallback(0.8).build();
  config_option<double> downgrade_stop_fraction =
    option<double>("downgrade_stop_fraction").validate(fraction{}).fallback(0.6).build();

  auto fields()
  {
    return field_list(
      usage_limit, reservation_limit, downgrade_trigger_fraction, downgrade_stop_fraction);
  }

  void validate(std::string_view path) const
  {
    check_downgrade_fractions(
      path, downgrade_trigger_fraction.get(), downgrade_stop_fraction.get());
  }
};

/// `sirius.memory.host`: one pinned host memory space per NUMA node.
struct host_memory_options : block_base<host_memory_options> {
  /// Each space's capacity: absolute bytes, or a fraction of its NUMA node's memory.
  config_option<bytes, double> capacity = option<bytes>("capacity_bytes")
                                            .validate(positive<bytes>{})
                                            .or_type<double>("capacity_fraction")
                                            .validate(positive_fraction{})
                                            .fallback(0.9)
                                            .build();

  config_option<bytes, double> reservation_limit = option<bytes>("reservation_limit_bytes")
                                                     .or_type<double>("reservation_limit_fraction")
                                                     .validate(positive_fraction{})
                                                     .fallback(1.0)
                                                     .build();

  config_option<double> downgrade_trigger_fraction =
    option<double>("downgrade_trigger_fraction").validate(fraction{}).fallback(0.9).build();
  config_option<double> downgrade_stop_fraction =
    option<double>("downgrade_stop_fraction").validate(fraction{}).fallback(0.8).build();

  config_option<bytes> block_size = option<bytes>("block_size")
                                      .validate(positive<bytes>{})
                                      .fallback(to_bytes(cucascade::memory::default_block_size))
                                      .build();
  config_option<std::size_t> pool_size = option<std::size_t>("pool_size")
                                           .validate(positive<std::size_t>{})
                                           .fallback(cucascade::memory::default_pool_size)
                                           .build();
  /// Unset: sized from the NUMA node's memory when the configuration is resolved.
  config_option<std::size_t> initial_number_pools =
    option<std::size_t>("initial_number_pools").build();

  auto fields()
  {
    return field_list(capacity,
                      reservation_limit,
                      downgrade_trigger_fraction,
                      downgrade_stop_fraction,
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

/// `sirius.memory.disk`: the disk tier batches spill to. Off unless downgrade_root_dirs is set and
/// capacity_bytes is nonzero.
struct disk_memory_options : block_base<disk_memory_options> {
  config_option<int> disk_id = option<int>("disk_id").fallback(0).build();
  config_option<bytes> capacity_bytes =
    option<bytes>("capacity_bytes").fallback(tebibytes(1)).build();
  /// The spill directory; empty turns the disk tier off.
  config_option<std::string> downgrade_root_dirs =
    option<std::string>("downgrade_root_dirs").fallback(std::string{}).build();

  auto fields() { return field_list(disk_id, capacity_bytes, downgrade_root_dirs); }
};

/// `sirius.memory`: memory spaces described per tier. Mutually exclusive with non-empty
/// sirius.space lists; a tier block counts as used when present, even empty.
struct memory_options : block_base<memory_options> {
  gpu_memory_options gpu;
  host_memory_options host;
  disk_memory_options disk;

  auto fields() { return field_list(sub("gpu", gpu), sub("host", host), sub("disk", disk)); }

  /// Whether any tier block is present.
  [[nodiscard]] bool configured() const noexcept
  {
    return gpu.user_defined() || host.user_defined() || disk.user_defined();
  }
};

}  // namespace sirius::config
