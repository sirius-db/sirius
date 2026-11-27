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
  config_option<ubytes, double> usage_limit =
    option<ubytes>("usage_limit_bytes")
      .or_type<double>("usage_limit_fraction")
      .validate(fraction{})
      .fallback(0.95)
      .mark_production()
      .describe("GPU memory capacity Sirius may use on each GPU")
      .build();

  /// How much of the capacity reservations may take: bytes or a fraction of it.
  config_option<ubytes, double> reservation_limit =
    option<ubytes>("reservation_limit_bytes")
      .or_type<double>("reservation_limit_fraction")
      .validate(positive_fraction{})
      .fallback(1.0)
      .mark_production()
      .describe("part of the GPU capacity that reservations may take")
      .build();

  config_option<double> downgrade_trigger_fraction =
    option<double>("downgrade_trigger_fraction")
      .validate(fraction{})
      .fallback(0.8)
      .mark_production()
      .describe("GPU memory usage that starts spilling to host")
      .build();
  config_option<double> downgrade_stop_fraction =
    option<double>("downgrade_stop_fraction")
      .validate(fraction{})
      .fallback(0.6)
      .mark_production()
      .describe("GPU memory usage at which spilling stops")
      .build();

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
  config_option<ubytes, double> capacity = option<ubytes>("capacity_bytes")
                                             .validate(positive<ubytes>{})
                                             .or_type<double>("capacity_fraction")
                                             .validate(positive_fraction{})
                                             .fallback(0.9)
                                             .mark_production()
                                             .describe("pinned host memory capacity per NUMA node")
                                             .build();

  config_option<ubytes, double> reservation_limit =
    option<ubytes>("reservation_limit_bytes")
      .or_type<double>("reservation_limit_fraction")
      .validate(positive_fraction{})
      .fallback(1.0)
      .mark_production()
      .describe("part of the host capacity that reservations may take")
      .build();

  config_option<double> downgrade_trigger_fraction =
    option<double>("downgrade_trigger_fraction")
      .validate(fraction{})
      .fallback(0.9)
      .mark_production()
      .describe("host memory usage that starts reclamation")
      .build();
  config_option<double> downgrade_stop_fraction =
    option<double>("downgrade_stop_fraction")
      .validate(fraction{})
      .fallback(0.8)
      .mark_production()
      .describe("host memory usage at which reclamation stops")
      .build();

  config_option<ubytes> block_size = option<ubytes>("block_size")
                                       .validate(positive<ubytes>{})
                                       .fallback(ubytes(cucascade::memory::default_block_size))
                                       .mark_production()
                                       .describe("allocation block size of the pinned host pools")
                                       .build();
  config_option<std::size_t> pool_size = option<std::size_t>("pool_size")
                                           .validate(positive<std::size_t>{})
                                           .fallback(cucascade::memory::default_pool_size)
                                           .mark_production()
                                           .describe("allocation blocks per pinned host pool")
                                           .build();
  /// Unset: sized from the NUMA node's memory when the configuration is resolved.
  config_option<std::size_t> initial_number_pools =
    option<std::size_t>("initial_number_pools")
      .mark_production()
      .describe("pinned host pools pre-allocated at startup")
      .build();

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
  config_option<int> disk_id = option<int>("disk_id")
                                 .fallback(0)
                                 .mark_production()
                                 .describe("identifier of the disk spill space")
                                 .build();
  config_option<ubytes> capacity_bytes = option<ubytes>("capacity_bytes")
                                           .fallback(tebibytes(1))
                                           .mark_production()
                                           .describe("maximum disk space for spill files")
                                           .build();
  /// The spill directory; empty turns the disk tier off.
  config_option<std::string> downgrade_root_dirs = option<std::string>("downgrade_root_dirs")
                                                     .fallback(std::string{})
                                                     .mark_production()
                                                     .describe("directory for disk spill files")
                                                     .build();

  auto fields() { return field_list(disk_id, capacity_bytes, downgrade_root_dirs); }
};

/// `sirius.memory`: memory spaces described per tier. Mutually exclusive with non-empty
/// sirius.space lists; a tier block counts as used when present, even empty.
struct memory_options : block_base<memory_options> {
  gpu_memory_options gpu;
  host_memory_options host;
  disk_memory_options disk;

  auto fields() { return field_list(sub("gpu", gpu), sub("host", host), sub("disk", disk)); }

  /// Whether any tier is in use: present in the file (even empty) or holding a user value.
  [[nodiscard]] bool configured() const
  {
    return gpu.has_user_values() || host.has_user_values() || disk.has_user_values();
  }
};

}  // namespace sirius::config
