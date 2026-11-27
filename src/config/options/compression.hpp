// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/block.hpp"
#include "config/option.hpp"
#include "config/size.hpp"
#include "config/validators.hpp"
#include "sirius_config.hpp"

#include <string>

namespace sirius::config {

namespace detail {
inline const compression_config& compression_defaults()
{
  static const compression_config value{};
  return value;
}
}  // namespace detail

/// `sirius.compression`: Simpatico compression of pinned tables (per connection) and of spilled
/// batches (process-wide).
struct compression_options : block_base<compression_options> {
  static const compression_config& defaults() { return detail::compression_defaults(); }

  config_option<bool> enable_pin_table_compression =
    option<bool>("enable_pin_table_compression")
      .fallback(defaults().enable_pin_table_compression)
      .describe("whether pin_table compresses by default")
      .build();
  /// Smaller chunks stay uncompressed; 0 = no threshold.
  config_option<ubytes> min_batch_size_bytes =
    option<ubytes>("min_batch_size_bytes")
      .fallback(ubytes(defaults().min_batch_size_bytes))
      .describe("smallest pinned batch worth compressing")
      .build();
  /// Keep the compressed form only at or below this share of the original size.
  config_option<double> max_compressed_fraction =
    option<double>("max_compressed_fraction")
      .validate(non_negative_finite{})
      .fallback(defaults().max_compressed_fraction)
      .describe("largest compressed/original size ratio kept when pinning")
      .build();
  /// Per-table plan files; empty selects plans automatically.
  config_option<std::string> input_plan_dir =
    option<std::string>("input_plan_dir")
      .fallback(defaults().input_plan_dir)
      .describe("directory of per-table compression plan overrides")
      .build();
  /// Smaller GPU-tier pins stay uncompressed: the share of the pinned GPUs' memory a pin must
  /// reach before plans are selected automatically; 0 disables the gate.
  config_option<double> min_table_device_fraction =
    option<double>("min_table_device_fraction")
      .validate(non_negative_finite{})
      .fallback(defaults().min_table_device_fraction)
      .describe("smallest GPU-tier pin, as a share of GPU memory, to compress")
      .build();
  /// SET spill_compression changes it for the whole process, so it is session-scoped.
  config_option<bool> enable_spill_compression =
    option<bool>("enable_spill_compression")
      .fallback(defaults().enable_spill_compression)
      .mark_production()
      .describe("compress batches spilled to host or disk")
      .build();
  config_option<ubytes> spill_min_batch_bytes =
    option<ubytes>("spill_min_batch_bytes")
      .fallback(ubytes(defaults().spill_min_batch_bytes))
      .describe("smallest spilled batch worth compressing")
      .build();
  config_option<double> spill_max_compressed_fraction =
    option<double>("spill_max_compressed_fraction")
      .validate(non_negative_finite{})
      .fallback(defaults().spill_max_compressed_fraction)
      .describe("largest compressed/original size ratio kept for spills")
      .build();
  /// 0 skips the headroom check.
  config_option<double> spill_encode_reserve_fraction =
    option<double>("spill_encode_reserve_fraction")
      .validate(fraction{})
      .fallback(defaults().spill_encode_reserve_fraction)
      .describe("GPU headroom, as a share of the batch, needed to encode a spill")
      .build();

  auto fields()
  {
    return field_list(enable_pin_table_compression,
                      min_batch_size_bytes,
                      max_compressed_fraction,
                      input_plan_dir,
                      min_table_device_fraction,
                      enable_spill_compression,
                      spill_min_batch_bytes,
                      spill_max_compressed_fraction,
                      spill_encode_reserve_fraction);
  }
};

}  // namespace sirius::config
