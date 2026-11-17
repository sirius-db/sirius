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
      .scope(option_scope::query)
      .build();
  /// Smaller chunks stay uncompressed; 0 = no threshold.
  config_option<bytes> min_batch_size_bytes = option<bytes>("min_batch_size_bytes")
                                                .fallback(to_bytes(defaults().min_batch_size_bytes))
                                                .scope(option_scope::query)
                                                .build();
  /// Keep the compressed form only at or below this share of the original size.
  config_option<double> max_compressed_fraction = option<double>("max_compressed_fraction")
                                                    .validate(non_negative_finite{})
                                                    .fallback(defaults().max_compressed_fraction)
                                                    .scope(option_scope::query)
                                                    .build();
  /// Per-table plan files; empty selects plans automatically.
  config_option<std::string> input_plan_dir = option<std::string>("input_plan_dir")
                                                .fallback(defaults().input_plan_dir)
                                                .scope(option_scope::query)
                                                .build();
  /// SET spill_compression changes it for the whole process, so it is session-scoped.
  config_option<bool> enable_spill_compression =
    option<bool>("enable_spill_compression").fallback(defaults().enable_spill_compression).build();
  config_option<bytes> spill_min_batch_bytes =
    option<bytes>("spill_min_batch_bytes")
      .fallback(to_bytes(defaults().spill_min_batch_bytes))
      .build();
  config_option<double> spill_max_compressed_fraction =
    option<double>("spill_max_compressed_fraction")
      .validate(non_negative_finite{})
      .fallback(defaults().spill_max_compressed_fraction)
      .build();
  /// 0 skips the headroom check.
  config_option<double> spill_encode_reserve_fraction =
    option<double>("spill_encode_reserve_fraction")
      .validate(fraction{})
      .fallback(defaults().spill_encode_reserve_fraction)
      .build();

  auto fields()
  {
    return field_list(enable_pin_table_compression,
                      min_batch_size_bytes,
                      max_compressed_fraction,
                      input_plan_dir,
                      enable_spill_compression,
                      spill_min_batch_bytes,
                      spill_max_compressed_fraction,
                      spill_encode_reserve_fraction);
  }
};

}  // namespace sirius::config
