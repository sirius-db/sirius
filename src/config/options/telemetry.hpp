// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/block.hpp"
#include "config/option.hpp"
#include "config/validators.hpp"
#include "sirius_config.hpp"

#include <string>

namespace sirius::config {

/// `sirius.telemetry`: Quent telemetry. Read before the hardware is discovered.
struct telemetry_options : block_base<telemetry_options> {
  static const telemetry_config& defaults()
  {
    static const telemetry_config value{};
    return value;
  }

  /// Off in released extensions, on otherwise.
  config_option<bool> enable_quent =
    option<bool>("enable_quent").fallback(telemetry_config::default_enable_quent).build();
  /// No effect unless enable_quent.
  config_option<bool> enable_nvtx =
    option<bool>("enable_nvtx").fallback(defaults().enable_nvtx).build();
  config_option<std::string> output_directory = option<std::string>("output_directory")
                                                  .validate(non_empty{})
                                                  .fallback(defaults().output_directory)
                                                  .build();
  config_option<std::string> engine_name = option<std::string>("engine_name")
                                             .validate(non_empty{})
                                             .fallback(defaults().engine_name)
                                             .build();
  /// Empty: Sirius's own injection library. NVTX_INJECTION64_PATH in the environment wins.
  config_option<std::string> nvtx_injection_lib =
    option<std::string>("nvtx_injection_lib").fallback(defaults().nvtx_injection_lib).build();

  auto fields()
  {
    return field_list(enable_quent, enable_nvtx, output_directory, engine_name, nvtx_injection_lib);
  }
};

}  // namespace sirius::config
