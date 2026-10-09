// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sirius_config.hpp"

#include <stdexcept>
#include <utility>

namespace sirius {
enum class configuration_load_error_code { io, malformed_yaml, invalid_configuration };

struct configuration_load_error : std::exception {
  configuration_load_error(configuration_load_error_code code,
                           const char* text,
                           const std::filesystem::path& path = {}) noexcept
    : code(code)
  {
    try {
      message = path.empty() ? std::string(text) : path.string() + ": " + text;
    } catch (...) {
      // The error code remains usable when a diagnostic cannot be allocated.
    }
  }
  const char* what() const noexcept override { return message.c_str(); }
  configuration_load_error_code code;
  std::string message;
};
parsed_sirius_config load_configuration(const std::filesystem::path& path);
}  // namespace sirius
