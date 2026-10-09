// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sirius_config.hpp"

#include <sirius/c/error.h>

#include <stdexcept>
#include <utility>

namespace sirius {
struct configuration_load_error : std::exception {
  configuration_load_error(sirius_status status,
                           const char* text,
                           const std::filesystem::path& path = {}) noexcept
    : status(status)
  {
    try {
      message = path.empty() ? std::string(text) : path.string() + ": " + text;
    } catch (...) {
      // The status remains usable when a diagnostic cannot be allocated.
    }
  }
  const char* what() const noexcept override { return message.c_str(); }
  sirius_status status;
  std::string message;
};
parsed_sirius_config load_configuration(const std::filesystem::path& path);
}  // namespace sirius
