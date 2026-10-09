// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sirius_config.hpp"

#include <atomic>
#include <utility>

struct sirius_config_builder {
  explicit sirius_config_builder(sirius::parsed_sirius_config value = {}) : config(std::move(value))
  {
  }
  std::atomic<std::size_t> references{1};
  const sirius::parsed_sirius_config config;
};
struct sirius_config {
  explicit sirius_config(const sirius::parsed_sirius_config& value) : config(value) {}
  std::atomic<std::size_t> references{1};
  const sirius::parsed_sirius_config config;
};
