// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "sirius_config.hpp"

#include <sirius/context/config.hpp>

namespace sirius {

struct ContextConfig::Impl {
  parsed_sirius_config config;
};

}  // namespace sirius
