// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "object_store_scope/settings.hpp"

#include <atomic>
#include <cstddef>

struct sirius_object_store_config_builder {
  std::atomic<std::size_t> references{1};
  const sirius::object_store_scope::settings value;
};
struct sirius_object_store_config {
  std::atomic<std::size_t> references{1};
  const sirius::object_store_scope::settings value;
};
