// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "object_store_scope/scope.hpp"

#include <atomic>
#include <cstddef>
#include <memory>

struct sirius_object_store_registry {
  std::atomic<std::size_t> references{1};
  const std::shared_ptr<const sirius::object_store_scope::scope_set> stores;
};
