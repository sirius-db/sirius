// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "object_store/settings.hpp"
#include "scan_manager/config.hpp"  // sirius::scan_manager::scan_manager_config

namespace sirius::object_store {

/// The object-store options honored by the IO backend that @p scan_manager selects.
[[nodiscard]] capabilities capabilities_of(
  const sirius::scan_manager::scan_manager_config& scan_manager);

}  // namespace sirius::object_store
