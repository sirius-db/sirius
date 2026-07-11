/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade column_metadata — ROCm stub.

#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace cucascade::memory {

struct column_metadata {
  std::string name;
  std::int32_t type_id{0};
  std::int32_t size{0};
  bool nullable{false};
  std::vector<column_metadata> children;
};

}  // namespace cucascade::memory
