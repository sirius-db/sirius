/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade disk_access_limiter — ROCm stub.

#pragma once
#include <cstddef>

namespace cucascade::memory {

class disk_access_limiter {
 public:
  explicit disk_access_limiter(std::size_t /*max_concurrent*/ = 1) {}
  bool try_acquire() { return false; }
  void release() {}
};

}  // namespace cucascade::memory
