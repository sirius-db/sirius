/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade chunked_resource_info — ROCm stub.

#pragma once
#include <cstddef>

namespace cucascade::memory {

class chunked_resource_info {
 public:
  virtual ~chunked_resource_info() = default;
  virtual std::size_t max_chunk_bytes() const { return 0; }
};

}  // namespace cucascade::memory
