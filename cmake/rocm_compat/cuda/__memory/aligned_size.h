/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! CCCL @c <cuda/__memory/aligned_size.h> redirect for ROCm.
//! Provides @c ::cuda::aligned_size_t<N> — used by @c cg::memcpy_async to
//! communicate the alignment of a copy. Sirius uses it at alignments 4, 8, 16.

#pragma once

#include <cstddef>

namespace cuda {

/// A size value tagged with a compile-time alignment. Used by
/// cooperative_groups::memcpy_async to select an optimized copy path.
template <std::size_t Alignment>
struct aligned_size_t {
  static constexpr std::size_t alignment = Alignment;
  std::size_t value;

  explicit constexpr aligned_size_t(std::size_t n) : value(n) {}
  constexpr operator std::size_t() const { return value; }
};

}  // namespace cuda
