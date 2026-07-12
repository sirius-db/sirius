/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade disk_access_limiter — ROCm real implementation.

#pragma once
#include <atomic>
#include <cstddef>

namespace cucascade::memory {

class disk_access_limiter {
 public:
  explicit disk_access_limiter(std::size_t max_concurrent = 1)
    : max_(max_concurrent), current_(0) {}

  bool try_acquire() {
    std::size_t expected = current_.load(std::memory_order_relaxed);
    while (expected < max_) {
      if (current_.compare_exchange_weak(expected, expected + 1,
                                          std::memory_order_acquire,
                                          std::memory_order_relaxed)) {
        return true;
      }
    }
    return false;
  }

  void release() {
    current_.fetch_sub(1, std::memory_order_release);
  }

 private:
  std::size_t max_;
  std::atomic<std::size_t> current_;
};

}  // namespace cucascade::memory
