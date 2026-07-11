/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuCascade stream pool — ROCm stub.

#pragma once

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_device.hpp>
#include <cstddef>
#include <stdexcept>

namespace cucascade::memory {

/// A borrowed stream RAII wrapper.
class borrowed_stream {
 public:
  borrowed_stream() = default;
  explicit borrowed_stream(rmm::cuda_stream stream) : stream_(stream) {}
  rmm::cuda_stream get() const { return stream_; }
  operator rmm::cuda_stream_view() const { return stream_.view(); }
  void reset() { stream_ = rmm::cuda_stream{}; }
 private:
  rmm::cuda_stream stream_{rmm::cuda_stream_per_thread};
};

/// Exclusive stream pool — manages a set of CUDA/HIP streams.
class exclusive_stream_pool {
 public:
  enum class stream_acquire_policy { GROW, BLOCK };

  exclusive_stream_pool(rmm::cuda_device_id device_id, std::size_t /*pool_size*/ = 0,
                        rmm::cuda_stream::flags /*flags*/ = {})
    : device_id_(device_id) {}

  borrowed_stream acquire_stream(stream_acquire_policy /*policy*/ = stream_acquire_policy::GROW) {
    return borrowed_stream{rmm::cuda_stream_per_thread};
  }

  std::size_t size() const { return 0; }
  rmm::cuda_device_id get_device_id() const { return device_id_; }

 private:
  rmm::cuda_device_id device_id_;
};

}  // namespace cucascade::memory
