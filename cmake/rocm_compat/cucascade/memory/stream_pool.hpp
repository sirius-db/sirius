/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file cuCascade stream pool — ROCm real implementation.
//! Uses hipStreamCreateWithFlags for real stream management.

#pragma once

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_device.hpp>
#include <cuda_runtime.h>  // shim → hip runtime
#include <cstddef>
#include <stdexcept>
#include <vector>

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

/// Exclusive stream pool — manages a set of HIP streams.
/// GROW policy: creates streams on demand. BLOCK: reuses existing.
class exclusive_stream_pool {
 public:
  enum class stream_acquire_policy { GROW, BLOCK };

  exclusive_stream_pool(rmm::cuda_device_id device_id, std::size_t pool_size = 0,
                        rmm::cuda_stream::flags flags = {})
    : device_id_(device_id), pool_size_(pool_size), flags_(flags) {
    // Pre-create pool_size streams
    for (std::size_t i = 0; i < pool_size; ++i) {
      hipStream_t s = nullptr;
      if (hipStreamCreateWithFlags(&s, 0) == hipSuccess) {
        streams_.push_back(s);
        free_.push_back(true);
      }
    }
  }

  ~exclusive_stream_pool() {
    for (auto s : streams_) {
      if (s) hipStreamDestroy(s);
    }
  }

  borrowed_stream acquire_stream(stream_acquire_policy policy = stream_acquire_policy::GROW) {
    // Try to find a free stream
    for (std::size_t i = 0; i < free_.size(); ++i) {
      if (free_[i]) {
        free_[i] = false;
        // Wrap the raw hipStream_t in an rmm::cuda_stream — this requires
        // rmm::cuda_stream to accept a raw stream. In hipMM, cuda_stream
        // wraps a hipStream_t, so we create a non-owning view.
        return borrowed_stream{rmm::cuda_stream{streams_[i]}};
      }
    }
    // GROW: create a new stream
    if (policy == stream_acquire_policy::GROW) {
      hipStream_t s = nullptr;
      if (hipStreamCreateWithFlags(&s, 0) == hipSuccess) {
        streams_.push_back(s);
        free_.push_back(false);
        return borrowed_stream{rmm::cuda_stream{s}};
      }
    }
    // Fallback: per-thread default stream
    return borrowed_stream{rmm::cuda_stream_per_thread};
  }

  std::size_t size() const { return streams_.size(); }
  rmm::cuda_device_id get_device_id() const { return device_id_; }

 private:
  rmm::cuda_device_id device_id_;
  std::size_t pool_size_{0};
  rmm::cuda_stream::flags flags_{};
  std::vector<hipStream_t> streams_;
  std::vector<bool> free_;
};

}  // namespace cucascade::memory
