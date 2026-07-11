/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade cuda/event — ROCm stub.

#pragma once
#include <rmm/cuda_stream.hpp>
#include <cuda_runtime.h>  // shim → hip runtime: cudaEvent_t, cudaEventDisableTiming
#include <stdexcept>

namespace cucascade::cuda {

namespace event {
enum class query_result { success, in_progress, error };
}

class cuda_event_view {
 public:
  cuda_event_view() = default;
  explicit cuda_event_view(cudaEvent_t e) : event_(e) {}
  cudaEvent_t value() const { return event_; }
  operator cudaEvent_t() const { return event_; }
  void record(rmm::cuda_stream_view /*stream*/) {}
  void wait(rmm::cuda_stream_view /*stream*/) {}
  void synchronize() {}
  query_result query() const { return query_result::success; }
  float elapsed_time(cuda_event_view /*start*/) const { return 0.0f; }
 private:
  cudaEvent_t event_{nullptr};
};

class cuda_event {
 public:
  explicit cuda_event(unsigned int /*flags*/ = cudaEventDisableTiming) {
    // Stub: would call cudaEventCreateWithFlags. Not allocating a real event
    // on ROCm since the stub doesn't need actual synchronization.
  }
  ~cuda_event() = default;

  cuda_event(cuda_event const&) = delete;
  cuda_event& operator=(cuda_event const&) = delete;

  cudaEvent_t get() const { return nullptr; }
  operator cudaEvent_t() const { return nullptr; }
  cuda_event_view view() const { return cuda_event_view{}; }

  void record(rmm::cuda_stream_view) {}
  void wait(rmm::cuda_stream_view) {}
  void synchronize() {}
  cudaError_t synchronize_no_throw() { return cudaSuccess; }
  float elapsed_time(cuda_event const& /*start*/) const { return 0.0f; }
  query_result query() const { return query_result::success; }
};

}  // namespace cucascade::cuda
