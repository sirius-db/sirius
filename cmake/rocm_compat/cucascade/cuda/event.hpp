/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade cuda/event — ROCm real implementation using hipEvent*.

#pragma once
#include <rmm/cuda_stream.hpp>
#include <cuda_runtime.h>  // shim → hip runtime
#include "cucascade/error.hpp"
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
  void record(rmm::cuda_stream_view stream) { CUCASCADE_CUDA_TRY(hipEventRecord(event_, stream.value())); }
  void wait(rmm::cuda_stream_view stream) { CUCASCADE_CUDA_TRY(hipStreamWaitEvent(stream.value(), event_, 0)); }
  void synchronize() { CUCASCADE_CUDA_TRY(hipEventSynchronize(event_)); }
  query_result query() const {
    cudaError_t err = hipEventQuery(event_);
    if (err == cudaSuccess) return query_result::success;
    if (err == hipErrorNotReady) return query_result::in_progress;
    return query_result::error;
  }
  float elapsed_time(cuda_event_view start) const {
    float ms = 0.f;
    CUCASCADE_CUDA_TRY(hipEventElapsedTime(&ms, start.value(), event_));
    return ms;
  }
 private:
  cudaEvent_t event_{nullptr};
};

class cuda_event {
 public:
  explicit cuda_event(unsigned int flags = cudaEventDisableTiming) {
    CUCASCADE_CUDA_TRY(hipEventCreateWithFlags(&event_, flags));
  }
  ~cuda_event() {
    if (event_) hipEventDestroy(event_);
  }
  cuda_event(cuda_event const&) = delete;
  cuda_event& operator=(cuda_event const&) = delete;
  cuda_event(cuda_event&& o) noexcept : event_(o.event_) { o.event_ = nullptr; }
  cuda_event& operator=(cuda_event&& o) noexcept {
    if (event_) hipEventDestroy(event_);
    event_ = o.event_; o.event_ = nullptr;
    return *this;
  }

  cudaEvent_t get() const { return event_; }
  operator cudaEvent_t() const { return event_; }
  cuda_event_view view() const { return cuda_event_view{event_}; }

  void record(rmm::cuda_stream_view stream) { CUCASCADE_CUDA_TRY(hipEventRecord(event_, stream.value())); }
  void wait(rmm::cuda_stream_view stream) { CUCASCADE_CUDA_TRY(hipStreamWaitEvent(stream.value(), event_, 0)); }
  void synchronize() { CUCASCADE_CUDA_TRY(hipEventSynchronize(event_)); }
  cudaError_t synchronize_no_throw() { return hipEventSynchronize(event_); }
  float elapsed_time(cuda_event const& start) const {
    float ms = 0.f;
    hipEventElapsedTime(&ms, start.event_, event_);
    return ms;
  }
  event::query_result query() const {
    cudaError_t err = hipEventQuery(event_);
    if (err == cudaSuccess) return event::query_result::success;
    if (err == hipErrorNotReady) return event::query_result::in_progress;
    return event::query_result::error;
  }
 private:
  cudaEvent_t event_{nullptr};
};

}  // namespace cucascade::cuda
