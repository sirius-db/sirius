/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "op/scan/scan_ring_buffer.hpp"

#include <log/logging.hpp>

#include <cuda_runtime.h>

#include <stdexcept>

namespace sirius::op::scan {

scan_ring_buffer::scan_ring_buffer(size_t num_slots, size_t slot_bytes)
  : slot_bytes_(slot_bytes)
{
  slots_.resize(num_slots);

  for (size_t i = 0; i < num_slots; ++i) {
    auto& s = slots_[i];

    // Allocate page-locked host memory for async H2D
    auto err = cudaHostAlloc(&s.host_pinned, slot_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "scan_ring_buffer: cudaHostAlloc failed for slot " + std::to_string(i) +
          " (" + std::to_string(slot_bytes) + " bytes): " + cudaGetErrorString(err));
    }

    // Allocate device staging memory
    err = cudaMalloc(&s.device_staging, slot_bytes);
    if (err != cudaSuccess) {
      cudaFreeHost(s.host_pinned);
      throw std::runtime_error(
          "scan_ring_buffer: cudaMalloc failed for slot " + std::to_string(i) +
          " (" + std::to_string(slot_bytes) + " bytes): " + cudaGetErrorString(err));
    }

    // Create events with timing disabled for lower overhead
    cudaEventCreateWithFlags(&s.h2d_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&s.compute_done, cudaEventDisableTiming);

    s.state = slot_state::FREE;
    s.used_bytes = 0;
  }

  SIRIUS_LOG_INFO(
      "[scan_ring_buffer] allocated {} slots x {:.1f}MB = {:.1f}MB pinned + {:.1f}MB device",
      num_slots,
      slot_bytes / (1024.0 * 1024.0),
      num_slots * slot_bytes / (1024.0 * 1024.0),
      num_slots * slot_bytes / (1024.0 * 1024.0));
}

scan_ring_buffer::~scan_ring_buffer()
{
  for (auto& s : slots_) {
    if (s.compute_done) {
      // Ensure all GPU work is done before freeing
      cudaEventSynchronize(s.compute_done);
    }
    if (s.host_pinned)     cudaFreeHost(s.host_pinned);
    if (s.device_staging)  cudaFree(s.device_staging);
    if (s.h2d_done)        cudaEventDestroy(s.h2d_done);
    if (s.compute_done)    cudaEventDestroy(s.compute_done);
  }
}

scan_ring_buffer::slot& scan_ring_buffer::acquire()
{
  auto& s = slots_[next_slot_ % slots_.size()];

  if (s.state != slot_state::FREE) {
    // Backpressure: wait for compute to finish on this slot before reusing.
    // This is the ONLY CPU-side sync in the pipeline loop (besides final drain).
    cudaEventSynchronize(s.compute_done);
    s.state = slot_state::FREE;
  }

  s.used_bytes = 0;
  next_slot_++;
  return s;
}

}  // namespace sirius::op::scan
