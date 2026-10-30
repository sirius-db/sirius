/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <rmm/error.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <algorithm>
#include <limits>
#include <memory>

namespace sirius::test {
// Measures allocations through the caller's MR, independently of the estimator.
// Copies share counters; returned columns may keep an erased resource copy alive.
class tracked_scan_resource {
 public:
  struct counters {
    std::size_t live = 0, peak = 0, attempts = 0;
    std::size_t fail_at          = std::numeric_limits<std::size_t>::max();
    cudaStream_t expected_stream = nullptr;
    bool wrong_stream            = false;
  };
  explicit tracked_scan_resource(rmm::device_async_resource_ref upstream, ::cuda::stream_ref stream)
    : state(std::make_shared<counters>()), upstream_(upstream)
  {
    state->expected_stream = stream.get();
  }
  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    state->wrong_stream |= stream.get() != state->expected_stream;
    attempt();
    auto p = upstream_.allocate(stream, bytes, alignment);
    state->live += bytes;
    state->peak = std::max(state->peak, state->live);
    return p;
  }
  void deallocate(::cuda::stream_ref stream,
                  void* p,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    state->wrong_stream |= stream.get() != state->expected_stream;
    upstream_.deallocate(stream, p, bytes, alignment);
    state->live -= bytes;
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    state->wrong_stream = true;
    attempt();
    auto p = upstream_.allocate_sync(bytes, alignment);
    state->live += bytes;
    state->peak = std::max(state->peak, state->live);
    return p;
  }
  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    state->wrong_stream = true;
    upstream_.deallocate_sync(p, bytes, alignment);
    state->live -= bytes;
  }
  bool operator==(tracked_scan_resource const& other) const noexcept
  {
    return state == other.state;
  }
  friend void get_property(tracked_scan_resource const&, ::cuda::mr::device_accessible) noexcept {}
  std::shared_ptr<counters> state;

 private:
  void attempt()
  {
    if (state->attempts++ == state->fail_at)
      throw rmm::out_of_memory("injected scan allocation failure");
  }
  rmm::device_async_resource_ref upstream_;
};
}  // namespace sirius::test
