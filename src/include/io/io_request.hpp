/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Backend-agnostic request-lifecycle primitives shared by every reactor
// (io_uring, REST/curl, ...).  A reactor splits one caller request into N
// per-chunk requests that share a single request_manager; each chunk reports
// completion or error, and the manager fulfills one future when all chunks are
// done.  device_cpy_request batches the host->device copies for GPU-bound
// reads.  rx_request_t is the per-reactor container that the templated_ioctx
// dispatch layer splits across the reactor pool.

#include "exec/semi_future.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <source_location>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::io {

/// Fan-in for one caller request that a reactor splits into @c total_chunks
/// per-chunk reads.  Each chunk calls @c chunk_complete (or @c report_error);
/// the future returned by @c get_future is fulfilled from the destructor once
/// the last owning reference drops — with the first reported error if any, or
/// @c bytes_requested otherwise.
class request_manager {
 public:
  using error_type = std::variant<std::exception_ptr, cudaError_t, std::error_code>;
  // @p bytes_requested is the number of bytes the *caller* asked for; it is the
  // value handed back through the future.  The reactor frequently reads more
  // than that (O_DIRECT/chunk alignment over-reads whole blocks), so the
  // physically-read total tracked in @c bytes_read is only used to assert that
  // the request was fully covered — it is never returned to the caller.
  explicit request_manager(std::size_t bytes_requested, std::size_t total_chunks)
    : bytes_requested(bytes_requested), total_chunks(total_chunks)
  {
  }

  ~request_manager()
  {
    if (has_error()) {
      promise.set_exception(first_exception);
    } else {
      assert(bytes_read >= bytes_requested &&
             "All chunks completed but fewer bytes were read than requested");
      assert(chunks_completed == total_chunks &&
             "All chunks completed but total chunks completed does not match expected");
      promise.set_value(bytes_requested);
    }
  }

  void chunk_complete(std::size_t n_bytes)
  {
    bytes_read.fetch_add(n_bytes, std::memory_order_acq_rel);
    chunks_completed.fetch_add(1, std::memory_order_acq_rel);
  }

  void report_error(const error_type& e, std::source_location loc = std::source_location::current())
  {
    if (!error_reported.exchange(true, std::memory_order_acq_rel)) {
      first_exception = to_exception_ptr(e, loc);
    }
  }

  [[nodiscard]] bool has_error() const noexcept
  {
    return error_reported.load(std::memory_order_acquire);
  }

  [[nodiscard]] exec::semi_future<size_t> get_future() noexcept
  {
    return promise.get_semi_future();
  }

  const std::size_t bytes_requested;
  const std::size_t total_chunks;

 private:
  [[nodiscard]] std::exception_ptr to_exception_ptr(const error_type& e,
                                                    std::source_location loc) const noexcept
  {
    if (std::holds_alternative<std::exception_ptr>(e)) {
      return std::get<std::exception_ptr>(e);
    } else if (std::holds_alternative<cudaError_t>(e)) {
      auto err = std::get<cudaError_t>(e);
      return std::make_exception_ptr(
        std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)) + " at " +
                           loc.file_name() + ":" + std::to_string(loc.line())));
    } else if (std::holds_alternative<std::error_code>(e)) {
      auto err = std::get<std::error_code>(e);
      return std::make_exception_ptr(std::system_error(
        err, "System error at " + std::string(loc.file_name()) + ":" + std::to_string(loc.line())));
    }
    return nullptr;  // Should never reach here
  }

  std::atomic<std::size_t> bytes_read{0};
  std::atomic<std::size_t> chunks_completed{0};
  std::atomic<bool> error_reported{false};
  std::exception_ptr first_exception{nullptr};
  exec::promise<size_t> promise;
};

struct device_cpy_request {
  // One host->device copy.  @c src is resolved as @c host_buffer + @c src_off
  // when @c src is null (the bounce-slot path assigns its host buffer late,
  // once a slot is acquired); otherwise @c src is an absolute caller-owned host
  // pointer (the multi-buffer readv path, whose buffers are separate host
  // allocations and so cannot share one base).
  struct copy {
    uint8_t* dst{nullptr};
    uint8_t* src{nullptr};
    size_t src_off{0};
    size_t size{0};
  };

  // Issue every copy on @p stream (a batch when there is more than one), then
  // record @p event once after the last so a single wait covers them all.
  cudaError_t copy_async(uint8_t* host_buffer,
                         [[maybe_unused]] size_t bytes,
                         cudaEvent_t event = nullptr) noexcept
  {
    assert(host_buffer != nullptr && "Caller must provide a valid host buffer for the copy.");
    rmm::cuda_set_device_raii device_guard(rmm::cuda_device_id{device_id});
    cudaError_t err = cudaSuccess;
    for (auto const& c : copies) {
      assert(c.dst != nullptr &&
             "Caller must provide a valid device destination buffer for the copy.");
      assert((c.src != nullptr || c.src_off + c.size <= bytes) &&
             "Caller must ensure the copy fits in the host buffer.");
      uint8_t* src_ptr = c.src != nullptr ? c.src : host_buffer + c.src_off;
      err              = cudaMemcpyAsync(c.dst, src_ptr, c.size, cudaMemcpyHostToDevice, stream);
      if (err != cudaSuccess) { return err; }
    }
    if (event != nullptr) { err = cudaEventRecord(event, stream); }
    return err;
  }

  std::vector<copy> copies;
  rmm::cuda_stream_view stream;
  int device_id{-1};
};

/// Per-reactor container of per-chunk requests for one split of a caller
/// request.  @c Chunk is the backend-specific chunk type (it must expose a
/// @c manager member of type @c std::shared_ptr<request_manager>).  The
/// templated_ioctx dispatch layer builds one of these, retrieves its future,
/// then @c splits it across the reactor pool and enqueues each split.
template <class Chunk>
struct rx_request_t {
  static std::unique_ptr<rx_request_t> create(std::vector<std::unique_ptr<Chunk>> reqs) noexcept
  {
    return std::unique_ptr<rx_request_t>(new rx_request_t(std::move(reqs)));
  }

  [[nodiscard]] std::size_t size() const noexcept { return requests.size(); }

  static std::vector<std::unique_ptr<rx_request_t>> splits(std::unique_ptr<rx_request_t> req,
                                                           std::size_t n_splits) noexcept
  {
    std::vector<std::unique_ptr<rx_request_t>> result;
    auto chunks = req->get_all_chunks();
    req.reset(nullptr);
    if (n_splits == 0 || chunks.empty()) return result;

    std::size_t chunks_per_split = (chunks.size() + n_splits - 1) / n_splits;
    std::vector<std::unique_ptr<Chunk>> current_batch;

    for (auto& chunk : chunks) {
      current_batch.push_back(std::move(chunk));
      if (current_batch.size() >= chunks_per_split) {
        result.push_back(create(std::move(current_batch)));
        current_batch.clear();
      }
    }

    if (!current_batch.empty()) { result.push_back(create(std::move(current_batch))); }
    return result;
  }

  std::unique_ptr<Chunk> get_next_chunk() noexcept
  {
    if (requests.empty()) return nullptr;
    auto next_req = std::move(requests.back());
    requests.pop_back();
    return next_req;
  }

  std::vector<std::unique_ptr<Chunk>> get_all_chunks() noexcept { return std::move(requests); }

  exec::semi_future<size_t> get_future() noexcept
  {
    if (requests.empty()) {
      // Nothing to read (e.g. all segments were already in flight): hand back a
      // ready, zero-byte future.  The semi_future must be retrieved BEFORE the
      // promise is satisfied — set_value() releases the shared core, after which
      // get_semi_future() would throw promise_already_satisfied.
      exec::promise<size_t> promise;
      auto fut = promise.get_semi_future();
      promise.set_value(0);
      return fut;
    }
    return requests.front()->manager->get_future();
  }

 private:
  std::vector<std::unique_ptr<Chunk>> requests;

  explicit rx_request_t(std::vector<std::unique_ptr<Chunk>> reqs) : requests(std::move(reqs)) {}
};

}  // namespace sirius::io
