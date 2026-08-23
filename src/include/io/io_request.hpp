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

#include "exec/invocable.hpp"
#include "exec/semi_future.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <io/types.hpp>

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
// class request_manager {
//  public:
//   using error_type = std::variant<std::exception_ptr, cudaError_t, std::error_code>;
//   // @p bytes_requested is the number of bytes the *caller* asked for; it is the
//   // value handed back through the future.  The reactor frequently reads more
//   // than that (O_DIRECT/chunk alignment over-reads whole blocks), so the
//   // physically-read total tracked in @c bytes_read is only used to assert that
//   // the request was fully covered — it is never returned to the caller.
//   explicit request_manager(std::size_t bytes_requested, std::size_t total_chunks)
//     : bytes_requested(bytes_requested), total_chunks(total_chunks)
//   {
//   }

//   ~request_manager()
//   {
//     if (has_error()) {
//       promise.set_exception(first_exception);
//     } else {
//       assert(bytes_read >= bytes_requested &&
//              "All chunks completed but fewer bytes were read than requested");
//       assert(chunks_completed == total_chunks &&
//              "All chunks completed but total chunks completed does not match expected");
//       promise.set_value(bytes_requested);
//     }
//   }

//   void chunk_complete(std::size_t n_bytes)
//   {
//     bytes_read.fetch_add(n_bytes, std::memory_order_acq_rel);
//     chunks_completed.fetch_add(1, std::memory_order_acq_rel);
//   }

//   void report_error(const error_type& e, std::source_location loc =
//   std::source_location::current())
//   {
//     if (!error_reported.exchange(true, std::memory_order_acq_rel)) {
//       first_exception = to_exception_ptr(e, loc);
//     }
//   }

//   [[nodiscard]] bool has_error() const noexcept
//   {
//     return error_reported.load(std::memory_order_acquire);
//   }

//   [[nodiscard]] exec::semi_future<size_t> get_future() noexcept
//   {
//     return promise.get_semi_future();
//   }

//   const std::size_t bytes_requested;
//   const std::size_t total_chunks;

//  private:
//   [[nodiscard]] std::exception_ptr to_exception_ptr(const error_type& e,
//                                                     std::source_location loc) const noexcept
//   {
//     if (std::holds_alternative<std::exception_ptr>(e)) {
//       return std::get<std::exception_ptr>(e);
//     } else if (std::holds_alternative<cudaError_t>(e)) {
//       auto err = std::get<cudaError_t>(e);
//       return std::make_exception_ptr(
//         std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)) + " at " +
//                            loc.file_name() + ":" + std::to_string(loc.line())));
//     } else if (std::holds_alternative<std::error_code>(e)) {
//       auto err = std::get<std::error_code>(e);
//       return std::make_exception_ptr(std::system_error(
//         err, "System error at " + std::string(loc.file_name()) + ":" +
//         std::to_string(loc.line())));
//     }
//     return nullptr;  // Should never reach here
//   }

//   std::atomic<std::size_t> bytes_read{0};
//   std::atomic<std::size_t> chunks_completed{0};
//   std::atomic<bool> error_reported{false};
//   std::exception_ptr first_exception{nullptr};
//   exec::promise<size_t> promise;
// };

struct grouped_io_request;

struct grouped_coordinator {
  using error_type = std::variant<std::exception_ptr, cudaError_t, std::error_code>;

  [[nodiscard]] bool should_continue() const noexcept { return !has_error(); }

  void on_complete();

  void report_error(const error_type& e,
                    std::source_location loc = std::source_location::current());

  [[nodiscard]] bool has_error() const noexcept;

  [[nodiscard]] exec::semi_future<void> get_future() noexcept;

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

  std::atomic<std::size_t> tasks_remaining{0};
  std::atomic<bool> error_reported{false};
  std::exception_ptr first_exception{nullptr};
  exec::promise<void> promise;
};

struct grouped_io_request {
  static std::unique_ptr<grouped_io_request> create(std::shared_ptr<const io_object> obj,
                                                    std::vector<prepared_io_slice> slices);

  std::shared_ptr<const io_object> obj;
  std::vector<prepared_io_slice> slices;
  std::shared_ptr<grouped_coordinator> coordinator;

 private:
  grouped_io_request(std::shared_ptr<const io_object> obj,
                     std::vector<prepared_io_slice> slices,
                     std::shared_ptr<grouped_coordinator> g_coordinator)
    : obj(std::move(obj)), slices(std::move(slices)), coordinator(std::move(g_coordinator))
  {
  }
};

struct device_cpy_request {
  range req_rng;
  device_buffer d_buffer;
  int device_id{-1};

  // Issue every copy on @p stream (a batch when there is more than one), then
  // record @p event once after the last so a single wait covers them all.
  cudaError_t copy_async(const range& io_rng,
                         std::span<iovec> host_buf,
                         cudaEvent_t event = nullptr) noexcept
  {
    /// from req_range and io_rng, compute the offset into the host buffers that should start
    /// copying into device host_buf is a span of iovec, that represent contiguous slices of
    /// buffers, starting from first byte in frin iovec and going to io_rng.size bytes
    /// copy them to d_buffer for req_rng.size bytes, and record event once after if the event is
    /// not null, you need to set the device using rmm::raii::set_device before issuing the copy,
    /// and reset it after the copy is issued
  }
};

template <typename Reactor>
struct io_op_request {
  Reactor::io_object_identifier_type
    file_id;     // this is whatever the reactor uses to identify the file, e.g. and do io
  range io_rng;  // range of the io operation, for example, O_DIRECT aligned range for io for a
                 // non-aligned request, this is the range that will be read from the file and
                 // copied to the device
  std::vector<iovec> iovecs;
  std::unique_ptr<device_cpy_request> device_copy;
  std::shared_ptr<grouped_coordinator> coordinator;
  exec::invocable<void(bool) noexcept> on_complete;
};

}  // namespace sirius::io
