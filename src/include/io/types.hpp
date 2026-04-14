/*
 * Copyright 2025, Sirius Contributors.
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

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sirius::io {

// ---------------------------------------------------------------------------
// IO constants
// ---------------------------------------------------------------------------

static constexpr size_t CHUNK_SIZE    = 1UL << 20;  ///< Bounce-buffer chunk (1 MiB).
static constexpr size_t NUM_CHUNKS    = 32;         ///< Bounce slots per reactor.
static constexpr size_t IO_BLOCK_SIZE = 4096;       ///< O_DIRECT alignment (bytes).

// ---------------------------------------------------------------------------
// request context
// ---------------------------------------------------------------------------

/**
 * @brief Shared completion state for one logical read call (host or device).
 *
 * A single read may be split into multiple sub-requests. All sub-requests
 * decrement @c pending; the last one resolves the promise.
 */
struct request_context {
  std::promise<size_t> promise;    ///< Resolved when all sub-requests finish.
  std::atomic<size_t> pending{0};  ///< Outstanding sub-request count.
  size_t total_bytes{0};           ///< Bytes originally requested (returned on success).
  std::atomic<bool> failed{false};
  std::mutex exc_mtx;
  std::exception_ptr exc;

  void chunk_done()
  {
    if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard lk{exc_mtx};
      if (failed.load(std::memory_order_relaxed))
        promise.set_exception(exc);
      else
        promise.set_value(total_bytes);
    }
  }

  void chunk_failed(std::exception_ptr e)
  {
    if (!failed.exchange(true, std::memory_order_relaxed)) {
      std::lock_guard lk{exc_mtx};
      exc = std::move(e);
    }
    chunk_done();
  }
};

// ---------------------------------------------------------------------------
// device_read_req / host_read_req
// ---------------------------------------------------------------------------

/**
 * @brief Descriptor for one aligned 1 MiB I/O chunk pushed to a reactor for
 *        a device (GPU) read.
 */
struct device_read_req {
  int fd_direct{-1};      ///< O_DIRECT file descriptor.
  size_t file_off{0};     ///< Page-aligned file offset for O_DIRECT.
  size_t io_size{0};      ///< Aligned read length (≤ CHUNK_SIZE).
  size_t data_off{0};     ///< Prefix bytes to skip inside the pinned buffer.
  size_t data_size{0};    ///< Bytes to actually copy to the device destination.
  uint8_t* dst{nullptr};  ///< Device destination pointer (pre-offset by caller).
  cudaStream_t stream{nullptr};
  std::shared_ptr<request_context> ctx;  ///< Shared completion context.
};

/**
 * @brief Descriptor for one buffered host read pushed to a reactor.
 */
struct host_read_req {
  int fd{-1};                            ///< Buffered @c O_RDONLY file descriptor.
  size_t offset{0};                      ///< Byte offset in the file.
  size_t size{0};                        ///< Bytes to read.
  uint8_t* dst{nullptr};                 ///< Host destination pointer.
  std::shared_ptr<request_context> ctx;  ///< Shared completion context.
};

// ---------------------------------------------------------------------------
// sirius_io_object_metadata
// ---------------------------------------------------------------------------

class sirius_io_object_metadata {
 public:
  ~sirius_io_object_metadata() = default;
};

// ---------------------------------------------------------------------------
// sirius_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Abstract per-file handle that provides file identity to a datasource.
 *
 * Decouples file location / cache-key logic from I/O mechanics.
 */
class sirius_io_object {
 public:
  virtual ~sirius_io_object() = default;

  [[nodiscard]] virtual const std::string& raw_file_cache_id() const noexcept = 0;

  [[nodiscard]] virtual size_t size() const noexcept = 0;
};

// ---------------------------------------------------------------------------
// sirius_io_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Abstract I/O reactor interface.
 *
 * Accepts device-read and host-read requests for asynchronous execution.
 */
class sirius_io_reactor {
 public:
  virtual ~sirius_io_reactor() = default;

  virtual void enqueue(device_read_req req)    = 0;
  virtual void enqueue_host(host_read_req req) = 0;
  virtual void interrupt()                     = 0;
  virtual void shutdown()                      = 0;
};

// ---------------------------------------------------------------------------
// sirius_datasource
// ---------------------------------------------------------------------------

class sirius_ioctx;

/**
 * @brief Abstract datasource extending cudf::io::datasource with batch read
 *        APIs.
 */
class io_datasource : public cudf::io::datasource {
 public:
  ~io_datasource() override = default;

  virtual std::future<size_t> host_read_ranges_async(
    std::vector<cudf::io::text::byte_range_info> const& ranges,
    std::span<cudf::host_span<std::byte>> dst) = 0;

  virtual size_t host_read_ranges(std::vector<cudf::io::text::byte_range_info> const& ranges,
                                  std::span<cudf::host_span<std::byte>> dst) = 0;
};

// ---------------------------------------------------------------------------
// sirius_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief Abstract shared context passed to every datasource.
 *
 * Holds resources shared across all datasources (ring pools, reactor threads,
 * …). Also exposes read APIs parameterised by @c sirius_io_object so that
 * callers can issue I/O without going through a full datasource wrapper.
 */
class sirius_ioctx : public std::enable_shared_from_this<sirius_ioctx> {
 public:
  virtual ~sirius_ioctx() = default;

  virtual void shutdown() = 0;

  virtual std::unique_ptr<cudf::io::datasource> make_datasource(
    std::unique_ptr<sirius_io_object> io_object) = 0;

  // -- Read APIs (parameterised by io_object) --------------------------------

  virtual size_t host_read(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst) = 0;
  virtual std::unique_ptr<cudf::io::datasource::buffer> host_read(sirius_io_object& obj,
                                                                  size_t offset,
                                                                  size_t size)              = 0;

  virtual std::future<size_t> host_read_async(sirius_io_object& obj,
                                              size_t offset,
                                              size_t size,
                                              uint8_t* dst) = 0;
  virtual std::future<std::unique_ptr<cudf::io::datasource::buffer>> host_read_async(
    sirius_io_object& obj, size_t offset, size_t size) = 0;

  virtual std::unique_ptr<cudf::io::datasource::buffer> device_read(
    sirius_io_object& obj, size_t offset, size_t size, rmm::cuda_stream_view stream) = 0;
  virtual size_t device_read(sirius_io_object& obj,
                             size_t offset,
                             size_t size,
                             uint8_t* dst,
                             rmm::cuda_stream_view stream)                           = 0;
  virtual std::future<size_t> device_read_async(sirius_io_object& obj,
                                                size_t offset,
                                                size_t size,
                                                uint8_t* dst,
                                                rmm::cuda_stream_view stream)        = 0;

  virtual std::future<size_t> host_read_ranges_async(
    sirius_io_object& obj,
    std::vector<cudf::io::text::byte_range_info> const& ranges,
    std::span<cudf::host_span<std::byte>> dst)                               = 0;
  virtual size_t host_read_ranges(sirius_io_object& obj,
                                  std::vector<cudf::io::text::byte_range_info> const& ranges,
                                  std::span<cudf::host_span<std::byte>> dst) = 0;
};

}  // namespace sirius::io
