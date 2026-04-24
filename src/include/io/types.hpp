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

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace sirius::io {

// Forward declarations (cache is owned by sirius_ioctx; defined in
// prefetching_cache.hpp).
class prefetching_cache;
class buffer_pool;

// ---------------------------------------------------------------------------
// Completion handler
// ---------------------------------------------------------------------------

/// Boost.Asio-style completion handler for async I/O.
/// @param bytes_transferred  Total bytes read on success.
/// @param ep                 Non-null on failure.
using io_completion_handler = std::function<void(size_t bytes_transferred, std::exception_ptr ep)>;

// ---------------------------------------------------------------------------
// IO constants
// ---------------------------------------------------------------------------

static constexpr size_t CHUNK_SIZE    = 1UL << 20;  ///< Bounce-buffer chunk size (1 MiB).
static constexpr size_t NUM_CHUNKS    = 32;         ///< Number of bounce slots per reactor.
static constexpr size_t IO_BLOCK_SIZE = 4096;       ///< O_DIRECT alignment requirement (bytes).

// ---------------------------------------------------------------------------
// sirius_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Abstract per-file handle that provides file identity to a datasource.
 *
 * Decouples file location / cache-key logic from I/O mechanics.
 *
 * Inherits from @c std::enable_shared_from_this so the prefetching cache can
 * take a reference to an io_object and safely extend its lifetime via
 * @c shared_from_this() — this enforces at call sites that every io_object
 * passed in is already owned by a @c std::shared_ptr.
 */
class sirius_io_object : public std::enable_shared_from_this<sirius_io_object> {
 public:
  virtual ~sirius_io_object() = default;

  [[nodiscard]] virtual const std::string& raw_file_cache_id() const noexcept = 0;
  [[nodiscard]] virtual size_t size() const noexcept                          = 0;
};

class sirius_io_object_metadata {
 public:
  virtual ~sirius_io_object_metadata() = default;
};

// ---------------------------------------------------------------------------
// request_context
// ---------------------------------------------------------------------------

/**
 * @brief Shared completion state for one logical read call (host or device).
 *
 * A single read may be split into multiple sub-requests. All sub-requests
 * decrement @c pending; the last one resolves the promise.
 */
struct request_context {
  io_completion_handler handler;
  std::atomic<size_t> pending{0};
  size_t total_bytes{0};
  std::atomic<bool> failed{false};
  std::mutex exc_mtx;
  std::exception_ptr exc;

  void chunk_done()
  {
    if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard lk{exc_mtx};
      if (failed.load(std::memory_order_relaxed))
        handler(0, exc);
      else
        handler(total_bytes, nullptr);
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
 *        a device (GPU) read.  Templated on the backend's native handle type
 *        (e.g. @c int for a POSIX file descriptor).
 */
template <typename Handle>
struct device_read_req {
  Handle handle{};
  size_t file_off{0};
  size_t io_size{0};
  size_t data_off{0};
  size_t data_size{0};
  uint8_t* dst{nullptr};
  cudaStream_t stream{nullptr};
  /// CUDA device index that owns @c dst and @c stream.  The reactor thread
  /// may be running with a different current device, so it must
  /// cudaSetDevice(device_id) before issuing the H2D copy in multi-GPU
  /// deployments.  -1 means "don't switch" (single-GPU fast path).
  int device_id{-1};
  std::shared_ptr<request_context> ctx;
};

/**
 * @brief Descriptor for one buffered host read pushed to a reactor.
 *        Templated on the backend's native handle type.
 */
template <typename Handle>
struct host_read_req {
  Handle handle{};
  size_t offset{0};
  size_t size{0};
  uint8_t* dst{nullptr};
  std::shared_ptr<request_context> ctx;
};

// ---------------------------------------------------------------------------
// sirius_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief Abstract shared context passed to every datasource.
 *
 * Holds resources that are shared across all datasources (ring pools,
 * reactor threads, ...). Extend this class to provide a concrete I/O backend.
 */
class sirius_ioctx : public std::enable_shared_from_this<sirius_ioctx> {
 public:
  sirius_ioctx();
  virtual ~sirius_ioctx();

  virtual void shutdown() = 0;

  virtual std::unique_ptr<cudf::io::datasource> make_datasource(
    std::shared_ptr<sirius_io_object> io_object) = 0;

  /// Construct the owned prefetching_cache.  Must be called before any
  /// read that should consult the cache; until then device_read falls
  /// through directly to device_read_io.
  void initialize_cache(buffer_pool& pool, size_t inflight_budget_chunks = 2048);

  [[nodiscard]] prefetching_cache* cache() noexcept { return _cache.get(); }

  // -- Read API ---------------------------------------------------------------

  virtual size_t host_read(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst) = 0;

  virtual std::unique_ptr<cudf::io::datasource::buffer> host_read(sirius_io_object& obj,
                                                                  size_t offset,
                                                                  size_t size) = 0;

  virtual void host_read_async(sirius_io_object& obj,
                               size_t offset,
                               size_t size,
                               uint8_t* dst,
                               io_completion_handler handler) = 0;

  // device_read / device_read_async: concrete in the base; consult the
  // cache first, fall through to device_read_io{,_async} on miss.
  std::unique_ptr<cudf::io::datasource::buffer> device_read(sirius_io_object& obj,
                                                            size_t offset,
                                                            size_t size,
                                                            rmm::cuda_stream_view stream);

  size_t device_read(
    sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream);

  void device_read_async(sirius_io_object& obj,
                         size_t offset,
                         size_t size,
                         uint8_t* dst,
                         rmm::cuda_stream_view stream,
                         io_completion_handler handler);

  // Backend-specific IO path (no cache lookup).  Implementations read
  // directly from the underlying device (e.g. O_DIRECT + cuFile).
  virtual std::unique_ptr<cudf::io::datasource::buffer> device_read_io(
    sirius_io_object& obj, size_t offset, size_t size, rmm::cuda_stream_view stream) = 0;

  virtual size_t device_read_io(sirius_io_object& obj,
                                size_t offset,
                                size_t size,
                                uint8_t* dst,
                                rmm::cuda_stream_view stream) = 0;

  virtual void device_read_io_async(sirius_io_object& obj,
                                    size_t offset,
                                    size_t size,
                                    uint8_t* dst,
                                    rmm::cuda_stream_view stream,
                                    io_completion_handler handler) = 0;

  virtual void host_read_ranges_async(sirius_io_object& obj,
                                      std::vector<cudf::io::text::byte_range_info> const& ranges,
                                      std::span<cudf::host_span<std::byte>> dst,
                                      io_completion_handler handler) = 0;

  virtual size_t host_read_ranges(sirius_io_object& obj,
                                  std::vector<cudf::io::text::byte_range_info> const& ranges,
                                  std::span<cudf::host_span<std::byte>> dst) = 0;

  // -- Physical range alignment ------------------------------------------------

  virtual cudf::io::text::byte_range_info compute_physical_range(
    cudf::io::text::byte_range_info logical, size_t file_size) const = 0;

 protected:
  std::unique_ptr<prefetching_cache> _cache;
};

// ---------------------------------------------------------------------------
// io_datasource
// ---------------------------------------------------------------------------

/**
 * @brief Extended datasource with batch range-read support.
 */
class io_datasource : public cudf::io::datasource {
 public:
  virtual void host_read_ranges_async(std::vector<cudf::io::text::byte_range_info> const& ranges,
                                      std::span<cudf::host_span<std::byte>> dst,
                                      io_completion_handler handler) = 0;

  virtual size_t host_read_ranges(std::vector<cudf::io::text::byte_range_info> const& ranges,
                                  std::span<cudf::host_span<std::byte>> dst) = 0;
};

}  // namespace sirius::io
