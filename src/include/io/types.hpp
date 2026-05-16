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

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::io {

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
static constexpr size_t NUM_CHUNKS    = 128;        ///< Number of bounce slots per reactor.
static constexpr size_t IO_BLOCK_SIZE = 4096;       ///< O_DIRECT alignment requirement (bytes).

// ---------------------------------------------------------------------------
// sirius_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Abstract per-file handle.  A passive bag of native handles
 * produced by a backend reactor (e.g. file descriptors, CURL easy
 * handles, S3 client state).  Performs no I/O of its own.
 *
 * Inherits from @c std::enable_shared_from_this so the prefetching cache can
 * take a reference to an io_object and safely extend its lifetime via
 * @c shared_from_this() — this enforces at call sites that every io_object
 * passed in is already owned by a @c std::shared_ptr.
 */
class sirius_io_object : public std::enable_shared_from_this<sirius_io_object> {
 public:
  virtual ~sirius_io_object() = default;

  /// Stable identifier used as the prefetching-cache key.  Often equal to
  /// @c object_path() but may differ for backends that need to distinguish
  /// otherwise-equal paths (versioned S3 keys, normalized URLs, …).
  [[nodiscard]] virtual const std::string& raw_file_cache_id() const noexcept = 0;

  /// The path / URL / key the caller used to construct this object.
  [[nodiscard]] virtual const std::string& object_path() const noexcept = 0;

  /// Total size of the underlying object, populated by the reactor at
  /// construction time and stored on the io_object thereafter.
  [[nodiscard]] virtual size_t size() const noexcept = 0;
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
 * decrement @c pending; the last one resolves the handler.
 *
 * Construction is gated by @c create() so callers can't forget to set
 * @c pending or @c handler (a missed setup would silently deadlock the
 * caller).  The destructor is a safety net: if @c handler hasn't been
 * fired by the time the last @c shared_ptr drops (e.g., a sub-request
 * was silently dropped somewhere in the dispatch chain), it fires with
 * an explicit error so the caller sees a failure instead of hanging.
 */
struct request_context {
 private:
  // Private tag so only create() can construct.  Public ctor signature so
  // std::make_shared still works (preserves the single-allocation control
  // block + object layout).
  struct create_passkey {};

 public:
  explicit request_context(create_passkey) noexcept {}

  request_context(request_context const&)            = delete;
  request_context& operator=(request_context const&) = delete;

  /// Construct a request_context expecting @p n_chunks chunk_done /
  /// chunk_failed calls.
  ///
  /// - If @p n_chunks == 0: invokes @p handler with (0, nullptr) immediately
  ///   and returns nullptr.  Caller checks `if (!ctx) return;`.
  /// - If @p handler is null and @p n_chunks > 0: throws
  ///   @c std::invalid_argument.  A pending count with no handler would
  ///   silently deadlock the caller.
  /// - Otherwise: returns a fully-populated shared_ptr.
  [[nodiscard]] static std::shared_ptr<request_context> create(size_t n_chunks,
                                                               size_t total_bytes,
                                                               io_completion_handler handler)
  {
    if (n_chunks == 0) {
      if (handler) {
        try {
          handler(0, nullptr);
        } catch (...) {
          // Caller's handler threw on the zero-work path; nothing useful
          // to do here.
        }
      }
      return nullptr;
    }
    if (!handler) {
      throw std::invalid_argument("request_context::create: handler is null but n_chunks > 0");
    }
    auto ctx         = std::make_shared<request_context>(create_passkey{});
    ctx->handler     = std::move(handler);
    ctx->total_bytes = total_bytes;
    ctx->pending.store(n_chunks, std::memory_order_relaxed);
    return ctx;
  }

  ~request_context() noexcept
  {
    // Safety net: if the handler hasn't been fired by the normal path
    // (e.g., a sub-request was silently dropped between enqueue and
    // completion), fire it now with an explicit error so the caller
    // doesn't hang forever waiting on a handler that will never come.
    bool expected = false;
    if (handler_fired.compare_exchange_strong(expected, true, std::memory_order_acq_rel) &&
        handler) {
      try {
        handler(0,
                std::make_exception_ptr(
                  std::runtime_error("request_context destructed before all chunks completed — "
                                     "handler resolved by safety net")));
      } catch (...) {
        // A throwing handler at destruction time can't propagate anywhere
        // useful — swallow to keep the destructor noexcept.
      }
    }
  }

  void chunk_done()
  {
    if (pending.fetch_sub(1, std::memory_order_acq_rel) != 1) return;

    // Last chunk — fire the handler exactly once.  The CAS guard makes
    // this idempotent against the destructor safety net and against any
    // future imbalance where chunk_done could be called extra times.
    bool expected = false;
    if (!handler_fired.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;

    if (failed.load(std::memory_order_relaxed)) {
      handler(0, exc);
    } else {
      handler(total_bytes, nullptr);
    }
  }

  void chunk_failed(std::exception_ptr e)
  {
    bool expected = false;
    if (failed.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
      exc = std::move(e);
    }
    chunk_done();
  }

  io_completion_handler handler;
  std::atomic<size_t> pending{0};
  size_t total_bytes{0};
  std::atomic<bool> failed{false};
  std::exception_ptr exc;
  /// Set (via CAS) by whichever path resolves the handler — normal
  /// completion in chunk_done() or the destructor safety net.
  /// Guarantees the handler fires at most once.
  std::atomic<bool> handler_fired{false};
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

}  // namespace sirius::io
