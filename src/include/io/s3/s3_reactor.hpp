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

#include "io/io_context.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/types.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

namespace cucascade::memory {
class fixed_size_host_memory_resource;
}  // namespace cucascade::memory

namespace sirius::io::s3 {

// ---------------------------------------------------------------------------
// s3_object_state / s3_async_io_object
// ---------------------------------------------------------------------------

/// Immutable per-object descriptor. Carried by value (shared_ptr) inside every
/// host/device read request so the object outlives the async transfer — the
/// templated_ioctx async path copies the native handle into each request and
/// does NOT capture the io_object's shared_ptr (unlike the synchronous
/// production s3_ioctx). string_view handles would dangle; a shared_ptr does
/// not.
struct s3_object_state {
  std::string bucket;
  std::string key;
  std::size_t object_size{0};
};

/// The native handle the reactor carries per request.
using s3_native_handle = std::shared_ptr<const s3_object_state>;

/// Experimental S3 io_object for the async-curl backend. Distinct from the
/// production @c s3_io_object. Satisfies templated_ioctx's @c io_object_c
/// (host_handle/device_handle returning @c s3_native_handle).
class s3_async_io_object : public sirius_io_object {
 public:
  s3_async_io_object(std::string uri, std::shared_ptr<const s3_object_state> state)
    : _uri(std::move(uri)), _state(std::move(state))
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _uri; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _uri; }
  [[nodiscard]] std::size_t size() const noexcept override { return _state->object_size; }

  [[nodiscard]] s3_native_handle host_handle() const noexcept { return _state; }
  [[nodiscard]] s3_native_handle device_handle() const noexcept { return _state; }

 private:
  std::string _uri;
  std::shared_ptr<const s3_object_state> _state;
};

// ---------------------------------------------------------------------------
// s3_reactor
// ---------------------------------------------------------------------------

/// Single-threaded async-S3 reactor over libcurl's multi interface. Models the
/// reactor concept consumed by @c templated_ioctx.
///
/// Phase 0: host reads over a curl_multi worker loop with bounded submit; sync
/// host_read/HEAD on a separate blocking easy path. Phase 1 adds: an async
/// retry state machine (transient curl / 408 / 429 / 5xx, honoring Retry-After),
/// a sync retry loop, range-response validation (Content-Range / overflow),
/// exception-safe transfer setup, and connection-cache limits. Device reads land
/// in Phase 2.
class s3_reactor {
 public:
  struct config {
    std::shared_ptr<s3_request_authorizer> creds;
    long request_timeout_s{20};
    std::string ca_bundle_path;
    bool tls_verify{true};
    std::size_t max_connections{4};
    cucascade::memory::fixed_size_host_memory_resource* host_memory_resource{nullptr};

    // Retry knobs (Phase 1). Defaults keep tests fast.
    std::size_t max_retry_attempts{4};                 // total attempts incl. the first
    std::chrono::milliseconds retry_backoff_base{50};  // exponential base
    std::chrono::milliseconds retry_jitter{20};        // +/- bound
    bool honor_retry_after{true};
  };

  using native_handle_type   = s3_native_handle;
  using io_object_type       = s3_async_io_object;
  using device_read_req_type = device_read_req<native_handle_type>;
  using host_read_req_type   = host_read_req<native_handle_type>;

  explicit s3_reactor(config cfg);
  ~s3_reactor();

  s3_reactor(s3_reactor const&)            = delete;
  s3_reactor& operator=(s3_reactor const&) = delete;

  // -- io_reactor_c surface --------------------------------------------------

  std::size_t host_read(native_handle_type handle,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* dst);

  void host_read_async(host_read_req_type req);
  void host_enqueue_bulk(std::span<host_read_req_type> batch);

  /// Phase 2.
  void enqueue_bulk(std::span<device_read_req_type> batch);

  void interrupt();
  void shutdown();

  /// HTTP has no physical alignment requirement; clip to the file size.
  static cudf::io::text::byte_range_info align_to_physical(cudf::io::text::byte_range_info logical,
                                                           std::size_t file_size);

  [[nodiscard]] static bool supports(std::string_view path);

  /// Present only to satisfy the concept's compile — the sole legal entry for
  /// building an io_object is @c s3_async_experimental_ioctx::create_io_object
  /// (it needs an instance HEAD via the authorizer). Always throws.
  static std::unique_ptr<s3_async_io_object> create_io_object(std::string path);

  [[nodiscard]] static std::size_t size(native_handle_type handle) noexcept
  {
    return handle->object_size;
  }

  // -- instance helpers used by the ioctx ------------------------------------

  std::size_t head_object_size(std::string_view bucket, std::string_view key);

  [[nodiscard]] std::uint64_t bytes_read_total() const noexcept
  {
    return _bytes_read_total.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t fsmr_borrows_total() const noexcept
  {
    return _fsmr_borrows_total.load(std::memory_order_relaxed);
  }

 private:
  struct transfer;  // per-async-request state (defined in the .cpp)

  void worker_loop();
  void drain_incoming();
  void promote_due_retries();
  void submit_pending();  // bounded by max_connections
  void build_and_add(transfer* t);
  void schedule_retry(transfer* t, std::chrono::steady_clock::duration delay);
  void finish(transfer* t, std::exception_ptr ep);  // chunk_done/chunk_failed + cleanup
  void cancel_all_on_shutdown();

  std::chrono::steady_clock::duration backoff_delay(
    std::size_t attempt, std::optional<std::chrono::milliseconds> retry_after) const;

  /// Blocking GET/HEAD shared by host_read + head_object_size, with the sync
  /// retry loop. dst may be null for a HEAD (size resolved via out_object_size).
  std::size_t blocking_request(std::string_view bucket,
                               std::string_view key,
                               s3_request_method method,
                               std::size_t offset,
                               std::size_t size,
                               std::uint8_t* dst,
                               std::size_t* out_object_size);

  config _cfg;

  void* _multi{nullptr};  // CURLM*
  std::thread _worker;
  std::atomic<bool> _stop{false};

  std::mutex _mtx;
  std::deque<host_read_req_type> _incoming;  // newly enqueued reqs (API side)
  std::deque<transfer*> _pending;            // wrapped, awaiting bounded submit
  std::multimap<std::chrono::steady_clock::time_point, transfer*> _retry_queue;
  std::unordered_map<void*, transfer*> _inflight;  // easy handle -> transfer

  std::atomic<std::uint64_t> _bytes_read_total{0};
  std::atomic<std::uint64_t> _fsmr_borrows_total{0};
};

}  // namespace sirius::io::s3
