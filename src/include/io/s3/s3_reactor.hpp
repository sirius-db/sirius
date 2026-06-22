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

#include <cuda_runtime.h>

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
#include <vector>

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
/// s3_blocking_ioctx). string_view handles would dangle; a shared_ptr does
/// not.
struct s3_object_state {
  std::string bucket;
  std::string key;
  std::size_t object_size{0};
};

/// The native handle the reactor carries per request.
using s3_native_handle = std::shared_ptr<const s3_object_state>;

/// S3 io_object for the async-curl backend. Distinct from the
/// blocking @c s3_blocking_io_object. Satisfies templated_ioctx's @c io_object_c
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
/// Host reads run over a curl_multi worker loop with bounded submit; sync
/// host_read/HEAD use a separate blocking easy path. A retriable completion
/// (transient curl / 408 / 429 / 5xx, honoring Retry-After) re-arms after a
/// backoff; range responses are validated (Content-Range / overflow); request
/// setup is exception-safe; the connection cache is bounded by max_connections.
/// Device reads stream through a bounded pool of pinned staging blocks: each
/// chunk borrows one block at submit time (so the pinned footprint tracks the
/// active window, not the whole request), the range is GET'd into it, then an
/// async cudaMemcpyAsync H2D copy is completed by a stream callback — no
/// per-chunk cudaStreamSynchronize on the read path.
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

    // Opt-in per-chunk perf instrumentation (ns-level accumulators). Default off:
    // when false the device path takes no steady_clock::now() and no extra
    // atomics (one predicted-not-taken branch per chunk). The retry / terminal
    // counters below are always-on regardless of this flag.
    bool s3_perf_instrumentation{false};
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
  /// building an io_object is @c s3_ioctx::create_io_object
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
  /// Number of completed H2D copies (stream-callback fires; success or failure).
  [[nodiscard]] std::uint64_t device_copies_total() const noexcept
  {
    return _device_copies_total.load(std::memory_order_relaxed);
  }
  /// `cudaStreamSynchronize` calls the reactor issued — 0 across any device read
  /// that completes without shutdown (the observable proxy for "no per-chunk
  /// sync"); only the shutdown drain bumps it.
  [[nodiscard]] std::uint64_t device_stream_sync_total() const noexcept
  {
    return _device_stream_sync_total.load(std::memory_order_relaxed);
  }
  /// High-water mark of staging blocks held concurrently (device GETs in flight
  /// plus H2D copies). Bounded by max_connections — the observable proxy for the
  /// pinned-memory window staying bounded regardless of request size.
  [[nodiscard]] std::uint64_t device_peak_inflight() const noexcept
  {
    return _device_peak_inflight.load(std::memory_order_relaxed);
  }

  // -- Per-chunk perf micro (gated by config.s3_perf_instrumentation) ---------
  // Raw ns totals/counts/max; the consumer computes means (ns_total / count).
  // All zero when instrumentation is off. Populated on the async device path
  // only (footer/HEAD go through the separate blocking path and are not seen).
  [[nodiscard]] std::uint64_t chunk_get_ns_total() const noexcept
  {
    return _chunk_get_ns_total.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t chunk_get_ns_count() const noexcept
  {
    return _chunk_get_ns_count.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t chunk_get_ns_max() const noexcept
  {
    return _chunk_get_ns_max.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t queue_wait_ns_total() const noexcept
  {
    return _queue_wait_ns_total.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t queue_wait_ns_count() const noexcept
  {
    return _queue_wait_ns_count.load(std::memory_order_relaxed);
  }
  /// memcpy issue -> worker-observed cuda_done. NOT a pure GPU copy time: it
  /// includes the CUDA-callback + wakeup + poll observation latency.
  [[nodiscard]] std::uint64_t h2d_observed_ns_total() const noexcept
  {
    return _h2d_observed_ns_total.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t h2d_observed_ns_count() const noexcept
  {
    return _h2d_observed_ns_count.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t h2d_observed_ns_max() const noexcept
  {
    return _h2d_observed_ns_max.load(std::memory_order_relaxed);
  }
  /// First GET submit -> first GET completion of the reactor's lifetime
  /// (excludes queue-wait by construction).
  [[nodiscard]] std::uint64_t ttfb_ns() const noexcept
  {
    return _ttfb_ns.load(std::memory_order_relaxed);
  }

  // -- Retry / failure counters (always-on, off-happy-path) -------------------
  [[nodiscard]] std::uint64_t retries_total() const noexcept
  {
    return _retries_total.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t terminal_failures_total() const noexcept
  {
    return _terminal_failures_total.load(std::memory_order_relaxed);
  }

 private:
  struct transfer;  // per-async-request state (defined in the .cpp)

  void worker_loop();
  void drain_incoming();
  void drain_incoming_device();  // borrows staging, builds device GET transfers
  void promote_due_retries();
  void submit_pending();  // bounded by max_connections
  void build_and_add(transfer* t);
  void schedule_retry(transfer* t, std::chrono::steady_clock::duration delay);
  void cleanup_easy(transfer* t);  // remove from multi + free easy/slist (no completion)
  /// Staging blocks held right now: device GETs in _inflight + H2D copies.
  [[nodiscard]] std::size_t device_blocks_in_flight() const;
  [[nodiscard]] bool has_device_in_flight() const  // a device GET/copy will free a block
  {
    return device_blocks_in_flight() > 0;
  }
  void start_device_copy(transfer* t);              // cudaMemcpyAsync H2D + stream callback
  void poll_device_copies();                        // reap callback-signaled copies -> chunk_done
  void finish(transfer* t, std::exception_ptr ep);  // chunk_done/chunk_failed + cleanup
  void cancel_all_on_shutdown();

  /// Stream callback (userdata = transfer*): records status, flags done, wakes
  /// the worker. Mirrors uring_reactor's cuda_copy_cb — fires even on error.
  static void CUDART_CB cuda_copy_done(cudaStream_t stream, cudaError_t status, void* userdata);

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
  std::deque<host_read_req_type> _incoming;           // newly enqueued host reqs (API side)
  std::deque<device_read_req_type> _incoming_device;  // newly enqueued device chunks (API side)
  std::deque<transfer*> _pending;                     // wrapped, awaiting bounded submit
  std::multimap<std::chrono::steady_clock::time_point, transfer*> _retry_queue;
  std::unordered_map<void*, transfer*> _inflight;  // easy handle -> transfer (GET in flight)
  std::vector<transfer*> _copying;                 // device transfers in the H2D-copy phase

  std::atomic<std::uint64_t> _bytes_read_total{0};
  std::atomic<std::uint64_t> _fsmr_borrows_total{0};
  std::atomic<std::uint64_t> _device_copies_total{0};
  std::atomic<std::uint64_t> _device_peak_inflight{0};
  std::atomic<std::uint64_t> _device_stream_sync_total{0};

  // Gated per-chunk perf micro (written by the worker thread only).
  std::atomic<std::uint64_t> _chunk_get_ns_total{0};
  std::atomic<std::uint64_t> _chunk_get_ns_count{0};
  std::atomic<std::uint64_t> _chunk_get_ns_max{0};
  std::atomic<std::uint64_t> _queue_wait_ns_total{0};
  std::atomic<std::uint64_t> _queue_wait_ns_count{0};
  std::atomic<std::uint64_t> _h2d_observed_ns_total{0};
  std::atomic<std::uint64_t> _h2d_observed_ns_count{0};
  std::atomic<std::uint64_t> _h2d_observed_ns_max{0};
  std::atomic<std::uint64_t> _ttfb_ns{0};
  // TTFB start point: the first GET submit of the reactor's lifetime. Worker-only.
  std::chrono::steady_clock::time_point _ttfb_first_submit{};
  bool _ttfb_first_submit_set{false};

  // Always-on retry / failure counters.
  std::atomic<std::uint64_t> _retries_total{0};
  std::atomic<std::uint64_t> _terminal_failures_total{0};
};

}  // namespace sirius::io::s3
