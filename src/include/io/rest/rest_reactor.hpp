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

#include "io/cache/types.hpp"
#include "io/rest/config.hpp"
#include "io/rest/types.hpp"
#include "io/s3/s3_object_ref.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/types.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <blockingconcurrentqueue.h>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace sirius::io::rest {

/// Parse the total object length out of a Content-Range value of the form
/// "bytes <first>-<last>/<total>".  Returns nullopt when the unit is not
/// "bytes", the range is unsatisfied ("bytes */..."), or the total is unknown
/// ("*") — i.e. any response the footer probe cannot trust.
[[nodiscard]] std::optional<std::size_t> content_range_total(std::string const& content_range);

// ---------------------------------------------------------------------------
// footer_probe
// ---------------------------------------------------------------------------

/// Result of a suffix-range footer probe: the object's total size plus the
/// trailing window [window_lo, object_size) captured in @c bytes.  @c bytes is
/// null when the probe could not be satisfied (the caller then falls back to a
/// HEAD).  Held by shared_ptr so the trailing bytes are shared, not copied, with
/// the io_object that carries them for this open.
struct footer_probe {
  std::size_t object_size{0};
  std::size_t window_lo{0};
  std::shared_ptr<const std::vector<std::uint8_t>> bytes;
};

// ---------------------------------------------------------------------------
// rest_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c sirius_io_object backed by a RESTful object-store key.
 *
 * Passive bag of identity: the original URL/path (also the cache id), the
 * bucket + key the reactor authorizes against, and the object size discovered
 * by a one-time HEAD at construction.  Does no I/O of its own.
 */
class rest_io_object : public sirius_io_object {
 public:
  rest_io_object(std::string path, std::string bucket, std::string key, size_t size)
    : _path(std::move(path)), _bucket(std::move(bucket)), _key(std::move(key)), _file_size(size)
  {
  }

  /// As above, but carrying a suffix-range footer stash: @p stash holds the
  /// object's bytes over [window_lo, object_size), so @c rest_reactor::host_read
  /// serves any read fully inside that window from memory instead of a GET.
  rest_io_object(std::string path,
                 std::string bucket,
                 std::string key,
                 size_t object_size,
                 size_t window_lo,
                 std::shared_ptr<const std::vector<std::uint8_t>> stash)
    : _path(std::move(path)),
      _bucket(std::move(bucket)),
      _key(std::move(key)),
      _file_size(object_size),
      _window_lo(window_lo),
      _stash(std::move(stash))
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override { return _file_size; }

  [[nodiscard]] const std::string& bucket() const noexcept { return _bucket; }
  [[nodiscard]] const std::string& key() const noexcept { return _key; }
  [[nodiscard]] s3::s3_object_ref object_ref() const { return s3::s3_object_ref{_bucket, _key}; }

  /// Trailing bytes prefetched at open (a suffix-range footer probe), or null
  /// when the object was opened without one.  A read fully inside
  /// [stash_window_lo, size) is served from here by @c host_read.
  [[nodiscard]] const std::shared_ptr<const std::vector<std::uint8_t>>& stash() const noexcept
  {
    return _stash;
  }
  [[nodiscard]] size_t stash_window_lo() const noexcept { return _window_lo; }

 private:
  std::string _path;
  std::string _bucket;
  std::string _key;
  size_t _file_size{0};
  size_t _window_lo{0};
  std::shared_ptr<const std::vector<std::uint8_t>> _stash;
};

// ---------------------------------------------------------------------------
// rest_perf_snapshot
// ---------------------------------------------------------------------------

/// Plain-value perf counters read out of a reactor, or summed across the pool
/// by @c rest_ioctx.  The ns totals/maxes and ttfb stay 0 unless the reactor's
/// @c perf_instrumentation is on; retry / terminal / device-stream-sync and
/// payload-bytes counts are populated regardless.
struct rest_perf_snapshot {
  std::uint64_t chunk_get_ns_total{0};
  std::uint64_t chunk_get_count{0};
  std::uint64_t chunk_get_ns_max{0};
  std::uint64_t queue_wait_ns_total{0};
  std::uint64_t queue_wait_count{0};
  std::uint64_t ttfb_ns{0};
  std::uint64_t h2d_observed_ns_total{0};
  std::uint64_t h2d_observed_count{0};
  std::uint64_t h2d_observed_ns_max{0};
  std::uint64_t retries_total{0};
  std::uint64_t terminal_failures_total{0};
  std::uint64_t device_stream_sync_total{0};
  // Always-on: HTTP response *body* bytes received (sink.total_received), summed
  // over every completed curl attempt incl. retries / partial / failed bodies.
  // Not TLS/header/TCP-frame bytes — this is the S3-scan payload byte budget.
  std::uint64_t payload_bytes_read_total{0};
  // perf_instrumentation-gated. Blocking host GETs remain part of chunk_get_*
  // and are also attributed to blocking_host_get_*. Stash hits issue no GET and
  // increment neither.
  std::uint64_t blocking_host_get_count{0};
  std::uint64_t blocking_host_get_wall_ns_total{0};
  std::uint64_t blocking_host_get_wall_ns_max{0};
};

// ---------------------------------------------------------------------------
// rest_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Single-threaded I/O reactor for RESTful object storage (s3://...).
 *
 * Owns one worker thread driving a libcurl multi handle over an epoll event
 * loop (curl_multi_socket_action), a pool of reusable easy handles, optional
 * pinned bounce slots for device staging, a timerfd + min-heap retry
 * scheduler, and an MPSC request queue.  Models the reactor concept consumed
 * by @c templated_ioctx.  Presigned GET/HEAD URLs come from a
 * @c s3_request_authorizer, re-issued on every attempt.
 */
class rest_reactor {
 public:
  /// Shared, immutable services for a pool of reactors.  One instance is built
  /// by @c rest_ioctx and shared (via shared_ptr) across every reactor in the
  /// pool, so it is the natural home for things that are shared rather than
  /// per-reactor: the presigning @c authorizer, the pinned bounce-staging
  /// resource, and — in future — a shared connection pool, a registered-buffer
  /// table, etc.  It also carries the primitive @c config (separating the
  /// injected collaborators from the plain, file-settable tunables).
  class reactor_context {
   public:
    reactor_context(config cfg,
                    std::shared_ptr<s3::s3_request_authorizer> authorizer,
                    cucascade::memory::fixed_size_host_memory_resource* host_mr = nullptr)
      : _config(std::move(cfg)), _authorizer(std::move(authorizer)), _host_mr(host_mr)
    {
    }

    [[nodiscard]] const config& cfg() const noexcept { return _config; }
    [[nodiscard]] const std::shared_ptr<s3::s3_request_authorizer>& authorizer() const noexcept
    {
      return _authorizer;
    }
    [[nodiscard]] cucascade::memory::fixed_size_host_memory_resource* host_memory_resource()
      const noexcept
    {
      return _host_mr;
    }

   private:
    config _config;
    std::shared_ptr<s3::s3_request_authorizer> _authorizer;
    cucascade::memory::fixed_size_host_memory_resource* _host_mr{nullptr};
  };

  using io_object_type       = rest_io_object;
  using request_type         = rest_rx_request;
  using request_type_ptr     = std::unique_ptr<rest_rx_request>;
  using reactor_config_type  = config;
  using reactor_context_type = reactor_context;

  explicit rest_reactor(std::shared_ptr<reactor_context> ctx,
                        std::string_view tname = "rest_reactor");
  ~rest_reactor();

  rest_reactor(rest_reactor const&)            = delete;
  rest_reactor& operator=(rest_reactor const&) = delete;

  /// The reactor's effective config (copied from its context at construction and
  /// clamped to legal values).  templated_ioctx reads its own _config from here
  /// so the config lives in one place — the context — rather than being passed
  /// in separately.
  [[nodiscard]] const reactor_config_type& get_config() const noexcept { return _config; }

  // -- request preparation (static: build chunk descriptions) --------------

  static request_type_ptr prep_host_rx_request(const reactor_config_type& cfg,
                                               const io_object_type& file,
                                               const io_object_segment& segment,
                                               bool perf_blocking_host_get = false);

  static request_type_ptr prep_host_rxv_request(const reactor_config_type& cfg,
                                                const io_object_type& file,
                                                std::span<io_object_segment> segments);

  static request_type_ptr prep_device_rx_request(const reactor_config_type& cfg,
                                                 const io_object_type& file,
                                                 uint8_t* dst,
                                                 size_t offset,
                                                 size_t size,
                                                 rmm::cuda_stream_view stream,
                                                 int device_id);

  static request_type_ptr prep_host_to_device_rx_request(const reactor_config_type& cfg,
                                                         const io_object_type& file,
                                                         std::span<io_object_segment> bounce,
                                                         uint8_t* dst,
                                                         size_t offset,
                                                         size_t size,
                                                         rmm::cuda_stream_view stream,
                                                         int device_id);

  // -- dispatch / lifecycle ------------------------------------------------

  /// Allocate the pinned bounce slots and launch the worker thread.  Split out
  /// of the constructor so a reactor can be built cheaply (it only copies its
  /// config and creates its wakeup fd) and parked until it is actually needed —
  /// see @c sirius_ioctx::start.  Idempotent: a second call (while the worker is
  /// already running) is a no-op.
  void start();

  void enqueue(request_type_ptr req);
  void interrupt();
  void shutdown();

  /// Synchronous buffered host read (blocking ranged GET).  Blocks the caller.
  size_t host_read(const io_object_type& file, size_t offset, size_t size, uint8_t* dst);

  /// Blocking HEAD to discover an object's size.  Used by the ioctx to build
  /// an @c rest_io_object.  @p bucket / @p key identify the object.
  size_t head_object_size(std::string_view bucket, std::string_view key);

  /// Blocking suffix-range GET of the last @p n bytes of an object, resolving
  /// the size and stashing the parquet footer in a single round-trip.  On a
  /// well-formed 206 the returned @c footer_probe carries the object size, the
  /// window origin, and the trailing bytes; on any unusable response (200 full
  /// body, missing / unsatisfied Content-Range) @c bytes is null so the caller
  /// falls back to a HEAD.  @p bucket / @p key identify the object.
  footer_probe fetch_footer_suffix(std::string_view bucket, std::string_view key, std::size_t n);

  /// Blocking bucket-level ListObjectsV2 GET for one page: returns the raw XML
  /// body on HTTP 200.  @p canonical_query is the pre-encoded, key-sorted
  /// request query (no auth params — authorization is added via
  /// @c authorize_list).  @p prefix is only for retry-log / error text.
  /// Control-plane op: retries/terminals are counted (and retries WARN-logged)
  /// like every retry loop here, but the XML body never touches the
  /// chunk-GET / payload byte counters.
  std::string list_page(std::string_view bucket,
                        std::string_view prefix,
                        std::string_view canonical_query);

  /// Snapshot of this reactor's perf counters.  Lock-free (relaxed atomic
  /// loads); safe to call while the reactor is running.
  [[nodiscard]] rest_perf_snapshot perf_snapshot() const noexcept;

  // -- capabilities / factory ----------------------------------------------

  /// True iff @p path is an s3:// URL this reactor can serve.
  [[nodiscard]] static bool supports(std::string_view path);

  /// Concept stub: real object creation needs a HEAD + authorizer and lives in
  /// @c rest_ioctx::create_io_object.  Always throws.
  static std::unique_ptr<io_object_type> create_io_object(std::string path);

  static constexpr cache::prefetching_stage preferred_prefetching_stage() noexcept
  {
    // Network round-trips are high-latency; read ahead on demand rather than
    // eagerly prefilling the whole working set.
    return cache::prefetching_stage::just_in_time;
  }

  /// REST has no physical block alignment, so this only coalesces overlapping /
  /// adjacent ranges (honoring a caller-supplied alignment >= 1 as a lower
  /// bound) into a minimal sorted set — fewer ranges means fewer GETs.
  static std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<const cudf::io::text::byte_range_info> ranges,
    std::optional<size_t> alignment = std::nullopt);

 private:
  void worker_loop(const std::stop_token& stop_token);

  /// Enqueue a batch of chunks with a single wake notification.
  void enqueue_chunks(std::span<std::unique_ptr<rest_chunked_rx_request>> batch);

  // Shared services + tunables for the whole reactor pool; kept alive for this
  // reactor's lifetime (the authorizer is used on every request).
  std::shared_ptr<reactor_context> _ctx;
  config _config;  // copy of _ctx->cfg() for hot-path access
  // Thread name prefix captured at construction; applied to the worker in start().
  std::string _tname;
  std::size_t _bounce_slot_size{0};

  // Keeps the bounce-slot blocks alive for the reactor's lifetime; the
  // allocation handle returns the blocks to the upstream resource when the
  // reactor is destroyed.  Null when no host_memory_resource is set.
  cucascade::memory::fixed_multiple_blocks_allocation _bounce_storage;

  // Cross-thread wakeup: written by enqueue()/interrupt() and the CUDA
  // copy-completion callback to break the worker out of epoll_wait.
  file_descriptor _wakeup_fd;

  std::stop_source _stop_source;
  duckdb_moodycamel::BlockingConcurrentQueue<std::unique_ptr<rest_chunked_rx_request>> _requests;

  // Instrumentation counters, owned by the reactor (not worker_loop locals) so
  // rest_ioctx can read them cross-thread.  Micro timings are stamped only under
  // perf_instrumentation; retries/terminal/device_stream_sync/payload_bytes are
  // always-on.
  struct perf_counters {
    std::atomic<std::uint64_t> chunk_get_ns_total{0};
    std::atomic<std::uint64_t> chunk_get_count{0};
    std::atomic<std::uint64_t> chunk_get_ns_max{0};
    std::atomic<std::uint64_t> queue_wait_ns_total{0};
    std::atomic<std::uint64_t> queue_wait_count{0};
    std::atomic<std::uint64_t> ttfb_ns{0};
    std::atomic<std::uint64_t> h2d_observed_ns_total{0};
    std::atomic<std::uint64_t> h2d_observed_count{0};
    std::atomic<std::uint64_t> h2d_observed_ns_max{0};
    std::atomic<std::uint64_t> retries_total{0};
    std::atomic<std::uint64_t> terminal_failures_total{0};
    std::atomic<std::uint64_t> device_stream_sync_total{0};
    std::atomic<std::uint64_t> payload_bytes_read_total{0};
    std::atomic<std::uint64_t> blocking_host_get_count{0};
    std::atomic<std::uint64_t> blocking_host_get_wall_ns_total{0};
    std::atomic<std::uint64_t> blocking_host_get_wall_ns_max{0};
  };
  perf_counters _perf;

  std::jthread _worker;
};

}  // namespace sirius::io::rest
