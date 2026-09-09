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
#include "io/rest/authorizer.hpp"
#include "io/rest/config.hpp"
#include "io/rest/types.hpp"
#include "io/types.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <blockingconcurrentqueue.h>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
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
[[nodiscard]] std::optional<std::size_t> content_range_total(std::string_view content_range);

// ---------------------------------------------------------------------------
// shared_byte_span
// ---------------------------------------------------------------------------

namespace detail {

/// Owns a byte buffer plus a span over it.  Exists so @ref make_shared_byte_span
/// can hand out a shared_ptr to the *span* (via the aliasing constructor) while
/// the shared_ptr's control block keeps the *buffer* alive.  Never held
/// directly by callers.
struct byte_storage {
  std::vector<std::uint8_t> bytes;
  std::span<const std::uint8_t> view;

  // `bytes` is declared first, so it is already initialised when `view` binds
  // to it — the span never sees a moved-from buffer.
  explicit byte_storage(std::vector<std::uint8_t> b) : bytes(std::move(b)), view(bytes) {}

  // Non-copyable, non-movable: `view` points into `bytes`, so copying would
  // deep-copy the buffer and leave the copy's span aimed at the original's
  // allocation.  Only ever built in place by make_shared, so neither is needed.
  byte_storage(byte_storage const&)            = delete;
  byte_storage& operator=(byte_storage const&) = delete;
  byte_storage(byte_storage&&)                 = delete;
  byte_storage& operator=(byte_storage&&)      = delete;
};

}  // namespace detail

/// A shared, immutable view over a byte buffer.
///
/// Deliberately a span rather than a @c vector: consumers only ever read
/// through it (@c data / @c size / @c subspan), so exposing the container type —
/// and with it its allocator, growth policy and mutation API — would leak an
/// implementation detail into the interface.  Ownership still rides along: the
/// shared_ptr is built with the aliasing constructor, so the control block
/// retains the underlying buffer while the pointer itself refers to the span.
using shared_byte_span = std::shared_ptr<const std::span<const std::uint8_t>>;

/// Take ownership of @p bytes and return a @ref shared_byte_span over it.
/// A single allocation: the buffer and its span live in one control block.
[[nodiscard]] shared_byte_span make_shared_byte_span(std::vector<std::uint8_t> bytes);

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
  shared_byte_span bytes;
  // ETag from the verified 206, quotes preserved; empty otherwise.
  std::string etag;
};

/// Result of a blocking HEAD: the object's size plus its ETag when the server
/// sent one (quotes preserved, empty otherwise).
struct head_object_result {
  std::size_t object_size{0};
  std::string etag;
};

// ---------------------------------------------------------------------------
// rest_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c io_object backed by a RESTful object-store key.
 *
 * Passive bag of identity: the original URL/path (also the cache id), the
 * bucket + key the reactor authorizes against, and the object size discovered
 * by a one-time HEAD at construction.  Does no I/O of its own.
 */
class rest_io_object : public io_object {
 public:
  rest_io_object(
    std::string path, std::string bucket, std::string key, size_t size, std::string etag = {})
    : _path(std::move(path)),
      _bucket(std::move(bucket)),
      _key(std::move(key)),
      _file_size(size),
      _etag(std::move(etag))
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
                 shared_byte_span stash,
                 std::string etag = {})
    : _path(std::move(path)),
      _bucket(std::move(bucket)),
      _key(std::move(key)),
      _file_size(object_size),
      _window_lo(window_lo),
      _stash(std::move(stash)),
      _etag(std::move(etag))
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override { return _file_size; }
  [[nodiscard]] std::string_view validation_tag() const noexcept override { return _etag; }

  [[nodiscard]] const std::string& bucket() const noexcept { return _bucket; }
  [[nodiscard]] const std::string& key() const noexcept { return _key; }
  [[nodiscard]] object_ref get_object_ref() const { return object_ref{_bucket, _key}; }

  /// Trailing bytes prefetched at open (a suffix-range footer probe), or null
  /// when the object was opened without one.  A read fully inside
  /// [stash_window_lo, size) is served from here by @c host_read.
  [[nodiscard]] shared_byte_span const& stash() const noexcept { return _stash; }
  [[nodiscard]] size_t stash_window_lo() const noexcept { return _window_lo; }

 private:
  std::string _path;
  std::string _bucket;
  std::string _key;
  size_t _file_size{0};
  size_t _window_lo{0};
  shared_byte_span _stash;
  std::string _etag;
};

// ---------------------------------------------------------------------------
// rest_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Single-threaded I/O reactor for RESTful object storage (s3://...).
 *
 * Owns one worker thread driving a libcurl multi handle over an epoll event
 * loop (curl_multi_socket_action), a pool of reusable easy handles, dynamic
 * CuCascade staging for device reads, a timerfd + min-heap retry
 * scheduler, and an MPSC request queue.  Models the reactor concept consumed
 * by @c templated_ioctx.  Presigned GET/HEAD URLs come from a
 * @c request_authorizer, re-issued on every attempt.
 */
class rest_reactor {
 public:
  /// Every read is an HTTP round trip, so what a request costs is dominated by
  /// the fact that it IS a request.  Given the whole range set at once the
  /// reactor can fuse adjacent ranges and keep every connection busy, which it
  /// cannot do when ranges arrive one at a time as a reader walks the file.
  static constexpr bool prefers_bulk_io = true;

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
                    std::shared_ptr<request_authorizer> authorizer,
                    cucascade::memory::fixed_size_host_memory_resource* host_mr = nullptr)
      : _config(std::move(cfg)), _authorizer(std::move(authorizer)), _host_mr(host_mr)
    {
    }

    [[nodiscard]] const config& cfg() const noexcept { return _config; }
    [[nodiscard]] const std::shared_ptr<request_authorizer>& authorizer() const noexcept
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
    std::shared_ptr<request_authorizer> _authorizer;
    cucascade::memory::fixed_size_host_memory_resource* _host_mr{nullptr};
  };

  using io_object_type       = rest_io_object;
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

  // -- dispatch / lifecycle ------------------------------------------------

  /// Launch the worker thread. Split out of the constructor so a reactor can be
  /// built cheaply and parked until it is actually needed. Idempotent while
  /// running; shutdown is terminal and a later start is ignored.
  void start();

  void enqueue(std::unique_ptr<grouped_io_request> req) noexcept;
  void interrupt();
  void shutdown() noexcept;

  /// Bytes of queued-but-not-yet-submitted work — the reactor's backlog, and the
  /// signal @c rest_ioctx::next_reactor balances dispatch against.  Counts only
  /// what is waiting: a chunk stops counting the moment a connection picks it up,
  /// because in-flight work is already bounded by @c max_connections and is
  /// therefore the same ceiling on every reactor, while the queue is where an
  /// unevenly-loaded pool actually diverges.
  ///
  /// A hint, not a synchronization point: it is read without ordering against
  /// the queue itself, so a concurrent enqueue or dequeue may not be reflected
  /// yet.  Dispatch only needs to be right on average.
  [[nodiscard]] std::size_t queued_bytes() const noexcept
  {
    return _queued_bytes.load(std::memory_order_relaxed);
  }

  /// Synchronous buffered host read (blocking ranged GET).  Blocks the caller.
  size_t host_read(const io_object_type& file, size_t offset, size_t size, uint8_t* dst);

  /// Ask the worker to open its connection pool against @p bucket before any
  /// read needs it.  Returns immediately: the worker does the HEADs on its own
  /// thread at the top of its next pass, because the connection cache it fills
  /// is thread-confined (see the @c curl_share warning) and is reachable from
  /// nowhere else.  Coalescing is the caller's job -- a second call before the
  /// first is serviced simply replaces the target.
  ///
  /// The request is a bucket-scoped @c ListObjectsV2 capped at zero keys, not a
  /// HEAD: a HEAD is signed per object and @c sigv4_authorizer refuses an empty
  /// key, whereas @c authorize_list already signs a bucket-only URI, so this
  /// keeps warm-up traffic off the query's data files without touching the
  /// signing path.  The response is discarded and never inspected -- the
  /// handshake is what is being bought, so even a 403 is a success.
  void warmup(std::string bucket);

  /// Blocking HEAD to discover an object's size and ETag.  Used by the ioctx to
  /// build an @c rest_io_object.  @p bucket / @p key identify the object.
  head_object_result head_object(std::string_view bucket, std::string_view key);

  /// Size-only convenience wrapper around @c head_object.
  size_t head_object_size(std::string_view bucket, std::string_view key);

  /// Blocking suffix-range GET of the last @p n bytes of an object, resolving
  /// the size and stashing the parquet footer in a single round-trip.  On a
  /// well-formed 206 the returned @c footer_probe carries the object size, the
  /// window origin, the trailing bytes, and the ETag; on any unusable response
  /// (200 full body, missing / unsatisfied Content-Range) @c bytes is null so the caller
  /// falls back to a HEAD.  @p bucket / @p key identify the object.
  footer_probe fetch_footer_suffix(std::string_view bucket, std::string_view key, std::size_t n);

  /// Blocking bucket-level ListObjectsV2 GET for one page: returns the raw XML
  /// body on HTTP 200.  @p canonical_query is the pre-encoded, key-sorted
  /// request query (no auth params — authorization is added via
  /// @c authorize_list).  @p prefix is only for retry-log / error text.
  /// Control-plane op: transient failures are retried and WARN-logged.
  std::string list_page(std::string_view bucket,
                        std::string_view prefix,
                        std::string_view canonical_query);

  // -- capabilities / factory ----------------------------------------------

  /// True iff @p path is an s3:// URL this reactor can serve.
  [[nodiscard]] static bool supports(std::string_view path);

  /// Concept stub: real object creation needs a HEAD + authorizer and lives in
  /// @c rest_ioctx::create_io_object.  Always throws.
  static std::unique_ptr<io_object_type> create_io_object(std::string path);

  /// REST has no physical block alignment, so this only coalesces overlapping /
  /// adjacent ranges (honoring a caller-supplied alignment >= 1 as a lower
  /// bound) into a minimal sorted set — fewer ranges means fewer GETs.
  static std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<const cudf::io::text::byte_range_info> ranges,
    std::optional<size_t> alignment = std::nullopt);

 private:
  void worker_loop(const std::stop_token& stop_token);

  // Shared services + tunables for the whole reactor pool; kept alive for this
  // reactor's lifetime (the authorizer is used on every request).
  std::shared_ptr<reactor_context> _ctx;
  config _config;  // copy of _ctx->cfg() for hot-path access
  // Thread name prefix captured at construction; applied to the worker in start().
  std::string _tname;

  // Set by warmup() on a caller thread, consumed by the worker at the top of a
  // pass.  The bucket is guarded because a std::string is not atomically
  // publishable; the flag is what the worker actually polls.
  std::atomic<bool> _warm_requested{false};
  std::mutex _warm_mtx;
  std::string _warm_bucket;

  // Cross-thread wakeup: written by enqueue()/interrupt() to break the worker
  // out of epoll_wait.
  file_descriptor _wakeup_fd;

  std::stop_source _stop_source;
  duckdb_moodycamel::BlockingConcurrentQueue<std::unique_ptr<grouped_io_request>> _requests;
  mutable std::mutex _enqueue_mutex;
  bool _running{false};
  bool _accepting{false};
  bool _stopped{false};

  // Logical bytes not yet assigned to a curl slot. Retries are already claimed
  // work and therefore never get counted a second time.
  std::atomic<std::size_t> _queued_bytes{0};

  std::jthread _worker;
};

}  // namespace sirius::io::rest
