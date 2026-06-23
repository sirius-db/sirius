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
#include "io/rest/types.hpp"
#include "io/s3/s3_object_ref.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/types.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <blockingconcurrentqueue.h>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

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

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override { return _file_size; }

  [[nodiscard]] const std::string& bucket() const noexcept { return _bucket; }
  [[nodiscard]] const std::string& key() const noexcept { return _key; }
  [[nodiscard]] s3::s3_object_ref object_ref() const { return s3::s3_object_ref{_bucket, _key}; }

 private:
  std::string _path;
  std::string _bucket;
  std::string _key;
  size_t _file_size{0};
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
  struct config {
    /// Whole-request timeout (seconds, 0 = no limit) and presigned-URL TTL.
    long request_timeout_s{30};

    /// TLS: optional CA bundle path; when @c tls_verify is false, peer/host
    /// verification is disabled (self-signed dev endpoints / MinIO).
    std::string ca_bundle_path;
    bool tls_verify{true};

    /// Max concurrent in-flight easy handles per reactor.
    std::size_t max_connections{16};

    /// Target maximum bytes per ranged GET for the vector / device-staging
    /// paths: file-adjacent segments are fused into one scatter GET up to this
    /// size, and an oversized segment is split into ceil(size / chunk_size)
    /// pieces.  A single contiguous host read instead splits by
    /// @c max_read_split (see prep_host_rx_request).
    std::size_t chunk_size{8UL << 20};

    /// Cap on destination buffers fused into a single scatter GET (i.e. how
    /// many file-adjacent segments may merge into one request).
    std::size_t max_n_chunks{16};

    /// How many parallel ranged GETs a single contiguous host read is broken
    /// into (@c prep_host_rx_request).  The split picks the largest chunk count
    /// <= max_read_split that keeps every piece at least 1 MiB; a read smaller
    /// than 2 MiB stays a single GET.
    std::size_t max_read_split{16};

    /// Bounce-slot size (bytes) for the reactor-staged device path, cached from
    /// the staging resource's block size by @c rest_ioctx.  Zero disables the
    /// reactor-staged device read (the static @c prep_device_rx_request needs
    /// this size without access to the live resource, which lives on the
    /// @c reactor_context).
    std::size_t bounce_block_size{0};

    /// Idle-connection keepalive.  While the reactor is idle, every
    /// @c upkeep_interval the worker calls @c curl_easy_upkeep on its pooled
    /// connections, which sends an HTTP/2 PING on any connection idle at least
    /// this long — keeping the endpoint from idle-closing it (and detecting
    /// dead ones).  No effect on HTTP/1.1 (TCP keepalive covers that).  Zero
    /// disables upkeep.
    std::chrono::milliseconds upkeep_interval{std::chrono::seconds{15}};

    /// How long curl may reuse a pooled connection before discarding it
    /// (CURLOPT_MAXAGE_CONN).  Pairs with @c upkeep_interval: upkeep keeps idle
    /// connections warm, so keep this within the endpoint's idle timeout so a
    /// reused connection is not one the server already closed.  Zero leaves
    /// curl's default.
    std::chrono::seconds conn_max_age{std::chrono::seconds{20}};

    // -- retry policy ------------------------------------------------------
    std::size_t max_retry_attempts{10};
    /// Bounded retries for an HTTP 403.  A presigned URL that expired while the
    /// request waited in the queue comes back as 403; since every attempt
    /// re-authorizes (a fresh presigned URL), a small number of retries can
    /// recover from expiry.  Kept low so a genuine AccessDenied fails fast.
    std::size_t max_auth_retry_attempts{3};
    std::chrono::milliseconds retry_backoff_base{50};
    std::chrono::milliseconds retry_jitter{50};
    bool honor_retry_after{true};
  };

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

  // -- request preparation (static: build chunk descriptions) --------------

  static request_type_ptr prep_host_rx_request(const reactor_config_type& cfg,
                                               const io_object_type& file,
                                               const io_object_segment& segment);

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
  std::jthread _worker;
};

}  // namespace sirius::io::rest
