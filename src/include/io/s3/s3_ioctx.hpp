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

#include "io/io_context.hpp"
#include "io/s3/credential_provider.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::exec {
class static_thread_pool;
}  // namespace sirius::exec

namespace sirius::io::s3 {

class s3_io_object;

/// Construction config for @c s3_ioctx.
///
/// @c creds is required and owns the credential / signer chain; the reactor
/// itself is signing-blind — every request path obtains a fully-qualified
/// presigned URL from @c creds and issues an unsigned HTTP request against
/// it. Endpoint / region / access-key material lives inside the provider's
/// concrete implementation (e.g. @c sirius_sigv4_credential_provider's
/// constructor takes the endpoint and region), so they intentionally do not
/// appear here.
struct s3_ioctx_config {
  std::shared_ptr<credential_provider> creds;
  std::size_t max_connections = 16;
  long request_timeout_s      = 60;

  /// Optional caller-owned thread pool for async paths. When non-null,
  /// @c host_read_async_io / @c host_read_ranges_async_io /
  /// @c device_read_async_io schedule work onto this pool's threads. When
  /// nullptr (default), async paths fall back to per-request
  /// @c std::thread().detach() — kept for standalone-test scenarios where
  /// no scan_manager / SiriusContext owns a pool. The s3_ioctx never owns
  /// or shuts down this pool; lifetime is the caller's responsibility.
  exec::static_thread_pool* async_thread_pool{nullptr};

  /// Maximum total attempts for retriable HTTP / libcurl failures during
  /// @c range_get and @c head_object_size. Includes the first attempt
  /// (so @c max_retry_attempts=1 means no retries). Non-retriable errors
  /// (4xx except 408/429, @c SignatureDoesNotMatch, @c NoSuchKey, etc.)
  /// surface immediately regardless of this value.
  std::size_t max_retry_attempts = 3;

  /// Base backoff between attempts; the wait before attempt N (0-indexed)
  /// is @c base * 2^N + uniform[0, jitter]. @c base=0 disables both the
  /// computed backoff and the jitter so retries fire as fast as the
  /// reactor allows.
  std::chrono::milliseconds retry_backoff_base = std::chrono::milliseconds{100};

  /// Maximum random jitter added on top of the exponential backoff.
  std::chrono::milliseconds retry_jitter = std::chrono::milliseconds{50};

  /// Honor server-supplied @c Retry-After header (HTTP 429 / 503) when
  /// present and well-formed; the parsed value overrides the computed
  /// backoff for that retry, capped at 30 seconds. Disable for
  /// deterministic tests.
  bool honor_retry_after = true;
};

/**
 * @brief S3 @c sirius_ioctx implemented with libcurl HTTP Range GETs over
 *        presigned URLs.
 *
 * Implements the @c sirius_ioctx contract from @c io/io_context.hpp: backend
 * reads are libcurl range GETs; device reads bounce through a host staging
 * buffer + H2D copy (S3 has no native device path). The caching / admission
 * hooks on the base class are opt-in via @c initialize_cache.
 *
 * Authentication is delegated entirely to the @c credential_provider passed
 * via @c s3_ioctx_config — this class never sees raw access keys, never
 * computes a SigV4 signature, and never injects an Authorization header.
 * Each request path acquires a presigned URL via
 * @c credential_provider::get_presigned_url and lets libcurl follow it; the
 * only request header this class emits is @c Range (unsigned, allowed by
 * the URL's @c SignedHeaders=host).
 *
 * @par Construction
 *
 * @code
 *   sirius::io::s3::static_credentials creds;
 *   creds.access_key_id     = "...";
 *   creds.secret_access_key = "...";
 *   auto provider = std::make_shared<
 *     sirius::io::s3::sirius_sigv4_credential_provider>(
 *       std::move(creds), "us-east-1", "https://s3.amazonaws.com",
 *       std::chrono::minutes{5});
 *
 *   sirius::io::s3::s3_ioctx_config scfg{};
 *   scfg.creds            = std::move(provider);
 *   scfg.max_connections  = 16;
 *   scfg.request_timeout_s = 60;
 *   auto s3 = std::make_shared<sirius::io::s3::s3_ioctx>(std::move(scfg));
 *
 *   // (Optional) wire the prefetching cache from a host buffer pool.
 *   s3->initialize_cache(host_buffer_pool, 2048);
 *
 *   // Use directly:
 *   auto ds = s3->open_datasource("s3://bucket/key.parquet");
 * @endcode
 *
 * Sirius-runtime wiring (placing this @c s3_ioctx on @c SiriusContext or
 * @c sirius_scan_manager, dispatching by @c supports(path) across multiple
 * backends, etc.) is the integration owner's responsibility and is not
 * shown here.
 *
 * @par Shutdown
 *
 * @c shutdown() closes the libcurl handle pool and prevents new handle
 * acquisition; it does *not* join already-detached async workers. Those
 * workers stay alive on their own through @c shared_from_this() captures
 * and finish whatever request they had in flight before returning.
 *
 * @par Resources NOT yet injected
 *
 * - **Thread pool**. Async paths (@c host_read_async_io,
 *   @c host_read_ranges_async_io, @c device_read_async_io) currently spawn
 *   a per-request @c std::thread().detach(); the worker captures
 *   @c shared_from_this() and @c obj.shared_from_this() so the ioctx +
 *   io_object stay alive through the detached worker. A future enhancement
 *   could inject a shared @c bounded_thread_pool — see the follow-up PR
 *   (thread-pool injection + S3 HTTP retry).
 * - **CUDA stream**. Caller-supplied per request on the device-read path.
 */
class s3_ioctx final : public sirius_ioctx {
 public:
  explicit s3_ioctx(s3_ioctx_config config);
  ~s3_ioctx() override;

  s3_ioctx(s3_ioctx const&)            = delete;
  s3_ioctx& operator=(s3_ioctx const&) = delete;

  void shutdown() override;

  std::unique_ptr<cudf::io::datasource> make_datasource(
    std::shared_ptr<sirius_io_object> io_object) override;

  /// Backend factory: parses @p path as @c s3://bucket/key, issues a HEAD
  /// request via @c head_object_size, and constructs an @c s3_io_object
  /// carrying the size + original path string. Throws
  /// @c std::invalid_argument on a non-S3 scheme, on empty bucket or key,
  /// and on malformed URI components (see @c sirius::io::parse for the
  /// strict-leading-slash semantics). Throws @c std::runtime_error when the
  /// HEAD request fails (e.g. 404 NoSuchKey, 403 AccessDenied,
  /// connectivity issues).
  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override;

  /// Capability check: returns @c true when @p path begins with a
  /// case-insensitive @c "s3://" prefix. No network call; no exceptions —
  /// rejection mode for empty / non-S3 paths is a @c false return, not a
  /// throw. Backends are expected to validate scheme membership cheaply
  /// here so multi-ioctx dispatch can find the right backend via
  /// @c find_if without paying HEAD-request cost on the rejected ones.
  [[nodiscard]] bool supports(std::string_view path) const override;

  /// HEAD request helper: issues HEAD against the bucket/key and returns the
  /// object size. Kept public for callers (typically @c create_io_object,
  /// but also exposed for ad-hoc reachability checks) that want the size
  /// without constructing an @c s3_io_object.
  std::size_t head_object_size(std::string_view bucket, std::string_view key);

  // -- Host reads -----------------------------------------------------------

  std::size_t host_read_io(sirius_io_object& obj,
                           std::size_t offset,
                           std::size_t size,
                           std::uint8_t* dst) override;

  void host_read_async_io(sirius_io_object& obj,
                          std::size_t offset,
                          std::size_t size,
                          std::uint8_t* dst,
                          io_completion_handler handler) override;

  /// Async multi-range host read. The implementation copies the @p dst span's
  /// descriptor array into owned storage before launching the async worker,
  /// so callers may drop the source container immediately after this returns.
  /// The byte buffers each @c host_span points at remain caller-owned and
  /// must outlive the completion handler — same contract as the
  /// @c uint8_t* @c dst in @c host_read_async_io.
  ///
  /// Each @c ranges[i] is clipped to
  /// @c min(ranges[i].size(), obj.size() - ranges[i].offset()) before
  /// validation; the @c dst[i].size() check is against the clipped size, so
  /// an EOF-crossing range with a dst sized for the actual returned bytes
  /// does not throw. Ranges starting at or beyond EOF contribute zero bytes.
  /// Throws @c std::invalid_argument (delivered via the completion handler's
  /// @c exception_ptr) when @c dst[i].size() is smaller than the clipped
  /// size for any range. Mirrors single-range @c host_read_io EOF semantics.
  void host_read_ranges_async_io(sirius_io_object& obj,
                                 std::vector<cudf::io::text::byte_range_info> const& ranges,
                                 std::span<cudf::host_span<std::byte>> dst,
                                 io_completion_handler handler) override;

  // -- Device reads ---------------------------------------------------------
  //
  // S3 has no native device read path. These implement a bounce strategy:
  // HTTP body lands in a host staging buffer, then cudaMemcpyAsync onto the
  // caller-supplied device pointer / stream. The base-class device_read /
  // device_read_async first consult the (optional) prefetching cache; these
  // overrides only run on cache miss.

  std::size_t device_read_io(sirius_io_object& obj,
                             std::size_t offset,
                             std::size_t size,
                             std::uint8_t* dst,
                             rmm::cuda_stream_view stream) override;

  void device_read_async_io(sirius_io_object& obj,
                            std::size_t offset,
                            std::size_t size,
                            std::uint8_t* dst,
                            rmm::cuda_stream_view stream,
                            io_completion_handler handler) override;

  // -- Physical range alignment --------------------------------------------

  /// S3 over HTTP has no alignment requirement; return the logical range
  /// clipped to file size.
  cudf::io::text::byte_range_info compute_physical_range(cudf::io::text::byte_range_info logical,
                                                         std::size_t file_size) const override;

 private:
  struct handle_slot;

  handle_slot acquire_handle();
  void release_handle(handle_slot slot);

  std::size_t range_get(std::string_view bucket,
                        std::string_view key,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* dst);

  /// Sync multi-range implementation. Used internally by
  /// @c host_read_ranges_async_io after its lambda owns the descriptor
  /// array; not exposed in the public contract because the new
  /// @c sirius_ioctx base only offers an async multi-range entry point.
  /// Applies the clip-then-validate semantics documented on
  /// @c host_read_ranges_async_io.
  std::size_t host_read_ranges_impl(sirius_io_object& obj,
                                    std::vector<cudf::io::text::byte_range_info> const& ranges,
                                    std::span<cudf::host_span<std::byte>> dst);

  struct handle_slot {
    s3_ioctx* owner{nullptr};
    void* easy{nullptr};

    handle_slot() = default;
    handle_slot(s3_ioctx* o, void* h) : owner(o), easy(h) {}
    handle_slot(handle_slot&& other) noexcept : owner(other.owner), easy(other.easy)
    {
      other.owner = nullptr;
      other.easy  = nullptr;
    }
    handle_slot& operator=(handle_slot&& other) noexcept
    {
      if (this != &other) {
        reset();
        owner       = other.owner;
        easy        = other.easy;
        other.owner = nullptr;
        other.easy  = nullptr;
      }
      return *this;
    }
    ~handle_slot() { reset(); }
    handle_slot(handle_slot const&)            = delete;
    handle_slot& operator=(handle_slot const&) = delete;

    void reset();
  };

  s3_ioctx_config _cfg;

  std::mutex _pool_mtx;
  std::condition_variable _pool_cv;
  std::vector<void*> _free_handles;
  std::size_t _total_handles{0};
  bool _shutdown{false};
};

}  // namespace sirius::io::s3
