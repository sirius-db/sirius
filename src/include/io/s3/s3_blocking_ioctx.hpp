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
#include "io/s3/s3_request_authorizer.hpp"

#include <atomic>
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

namespace cucascade::memory {
class fixed_size_host_memory_resource;
}  // namespace cucascade::memory

namespace sirius::io::s3 {

class s3_blocking_io_object;

/// Construction config for @c s3_blocking_ioctx.
///
/// @c creds is required and owns the credential / signer chain; the reactor
/// itself is signing-blind — every request path obtains a fully-qualified
/// URL plus headers from @c creds and issues the HTTP request against them.
/// Endpoint / region / access-key material lives inside the authorizer's
/// concrete implementation (e.g. @c sirius_sigv4_presigned_authorizer's
/// constructor takes the endpoint and region), so they intentionally do not
/// appear here.
struct s3_ioctx_config {
  std::shared_ptr<s3_request_authorizer> creds;
  std::size_t max_connections = 16;
  long request_timeout_s      = 60;

  /// Optional caller-owned thread pool for async paths. When non-null,
  /// @c host_read_async_io / @c host_read_ranges_async_io /
  /// @c device_read_async_io schedule work onto this pool's threads. When
  /// nullptr (default), async paths fall back to per-request
  /// @c std::thread().detach() — kept for standalone-test scenarios where
  /// no scan_manager / SiriusContext owns a pool. The s3_blocking_ioctx never owns
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

  /// Optional caller-owned host memory resource for device-read staging.
  /// When non-null, @c device_read_io stages the H2D bounce through one
  /// bounded FSMR block (reused across chunks) instead of a per-call
  /// @c std::vector sized to the whole request, so transient host memory is
  /// capped at one block. When nullptr (default), device reads keep the
  /// @c std::vector fallback — preserved for standalone / unit-test
  /// construction. The s3_blocking_ioctx never owns this resource; its lifetime is
  /// the caller's (sirius_scan_manager) responsibility.
  cucascade::memory::fixed_size_host_memory_resource* host_memory_resource{nullptr};

  /// TLS for https endpoints. @c ca_bundle_path (non-empty) -> CURLOPT_CAINFO
  /// so a custom / self-signed CA verifies; empty uses the system CA bundle
  /// (AWS). @c tls_verify=false disables peer+host verification
  /// (CURLOPT_SSL_VERIFYPEER / CURLOPT_SSL_VERIFYHOST = 0) — INSECURE, dev/test.
  std::string ca_bundle_path;
  bool tls_verify = true;
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
 * Authentication is delegated entirely to the @c s3_request_authorizer passed
 * via @c s3_ioctx_config — this class never sees raw access keys and never
 * computes a SigV4 signature itself. Each request path acquires an
 * @c s3_authorized_request via @c s3_request_authorizer::authorize and lets
 * libcurl follow the returned URL, attaching any returned headers verbatim.
 * For the default presigned authorizer the headers are empty and the only
 * request header this class emits is @c Range (unsigned, allowed by the URL's
 * @c SignedHeaders=host); a header-signing authorizer may additionally return
 * Authorization / x-amz-* headers that this class attaches as-is.
 *
 * @par Construction
 *
 * @code
 *   sirius::io::s3::static_credentials creds;
 *   creds.access_key_id     = "...";
 *   creds.secret_access_key = "...";
 *   auto provider = std::make_shared<
 *     sirius::io::s3::sirius_sigv4_presigned_authorizer>(
 *       std::move(creds), "us-east-1", "https://s3.amazonaws.com",
 *       std::chrono::minutes{5});
 *
 *   sirius::io::s3::s3_ioctx_config scfg{};
 *   scfg.creds            = std::move(provider);
 *   scfg.max_connections  = 16;
 *   scfg.request_timeout_s = 60;
 *   auto s3 = std::make_shared<sirius::io::s3::s3_blocking_ioctx>(std::move(scfg));
 *
 *   // (Optional) wire the prefetching cache from a host buffer pool.
 *   s3->initialize_cache(host_buffer_pool, 2048);
 *
 *   // Use directly:
 *   auto ds = s3->open_datasource("s3://bucket/key.parquet");
 * @endcode
 *
 * Sirius-runtime wiring (placing this @c s3_blocking_ioctx on @c SiriusContext or
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
 * @par Resource injection
 *
 * - **Thread pool**. Async paths (@c host_read_async_io,
 *   @c host_read_ranges_async_io, @c device_read_async_io) schedule their
 *   work on the caller-injected @c s3_ioctx_config::async_thread_pool when
 *   one is provided (the engine's @c sirius_scan_manager owns an 8-thread
 *   pool by default and wires it in). When no pool is injected, async
 *   paths fall back to per-request @c std::thread().detach() for
 *   standalone-test scenarios. Either way, the worker captures
 *   @c shared_from_this() and @c obj.shared_from_this() so the ioctx +
 *   io_object stay alive through the in-flight request.
 *   @c host_read_ranges_async_io fans out one task per range when a pool
 *   is wired, mirroring the worker count and avoiding the serial-latency
 *   stacking that the single-task path would exhibit.
 * - **CUDA stream**. Caller-supplied per request on the device-read path.
 */
class s3_blocking_ioctx final : public sirius_ioctx {
 public:
  explicit s3_blocking_ioctx(s3_ioctx_config config);
  ~s3_blocking_ioctx() override;

  s3_blocking_ioctx(s3_blocking_ioctx const&)            = delete;
  s3_blocking_ioctx& operator=(s3_blocking_ioctx const&) = delete;

  void shutdown() override;

  std::unique_ptr<io::sirius_datasource> make_datasource(
    std::shared_ptr<sirius_io_object> io_object) override;

  /// Backend factory: parses @p path as @c s3://bucket/key, issues a HEAD
  /// request via @c head_object_size, and constructs an @c s3_blocking_io_object
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
  /// without constructing an @c s3_blocking_io_object.
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
  /// When @c s3_ioctx_config::async_thread_pool is set, the implementation
  /// schedules one task per range onto the pool so the per-cURL latency of
  /// independent ranges overlaps (P1.6 fan-out). The user handler fires
  /// exactly once from a pool thread after every range either completes or
  /// errors — successful byte counts are summed; the first observed
  /// exception (if any) is delivered through the handler's
  /// @c exception_ptr. When no pool is wired, the implementation falls
  /// back to a single-task serial walk on a detached @c std::thread.
  ///
  /// Each @c ranges[i] is clipped to
  /// @c min(ranges[i].size(), obj.size() - ranges[i].offset()) before
  /// validation; the @c dst[i].size() check is against the clipped size, so
  /// an EOF-crossing range with a dst sized for the actual returned bytes
  /// does not error. Ranges starting at or beyond EOF contribute zero bytes.
  ///
  /// Validation errors (@c ranges/dst size mismatch, @c dst[i].size() smaller
  /// than the clipped size) and lookup errors (@c bad_weak_ptr from
  /// @c shared_from_this on an unowned object, @c bad_cast from a
  /// non-S3 @c sirius_io_object) are *not* thrown synchronously — they
  /// are caught and delivered through the completion handler's
  /// @c exception_ptr from a pool / detached worker, preserving the
  /// async-only "exactly one handler invocation" contract that the cache
  /// worker (and other callers without their own try/catch) rely on.
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

  // -- Observability --------------------------------------------------------

  /// Cumulative bytes returned by @c range_get since construction. Each
  /// successful HTTP body delivered into @c dst is added exactly once
  /// (retried attempts of the same range are not double-counted because
  /// the counter is bumped only on the success-return path). Intended for
  /// perf benchmarks (P7) and observability — every call to
  /// @c host_read_io / @c host_read_async_io / @c host_read_ranges_async_io /
  /// @c device_read_io / @c device_read_async_io that misses the cache
  /// passes through @c range_get and increments this counter.
  [[nodiscard]] std::uint64_t bytes_read_total() const noexcept
  {
    return _bytes_read_total.load(std::memory_order_relaxed);
  }

  /// Cumulative count of successful FSMR block borrows performed by
  /// @c device_read_io since construction. Each successful
  /// @c allocate_multiple_blocks on the FSMR-staged path increments by one;
  /// the vector-fallback path (no caller-supplied FSMR) does not contribute.
  /// Failed allocations (out-of-memory etc.) are not counted. Intended for
  /// tests asserting that F1's chunked-staging path is exercised at SQL
  /// scale.
  [[nodiscard]] std::uint64_t fsmr_borrows_total() const noexcept
  {
    return _fsmr_borrows_total.load(std::memory_order_relaxed);
  }

 private:
  struct handle_slot;

  handle_slot acquire_handle();
  void release_handle(handle_slot slot);

  std::size_t range_get(std::string_view bucket,
                        std::string_view key,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* dst);

  /// Serial multi-range implementation. Used by @c host_read_ranges_async_io
  /// only when @c _cfg.async_thread_pool is null (test/standalone scenarios
  /// where no caller pool is wired in). When a pool is present, the async
  /// entry point fans out one task per range instead — see P1.6 in the
  /// PR4+5 runtime integration plan. Not exposed in the public contract
  /// because the new @c sirius_ioctx base only offers an async multi-range
  /// entry point. Applies the clip-then-validate semantics documented on
  /// @c host_read_ranges_async_io.
  std::size_t host_read_ranges_serial_impl(
    sirius_io_object& obj,
    std::vector<cudf::io::text::byte_range_info> const& ranges,
    std::span<cudf::host_span<std::byte>> dst);

  struct handle_slot {
    s3_blocking_ioctx* owner{nullptr};
    void* easy{nullptr};

    handle_slot() = default;
    handle_slot(s3_blocking_ioctx* o, void* h) : owner(o), easy(h) {}
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

  /// See @c bytes_read_total(). Updated only inside @c range_get on the
  /// success-return path (HTTP 206 / 200 with validated body length).
  std::atomic<std::uint64_t> _bytes_read_total{0};

  /// See @c fsmr_borrows_total(). Updated only on the FSMR-staged branch of
  /// @c device_read_io, after @c allocate_multiple_blocks succeeds.
  std::atomic<std::uint64_t> _fsmr_borrows_total{0};
};

}  // namespace sirius::io::s3
