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

#include "io/s3/credential_provider.hpp"
#include "io/types.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

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
};

/**
 * @brief S3 @c sirius_ioctx implemented with libcurl HTTP Range GETs over
 *        presigned URLs.
 *
 * Targets the @c sirius_ioctx contract from PR #675: host reads map to
 * libcurl range GETs; device reads bounce through a host staging buffer +
 * H2D copy (S3 has no native device path). The caching / admission hooks on
 * the base class are opt-in via @c initialize_cache (see "Bootstrap" below).
 *
 * Authentication is delegated entirely to the @c credential_provider passed
 * via @c s3_ioctx_config — this class never sees raw access keys, never
 * computes a SigV4 signature, and never injects an Authorization header.
 * Each request path acquires a presigned URL via
 * @c credential_provider::get_presigned_url and lets libcurl follow it; the
 * only request header this class emits is @c Range (unsigned, allowed by
 * the URL's @c SignedHeaders=host).
 *
 * @par Bootstrap (typical usage in @c SiriusContext::initialize)
 *
 * The integration PR composes the types from PR1 / PR2 / PR3 inside
 * @c SiriusContext so the registry, the credential provider, and the
 * reactor share @c SiriusContext-scoped lifetime, **not** per-query
 * @c sirius_engine lifetime (mirroring @c task_scheduler_ /
 * @c scan_manager_ / @c task_creator_ already on @c SiriusContext).
 *
 * The code below is illustrative pseudocode — it references methods (e.g.
 * @c siriusContext.host_buffer_pool, the teardown iteration helpers) that
 * the integration PR is expected to introduce / surface; use whatever shape
 * the integration PR chooses, but keep the lifetime scoping the same.
 *
 * @code
 *   // 1. From object_store_config string fields → static_credentials.
 *   sirius::io::s3::static_credentials creds;
 *   creds.access_key_id     = osc.access_key;
 *   creds.secret_access_key = osc.secret_key;
 *   // (session_token / expires_at left empty for long-lived keys.)
 *
 *   // 2. Wrap in the default SigV4 provider. Downstream forks that want
 *   //    refresh-aware credentials (AWS SDK / IMDS / STS / SSO / internal
 *   //    auth broker) plug in their own credential_provider subclass here
 *   //    without source changes to Sirius.
 *   auto provider = std::make_shared<
 *     sirius::io::s3::sirius_sigv4_credential_provider>(
 *       std::move(creds), osc.region, osc.endpoint,
 *       std::chrono::minutes{5});
 *
 *   // 3. Construct the reactor and register it in the engine's registry.
 *   sirius::io::s3::s3_ioctx_config scfg{};
 *   scfg.creds            = std::move(provider);
 *   scfg.max_connections  = 16;   // libcurl handle pool size — internal,
 *                                 // SiriusContext-scoped through the
 *                                 // s3_ioctx instance.
 *   scfg.request_timeout_s = 60;
 *   auto s3 = std::make_shared<sirius::io::s3::s3_ioctx>(std::move(scfg));
 *
 *   // 4. (Optional, recommended) Wire the prefetching cache. Only this
 *   //    seam takes external resources from the surrounding context.
 *   //    The buffer_pool accessor below is pseudocode — pick whichever
 *   //    name the integration PR exposes on SiriusContext. Its lifetime
 *   //    mirrors task_scheduler_ etc., so the cache and any pinned chunks
 *   //    it holds inherit SiriusContext-scoped lifetime, not per-query
 *   //    sirius_engine lifetime.
 *   s3->initialize_cache(siriusContext.host_buffer_pool(),
 *                        2048);  // inflight_budget_chunks
 *
 *   // 5. Register under the "s3" scheme. The registry sits on
 *   //    SiriusContext (SiriusContext-scoped lifetime) — never on
 *   //    sirius_engine (per-query lifetime), per the #742 review thread.
 *   datasource_registry_.register_ioctx("s3", std::move(s3));
 * @endcode
 *
 * @par Teardown (in @c SiriusContext::terminate)
 *
 * @c datasource_registry::clear() is a passive @c shared_ptr drop — it
 * does *not* call @c shutdown() on each ioctx. The owner has to do that
 * first, mirroring how nothing else on @c SiriusContext self-shuts in its
 * destructor.
 *
 * Note: @c s3_ioctx::shutdown() closes the libcurl handle pool and prevents
 * new handle acquisition; it does *not* join already-detached async
 * workers. Those workers stay alive on their own through
 * @c shared_from_this() captures (see "Async lifetime safety" in the PR
 * description) and finish whatever request they had in flight before
 * returning.
 *
 * @code
 *   // Pseudocode — use whatever iteration / clear API
 *   // datasource_registry exposes when the integration PR lands.
 *   for (auto const& scheme : datasource_registry_.schemes()) {
 *     if (auto ioctx = datasource_registry_.lookup(scheme)) {
 *       ioctx->shutdown();
 *     }
 *   }
 *   datasource_registry_.clear();
 * @endcode
 *
 * @par Downstream callers (after the integration PR re-adds
 *      @c datasource_factory's S3 dispatch branch)
 *
 * Once @c datasource_factory::create dispatches the @c s3 scheme through
 * @c registry.lookup → @c s3_ioctx::head_object_size → @c s3_io_object
 * construction → @c make_datasource, query-time code reads from S3 with no
 * S3-specific knowledge:
 *
 * @code
 *   auto ds = sirius::io::datasource_factory::create(
 *               "s3://bucket/key.parquet", registry, sirius_config);
 *   // ds is a cudf::io::datasource backed by this s3_ioctx.
 * @endcode
 *
 * For a runnable end-to-end demonstration of steps 1–3 against a real
 * MinIO instance, see @c make_live_ioctx in
 * @c test/cpp/io/s3/test_s3_ioctx.cpp (gated on @c SIRIUS_TEST_S3_*
 * environment variables; brought up via `make s3-up`).
 *
 * @par Resources NOT injected by PR3
 *
 * - **Thread pool**. Async paths (@c host_read_async,
 *   @c device_read_io_async, @c host_read_ranges_async) currently spawn a
 *   per-request @c std::thread().detach(); the worker captures
 *   @c shared_from_this() and @c obj.shared_from_this() so the ioctx +
 *   io_object stay alive through the detached worker. A future enhancement
 *   could inject a shared @c bounded_thread_pool from @c SiriusContext, but
 *   the @c sirius_ioctx base class does not yet expose that seam.
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

  /// HEAD request helper used by the factory before constructing an s3_io_object
  /// so that @c sirius_io_object::size() can remain @c noexcept.
  std::size_t head_object_size(std::string_view bucket, std::string_view key);

  // -- Host reads -----------------------------------------------------------

  std::size_t host_read(sirius_io_object& obj,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* dst) override;

  std::unique_ptr<cudf::io::datasource::buffer> host_read(sirius_io_object& obj,
                                                          std::size_t offset,
                                                          std::size_t size) override;

  void host_read_async(sirius_io_object& obj,
                       std::size_t offset,
                       std::size_t size,
                       std::uint8_t* dst,
                       io_completion_handler handler) override;

  /// Async multi-range host read. The implementation copies the @p dst span's
  /// descriptor array into owned storage before launching the async worker,
  /// so callers may drop the source container immediately after this returns.
  /// The byte buffers each @c host_span points at remain caller-owned and
  /// must outlive the completion handler — same contract as the @c uint8_t*
  /// in @c host_read_async.
  void host_read_ranges_async(sirius_io_object& obj,
                              std::vector<cudf::io::text::byte_range_info> const& ranges,
                              std::span<cudf::host_span<std::byte>> dst,
                              io_completion_handler handler) override;

  /// Synchronous multi-range host read. Each @c ranges[i] is clipped to
  /// @c min(ranges[i].size(), obj.size() - ranges[i].offset()) before
  /// validation; the @c dst[i].size() check is against the clipped size, so
  /// an EOF-crossing range with a dst sized for the actual returned bytes
  /// does not throw. Ranges starting at or beyond EOF contribute zero bytes.
  /// Throws @c std::invalid_argument when @c dst[i].size() is smaller than
  /// the clipped size for any range. Mirrors single-range @c host_read EOF
  /// semantics.
  std::size_t host_read_ranges(sirius_io_object& obj,
                               std::vector<cudf::io::text::byte_range_info> const& ranges,
                               std::span<cudf::host_span<std::byte>> dst) override;

  // -- Device reads ---------------------------------------------------------
  //
  // S3 has no native device read path. These implement a bounce strategy:
  // HTTP body lands in a host staging buffer, then cudaMemcpyAsync onto the
  // caller-supplied device pointer / stream. The base-class device_read()
  // consults the (currently unused) cache before falling through to these.

  std::unique_ptr<cudf::io::datasource::buffer> device_read_io(
    sirius_io_object& obj,
    std::size_t offset,
    std::size_t size,
    rmm::cuda_stream_view stream) override;

  std::size_t device_read_io(sirius_io_object& obj,
                             std::size_t offset,
                             std::size_t size,
                             std::uint8_t* dst,
                             rmm::cuda_stream_view stream) override;

  void device_read_io_async(sirius_io_object& obj,
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
