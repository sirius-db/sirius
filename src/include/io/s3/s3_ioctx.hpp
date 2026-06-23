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

#include "io/s3/s3_reactor.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/templated_ioctx.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace sirius::io::s3 {

struct s3_ioctx_config;  // defined in io/s3/s3_blocking_ioctx.hpp

/**
 * @brief Production async-S3 backend (libcurl-multi reactor).
 *
 * The default S3 io context: the datasource factory builds this when
 * @c object_store_config::s3_use_async_backend is true. A thin
 * @c templated_ioctx<s3_reactor> with the S3-specific overrides the generic
 * plumbing can't cover: an instance @c create_io_object (HEAD via the
 * authorizer), strict @c host_read_ranges_async_io validation (errors through
 * the handler, never a sync throw / silent skip), and reactor-aggregated
 * counters. Reads run as concurrent range GETs with pipelined device copies
 * (async GET into a pinned staging block + async H2D, no per-chunk
 * cudaStreamSynchronize). The legacy serial backend is @c s3_blocking_ioctx,
 * selectable as a fallback via the same flag.
 */
class s3_ioctx : public templated_ioctx<s3_reactor> {
 public:
  // Retry knobs are optional; when omitted (the 6-arg call sites) the reactor
  // keeps its OWN config defaults — they are NOT duplicated here, so they can't
  // silently drift from s3_reactor::config. The production factory passes the
  // s3_ioctx_config values so this backend is config-equivalent to the
  // blocking one.
  s3_ioctx(std::shared_ptr<s3_request_authorizer> creds,
           long request_timeout_s,
           std::string ca_bundle_path,
           bool tls_verify,
           std::size_t max_connections,
           cucascade::memory::fixed_size_host_memory_resource* host_mr,
           std::optional<std::size_t> max_retry_attempts               = std::nullopt,
           std::optional<std::chrono::milliseconds> retry_backoff_base = std::nullopt,
           std::optional<std::chrono::milliseconds> retry_jitter       = std::nullopt,
           std::optional<bool> honor_retry_after                       = std::nullopt);

  /// Construct from the shared @c s3_ioctx_config (the same struct the blocking
  /// backend and scan_manager use). Maps every field — including
  /// @c s3_perf_instrumentation — into the reactor config. Preferred over the
  /// multi-arg ctor for new call sites.
  explicit s3_ioctx(s3_ioctx_config config);

  // -- F1: instance create_io_object (HEAD needs the authorizer) -------------
  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override;

  // -- F3: strict ranges validation (errors via the handler) -----------------
  void host_read_ranges_async_io(sirius_io_object& obj,
                                 std::vector<cudf::io::text::byte_range_info> const& ranges,
                                 std::span<cudf::host_span<std::byte>> dst,
                                 io_completion_handler handler) override;

  // Device reads use the inherited templated_ioctx path (1 MiB chunking ->
  // reactor().enqueue_bulk): async GET into a pinned staging block + async H2D
  // copy completed by a stream callback, no per-chunk cudaStreamSynchronize.

  // -- F5: observability aggregated across reactors --------------------------
  [[nodiscard]] std::uint64_t bytes_read_total() const noexcept;
  [[nodiscard]] std::uint64_t fsmr_borrows_total() const noexcept;
  [[nodiscard]] std::uint64_t device_copies_total() const noexcept;
  [[nodiscard]] std::uint64_t device_stream_sync_total() const noexcept;
  [[nodiscard]] std::uint64_t device_peak_inflight() const noexcept;

  // Per-chunk perf micro (gated by s3_perf_instrumentation) + always-on retry
  // counters; each sums over the reactor(s).
  [[nodiscard]] std::uint64_t chunk_get_ns_total() const noexcept;
  [[nodiscard]] std::uint64_t chunk_get_ns_count() const noexcept;
  [[nodiscard]] std::uint64_t chunk_get_ns_max() const noexcept;
  [[nodiscard]] std::uint64_t queue_wait_ns_total() const noexcept;
  [[nodiscard]] std::uint64_t queue_wait_ns_count() const noexcept;
  [[nodiscard]] std::uint64_t h2d_observed_ns_total() const noexcept;
  [[nodiscard]] std::uint64_t h2d_observed_ns_count() const noexcept;
  [[nodiscard]] std::uint64_t h2d_observed_ns_max() const noexcept;
  [[nodiscard]] std::uint64_t ttfb_ns() const noexcept;
  [[nodiscard]] std::uint64_t retries_total() const noexcept;
  [[nodiscard]] std::uint64_t terminal_failures_total() const noexcept;

  std::size_t head_object_size(std::string_view bucket, std::string_view key);

 private:
  [[nodiscard]] s3_reactor& reactor() noexcept { return *_reactors.front(); }
};

}  // namespace sirius::io::s3
