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

#include "io/s3/s3_list_parser.hpp"

#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace sirius::io::rest {

// 64x the production default and 32x the largest committed integration-test
// value. This leaves ample tuning headroom while bounding per-reactor slot,
// curl-handle, event, and pinned-bounce allocations.
inline constexpr std::size_t max_connection_limit = 1024;

inline std::size_t checked_bounce_storage_bytes(std::size_t connections, std::size_t bytes_per_slot)
{
  if (connections != 0 && bytes_per_slot > std::numeric_limits<std::size_t>::max() / connections) {
    throw std::overflow_error("rest_reactor: bounce allocation size overflow");
  }
  return connections * bytes_per_slot;
}

struct config {
  /// Whole-request timeout (seconds, 0 = no limit) and presigned-URL TTL.
  long request_timeout_s{30};

  /// TLS: optional CA bundle path; when @c tls_verify is false, peer/host
  /// verification is disabled (self-signed dev endpoints / MinIO).
  std::string ca_bundle_path;
  bool tls_verify{true};

  /// Max concurrent in-flight easy handles per reactor. Bounded because the
  /// reactor allocates one slot and, when staging is active, one bounce buffer
  /// per connection.
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

  /// When set, the reactor records the per-chunk micro timings (chunk_get,
  /// queue_wait, ttfb, h2d_observed) into its perf counters.  The retry,
  /// terminal-failure and device-stream-sync counters are always recorded,
  /// independent of this flag.
  bool perf_instrumentation{false};

  /// Suffix-range window (bytes) for the parquet footer probe
  /// (@c open_hint::parquet_footer_probe): one `Range: bytes=-N` GET resolves the
  /// object size and stashes its last N bytes, so cuDF's trailer/footer reads are
  /// served locally.  Tradeoff — a parquet footer is ~0.037% of the file (SF1
  /// lineitem 207 MB -> 78 KiB, SF10 2.2 GB -> 771 KiB): N must cover the footer,
  /// else the probe wastes the suffix and re-GETs the footer body (worse than a
  /// plain HEAD), so err large; the over-read when N exceeds the footer is a
  /// one-time bind transfer (~10 ms on a high-bandwidth link).  The 512 KiB
  /// default covers files up to ~1.4 GB in one GET (the common range); raise it
  /// for multi-GB single files, lower it for many-tiny-file / low-bandwidth
  /// workloads.
  std::size_t footer_probe_bytes{512UL << 10};  // 512 KiB

  /// S3 LIST / glob safety caps (both throw "narrow the glob prefix", never
  /// truncate).  @c list_max_matches bounds the files a glob keeps / a
  /// whole-listing accumulates (result memory); @c list_max_scanned bounds the
  /// objects a LIST sweep looks at across pages (time / LIST round-trips).  The
  /// two axes diverge when a prefix is huge but few keys match, so both exist.
  std::size_t list_max_matches{s3::default_max_list_objects};     // 100'000
  std::size_t list_max_scanned{s3::default_max_scanned_objects};  // 1'000'000
};

}  // namespace sirius::io::rest
