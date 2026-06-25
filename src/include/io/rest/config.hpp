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

#include <chrono>
#include <cstddef>

namespace sirius::io::rest {

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

}  // namespace sirius::io::rest