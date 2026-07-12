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
#include <limits>
#include <optional>

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

  /// Bounce-slot size (bytes) for the reactor-staged device path — also the
  /// window grain @c prep_device_rx_request splits device reads by (the
  /// static prep cannot reach the live staging resource).  Nonzero is
  /// authoritative (whole multiple of the host block size,
  /// <= @c max_bounce_block_size); a larger-than-block grain stages through
  /// one budget-booked contiguous span instead of per-slot staging blocks.
  /// 0 = auto: the host block size, or disabled when no staging resource
  /// exists.
  std::size_t bounce_block_size{0};

  /// RESOLVED host reservation limit, stamped by sirius_config after
  /// memory-space resolution so runtime budget failures can report exact
  /// needed/limit/shortfall (the staging pool exposes no limit getter).
  /// Internal carrier — NOT a YAML key; 0 = unknown, errors then fall back to
  /// the capacity-headroom upper bound.
  std::size_t resolved_host_reservation_limit{0};

  /// Per-slot ceiling for @c bounce_block_size; the pool-level bound is
  /// @c max_bounce_pool_bytes.
  static constexpr std::size_t max_bounce_block_size{1UL << 30};

  /// Ceiling for the whole bounce pool: reactors * max_connections *
  /// bounce_block_size, whichever knob produced the product.
  static constexpr std::size_t max_bounce_pool_bytes{2UL << 30};

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
};

/// Overflow-safe reactors * connections * grain: the single bounce-pool
/// sizing formula for every admission site (YAML preflight, factory, reactor
/// context).  nullopt on size_t overflow.
[[nodiscard]] inline std::optional<std::size_t> checked_bounce_pool_bytes(
  std::size_t n_reactors, std::size_t max_connections, std::size_t grain) noexcept
{
  if (n_reactors != 0 && max_connections > std::numeric_limits<std::size_t>::max() / n_reactors) {
    return std::nullopt;
  }
  std::size_t const slots = n_reactors * max_connections;
  if (slots != 0 && grain > std::numeric_limits<std::size_t>::max() / slots) {
    return std::nullopt;
  }
  return slots * grain;
}

}  // namespace sirius::io::rest
