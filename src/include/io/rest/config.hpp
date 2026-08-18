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

#include "exec/config.hpp"
#include "io/rest/s3/list_parser.hpp"

#include <chrono>
#include <cstddef>

namespace sirius::io::rest {

struct config {
  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once.  Zero disables readahead for it entirely.
  ///
  /// Object-store reads are latency-bound rather than bandwidth-bound, so more
  /// concurrency is needed to cover the round trips before the link itself is
  /// the limit.  This struct default can only name the compile-time pipeline
  /// width; @c sirius_config::derive_rest_scan_budget scales it to the
  /// configured pipeline pool size unless the config sets it explicitly.
  std::size_t n_max_concurrent_scans{8};

  /// Whether the config named @c n_max_concurrent_scans explicitly.  Needed
  /// because the derived default is computed from the pipeline width and can
  /// legitimately land on the struct default -- without this flag, a config that
  /// sets the value to exactly the struct default is indistinguishable from one
  /// that says nothing, and gets silently overridden.
  bool n_max_concurrent_scans_explicit{false};

  /// Whole-request timeout (seconds, 0 = no limit) and presigned-URL TTL.
  long request_timeout_s{30};

  /// TLS: optional CA bundle path; when @c tls_verify is false, peer/host
  /// verification is disabled (self-signed dev endpoints / MinIO).
  std::string ca_bundle_path;
  bool tls_verify{true};

  /// Max concurrent in-flight easy handles per reactor, i.e. the ceiling on
  /// simultaneous ranged GETs this reactor drives.  Fixed at 64, not exposed to
  /// YAML: the useful value is a property of one reactor thread rather than of a
  /// deployment.
  ///
  /// More is not better, and past the point where the link is full it is
  /// actively worse: extra connections only split the same bandwidth into
  /// thinner streams.  Each socket then delivers a few KiB per read, so the
  /// reactor thread spends its time in per-read callback and TLS-record overhead
  /// rather than moving bytes, and time-to-first-byte climbs because it cannot
  /// service that many sockets promptly.  To drive more concurrency, add
  /// reactors (@c rest_n_reactors) rather than sockets per reactor — each
  /// reactor brings its own thread to service them.
  ///
  /// Note this is also the reactor's host-staging footprint: one pinned bounce
  /// slot is parked per connection, so a reactor reserves
  /// 64 x @c bounce_block_size of host memory up front.
  std::size_t max_connections{64};

  /// Chunk floor (bytes) for every read the reactor splits: a coalesced read
  /// span of S bytes becomes max(1, S / max_chunk_size) requests with S balanced
  /// evenly across them, so each request carries AT LEAST this many bytes and at
  /// most twice it.  Integer division is the point — 31 MiB stays a single
  /// 31 MiB GET rather than becoming two half-sized ones, while 33 MiB splits
  /// into 17 + 16.
  ///
  /// Against an object store every extra request is a full round trip, so the
  /// floor exists to stop a large read from being shredded into latency-bound
  /// pieces; it is not a ceiling on request size.
  std::size_t max_chunk_size{16UL << 20};  // 16 MiB

  /// How far apart two segments may sit and still be fetched by one GET.  Reads
  /// arrive as many small destination buffers (one per cache chunk, one per
  /// storage block) that are near each other in the file but not adjacent —
  /// separated by an already-cached chunk, a block header, a skipped column.
  /// Bridging a gap up to this size costs the gap's bytes (fetched, then
  /// discarded on arrival) and saves a round trip; leaving it costs a round
  /// trip and saves the bytes.  On a high-bandwidth link to an object store the
  /// round trip is the expensive half, so the default bridges generously.
  ///
  /// Zero fuses only genuinely adjacent segments.
  std::size_t merge_max_gap{512UL << 10};  // 0.5 MiB

  /// Bounce-slot size (bytes) for the reactor-staged device path.  Not a YAML
  /// key: it is stamped at construction from the host staging resource's block
  /// size (@c make_rest_ioctx_factory), because a slot that did not match the
  /// pool's block size could not be handed out by that pool at all.  Configure
  /// it through @c scan_manager.memory.host.block_size.
  ///
  /// Zero means no host staging resource was configured, which disables the
  /// reactor-staged device read (host reads still work).  The static
  /// @c prep_device_rx_request needs this size without access to the live
  /// resource, which lives on the @c reactor_context.
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
