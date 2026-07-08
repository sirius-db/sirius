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

#include "io/rest/rest_reactor.hpp"
#include "io/s3/s3_list_parser.hpp"
#include "io/templated_ioctx.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::io::rest {

// ---------------------------------------------------------------------------
// rest_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief RESTful object-store (s3://) ioctx. Specialisation of
 *        @c templated_ioctx<rest_reactor>.
 *
 * Owns a pool of @c rest_reactor workers (round-robined by the base) that share
 * one @p authorizer.  Overrides @c create_io_object to resolve an object's size
 * via a blocking HEAD before constructing the @c rest_io_object — the static
 * reactor factory cannot do this since it needs the authorizer + a round-trip.
 */
class rest_ioctx : public templated_ioctx<rest_reactor> {
 public:
  /// Build a pool of @p n_reactors reactors, all sharing @p ctx (one context per
  /// pool: it carries the per-reactor @c config, the presigning authorizer, and
  /// the pinned bounce-staging resource — all of which must outlive this ioctx).
  /// The ioctx config is sourced from the reactors themselves — see
  /// @c templated_ioctx.
  rest_ioctx(std::size_t n_reactors, std::shared_ptr<rest_reactor::reactor_context> ctx);

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::restful; }

  /// Pool-aggregated perf counters: every reactor's snapshot summed (ns totals,
  /// counts, retries, terminal, device-sync), maxes maxed, and ttfb the first
  /// non-zero reactor value.  Lock-free; drives the s3-bench JSON baseline.
  [[nodiscard]] rest_perf_snapshot perf_snapshot() const noexcept;

  /// Stream a bucket's ListObjectsV2 pages under @p prefix to @p sink, one call
  /// per page (a page holds at most 1000 entries, so peak memory is one page
  /// regardless of bucket population).  @p sink returns false to stop early —
  /// no further LIST requests are issued.  @p page_size is clamped to [1,1000]
  /// (0 and >1000 mean 1000).  Throws (never truncates) on a truncated page
  /// without a continuation token, and once more than @p max_scanned entries
  /// have been scanned across pages (bounds time / request count on a prefix
  /// whose population dwarfs the caller's matches).
  void list_objects_paged(std::string_view bucket,
                          std::string_view prefix,
                          std::size_t page_size,
                          std::function<bool(s3::list_objects_v2_page const&)> const& sink,
                          std::size_t max_scanned = s3::default_max_scanned_objects);

  /// Whole-listing convenience over @c list_objects_paged: every object under
  /// @p prefix, in document order, with sizes.  Throws (never truncates) when
  /// the accumulated entries would exceed @p max_keys — a partial key set would
  /// resolve a glob to a silently incomplete table.
  [[nodiscard]] std::vector<s3::list_entry> list_objects(
    std::string_view bucket,
    std::string_view prefix,
    std::size_t page_size = 1000,
    std::size_t max_keys  = s3::default_max_list_objects);

 protected:
  /// Backend hook invoked by @c sirius_ioctx::open_datasource: parse @p path
  /// (s3://bucket/key), HEAD it for the size, and build a @c rest_io_object.
  /// Throws on a non-s3 scheme or a failed HEAD.
  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override;

  /// @c open_hint::parquet_footer_probe resolves the size and stashes the
  /// parquet footer together via a single suffix-range GET, carried on the
  /// returned io_object; every other hint falls back to the plain HEAD path above.
  std::shared_ptr<sirius_io_object> create_io_object(std::string path, open_hint hint) override;

  /// Known-size open: the caller already learned the object's size (e.g. from a
  /// ListObjectsV2 response), so the io_object is built with ZERO network — no
  /// HEAD, no probe.
  std::shared_ptr<sirius_io_object> create_io_object(std::string path,
                                                     std::uint64_t known_size) override;

 private:
  /// Resolve @p path with a single suffix-range GET: it discovers the size and
  /// stashes the object's trailing bytes on the returned io_object so cuDF's
  /// footer reads are served locally by @c rest_reactor::host_read.  Falls back
  /// to a HEAD when the suffix response is unusable.  The stash lives only as
  /// long as the returned io_object — a per-open transport shortcut, not a cache.
  std::shared_ptr<sirius_io_object> create_footer_probe_object(std::string path);
};

}  // namespace sirius::io::rest
