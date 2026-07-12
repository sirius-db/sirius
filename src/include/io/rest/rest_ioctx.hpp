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
#include "io/templated_ioctx.hpp"

#include <cstddef>
#include <memory>
#include <string>

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

  /// Prepares the context's shared bounce span (large-grain staging: one
  /// accounted contiguous allocation for the whole pool) BEFORE the base class
  /// starts any reactor worker — budget failures surface here, with no thread
  /// or GET in flight.  Then delegates to the base start (idempotent).
  void start() override;

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::restful; }

  /// Pool-aggregated perf counters: every reactor's snapshot summed (ns totals,
  /// counts, retries, terminal, device-sync), maxes maxed, and ttfb the first
  /// non-zero reactor value.  Lock-free; drives the s3-bench JSON baseline.
  [[nodiscard]] rest_perf_snapshot perf_snapshot() const noexcept;

  /// One entry per reactor whose staging is backed by the dedicated
  /// large-grain bounce pool (empty on the default block-carve path), with
  /// @c reactor_index assigned by pool position.  Exists so the shared-span
  /// slice invariants (one allocation, r * conns * grain offsets, no overlap)
  /// are externally verifiable — see rest_bounce_slice_snapshot.
  [[nodiscard]] std::vector<rest_bounce_slice_snapshot> bounce_slice_snapshots() const;

 protected:
  /// Backend hook invoked by @c sirius_ioctx::open_datasource: parse @p path
  /// (s3://bucket/key), HEAD it for the size, and build a @c rest_io_object.
  /// Throws on a non-s3 scheme or a failed HEAD.
  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override;

  /// @c open_hint::parquet_footer_probe resolves the size and stashes the
  /// parquet footer together via a single suffix-range GET, carried on the
  /// returned io_object; every other hint falls back to the plain HEAD path above.
  std::shared_ptr<sirius_io_object> create_io_object(std::string path, open_hint hint) override;

 private:
  /// The pool's shared context (also captured by every reactor); kept here so
  /// start() can prepare the bounce span before the reactors run.
  std::shared_ptr<rest_reactor::reactor_context> _pool_ctx;

  /// Resolve @p path with a single suffix-range GET: it discovers the size and
  /// stashes the object's trailing bytes on the returned io_object so cuDF's
  /// footer reads are served locally by @c rest_reactor::host_read.  Falls back
  /// to a HEAD when the suffix response is unusable.  The stash lives only as
  /// long as the returned io_object — a per-open transport shortcut, not a cache.
  std::shared_ptr<sirius_io_object> create_footer_probe_object(std::string path);
};

}  // namespace sirius::io::rest
