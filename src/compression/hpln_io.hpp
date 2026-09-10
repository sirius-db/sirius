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

// Where a .hpln's bytes come from, and how many requests it takes to get them.
//
// The format was built for a transport that charges per request: a fixed trailer locates a
// postscript, which locates every segment, so ONE tail read tells a reader where the metadata and
// each chunk live (CHUNK_SKIPPING_PLAN.md 7.5). What that buys is only realizable if the reads
// actually go through an object store, which is what this file is for: one staging/parsing path
// in simpatico_file_ingest.cpp, reading through a pluggable @ref hpln_source that is either the
// local filesystem or a sirius io_context (uring for local paths, REST for `s3://`).
//
// The coalescing policy is 7.7's, as a formula rather than a constant:
// `max_gap_bytes ~= per_request_cost x achievable_bandwidth`. S3 was measured to charge for
// REQUESTS, not for skipped bytes (7.6) -- so bridging a small gap to avoid a second request is
// a win, and the break-even is where the bridged bytes cost as much as the request would have.
// The defaults below are one machine's measurement (a g7e.2xlarge, in-region: ~0.29 ms/request
// idle, ~2.6 GB/s), so they are tunables carrying that value, not truths.

#include "io/io_context.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius {

//===----------------------------------------------------------------------===//
// policy
//===----------------------------------------------------------------------===//

/// How reads of a .hpln are turned into requests. Defaults measured on a g7e.2xlarge against
/// in-region S3 (CHUNK_SKIPPING_PLAN.md 7.7); the per-request cost moved by 2.6x with instance
/// load alone, so treat them as a starting point that a caller may override.
struct hpln_io_policy {
  /// Bridge a gap smaller than this rather than pay for a second request; the bridged bytes are
  /// read into scratch and thrown away. 0.286 ms x 2.6 GB/s ~= 0.75 MB.
  std::uint64_t max_gap_bytes = 768ull << 10;
  /// Requests are cut to about this size. Below ~4 MB the measured throughput falls off (0.77 vs
  /// 0.96 GB/s at equal volume), and 16 MB is where it flattens.
  std::uint64_t target_request_bytes = 16ull << 20;
  /// Bytes allowed outstanding across concurrent requests. 64 x 16 MB is what kept the measured
  /// pipe full; less starves it regardless of how well the ranges were merged.
  std::uint64_t max_bytes_in_flight = 1ull << 30;

  /// A policy that coalesces nothing and splits nothing -- one request per extent, as the
  /// pre-io_context reader behaved. Useful as a control.
  [[nodiscard]] static hpln_io_policy verbatim()
  {
    return hpln_io_policy{0, 0, std::numeric_limits<std::uint64_t>::max()};
  }
};

//===----------------------------------------------------------------------===//
// read planning
//===----------------------------------------------------------------------===//

/// A landing place for part of a file range.
struct hpln_dst {
  std::uint8_t* data  = nullptr;
  std::uint64_t bytes = 0;
};

/// A caller's want: `[offset, offset + sum(dst sizes))` delivered into @ref dst in file order.
/// Several destinations because a chunk's payload lands in pinned BLOCKS, which are not
/// contiguous -- and a range that is one request in the file may still be many buffers in memory.
struct hpln_extent {
  std::uint64_t offset = 0;
  std::vector<hpln_dst> dst;

  [[nodiscard]] std::uint64_t bytes() const noexcept;
};

/// One physical read: a contiguous file range scattered into @ref dst.
struct hpln_request {
  std::uint64_t offset = 0;
  std::uint64_t bytes  = 0;
  std::vector<hpln_dst> dst;
};

/// The requests that serve a set of extents, plus the scratch the bridged gaps land in.
///
/// The scratch is owned here because the requests point INTO it: a bridged gap is bytes nobody
/// asked for, and they still need somewhere to go. Moving the plan is safe (a vector move keeps
/// its buffer); copying it is not, so it is move-only.
struct hpln_read_plan {
  std::vector<hpln_request> requests;
  std::vector<std::uint8_t> scratch;
  /// Bytes read only to avoid a request. The cost side of the 7.7 trade, so it is measurable.
  std::uint64_t bridged_bytes = 0;
  /// Bytes the caller actually asked for.
  std::uint64_t wanted_bytes = 0;

  hpln_read_plan()                                 = default;
  hpln_read_plan(hpln_read_plan&&)                 = default;
  hpln_read_plan& operator=(hpln_read_plan&&)      = default;
  hpln_read_plan(hpln_read_plan const&)            = delete;
  hpln_read_plan& operator=(hpln_read_plan const&) = delete;
};

/// Turn @p extents into as few, as large and as well-placed requests as @p policy allows.
///
/// Extents are sorted by offset and merged while the gap between them is at most
/// `policy.max_gap_bytes` and the merged run stays within `policy.target_request_bytes`; a run
/// longer than that is then cut into pieces of at most that size. Every requested byte lands in
/// exactly one destination, in file order -- a planner that dropped or reordered one would not
/// fault, it would hand the decoder a neighbour's bytes as values, so
/// @ref plan_hpln_reads is exercised directly by the unit tests.
///
/// Overlapping extents are rejected (throws): two destinations for one byte is a caller bug that
/// silently half-works.
[[nodiscard]] hpln_read_plan plan_hpln_reads(std::vector<hpln_extent> extents,
                                             hpln_io_policy const& policy);

//===----------------------------------------------------------------------===//
// transport
//===----------------------------------------------------------------------===//

/// What a source did, so a caller (or a test) can tell WHICH transport served a read and how many
/// requests it took. A silent fall back to the filesystem when an io_context was wanted is
/// otherwise invisible: the bytes are identical.
struct hpln_io_stats {
  std::string transport;
  std::size_t requests       = 0;
  std::size_t extents        = 0;
  std::uint64_t bytes_read   = 0;  ///< including bridged gaps
  std::uint64_t bytes_wanted = 0;
};

/// A .hpln open for reading, whatever it is stored on.
class hpln_source {
 public:
  virtual ~hpln_source();

  hpln_source(hpln_source const&)            = delete;
  hpln_source& operator=(hpln_source const&) = delete;

  [[nodiscard]] virtual std::uint64_t size() const noexcept = 0;

  /// Identifies the transport in errors, logs and tests: "ifstream", or "io_context:<backend>".
  [[nodiscard]] virtual std::string_view transport() const noexcept = 0;

  /// Serve @p requests, throwing on a short read. Implementations may run them concurrently, so
  /// the destinations must be disjoint (which @ref plan_hpln_reads guarantees).
  virtual void submit(std::span<hpln_request const> requests,
                      hpln_io_policy const& policy,
                      char const* what) = 0;

  /// Plan @p extents under @p policy and serve them.
  void read_extents(std::vector<hpln_extent> extents,
                    hpln_io_policy const& policy,
                    char const* what);

  /// Read exactly `[offset, offset + bytes)` into @p dst.
  void read_at(std::uint64_t offset, std::uint64_t bytes, void* dst, char const* what);

  /// Read exactly `[offset, offset + bytes)` into a fresh buffer.
  [[nodiscard]] std::vector<std::uint8_t> read_range(std::uint64_t offset,
                                                     std::uint64_t bytes,
                                                     char const* what);

  /// Read the last `min(want, size())` bytes -- the read that locates a .hpln.
  [[nodiscard]] std::vector<std::uint8_t> read_tail(std::uint64_t want, char const* what);

  /// Read the first `min(want, size())` bytes, for a file that predates the trailer.
  [[nodiscard]] std::vector<std::uint8_t> read_prefix(std::uint64_t want, char const* what);

  [[nodiscard]] hpln_io_stats const& stats() const noexcept { return _stats; }

 protected:
  hpln_source() = default;
  hpln_io_stats _stats;
};

/// Open @p path for reading.
///
/// With an @p io_ctx the reads go through that backend -- which is the only way an `s3://` path
/// can be read at all. Without one they go through the local filesystem, and a path carrying a
/// URI scheme is REFUSED rather than quietly handed to `std::ifstream`: a remote read that
/// silently became a local one would fail with "no such file" at best, and at worst find a
/// same-named local file and return the wrong data.
///
/// @p who names the caller in error messages. Throws std::runtime_error if the file cannot be
/// opened or its size cannot be resolved.
[[nodiscard]] std::unique_ptr<hpln_source> open_hpln_source(
  std::string const& path, std::shared_ptr<io::sirius_ioctx> io_ctx, char const* who);

}  // namespace sirius
