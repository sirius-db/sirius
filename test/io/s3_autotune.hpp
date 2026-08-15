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

// Request-shaping model for s3_autotune_throughput_bench: how many connections
// to open, and which of a file's wanted byte ranges are worth merging into one
// GET.
//
// Two decisions, one model.  Both follow from the two numbers that describe an
// object store across a network: R, the MB/s a single connection sustains, and
// L, the round trip before its first byte arrives.
//
//   * How many connections?  One stream gives R and no more, so filling a NIC
//     takes ceil(NIC / R) of them -- see @ref plan_connections.
//   * How big is a GET, and which of the ranges a query wants share one?  A GET
//     of S bytes takes L + S/R, so the product L*R is what one request costs
//     measured in bytes.  That single quantity sets both the GET size that
//     reaches a target efficiency and the gap that is cheaper to read through
//     than to skip -- see @ref coalesce_and_chunk.
//
// The ranges @ref coalesce_and_chunk merges come from a real query: the
// benchmark asks a @c hybrid_scan_reader for the column-chunk byte ranges a
// TPC-H query's projected columns need from each file, and this model decides
// how those get shaped into GETs -- it does not generate a workload itself.
//
// Sizes are bytes throughout; rates are decimal (MB = 10^6 B) so a Gbps line
// rate converts exactly and the derived byte counts line up with the GB/s the
// benchmark reports.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sirius::bench::autotune {

inline constexpr double bytes_per_mb_v = 1e6;
inline constexpr double bytes_per_gb_v = 1e9;

/// A byte span of one object: [offset, offset + length).
struct byte_span {
  std::size_t offset{0};
  std::size_t length{0};

  [[nodiscard]] std::size_t end() const noexcept { return offset + length; }
};

/// What the link and the endpoint are worth, measured or assumed.
struct tuning_params {
  /// Line rate of the NIC we are trying to fill.
  double nic_gbps{100.0};

  /// MB/s one S3 connection sustains.  This is the endpoint's per-stream
  /// ceiling, not a share of the NIC -- it is the whole reason more than one
  /// connection is needed.
  double stream_mbps{100.0};

  /// Time to first byte of a ranged GET, seconds.
  double rtt_s{0.015};

  /// Target share of the wall time of a GET spent moving bytes rather than
  /// waiting out the round trip.  Sets the GET size; 1.0 is unreachable.
  double transfer_efficiency{0.85};

  /// Minimum share of a coalesced GET that the caller actually asked for.  Its
  /// complement is the bandwidth we are willing to throw away in order to skip
  /// a round trip.
  double gap_efficiency{0.80};
};

// ---------------------------------------------------------------------------
// connection sizing
// ---------------------------------------------------------------------------

struct connection_plan {
  /// Streams it takes to fill the NIC.  A ceiling: independent of the workload.
  std::size_t nic_connections{1};

  /// The above, scaled back to what this particular batch can keep busy.
  std::size_t active_connections{1};

  /// GET size that reaches @c transfer_efficiency.
  std::size_t chunk_bytes{0};

  /// L*R -- what one round trip is worth in bytes, i.e. the largest gap worth
  /// reading through instead of paying for another GET.
  std::size_t rtt_bytes{0};
};

/**
 * @brief Size the connection pool and the GET for one batch of reads.
 *
 * @param p                        link / endpoint model
 * @param workload_bytes           bytes the batch needs (useful bytes, not wire
 *                                 bytes -- coalescing has not happened yet)
 * @param min_chunks_per_connection GETs a connection must have queued to be
 *                                 worth opening.  With fewer, its round trip is
 *                                 never hidden behind a transfer and its
 *                                 congestion window never opens.
 * @param min_chunk_bytes          floor on the GET size, whatever the
 *                                 efficiency solve asks for.
 */
[[nodiscard]] inline connection_plan plan_connections(tuning_params const& p,
                                                      std::size_t workload_bytes,
                                                      std::size_t min_chunks_per_connection = 2,
                                                      std::size_t min_chunk_bytes = 8UL << 20)
{
  connection_plan plan;

  // 1 Gbps == 125 MB/s.  Both sides are MB/s, so the ratio is just how many
  // per-stream ceilings the link has room for.
  double const stream   = std::max(p.stream_mbps, 1e-9);
  double const nic_mbps = p.nic_gbps * 125.0;
  plan.nic_connections =
    std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(nic_mbps / stream)));

  // A GET of S bytes takes L + S/R, so the share of it that moves bytes is
  // (S/R) / (L + S/R).  Setting that to eta and solving gives
  // S = eta/(1-eta) * L * R.  eta is clamped below 1 because the solve diverges
  // there: no finite GET is 100% transfer.
  double const eta = std::clamp(p.transfer_efficiency, 0.0, 0.99);
  plan.rtt_bytes   = static_cast<std::size_t>(std::ceil(p.rtt_s * stream * bytes_per_mb_v));
  plan.chunk_bytes = std::max(
    min_chunk_bytes,
    static_cast<std::size_t>(std::ceil((eta / (1.0 - eta)) * p.rtt_s * stream * bytes_per_mb_v)));

  // Opening a connection only pays off if it has min_chunks_per_connection GETs
  // to run back to back, so the batch itself caps the count: past
  // workload / (k * chunk) every extra connection gets one short GET, and a
  // short GET is mostly round trip.
  std::size_t const per_connection =
    std::max<std::size_t>(1, min_chunks_per_connection) * plan.chunk_bytes;
  plan.active_connections =
    std::clamp<std::size_t>(workload_bytes / per_connection, 1, plan.nic_connections);
  return plan;
}

// ---------------------------------------------------------------------------
// request shaping
// ---------------------------------------------------------------------------

/// What @ref coalesce_and_chunk decided.
struct request_plan {
  /// Exactly what goes on the wire: one ranged GET per entry.
  std::vector<byte_span> requests;

  /// Coalesced blocks, before the size cap cut them up.
  std::size_t n_blocks{0};

  /// Bytes of the input segments (overlaps counted once).
  std::size_t useful_bytes{0};

  /// Bytes actually fetched: useful bytes plus the gaps read through.
  std::size_t wire_bytes{0};
};

/**
 * @brief Merge segments across gaps that cost less to read than to skip, then
 *        cut the result into GETs of at most @p max_chunk_bytes.
 *
 * Merging: a gap of g bytes costs g/R to read through and saves L by not
 * issuing another GET, so it is worth reading whenever g <= L*R
 * (@p rtt_bytes).  That test is per gap; @p max_waste_ratio is the cumulative
 * guard, because many individually cheap gaps still add up to a request that is
 * mostly bytes nobody asked for.
 *
 * Chunking: one fat GET is a straggler the whole batch waits on, and it can
 * only ever occupy a single connection.  Cutting to a uniform cap spreads the
 * block across the pool and bounds the tail.
 *
 * @p merge_within_chunk additionally bounds a merge by @p max_chunk_bytes.  Off
 * (how the model was first written) a merge may grow past the cap and then be
 * split again -- which can leave MORE requests than not merging at all, and pay
 * for the gap bytes on top: two 8 MiB segments either side of a 1 MiB gap are 2
 * GETs apart but 3 GETs merged.  On, a merge only happens when it actually
 * removes a round trip.  The benchmark exposes both so the difference can be
 * measured rather than argued.
 */
[[nodiscard]] inline request_plan coalesce_and_chunk(std::vector<byte_span> segments,
                                                     std::size_t rtt_bytes,
                                                     double max_waste_ratio,
                                                     std::size_t max_chunk_bytes,
                                                     bool merge_within_chunk = false)
{
  request_plan plan;
  if (segments.empty()) { return plan; }

  std::size_t const cap = std::max<std::size_t>(max_chunk_bytes, 1);
  std::ranges::sort(segments, {}, &byte_span::offset);

  struct block {
    std::size_t offset;
    std::size_t length;  ///< span covered, gaps included
    std::size_t useful;
    std::size_t gaps;
  };

  std::vector<block> blocks;
  blocks.reserve(segments.size());
  block current{segments[0].offset, segments[0].length, segments[0].length, 0};

  for (std::size_t i = 1; i < segments.size(); ++i) {
    auto const& next           = segments[i];
    std::size_t const cur_end  = current.offset + current.length;
    std::size_t const next_end = next.end();

    // Sorted by offset, so `next` can only ever extend the block to the right.
    // Overlapping input is not what a scan produces, but a segment contained in
    // the block must neither shrink it nor have its bytes counted twice.
    std::size_t const gap   = next.offset > cur_end ? next.offset - cur_end : 0;
    std::size_t const fresh = next_end > cur_end ? next_end - std::max(next.offset, cur_end) : 0;

    std::size_t const cand_useful = current.useful + fresh;
    std::size_t const cand_gaps   = current.gaps + gap;
    std::size_t const cand_total  = cand_useful + cand_gaps;
    double const cand_waste =
      cand_total > 0 ? static_cast<double>(cand_gaps) / static_cast<double>(cand_total) : 0.0;

    if (gap <= rtt_bytes && cand_waste <= max_waste_ratio &&
        (!merge_within_chunk || cand_total <= cap)) {
      current.length = std::max(cur_end, next_end) - current.offset;
      current.useful = cand_useful;
      current.gaps   = cand_gaps;
    } else {
      blocks.push_back(current);
      current = {next.offset, next.length, next.length, 0};
    }
  }
  blocks.push_back(current);

  plan.n_blocks = blocks.size();
  plan.requests.reserve(blocks.size());
  for (auto const& b : blocks) {
    plan.useful_bytes += b.useful;
    plan.wire_bytes += b.length;
    for (std::size_t pos = 0; pos < b.length; pos += cap) {
      plan.requests.push_back({b.offset + pos, std::min(cap, b.length - pos)});
    }
  }
  return plan;
}

}  // namespace sirius::bench::autotune
