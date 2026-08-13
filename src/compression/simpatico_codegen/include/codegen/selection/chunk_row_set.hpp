// SPDX-License-Identifier: Apache-2.0
//
// A row selection that arrives AFTER the scan — post-join survivor ids — in the
// form the decode can consume.
//
// The selection types in selection.hpp describe rows the DECODE itself chose: a
// bitmap it balloted, or the ascending index list built from that bitmap. Both
// are dense over the batch's chunks, because every chunk was a candidate. A
// selection handed back later is not: a join may leave a few thousand rows
// spread over a handful of chunks out of tens of thousands, and the difference
// between "every chunk, most of them empty" and "only the chunks that survive"
// is the whole cost at that density.
//
// So this is CSR (compressed sparse row) over chunks — "rows" being 1024-row
// chunks, "entries" being surviving rows:
//
//   chunk_ids[b]        which chunk block b serves        (ascending, T entries)
//   block_offsets[b]    where block b's output starts     (T + 1 entries)
//   in_chunk_rows[]     positions 0..1023 within a chunk  (S entries, uint16)
//
// Block b reads in_chunk_rows[block_offsets[b] .. block_offsets[b+1]) and writes
// from block_offsets[b]. Empty chunks are absent rather than launched-and-
// skipped, and the output is in ascending row order by construction.
//
// Why uint16 positions: they cannot address another chunk. The index list's
// global ids are turned into in-chunk positions by subtracting chunk_start,
// which is correct only while every id in a block's slice really belongs to that
// block's chunk — an invariant the mask->indices wave supplies but a
// post-join caller would have to be trusted for. A position that cannot exceed
// 1023 makes the invariant structural.
//
// Size, for S survivors over T touched chunks: 2S + 8T bytes, against 4S + 4(C+1)
// for the index list over C chunks. The crossover is DENSITY, not clustering:
// survivors falling at random still leave most chunks empty once density is low
// (a chunk is touched with probability 1-(1-d)^1024), so T/C collapses on its
// own. On TPC-H sf1000 q17 and q19 sit exactly on that random baseline — no
// clustering at all — and still skip 5.4M of 5.86M empty blocks; q18 is the one
// genuinely clustered case, at T/C 0.011 against a 0.073 baseline.
//
// So this form never launches more blocks than the index list and never occupies
// more memory. What it costs instead is construction: the index list falls out
// of the mask->indices wave, while the CSR has to be bucketed.

#pragma once

#include "codegen/jit/fused_tree.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>

namespace sirius::codegen {

/// Non-owning device view. The buffers belong to whoever built the selection;
/// the decode only reads them.
struct chunk_row_set {
  /// Batch-local chunk ids, ascending, one per launched block.
  std::uint32_t const* chunk_ids = nullptr;
  /// Exclusive prefix of per-block survivor counts; num_touched + 1 entries,
  /// last == num_survivors. Indexed by BLOCK, not by chunk id — that is what
  /// lets the grid skip untouched chunks.
  std::uint32_t const* block_offsets = nullptr;
  /// In-chunk positions, grouped by block and ascending within each.
  std::uint16_t const* in_chunk_rows = nullptr;

  std::int64_t num_touched   = 0;  ///< T: blocks to launch
  std::int64_t num_survivors = 0;  ///< S: rows the decode will write
  std::int64_t num_rows      = 0;  ///< the batch's row count, for bounds checks

  [[nodiscard]] bool valid() const noexcept
  {
    if (num_survivors == 0) { return true; }  // an empty selection needs no arrays
    return chunk_ids != nullptr && block_offsets != nullptr && in_chunk_rows != nullptr &&
           num_touched > 0 && num_touched <= num_survivors &&
           num_touched <= (num_rows + ::codegen::kChunkSize - 1) / ::codegen::kChunkSize;
  }
};

/// Owning storage behind a chunk_row_set. The view is non-owning by design —
/// the decode only reads — so this is what a builder returns and what the
/// caller must keep alive for as long as any launch is still reading it.
struct chunk_row_set_owner {
  rmm::device_buffer chunk_ids;      // uint32 x num_touched
  rmm::device_buffer block_offsets;  // uint32 x (num_touched + 1)
  rmm::device_buffer in_chunk_rows;  // uint16 x num_survivors

  std::int64_t num_touched   = 0;
  std::int64_t num_survivors = 0;
  std::int64_t num_rows      = 0;

  [[nodiscard]] chunk_row_set view() const noexcept
  {
    return chunk_row_set{static_cast<std::uint32_t const*>(chunk_ids.data()),
                         static_cast<std::uint32_t const*>(block_offsets.data()),
                         static_cast<std::uint16_t const*>(in_chunk_rows.data()),
                         num_touched,
                         num_survivors,
                         num_rows};
  }
};

/// Bucket a selection that arrived after the scan into the CSR above.
///
/// ``row_ids`` are batch-local row ids on device, STRICTLY INCREASING and each
/// in [0, num_rows). A join may well hand the same row back many times, but a
/// repeat here would decode that row once per reference; deduplication belongs
/// upstream, where sort_unique_global_ids drops the repeats and keeps the ranks
/// that replay them from the compact output (row_id_space.hpp). So a duplicate
/// reaching this point is an upstream bug, and is rejected as one.
///
/// Cost is O(num_ids) — no pass over the batch's chunks. That is the point: a
/// selection touching 1% of chunks must not pay for the 99% it skips, which is
/// exactly what a per-chunk counter array would charge. One host sync, for the
/// touched-chunk count, because that count is the grid the launcher needs.
///
/// Throws if the ids are out of order or out of range, rather than building a
/// row set that would decode the wrong rows.
chunk_row_set_owner build_chunk_row_set(std::int32_t const* row_ids,
                                        std::int64_t num_ids,
                                        std::int64_t num_rows,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);

}  // namespace sirius::codegen
