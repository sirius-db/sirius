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
// for the index list over C chunks. The crossover is clustering, not density —
// scattered survivors touch every chunk and gain only the narrower ids.

#pragma once

#include "codegen/jit/fused_tree.hpp"

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

}  // namespace sirius::codegen
