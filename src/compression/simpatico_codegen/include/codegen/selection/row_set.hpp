// SPDX-License-Identifier: Apache-2.0
//
// Chunk-bucketed row sets for late materialization (SIRIUS_EXP_LATE_MAT).
//
// NEW header — nothing in the fused scan-filter pipeline includes it; the
// shipped selection types stay in codegen/selection/selection.hpp untouched.
//
// A chunk_row_set is the canonical device form of an arbitrary row selection
// arriving AFTER scan time (post-join survivor ids): survivors grouped by
// 1024-row chunk in CSR form, in-chunk positions stored as uint16.
//
// Representation arithmetic (S survivors, T touched chunks, n rows):
//   u64 flat id list      8*S bytes
//   chunk-CSR (this)      2*S + 8*T (+4) bytes    — wins when S/T > 4/3
//   fused bitvector       n/8 + 4*(n/1024+1)      — wins above ~6% density
// Plain u32 global ids do NOT cover lineitem SF1000 (6.0e9 rows > 2^32), so
// the flat boundary form is uint64 and the compact interior form is this CSR.
//
// The sparse decode variants (DecodeVariant::sparse_*) consume exactly these
// three arrays: grid = num_touched_chunks blocks, block b serves chunk
// chunk_ids[b], its survivors at in_chunk_offsets[chunk_out_offsets[b] ..
// chunk_out_offsets[b+1]), writing compacted output from base
// chunk_out_offsets[b]. Output order is ascending row order by construction.

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>

namespace sirius::codegen {

// Non-owning device views + host-known counts. All device arrays live on
// stream-ordered RMM allocations owned by the caller (late_mat's
// prepared_selection); consumers only read.
struct chunk_row_set {
  // Batch-local chunk ids of touched chunks, ascending. num_touched entries.
  std::uint32_t const* chunk_ids = nullptr;
  // Exclusive prefix of per-touched-chunk survivor counts; num_touched + 1
  // entries, last == num_survivors. Indexed by BLOCK (position in chunk_ids),
  // not by chunk id — this is what lets the launch grid skip empty chunks.
  std::uint32_t const* chunk_out_offsets = nullptr;
  // In-chunk row positions (0..1023), grouped by touched chunk, ascending
  // within each chunk. num_survivors entries.
  std::uint16_t const* in_chunk_offsets = nullptr;

  std::int64_t num_touched = 0;    // T
  std::int64_t num_survivors = 0;  // S
  std::int64_t num_rows = 0;       // n — the batch's total rows (grid domain)

  [[nodiscard]] bool valid() const noexcept
  {
    return (num_survivors == 0) ||
           (chunk_ids != nullptr && chunk_out_offsets != nullptr &&
            in_chunk_offsets != nullptr && num_touched > 0);
  }
};

// Owning form of one batch's bucketed selection (buffers on `stream`).
struct owned_chunk_row_set {
  rmm::device_buffer chunk_ids;          // uint32 x num_touched
  rmm::device_buffer chunk_out_offsets;  // uint32 x (num_touched + 1)
  rmm::device_buffer in_chunk_offsets;   // uint16 x num_survivors
  std::int64_t num_touched = 0;
  std::int64_t num_survivors = 0;
  std::int64_t num_rows = 0;

  [[nodiscard]] chunk_row_set view() const noexcept
  {
    return chunk_row_set{static_cast<std::uint32_t const*>(chunk_ids.data()),
                         static_cast<std::uint32_t const*>(chunk_out_offsets.data()),
                         static_cast<std::uint16_t const*>(in_chunk_offsets.data()),
                         num_touched,
                         num_survivors,
                         num_rows};
  }

  void set_stream(rmm::cuda_stream_view stream)
  {
    if (chunk_ids.size() != 0) chunk_ids.set_stream(stream);
    if (chunk_out_offsets.size() != 0) chunk_out_offsets.set_stream(stream);
    if (in_chunk_offsets.size() != 0) in_chunk_offsets.set_stream(stream);
  }
};

// ── Bucketing (src/selection/latemat_rowset.cu) ─────────────────────────────
// All entry points are asynchronous on `stream` unless noted; they throw
// std::runtime_error on CUDA failure.

// Bucket SORTED-ASCENDING batch-local row ids (uint32, in [0, num_rows)) into
// a chunk-CSR. `sorted_local_ids` must be duplicate-free. This is the
// per-batch workhorse — the id-space decompose (global u64 -> per-batch u32)
// is the caller's, so this file stays independent of the global-id
// definition. HOST-SYNCS `stream` once (T must reach the host to size the
// CSR arrays exactly; late_mat calls this at prepare time only, per the
// scheduling contract).
owned_chunk_row_set bucket_sorted_local_ids(std::uint32_t const* sorted_local_ids,
                                            std::int64_t count,
                                            std::int64_t num_rows,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

// Expand a chunk-CSR into ascending batch-local int32 row indices
// (out[chunk_out_offsets[b] + k] = chunk_ids[b]*1024 + in_chunk_offsets[..k]),
// the shape the shipped K4 launcher / cudf::gather consume. `out` must hold
// num_survivors int32.
void row_set_to_local_indices(chunk_row_set const& set,
                              std::int32_t* out,
                              rmm::cuda_stream_view stream);

// ── Prepare-time id-space helpers (latemat_rowset.cu) ───────────────────────
// Global row ids are PIN-ORDER POSITIONS (a row's index in the concatenation
// of the pinned entry's batches in emission order — the
// late_mat/column_origin.hpp addressing). Batch-local conversion is ONE
// subtract (gid - batch_row_start), and the batch-local chunk/offset
// decompose is shifts/masks (local >> 10, local & 1023).

// Sort an arbitrary u64 global-id list ascending, deduplicate, and emit the
// order-restoration ranks: restore_rank[i] = position of ids[i]'s value in
// the deduped ascending array (int32 — one materialize output is one cudf
// column, < 2^31 rows).
//
// FULLY ASYNCHRONOUS (sync-surgery rev): no host sync inside. `ids` is
// allocated worst-case (original count entries; the first unique_count are
// valid, ascending — the unique scatter is idempotent, duplicates write the
// same value to the same rank slot), and the unique count is left in
// `count_dev` (one int32, device). The caller reads it back inside its own
// single boundary sync (split_sorted_ids_by_batch does this for
// prepare_selection). Prepare-time only.
struct sorted_unique_ids {
  rmm::device_buffer ids;           // uint64 x original count (worst-case; first
                                    // unique_count entries valid, ascending)
  rmm::device_buffer restore_rank;  // int32 x original count
  rmm::device_buffer count_dev;     // int32 x 1: the unique count (device)
  std::int64_t original_count = 0;
};
sorted_unique_ids sort_unique_global_ids(std::uint64_t const* ids,
                                         std::int64_t count,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

// Per-batch slice boundaries of a SORTED global-id list: returns B+1 start
// indices (host), starts[k] = first index with id >= batch_row_start[k]
// (batch_row_start has B+1 entries, exclusive scan of per-batch rows, last =
// total pinned rows).
//
// THE one boundary host sync of the canonical prepare path lives here: batch
// slicing and per-batch allocation are host-driven, so the starts must reach
// the host. Everything foldable is folded into that same sync:
//   * `count_dev` (nullable): the actual element count as a device scalar
//     (sort_unique_global_ids's unique count). The search kernel bounds
//     itself from the device value, so no separate count sync is needed
//     upstream; `max_count` is only the search-space upper bound.
//   * `count_out` (nullable): receives the actual count on the host, read in
//     the same sync (== max_count when count_dev is null).
// Prepare-time only.
std::vector<std::int64_t> split_sorted_ids_by_batch(
  std::uint64_t const* sorted_ids,
  std::int64_t max_count,
  std::int32_t const* count_dev,
  std::vector<std::int64_t> const& batch_row_start,
  std::int64_t* count_out,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

// Convert one batch's slice of sorted global ids to sorted batch-local u32
// ids (the bucket_sorted_local_ids input): out[i] = ids[i] -
// batch_row_start. Asynchronous.
void global_slice_to_local(std::uint64_t const* ids,
                           std::int64_t count,
                           std::int64_t batch_row_start,
                           std::uint32_t* out_local,
                           rmm::cuda_stream_view stream);

// ── Multi-source raw gather (SIRIUS_EXP_LATE_MAT_V2; latemat_rowset.cu) ─────
// One-pass gather from B per-batch base pointers by GLOBAL pin-order id: for
// each element, binary-search the batch (row_start, B+1 exclusive starts —
// B is small, the array stays cache-resident) and copy elem_size bytes from
// bases[b] + (id - row_start[b])*elem_size. Caller order preserved,
// duplicates and disorder fine (gather semantics), NO sort/restore — this is
// what replaces the canonical u64 sort for multi-batch UNCOMPRESSED
// fixed-width origins (~130-190 B/row of sort traffic -> S*(elem_size+8)
// bytes). elem_size in {1,2,4,8,16}. `bases`/`row_start` are DEVICE arrays
// (upload once per call; a few hundred bytes). Ids must be valid pin-order
// positions (same DONT_CHECK contract as the single-batch raw gather).
// Asynchronous on `stream`.
void multi_source_gather_fixed(void const* const* bases_dev,
                               std::int64_t const* row_start_dev,
                               std::int32_t num_batches,
                               std::size_t elem_size,
                               std::uint64_t const* ids,
                               std::int64_t count,
                               void* out,
                               rmm::cuda_stream_view stream);

// Expand a chunk-CSR into fused-format selection-mask words +
// per-ALL-chunks exclusive survivor offsets (the shipped K3/CNT shape):
// mask_words must hold selection_mask::WordsFor(num_rows) words and is
// fully written (untouched chunks/tail zero); all_chunk_offsets must hold
// ChunksFor(num_rows)+1 uint32 and is fully written. Enables the 100%-shipped
// mask route without re-running CNT. `mr` provides the scan scratch.
void row_set_to_mask(chunk_row_set const& set,
                     std::uint32_t* mask_words,
                     std::uint32_t* all_chunk_offsets,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr);

}  // namespace sirius::codegen
