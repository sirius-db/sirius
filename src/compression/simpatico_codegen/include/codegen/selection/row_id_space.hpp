// SPDX-License-Identifier: Apache-2.0
//
// Turning an arbitrary post-join id list into what a chunk CSR can be built
// from (codegen/selection/chunk_row_set.hpp).
//
// A join hands back row ids in ITS order, with repeats, spanning every batch of
// the pinned entry. The bucketer wants the opposite: one batch's worth, sorted,
// duplicate-free. This is that conversion, and the three things it has to
// arrange are independent enough to be separate calls:
//
//   sort_unique_global_ids     order + dedup, keeping the ranks that undo it
//   split_sorted_ids_by_batch  where each batch's slice of the sorted list is
//   global_slice_to_local      global ids -> one batch's local ids
//
// WHY DEDUP IS WORTH A SORT. A repeat costs a full decode of that row, so a
// many-to-many join referencing one row twenty times would decode it twenty
// times. Deduplicating collapses that to one decode plus a rank lookup, and the
// ranks are what let the caller rebuild its own order — including the repeats —
// with a gather over the compact output, which is narrow by construction.
//
// WHY IDS ARE 64-BIT HERE AND 32-BIT BELOW. A global id addresses the whole
// pinned entry, and lineitem at sf1000 is 6.0e9 rows — past what 32 bits hold.
// Batch-local ids are not: one batch is far below 2^31, so the id narrows to
// int32 at exactly the point it becomes batch-local, which is also where it
// stops being ambiguous about which batch it means.

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <vector>

namespace sirius::codegen {

/// A sorted, deduplicated id list plus the ranks that restore the caller's
/// order. ``restore_rank[i]`` is where the caller's element i landed in the
/// deduplicated array, so the caller reorders a compact result of
/// ``unique_count`` values back to its own ``original_count`` rows — repeats
/// included — with one gather by ``restore_rank``.
struct sorted_unique_ids {
  rmm::device_buffer ids;           ///< uint64 x original_count; the first
                                    ///< unique_count entries are the ascending
                                    ///< distinct ids, the rest are scratch.
  rmm::device_buffer restore_rank;  ///< int32 x original_count. Indexes into the deduplicated
                                    ///< array, so its values run up to unique_count -- which is
                                    ///< bounded differently than the batch-local ids below (see
                                    ///< "WHY IDS ARE 64-BIT HERE AND 32-BIT BELOW" above): it is
                                    ///< the distinct-id count over the WHOLE original_count input,
                                    ///< not a single batch. sort_unique_global_ids enforces the
                                    ///< int32 fit directly -- it throws when original_count (an
                                    ///< upper bound on unique_count) exceeds INT32_MAX -- rather
                                    ///< than relying on callers staying under it by construction.
  rmm::device_buffer count_dev;     ///< int32 x 1: unique_count, ON DEVICE
  std::int64_t original_count = 0;
};

/// Sort ``ids`` ascending, drop the repeats, and record the restoring ranks.
///
/// Deliberately does NOT sync: the unique count is left on device, for the
/// caller to fold into a sync it already has to make. ``ids`` is sized for the
/// worst case (no duplicates at all) rather than for the answer, which is what
/// makes that possible — sizing exactly would need the count on the host, and
/// that is the sync this avoids.
sorted_unique_ids sort_unique_global_ids(std::uint64_t const* ids,
                                         std::int64_t count,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

/// Where each batch's slice of a SORTED global id list begins.
///
/// ``batch_row_start`` is the exclusive scan of per-batch row counts (B + 1
/// entries, last == total pinned rows); the result has B + 1 entries, with
/// entry k the first index whose id is >= batch_row_start[k]. Batch k's ids are
/// therefore [result[k], result[k+1]).
///
/// This is the one host sync of the conversion, and it is structural rather
/// than incidental: slicing the list and sizing each batch's buffers are host
/// decisions, so the boundaries have to reach the host. Everything else that
/// needs a sync is folded into this one — pass ``count_dev`` from
/// sort_unique_global_ids and the search bounds itself by the device value,
/// with the count landing in ``count_out`` on the same trip (``max_count`` is
/// then only an upper bound on the search space).
std::vector<std::int64_t> split_sorted_ids_by_batch(
  std::uint64_t const* sorted_ids,
  std::int64_t max_count,
  std::int32_t const* count_dev,
  std::vector<std::int64_t> const& batch_row_start,
  std::int64_t* count_out,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/// One batch's slice of global ids to batch-local ids: ``out[i] = ids[i] -
/// batch_row_start``. Asynchronous.
///
/// An id that does not belong to this batch is not checked for here, because it
/// cannot escape: the subtraction puts it outside [0, batch rows), which is
/// exactly what build_chunk_row_set already rejects. One check, at the point
/// that has to make it anyway.
void global_slice_to_local(std::uint64_t const* ids,
                           std::int64_t count,
                           std::int64_t batch_row_start,
                           std::int32_t* out_local,
                           rmm::cuda_stream_view stream);

}  // namespace sirius::codegen
