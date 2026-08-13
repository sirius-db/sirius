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

// Turning one selection into per-batch work, once (env gate:
// SIRIUS_EXP_LATE_MAT).
//
// A deferred selection is resolved against SEVERAL columns of the same origin
// table — that is the whole point of deferring, since the rows survive the join
// once and each deferred column then has to be produced for them. Sorting,
// deduplicating, splitting by batch and bucketing into chunk CSRs depends only
// on the ids and the table's batch layout, not on which column is being
// produced. So it happens once here, and the per-column materialization reads
// the result.
//
// The pipeline is codegen/selection/row_id_space.hpp end to end: sort + dedup
// (keeping the ranks that restore the caller's order), split at batch
// boundaries, narrow to batch-local ids, bucket each batch into a chunk CSR.
//
// DENSE BATCHES ARE NOT PREPARED. A batch whose rows all survive needs no
// selection at all — it is a full decode or a copy — so it carries no CSR and
// no index list, and skips the host sync that building one would cost. This is
// not a rare case: a selection that is sparse over the table is often dense
// over the batches it does touch.

#include "late_mat/column_origin.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <codegen/selection/chunk_row_set.hpp>

#include <cstdint>
#include <vector>

namespace sirius::late_mat {

/// Column-agnostic layout of a pinned table: how its rows are distributed over
/// batches, in emission order. Shared by every column prepared from it.
struct pinned_table_layout {
  std::vector<std::int64_t> batch_rows;       ///< rows per batch, emission order
  std::vector<std::int64_t> batch_row_start;  ///< B+1, exclusive scan of batch_rows,
                                              ///< last == total pinned rows
  pin_generation_t pin_generation = 0;

  /// Derive batch_row_start from batch_rows, so the two cannot disagree.
  static pinned_table_layout from_batch_rows(std::vector<std::int64_t> rows,
                                             pin_generation_t generation = 0);

  [[nodiscard]] std::size_t num_batches() const noexcept { return batch_rows.size(); }
  [[nodiscard]] std::int64_t total_rows() const noexcept
  {
    return batch_row_start.empty() ? 0 : batch_row_start.back();
  }
};

/// A device list of global row ids — positions in pinned-table order.
///
/// Set `sorted_unique` when the caller can promise ascending, duplicate-free
/// ids: the sort and dedup are then skipped and the output stays in table
/// order, with no restoring gather. Otherwise the ids may arrive in any order
/// with repeats, which is what a join hands back.
///
/// LIFETIME: the buffer behind `ids` must outlive the prepare call. It is read,
/// never adopted.
struct row_id_list {
  std::uint64_t const* ids = nullptr;  ///< device
  std::int64_t count       = 0;
  bool sorted_unique       = false;
};

/// One batch's share of a prepared selection.
struct batch_selection {
  codegen::chunk_row_set_owner rows;  ///< batch-local chunk CSR; empty when dense
  rmm::device_buffer local_indices;   ///< int32 x survivors, ascending, batch-local;
                                      ///< empty when dense
  std::int64_t survivors = 0;
  double density         = 0.0;    ///< survivors / batch rows
  bool dense             = false;  ///< every row of the batch survives
};

/// One selection, resolved against a table's batch layout and ready for any
/// number of columns of that table. Immutable once prepared.
struct prepared_selection {
  pinned_table_layout layout;
  std::vector<batch_selection> batches;  ///< parallel to layout.batch_rows
  std::vector<std::int64_t> out_base;    ///< B+1, exclusive scan of per-batch
                                         ///< survivors; where batch b's rows start
                                         ///< in the materialized output
  std::int64_t total_survivors = 0;
  std::int64_t original_count  = 0;  ///< ids the caller passed in

  /// int32 x original_count, mapping each of the caller's ids to its row in the
  /// materialized output. Empty when the input was already sorted and unique,
  /// which is exactly when the output is already in the caller's order.
  rmm::device_buffer restore_rank;

  [[nodiscard]] bool needs_restore() const noexcept { return restore_rank.size() != 0; }
};

/// Resolve `ids` against `layout`.
///
/// Costs one host sync for the batch boundaries, plus one per non-dense batch
/// for its CSR (the touched-chunk count is a grid, and a grid is a host-side
/// value). Both are prepare-time, paid once for all the columns that follow.
///
/// Throws if the ids do not lie within the layout's rows — an id outside the
/// pinned table means the caller's addressing disagrees with the pin, and
/// materializing under that disagreement would produce plausible wrong rows.
prepared_selection prepare_selection(pinned_table_layout const& layout,
                                     row_id_list const& ids,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

}  // namespace sirius::late_mat
