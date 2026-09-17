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

// Turning one selection into per-batch work, at most once (env gate:
// SIRIUS_EXP_LATE_MAT).
//
// A deferred selection is resolved against SEVERAL columns of the same origin
// table — that is the point of deferring, since the rows survive the join once
// and each deferred column then has to be produced for them. Sorting,
// deduplicating, splitting by batch and bucketing into chunk CSRs depends only
// on the ids and the table's batch layout, not on which column is produced, so
// it is done once and shared.
//
// BUT NOT EAGERLY, because for some consumers it is pure loss. A gather needs
// neither sorted nor unique ids and emits in the caller's own order, so an
// uncompressed column materializes directly from the raw id list; making it
// sorted and unique first only adds a sort, and then a second gather to undo
// the ordering. Deduplication pays when a row costs a lot to produce — a
// compressed row is decoded, so producing it twice is a real waste — and costs
// when a row is merely copied.
//
// So the canonical form is built on FIRST USE that needs it and shared from
// then on: a table of uncompressed columns never builds one, a table with a
// compressed column builds it once for all of them. The prepared selection
// stays logically immutable, so concurrent materializes remain safe.
//
// DENSE BATCHES ARE NOT PREPARED. A batch whose rows all survive needs no
// selection — it is a full decode or a copy — so it carries no CSR and no index
// list, and skips the host sync building one would cost. Not a rare case: a
// selection sparse over a table is often dense over the batches it touches.

#include "late_mat/column_origin.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <codegen/selection/chunk_row_set.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
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
/// ids: canonicalizing is then a no-op and is skipped even when a consumer asks
/// for it. Otherwise the ids may arrive in any order with repeats, which is
/// what a join hands back.
///
/// LIFETIME: the buffer behind `ids` is BORROWED, not adopted, and must outlive
/// every materialize against the prepared selection — not merely the prepare
/// call. That is what lets an uncompressed column read it directly as a gather
/// map instead of copying it into a canonical form it does not need.
struct row_id_list {
  std::uint64_t const* ids = nullptr;  ///< device
  std::int64_t count       = 0;
  bool sorted_unique       = false;
};

/// One batch's share of the canonical form.
struct batch_selection {
  codegen::chunk_row_set_owner rows;  ///< batch-local chunk CSR; empty when dense
  rmm::device_buffer local_indices;   ///< int32 x survivors, ascending, batch-local;
                                      ///< empty when dense
  std::int64_t survivors = 0;
  double density         = 0.0;    ///< survivors / batch rows
  bool dense             = false;  ///< every row of the batch survives
};

/// The selection sorted, deduplicated and split across the table's batches —
/// what a consumer needs when producing a row twice would mean decoding it
/// twice.
struct canonical_selection {
  std::vector<batch_selection> batches;  ///< parallel to layout.batch_rows
  std::vector<std::int64_t> out_base;    ///< B+1, exclusive scan of per-batch
                                         ///< survivors; where batch b's rows start
  std::int64_t total_survivors = 0;

  /// int32 x original_count, mapping each of the caller's ids to its row in the
  /// canonical output. Empty when the input was already sorted and unique,
  /// which is exactly when that output is already in the caller's order.
  rmm::device_buffer restore_rank;

  [[nodiscard]] bool needs_restore() const noexcept { return restore_rank.size() != 0; }
};

/// One selection against one table's layout, ready for any number of its
/// columns. Construction does no device work.
class prepared_selection {
 public:
  prepared_selection(pinned_table_layout layout, row_id_list ids);

  [[nodiscard]] pinned_table_layout const& layout() const noexcept { return _layout; }
  [[nodiscard]] row_id_list const& ids() const noexcept { return _ids; }
  [[nodiscard]] std::int64_t original_count() const noexcept { return _ids.count; }

  /// The canonical form, built on first call and shared thereafter.
  ///
  /// Costs one host sync for the batch boundaries plus one per non-dense batch
  /// for its CSR — a touched-chunk count is a grid, and a grid is a host-side
  /// value. Callers that do not need it must not ask for it; that is the whole
  /// reason it is a call and not a field.
  ///
  /// Throws if the ids do not lie within the layout's rows: an id outside the
  /// pinned table means the caller's addressing disagrees with the pin, and
  /// materializing under that disagreement would produce plausible wrong rows.
  [[nodiscard]] canonical_selection const& canonical(rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr) const;

  /// Whether anything has needed the canonical form yet. Diagnostics and tests
  /// — the point of the laziness is that this stays false for a table of
  /// uncompressed columns.
  [[nodiscard]] bool has_canonical() const noexcept { return _canonical != nullptr; }

 private:
  pinned_table_layout _layout;
  row_id_list _ids;
  mutable std::once_flag _once;
  mutable std::unique_ptr<canonical_selection> _canonical;
};

}  // namespace sirius::late_mat
