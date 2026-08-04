// SPDX-License-Identifier: Apache-2.0
//
// Late materializer (SIRIUS_EXP_LATE_MAT) — materialize a deferred pinned
// column from (column origin, row selection), decoding/gathering ONLY the
// selected rows.
//
// Two-phase, per the scheduling contract:
//   prepare_selection  — once per (origin table, selection). Sorts/dedups/
//                        chunk-buckets the ids, splits by pinned batch, does
//                        the boundary host syncs (per-batch counts). Called
//                        by the scheduler when the selection becomes known.
//   materialize        — once per column, reusing the prepared selection.
//                        Stream-ordered on `stream`; the numeric routes
//                        enqueue and return (documented exceptions: the
//                        shipped launchers' internal syncs block the ISSUING
//                        thread only; the strings route has one
//                        data-dependent D2H sizing the chars buffer).
// Several materializations may be in flight against one prepared_selection —
// it is immutable after build; v1 requires the same stream (or streams the
// caller ordered after the prepare stream).
//
// ORIGIN MODEL. late_mat/column_origin.hpp is the engine-side source of
// truth: a column_origin resolves (generation-checked) to a pinned entry,
// whose per-batch chunks the scan-manager-side resolver renders into the
// batch_source / pinned_table_layout views below. Global row ids are
// PIN-ORDER POSITIONS (a row's index in the concatenation of the entry's
// chunks in emission order — the column_origin.hpp addressing): batch k
// covers [batch_row_start[k], batch_row_start[k+1]); batch-local id =
// gid - batch_row_start[k]; simpatico decode coordinates are batch-local
// (local/1024 chunks, local%1024 offsets — the fused selection_mask
// geometry).

#pragma once

#include "codegen/selection/row_set.hpp"
#include "late_mat/column_origin.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace simpatico {
class compressed_table;
}

namespace sirius::late_mat {

// ── Origin-side views (stub; see header comment) ────────────────────────────

/// One pinned batch of the origin column (device_pin_chunk shape): exactly
/// one of {compressed, uncompressed} is populated.
struct batch_source {
  simpatico::compressed_table const* compressed = nullptr;
  std::size_t column_index = 0;        ///< into compressed->columns
  cudf::column_view uncompressed{};    ///< valid iff compressed == nullptr
  std::int64_t num_rows = 0;
};

/// Column-agnostic layout of the pinned table — shared by prepare_selection
/// across every column materialized from the same origin table.
struct pinned_table_layout {
  std::vector<std::int64_t> batch_rows;       ///< rows per batch, emission order
  std::vector<std::int64_t> batch_row_start;  ///< B+1: exclusive scan of batch_rows,
                                              ///< last = total pinned rows
  std::uint64_t pin_generation = 0;

  /// Build batch_row_start from batch_rows.
  static pinned_table_layout from_batch_rows(std::vector<std::int64_t> rows,
                                             std::uint64_t generation = 0);
};

/// Per-column origin view; batches must be positionally consistent with the
/// layout handed to prepare_selection.
struct pinned_column_view {
  std::vector<batch_source> batches;
  cudf::data_type dtype{cudf::type_id::EMPTY};
  std::uint64_t pin_generation = 0;
};

/// Device list of global row ids (pin-order positions). When sorted_unique
/// is set the output column keeps ascending table order and no restoration
/// gather runs; otherwise materialize returns rows in the given order,
/// duplicates included (gather semantics).
///
/// LIFETIME: the device buffer behind `ids` must remain valid until the LAST
/// materialize against the prepared_selection returns. On the single-batch
/// raw-gather fast path the list is BORROWED (zero prep — the whole point,
/// per the +61 ms q9-nation attribution) and consumed directly as the gather
/// map / canonicalization input at materialize time. The port materializer's
/// pattern (prepare + N materializes inside one prepare_for_processing, the
/// rowid column alive on the batch throughout) satisfies this by
/// construction.
struct row_id_list {
  std::uint64_t const* ids = nullptr;  ///< device
  std::int64_t count = 0;
  bool sorted_unique = false;
};

// ── Prepared selection ───────────────────────────────────────────────────────

/// Immutable after prepare; shared (shared_ptr) across the N columns being
/// materialized from one origin table.
class prepared_selection {
 public:
  struct batch_selection {
    sirius::codegen::owned_chunk_row_set rows;  ///< batch-local chunk-CSR (id-derived
                                                ///< selections; empty for dense/mask)
    rmm::device_buffer local_indices;           ///< int32 x survivors (gather map /
                                                ///< shipped-K4 shape), built at prepare
                                                ///< (empty for dense batches)
    double density = 0.0;                       ///< survivors / batch rows
    bool dense = false;                         ///< whole batch live: deep copy /
                                                ///< full decode, no selection kernels
    /// Fused-capture mask form (annotation kind=mask): the wave-1 buffers ride
    /// here shared, and the compressed decode goes STRAIGHT through the
    /// shipped mask route (decompress_column_compacted) — zero conversion.
    std::shared_ptr<rmm::device_buffer> mask_words;
    std::shared_ptr<rmm::device_buffer> mask_chunk_offsets;
  };

  pinned_table_layout layout;
  std::vector<batch_selection> batches;    ///< parallel to layout.batch_rows
  std::vector<std::int64_t> out_base;      ///< B+1: exclusive scan of per-batch survivors
  std::int64_t total_survivors = 0;
  std::int64_t original_count  = 0;        ///< input id count (== total when sorted_unique)
  rmm::device_buffer restore_rank;         ///< int32 x original_count; empty when the
                                           ///< input was sorted_unique
  [[nodiscard]] bool needs_restore() const noexcept { return restore_rank.size() != 0; }

  // ── Raw-gather fast path (single-batch layouts) ────────────────────────────
  // When the layout has ONE batch, prepare_selection performs NO device work:
  // the id list is borrowed (see row_id_list lifetime note) and, since batch 0
  // starts at global row 0, it IS the batch-local gather map. An uncompressed
  // column then materializes as one direct cudf::gather with the raw u64 map
  // (duplicates and disorder are exactly gather semantics — no sort, no
  // unique, no restore, no host sync). A compressed column still needs the
  // chunk-CSR canonical form: it is built ON FIRST USE via canonical() below
  // (std::call_once — the prepared_selection stays logically immutable and
  // concurrent materializes stay safe) and shared by every later compressed
  // consumer. Multi-batch layouts keep the eager canonical prepare (splitting
  // ids by batch requires ordering them; the 8-bit batch-key partition
  // variant is a flagged follow-up if a multi-batch uncompressed origin ever
  // shows up hot).
  std::uint64_t const* raw_ids = nullptr;  ///< non-null => raw mode
  std::int64_t raw_count = 0;
  bool raw_sorted_unique = false;

  /// The canonical (sorted/bucketed) equivalent of a raw-mode selection,
  /// built on first use on the caller's stream. Only valid in raw mode.
  [[nodiscard]] prepared_selection const& canonical(rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr) const;

 private:
  mutable std::once_flag _canon_once;
  mutable std::shared_ptr<prepared_selection> _canon;
};

/// Build the shared selection state. Performs the boundary host syncs
/// (unique count, per-batch survivor/touched counts) — a few KB of D2H.
/// Throws std::runtime_error on CUDA failure or malformed inputs (id out of
/// range, layout inconsistent).
std::shared_ptr<prepared_selection> prepare_selection(
  pinned_table_layout const& layout,
  row_id_list const& rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/// Build the shared selection state from PER-BATCH selections in the scan
/// annotation contract (late_mat/column_origin.hpp row_selection), parallel
/// to layout.batch_rows. All three forms are consumed:
///   dense   — whole batch live: no selection kernels; materialize deep-copies
///             (uncompressed) or full-decodes (compressed) the batch.
///   mask    — fused wave-1 capture: the shared mask/chunk_offsets buffers are
///             kept and the compressed decode rides the shipped mask route
///             directly; an int32 id expansion is built once at prepare (the
///             gather map for uncompressed / tier_b fallbacks).
///   id_list — batch-local ascending int32 ids: bucketed to the chunk-CSR
///             (u16) exactly like the u64 entry point, minus sort/split.
/// Output row order is ascending pin order (per-batch forms are ascending by
/// contract), so no restoration gather is ever needed on this path.
/// A row_selection whose range is filled must agree with the layout (rows
/// mismatch throws); an all-zero range is accepted (annotation carriers may
/// leave it unset).
std::shared_ptr<prepared_selection> prepare_selection_from_batch(
  pinned_table_layout const& layout,
  std::span<sirius::late_mat::row_selection const> batch_selections,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

// ── Materialization ─────────────────────────────────────────────────────────

/// Materialize one deferred column for the prepared selection: per batch,
/// gather (uncompressed) or row-decode (compressed — sparse K8/K5s/K6s below
/// SIRIUS_LATE_MAT_MASK_SEL, the shipped mask route below
/// SIRIUS_LATE_MAT_DENSE_SEL, full decode + cudf::gather above it and for
/// tier_b plans), then assemble and (if the selection was not sorted_unique)
/// restore caller order with one survivor-sized gather.
/// v1 refusals (throws std::runtime_error): nullable columns, generation
/// mismatch, total survivors >= 2^31.
std::unique_ptr<cudf::column> materialize(pinned_column_view const& origin,
                                          prepared_selection const& sel,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

}  // namespace sirius::late_mat
