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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cudf {
class column;
class column_view;
class table;
}  // namespace cudf

namespace simpatico {
class compressed_table;
}  // namespace simpatico

namespace sirius {

//===----------------------------------------------------------------------===//
// Decoding a compressed chunk under a scan's filter
//===----------------------------------------------------------------------===//
//
// A scan can hand the decompressor its row filter instead of evaluating it
// afterwards. Where a column's compression plan allows it, the filter is
// answered from the compressed form and the surviving rows are written out
// already compacted, so the rows a filter rejects are never fully decoded.
//
// This header is the whole boundary between the scan and that machinery. The
// scan says what it wants (@ref pushdown_request, built from its pushed-down
// filter by @c sirius::op::analyze_scan_filters), the decoder answers with what
// it managed to do (@ref pushdown_outcome). Which parts of the request a given
// chunk can honour is decided per chunk, inside — the scan never inspects a
// compression plan and never names a kernel.
//
// The mechanism is experimental and gated on SIRIUS_EXP_FUSED_SCAN_FILTER;
// with the gate off every request is simply left unapplied, which is
// indistinguishable from an ordinary decompress.

/// Inclusive bounds in the DECODED integer domain — the values a decoder
/// reconstructs: DATE → stored day count, DECIMAL → unscaled integer at the
/// *column's* scale, plain integers as-is. @c lo > @c hi is a provably empty
/// range (e.g. an equality against a constant the column's scale cannot
/// represent) and legitimately selects nothing.
struct decode_range {
  std::int64_t lo = 0;
  std::int64_t hi = 0;
};

/// Type-erased set-membership test for a dynamic join filter: evaluates the
/// published device structure (small in-list / hash set / Bloom) over a decoded
/// key column and returns a BOOL8 keep-mask. The closure co-owns the filter for
/// the call's duration and must enqueue only on the handed stream.
using membership_probe_fn = std::function<std::unique_ptr<cudf::column>(
  cudf::column_view const&, rmm::cuda_stream_view, rmm::device_async_resource_ref)>;

/// One membership test plus the signal used to order it.
///
/// The decoder can only carry a bounded number of filters into one decode, and
/// it keeps a prefix of the list, so the caller ranks by ascending EXPECTED
/// keep-rate — strongest first. Best static signals, in priority order:
///   @c selectivity_rank — 0 = small in-list, 1 = hash in-list set, 2 = Bloom
///   (the set forms are exact; a Bloom filter over-keeps by construction);
///   255 = unknown, ordered last.
///   @c num_keys — build-side key count where the filter exposes it (fewer ⇒
///   stronger); 0 = unknown. Ties keep publication order.
struct membership_probe {
  membership_probe_fn probe;
  std::uint8_t selectivity_rank = 255;
  std::uint64_t num_keys        = 0;
};

/// What one scan asks the decoder to do to one chunk.
///
/// Immutable once built and shared by every batch; per-batch variations (a
/// fresher join-filter set, or dropping the row selection once measured
/// unprofitable) are made by copy — see @ref decompression_pushdown_scan.
struct pushdown_request {
  /// One selected column's share of the filter. Entries are parallel to the
  /// column list the decode is asked for; a shorter vector leaves the tail
  /// unfiltered.
  struct column_entry {
    /// Non-empty ⇒ this column's whole filter is `value IN {…}` over string
    /// constants, and the column is never projected. A dictionary-rooted plan
    /// answers it off its key set, so the column arrives as the BOOL8 ANSWER
    /// rather than its declared type (see @ref pushdown_outcome::predicate_columns).
    std::vector<std::string> equals_any;
    /// Row-restricting bounds on this column, evaluated while it decodes.
    std::optional<decode_range> range;
    /// Join filters to test this column against while it decodes. Dynamic:
    /// snapshotted per batch, since builds publish mid-scan.
    std::vector<membership_probe> membership;

    [[nodiscard]] bool empty() const noexcept
    {
      return equals_any.empty() && !range.has_value() && membership.empty();
    }
  };

  std::vector<column_entry> columns;

  /// Version of the dynamic filter set the @c membership probes were taken
  /// from (0 = none). Echoed back on the outcome so a scan that stopped using
  /// the request can reconsider when a tighter set arrives.
  std::uint64_t membership_generation = 0;

  /// True iff the @c range entries are the scan's WHOLE row-restricting
  /// filter. Only then can a chunk come back needing no filter at all; a
  /// partial request always leaves a residual for the scan to evaluate.
  bool ranges_cover_whole_filter = false;

  /// Set by @ref decompression_pushdown_scan::without_row_selection. Refuses
  /// the compacting decode outright, regardless of what sources this request
  /// still carries: an equality answer alone is sufficient to drive
  /// compaction (its ballot feeds the same combined mask as a range or
  /// membership source), and a later per-batch refresh (join filters
  /// publishing mid-scan) can repopulate @c membership on a copy of this
  /// request after row selection was deliberately dropped. Neither may
  /// silently re-enable compaction, so the refusal is a sticky property of
  /// the request rather than something inferred from which fields are empty
  /// — it survives @ref decompression_pushdown_scan::for_chunk and
  /// @ref decompression_pushdown_scan::with_membership_probes, both of which
  /// copy it along with everything else.
  bool row_selection_disabled = false;

  [[nodiscard]] bool empty() const noexcept;
  /// True iff any entry asks for rows to be DROPPED (as opposed to a column
  /// being answered in place, which changes no row count).
  [[nodiscard]] bool selects_rows() const noexcept;
};

/**
 * @brief What the decode did to a batch, beyond producing its columns.
 *
 * Facts the decoder knows exactly and the scan would otherwise have to infer.
 * Carried as a value so it survives copying and can grow a field without
 * growing a class.
 */
struct pushdown_outcome {
  /// The decode applied the scan's ENTIRE row filter and every column is
  /// compacted to the surviving rows, so the scan can skip filtering
  /// altogether and only project.
  ///
  /// A partially applied request must leave this false: the residual conjuncts
  /// still have to run, and re-checking already-applied ones on the surviving
  /// rows is idempotent.
  bool row_filtered = false;

  /// The decode measured too many surviving rows for compaction to pay for
  /// itself and produced the ordinary full-width columns instead (NOT
  /// row-filtered).
  ///
  /// Selectivity is near-uniform across one scan's batches (chunks are
  /// unclustered), so one such batch predicts the rest: the scan drops the row
  /// selection from its remaining batches and stops paying for the attempt.
  bool selection_unprofitable = false;

  /// Positions in the decoded table delivered as a BOOL8 predicate RESULT
  /// rather than the column's declared type (see
  /// @ref pushdown_request::column_entry::equals_any).
  ///
  /// Reported because the decoder knows it exactly. The alternative — the scan
  /// re-deriving it per batch by testing each candidate column for
  /// type_id::BOOL8 — is only unambiguous while candidates are VARCHAR-only;
  /// the day the substitution covers numeric or boolean equality, a genuine
  /// BOOL8 column becomes indistinguishable from a substituted one.
  std::vector<std::size_t> predicate_columns;

  /// The decode also APPLIED those answers to the rows: every equality it
  /// answered was folded into the row selection, so the surviving rows already
  /// satisfy those conjuncts and the scan can drop them from its residual
  /// rather than AND the answer back in.
  ///
  /// False when the filtering declined and the columns came back through the
  /// plain predicated decode instead — the answers are there, but no row was
  /// dropped for them, so the scan must still evaluate them.
  bool predicates_enforced = false;

  [[nodiscard]] bool any() const noexcept
  {
    return row_filtered || selection_unprofitable || !predicate_columns.empty();
  }
};

/// A decompressed chunk and what its decode did.
struct decompress_result {
  std::unique_ptr<cudf::table> table;
  pushdown_outcome outcome;
};

/**
 * @brief One logical scan's decode-time filtering, as the decoder sees it.
 *
 * Built once per scan from the analysed filter and attached to each compressed
 * batch the scan serves; the batch's converter decodes against it. Held by
 * @c shared_ptr and never mutated, so the per-batch adjustments below hand back
 * a new object rather than editing one that other batches are reading.
 */
class decompression_pushdown_scan {
 public:
  explicit decompression_pushdown_scan(pushdown_request request) : _request(std::move(request)) {}

  [[nodiscard]] pushdown_request const& request() const noexcept { return _request; }

  /// This scan without its row-dropping conjuncts — what to attach to the
  /// remaining batches once a decode has reported
  /// @c pushdown_outcome::selection_unprofitable. Columns answered in place keep
  /// working; only the attempt to compact is given up. Null if nothing is left
  /// to ask for.
  [[nodiscard]] std::shared_ptr<const decompression_pushdown_scan> without_row_selection() const;

  /// This scan with its join filters replaced by a fresher snapshot. @p probes
  /// is parallel to the selected column list.
  [[nodiscard]] std::shared_ptr<const decompression_pushdown_scan> with_membership_probes(
    std::vector<std::vector<membership_probe>> probes, std::uint64_t generation) const;

  /// This scan narrowed to what @p chunk is worth asking: a column is only
  /// answered in place where its compression plan can resolve the predicate
  /// without materialising the column. Null when nothing is left to ask for.
  ///
  /// The row-selecting conjuncts are NOT narrowed here — which of those a chunk
  /// can evaluate is decided inside the decode, where the answer is used.
  [[nodiscard]] std::shared_ptr<const decompression_pushdown_scan> for_chunk(
    simpatico::compressed_table const& chunk, std::span<const std::size_t> selected) const;

  /// What a decode of @p selected columns of @p chunk is expected to do to the
  /// batch size, from host metadata alone (plan-tree walks, no device work).
  /// Used to reserve memory before the decode runs.
  struct compaction_forecast {
    /// The decode is expected to hand back compacted columns.
    bool compacts = false;
    /// The surviving row count is bounded by the decoder's selectivity ceiling
    /// (above it the decode gives up compaction and returns full-width
    /// columns). False when a column is exempt from that ceiling, in which case
    /// the reservation must cover the full width.
    bool survivors_bounded = false;
  };
  [[nodiscard]] compaction_forecast forecast_compaction(
    simpatico::compressed_table const& chunk, std::span<const std::size_t> selected) const;

 private:
  pushdown_request _request;
};

/**
 * @brief Decompress @p selected columns of @p chunk into a cuDF table.
 *
 * With @p scan non-null, as much of its request as this chunk's compression
 * plans allow is applied during the decode; the rest is simply not applied,
 * which is always sound — the request is a conjunction, so an unapplied part
 * only means rows the scan must still reject itself. @ref decompress_result::outcome
 * says what happened. With @p scan null this is an ordinary decompress.
 *
 * Never returns a null table: every way the filtering can decline ends in the
 * plain decode of the same columns.
 */
decompress_result decompress_chunk(simpatico::compressed_table const& chunk,
                                   std::span<const std::size_t> selected,
                                   decompression_pushdown_scan const* scan,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

}  // namespace sirius
