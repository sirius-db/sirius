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

// Pin-time proof that a column's values are distinct across a whole pinned
// table (env gate: SIRIUS_LATE_MAT_PIN_UNIQUE_COLS).
//
// This fact is what lets a ride survive a GROUP BY: if the deferred columns are
// functionally determined by a key that is unique over the pinned table, the
// aggregate can group by the rowid instead of the wide keys and materialize at
// its OUTPUT (one row per group) instead of its INPUT (one row per join match).
// On q10 that is the difference between ~150k gathered rows and tens of
// millions.
//
// The cheap stage is INTEGER-ONLY, but that is a limit of the cheap stage, not
// of the fact: a column it cannot judge (a string, say) is reported UNDECIDED so
// the exact check still gets its turn. `nation.n_name` is exactly that case, and
// it is the column a q10 rider needs proven.
//
// The proof is chunk-local plus a range check, so it costs one sortedness test,
// one run count and one minmax per observed column per chunk, and never holds
// more than the current chunk:
//
//   a column is distinct across the pinned table  IFF
//     every chunk has no nulls, and
//     every chunk's distinct count equals its row count, and
//     the chunks' [min, max] ranges are pairwise disjoint.
//
// The count comes off SORTEDNESS rather than a hash set on purpose: pinned
// chunks are sized in gigabytes, and a hash-based distinct_count over one of
// them overruns cuco's representable extent and fails the whole pin. An
// unsorted chunk is reported UNDECIDED instead, for the exact stage to settle.

//
// Pairwise-disjoint, NOT strictly increasing in emission order: our `orders`
// and `lineitem` files are named `part.0 … part.14` and glob
// lexicographically, so `part.10` is read before `part.2` and any test phrased
// against emission order fails on data that is in fact unique. Disjointedness is
// the property that actually matters and is order-free; with ~15 chunks the
// sort that checks it is free.
//
// ABSENCE OF A FACT MUST READ AS *UNKNOWN*, NEVER AS "NOT UNIQUE" — and a
// false positive is far worse than a missed one. Claiming uniqueness that does
// not hold collapses distinct groups into one, which is wrong answers rather
// than slow ones. So every path that cannot decide (unsupported type, a null,
// an empty selection, a chunk whose column count disagrees with the selection)
// drops the column to "not proven" and the probe never guesses.
//
// The probe observes the table BEFORE narrowing: the fact describes VALUES, and
// same-family narrowing preserves them, so pre-narrow observation is both
// cheaper (native carriers) and equally valid.

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::late_mat {

/// Which pinned columns to observe, from `SIRIUS_LATE_MAT_PIN_UNIQUE_COLS`,
/// positional with @p column_names:
///
///   unset / "" / "0" / "none"  -> nothing observed (the probe is off)
///   "all"                      -> every column (the probe still skips the
///                                 ones it cannot decide, e.g. non-integers)
///   "c_custkey,o_orderkey"     -> exactly those names, case-insensitively
///
/// A name-list is the usual setting: distinct_count over every integer column
/// of a wide fact table is pin-time work nobody asked for, and the columns a
/// group-by-rowid ride can use are known in advance.
[[nodiscard]] std::vector<bool> pin_unique_probe_selection(
  std::span<std::string const> column_names);

/// Row cap for the exact stage (SIRIUS_LATE_MAT_EXACT_MAX_ROWS, default 300M).
///
/// The exact check sorts the whole assembled column, so on a fact table it is
/// pin-time work measured in seconds — for a fact nobody asked for, since the
/// tables whose keys unlock a ride are dimensions. Above the cap the column
/// stays UNKNOWN, which costs an optimization and never an answer.
[[nodiscard]] std::size_t exact_uniqueness_row_cap();

/// What the per-chunk pass concluded about one pinned column. The distinction
/// that matters is @c refused vs @c undecided: a refused column is one the
/// exact check cannot help either (it repeats a value, is nullable, or is not
/// an integer), so running one would be pure pin-time cost, while an undecided
/// column is one whose chunks merely overlap in range — the common case on
/// real pins, and exactly what the exact check is for.
enum class unique_verdict : std::uint8_t {
  not_observed = 0,  ///< the column was not selected for the probe
  refused,           ///< cannot be unique, or cannot be decided by any check here
  undecided,         ///< chunk-distinct everywhere, but the chunk ranges overlap
  proven,            ///< distinct across the whole pinned table
};

/// Exact whole-column check, for the columns the cheap per-chunk pass could not
/// decide: assemble @p chunks into one column, sort it, and count consecutive
/// equal runs (over sorted data that IS the exact distinct count — nothing
/// approximate is ever recorded, since the consumer is a correctness
/// transform). True iff the assembled column holds @p chunks 's rows with no
/// repeat.
///
/// This is not a luxury: on our data the cheap pass decides almost nothing.
/// Pinning SF1000 `customer` produces chunks whose c_custkey ranges OVERLAP —
/// the coalescer interleaves the two files' row groups rather than partitioning
/// the key space — so the range test cannot conclude, even though the column is
/// in fact unique. Cost is one sort of one column, paid once at pin time.
///
/// Works on anything cuDF can sort — strings included, which is what lets a
/// dimension's name column carry a rider's proof.
///
/// Returns nullopt when the check could not run at all (a chunk is nullable or
/// of a type with no ordering, the chunks disagree on type, or the assembled
/// column would exceed a cudf column) — undecidable stays UNKNOWN and must not
/// be confused with a column shown to repeat.
[[nodiscard]] std::optional<bool> exact_distinct_over_chunks(
  std::span<cudf::column_view const> chunks, rmm::cuda_stream_view stream);

/**
 * @brief Accumulates the per-chunk evidence behind a whole-table distinctness proof.
 *
 * One probe per pin, fed every materialized chunk in any order. Columns it
 * cannot decide are dropped as they are observed, so a table whose first chunk
 * disproves every candidate costs nothing on the remaining chunks.
 */
class unique_probe {
 public:
  /// @param selected  positional with the pinned columns; true = observe.
  explicit unique_probe(std::vector<bool> selected);

  /// True while at least one column is still a candidate — the drivers use this
  /// to skip the call (and its stream work) entirely.
  [[nodiscard]] bool active() const noexcept { return _live_candidates > 0; }

  /// Observe one materialized chunk. @p chunk must have one column per entry of
  /// the selection; a mismatch abandons the whole proof (fails closed) rather
  /// than risking a positional misread. Runs on @p stream and synchronizes it
  /// via the scalar reads it performs.
  void observe(cudf::table_view const& chunk, rmm::cuda_stream_view stream);

  /// Per-column verdicts, positional with the selection.
  [[nodiscard]] std::vector<unique_verdict> verdicts() const;

  /// Shorthand for `verdicts()[i] == unique_verdict::proven`.
  [[nodiscard]] std::vector<bool> proven() const;

  /// The proven columns' names, for the attach-by-name path and for logging.
  [[nodiscard]] std::vector<std::string> proven_names(
    std::span<std::string const> column_names) const;

 private:
  /// Value domain of the observed ranges. Wide enough to hold INT64 and UINT64
  /// together, so signed and unsigned columns compare without a special case.
  using value_type = __int128;

  struct range {
    value_type min{0};
    value_type max{0};
  };

  struct column_state {
    bool observed{false};   ///< selected for the probe
    bool candidate{false};  ///< still possibly unique
    /// The cheap stage cannot judge this column (not an integer, or its
    /// min/max could not be read). Not a refusal — the exact check may still
    /// decide it, so the verdict is `undecided` rather than `refused`.
    bool cheap_undecidable{false};
    std::vector<range> ranges;  ///< one per observed non-empty chunk
  };

  std::vector<column_state> _columns;
  std::size_t _live_candidates{0};

  void drop(std::size_t column_pos, std::string_view why);
};

}  // namespace sirius::late_mat
